import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stochastic-model"))

from rl_config import get_rl_config
from rl_algorithms import DDPGAgent, StateEncoder
from train_rl import build_system_config
from config import get_default_config
from data import generate_dataset
from simulator import EnergySystemSimulator
from stochastic import generate_episode_scenario

dataset = generate_dataset(year=2026, days=365, seed=21)
base_config = get_default_config()  # full-scale: episode_hours=168, unmodified

rl_config = get_rl_config("ddpg_lognormal")
FIXED_DESIGN = np.array([6.6, 4.5])
system_config = build_system_config(base_config, FIXED_DESIGN)

ITERATIONS = 500
EPISODES_PER_ITER = 32
WARMUP_EPISODES = 20
TOTAL_TRAINING_STEPS = ITERATIONS * EPISODES_PER_ITER * base_config["simulation"]["episode_hours"]


# --- Improvement 1: prioritized experience replay -------------------------

class SumTree:
    # Array-backed binary tree: leaves hold priorities, internal nodes hold
    # the sum of their children. Works for any capacity, not just powers of
    # two - node i's children are always 2i and 2i+1, leaves occupy
    # [capacity, 2*capacity).
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[1])

    def max(self) -> float:
        leaves = self.tree[self.capacity : 2 * self.capacity]
        nonzero = leaves[leaves > 0.0]
        return float(nonzero.max()) if len(nonzero) else 1.0

    def update(self, data_index: int, priority: float) -> None:
        tree_index = data_index + self.capacity
        self.tree[tree_index] = priority
        tree_index //= 2
        while tree_index >= 1:
            self.tree[tree_index] = self.tree[2 * tree_index] + self.tree[2 * tree_index + 1]
            tree_index //= 2

    def get(self, value: float) -> int:
        index = 1
        while index < self.capacity:
            left = 2 * index
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = left + 1
        return index - self.capacity


class PrioritizedReplayBuffer:
    # Proportional prioritization (Schaul et al., 2015): transitions with
    # larger critic TD-error are sampled more often, correction for the
    # resulting bias is applied via importance-sampling weights.
    def __init__(
        self,
        capacity: int,
        seed: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1,
        epsilon: float = 1e-3,
    ) -> None:
        self.capacity = capacity
        self.data = [None] * capacity
        self.tree = SumTree(capacity)
        self.size = 0
        self.write_index = 0
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, beta_frames)
        self.epsilon = epsilon
        self.frame = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        max_priority = self.tree.max() if self.size > 0 else 1.0
        self.data[self.write_index] = (
            np.asarray(observation, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_observation, dtype=np.float32),
            bool(terminated),
        )
        self.tree.update(self.write_index, max_priority)
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

        segment = self.tree.total() / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        for slot in range(batch_size):
            low = segment * slot
            high = segment * (slot + 1)
            value = self.rng.uniform(low, high)
            indices[slot] = self.tree.get(value)

        priorities = self.tree.tree[indices + self.capacity]
        probabilities = priorities / self.tree.total()
        weights = (self.size * probabilities) ** (-beta)
        weights = weights / weights.max()

        observations, actions, rewards, next_observations, terminated = zip(
            *(self.data[index] for index in indices)
        )
        return (
            torch.as_tensor(np.stack(observations), dtype=torch.float32),
            torch.as_tensor(np.stack(actions), dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_observations), dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.float32),
            indices,
            torch.as_tensor(weights, dtype=torch.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for index, priority in zip(indices, priorities):
            self.tree.update(int(index), float(priority))


class PERDDPGAgent(DDPGAgent):
    # Same actor/critic/target networks and act() as DDPGAgent - only the
    # replay buffer and the update step (prioritized sampling + IS-weighted
    # critic loss + priority refresh) differ.
    def __init__(self, config: dict, seed: int, beta_frames: int) -> None:
        super().__init__(config, seed)
        algorithm = config["algorithm"]
        self.replay = PrioritizedReplayBuffer(
            capacity=int(algorithm["replay_capacity"]),
            seed=seed,
            beta_frames=beta_frames,
        )

    def update(self) -> dict | None:
        required_samples = max(self.learning_starts, self.minibatch_size)
        if len(self.replay) < required_samples:
            return None

        observations, actions, rewards, next_observations, terminated, indices, weights = self.replay.sample(
            self.minibatch_size
        )
        with torch.no_grad():
            next_actions = self.target_actor(next_observations)
            targets = rewards + self.gamma * (1.0 - terminated) * self.target_critic(
                next_observations, next_actions
            )

        current_values = self.critic(observations, actions)
        td_errors = current_values - targets
        critic_loss = (weights * td_errors.square()).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_norm)
        self.critic_optimizer.step()

        self.replay.update_priorities(indices, td_errors.detach().numpy())

        actor_loss = -self.critic(observations, self.actor(observations)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            for target, current in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.mul_(1.0 - self.target_update_rate)
                target.add_(self.target_update_rate * current)
            for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.mul_(1.0 - self.target_update_rate)
                target.add_(self.target_update_rate * current)

        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}


# --- Improvement 2: warm-start the buffer from the rule-based baseline ----

def rule_based_action(state: dict) -> np.ndarray:
    # Same heuristic as runs/rule_based_baseline_check.py: discharge the
    # battery when load exceeds PV, charge it when PV exceeds load.
    net_load_kw = state["load_kw"] - state["pv_power_kw"]
    battery_action = float(np.clip(net_load_kw / 6.0, -1.0, 1.0))
    if state["ev_present"]:
        ev_action = float(np.clip(net_load_kw / 5.0, -1.0, 1.0))
    else:
        ev_action = 0.0
    return np.array([battery_action, ev_action], dtype=np.float32)


agent = PERDDPGAgent(rl_config, seed=99, beta_frames=TOTAL_TRAINING_STEPS)
encoder = StateEncoder(system_config, rl_config)

print(f"Warming up replay buffer with {WARMUP_EPISODES} rule-based episodes...", flush=True)
for episode in range(WARMUP_EPISODES):
    seed = 900_000 + episode
    scenario = generate_episode_scenario(data=dataset, config=system_config, split="train", seed=seed)
    simulator = EnergySystemSimulator(system_config)
    state = simulator.reset(scenario)
    while not simulator.done:
        obs = encoder.encode(state, FIXED_DESIGN)
        action = rule_based_action(state)
        next_state, reward, done, transition = simulator.step(action)
        next_obs = obs if done else encoder.encode(next_state, FIXED_DESIGN)
        agent.observe(obs, action, reward, next_obs, False)
        state = next_state
print(f"Replay buffer warm-started with {len(agent.replay)} rule-based transitions.", flush=True)

records = []
for iteration in range(ITERATIONS):
    iter_rewards = []
    for episode in range(EPISODES_PER_ITER):
        seed = iteration * 1000 + episode
        scenario = generate_episode_scenario(data=dataset, config=system_config, split="train", seed=seed)
        simulator = EnergySystemSimulator(system_config)
        state = simulator.reset(scenario)
        while not simulator.done:
            obs = encoder.encode(state, FIXED_DESIGN)
            action = agent.act(obs, explore=True)
            next_state, reward, done, transition = simulator.step(action)
            next_obs = obs if done else encoder.encode(next_state, FIXED_DESIGN)
            agent.observe(obs, action, reward, next_obs, False)
            agent.update()
            state = next_state
        iter_rewards.append(simulator.cumulative_reward)
    records.append(float(np.mean(iter_rewards)))
    if (iteration + 1) % 5 == 0:
        print(
            f"iteration {iteration + 1:>3d}/{ITERATIONS}  mean_train_reward={records[-1]:.3f}",
            flush=True,
        )

print()
print("First 10 iterations avg reward:", np.mean(records[:10]))
print("Last 10 iterations avg reward:", np.mean(records[-10:]))

# Fair, clean comparison: deterministic policy (no exploration noise), same
# 100 episodes/seeds used for the rule-based baseline.
print()
print("Running deterministic evaluation (explore=False), same 100 seeds as rule-based baseline...")
eval_rewards = []
for episode in range(100):
    seed = episode
    scenario = generate_episode_scenario(data=dataset, config=system_config, split="train", seed=seed)
    simulator = EnergySystemSimulator(system_config)
    state = simulator.reset(scenario)
    while not simulator.done:
        obs = encoder.encode(state, FIXED_DESIGN)
        action = agent.act(obs, explore=False)
        next_state, reward, done, transition = simulator.step(action)
        state = next_state
    eval_rewards.append(simulator.cumulative_reward)

eval_rewards = np.array(eval_rewards)
print(f"Deterministic DDPG policy over 100 episodes:")
print(f"mean reward: {eval_rewards.mean():.3f}  std: {eval_rewards.std():.3f}  min: {eval_rewards.min():.3f}  max: {eval_rewards.max():.3f}")
