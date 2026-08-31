from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


class StateEncoder:
    def __init__(self, system_config: dict, rl_config: dict) -> None:
        observation = rl_config["observation"]
        self.load_scale = float(observation["load_scale_kw"])
        self.price_scale = float(observation["price_scale_chf_per_kwh"])
        self.pv_design_scale = float(observation["pv_design_scale_kwp"])
        self.battery_design_scale = float(observation["battery_design_scale_kwh"])
        self.ev_capacity = float(system_config["ev"]["capacity_kwh"])
        self.size = int(observation["size"])

    def encode(self, state: dict, design: np.ndarray) -> np.ndarray:
        # Scale simulator state for the networks
        design = np.asarray(design, dtype=np.float32)
        if design.shape != (2,):
            raise ValueError("design must contain PV and battery capacities")

        day = state["timestamp"].dayofyear - 1
        battery_capacity = max(float(design[1]), 1e-8)
        encoded = np.array(
            [
                state["hour"] / 23.0,
                day / 364.0,
                state["battery_soc_kwh"] / battery_capacity,
                state["pv_power_kw"] / self.pv_design_scale,
                state["load_kw"] / self.load_scale,
                state["import_price"] / self.price_scale,
                state["export_price"] / self.price_scale,
                float(state["ev_present"]),
                state["ev_soc_kwh"] / self.ev_capacity,
                design[0] / self.pv_design_scale,
                design[1] / self.battery_design_scale,
            ],
            dtype=np.float32,
        )
        if encoded.shape != (self.size,):
            raise RuntimeError("encoded observation has the wrong size")
        if not np.isfinite(encoded).all():
            raise ValueError("encoded observation contains non-finite values")
        return encoded


def _activation(name: str) -> type[nn.Module]:
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unknown activation '{name}'")


def _mlp(input_size: int, hidden_sizes: tuple[int, ...], output_size: int, activation_name: str) -> nn.Sequential:
    activation = _activation(activation_name)
    layers = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(previous_size, hidden_size), activation()))
        previous_size = hidden_size
    layers.append(nn.Linear(previous_size, output_size))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, config: dict) -> None:
        super().__init__()
        hidden_sizes = tuple(config["hidden_sizes"])
        activation = config["activation"]
        self.actor = _mlp(observation_size, hidden_sizes, action_size, activation)
        self.critic = _mlp(observation_size, hidden_sizes, 1, activation)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        means = self.actor(observations)
        standard_deviations = torch.exp(self.log_std).expand_as(means)
        return torch.distributions.Normal(means, standard_deviations)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)


def _squashed_log_prob(distribution: torch.distributions.Normal, latent_actions: torch.Tensor) -> torch.Tensor:
    # Correct probabilities after tanh squashing
    actions = torch.tanh(latent_actions)
    correction = torch.log(1.0 - actions.square() + 1e-6).sum(dim=-1)
    return distribution.log_prob(latent_actions).sum(dim=-1) - correction


class PPOAgent:
    def __init__(self, config: dict, seed: int) -> None:
        torch.manual_seed(seed)
        algorithm = config["algorithm"]
        self.network = ActorCritic(config["observation"]["size"], 2, algorithm)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=algorithm["learning_rate"])
        self.generator = torch.Generator().manual_seed(seed)
        self.gamma = float(algorithm["discount_factor"])
        self.gae_lambda = float(algorithm["gae_lambda"])
        self.clip_ratio = float(algorithm["clip_ratio"])
        self.value_coefficient = float(algorithm["value_coefficient"])
        self.entropy_coefficient = float(algorithm["entropy_coefficient"])
        self.epochs = int(algorithm["epochs"])
        self.minibatch_size = int(algorithm["minibatch_size"])
        self.gradient_clip_norm = float(config["training"]["gradient_clip_norm"])

    def act(self, observation: np.ndarray, deterministic: bool = False) -> dict:
        observations = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution = self.network.distribution(observations)
            latent_action = distribution.mean if deterministic else distribution.sample()
            action = torch.tanh(latent_action)
            log_prob = _squashed_log_prob(distribution, latent_action)
            value = self.network.value(observations)
        return {
            "action": action.squeeze(0).numpy(),
            "latent_action": latent_action.squeeze(0).numpy(),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        }

    def value(self, observation: np.ndarray) -> float:
        observations = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self.network.value(observations).item())

    def _advantages(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # Estimate discounted advantages with GAE
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)
        values = torch.as_tensor(batch["values"], dtype=torch.float32)
        next_values = torch.as_tensor(batch["next_values"], dtype=torch.float32)
        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = torch.zeros_like(rewards)
        running_advantage = torch.tensor(0.0)
        for index in range(len(rewards) - 1, -1, -1):
            running_advantage = deltas[index] + (
                self.gamma * self.gae_lambda * (1.0 - dones[index]) * running_advantage
            )
            advantages[index] = running_advantage
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages, returns

    def update(self, batch: dict) -> dict:
        observations = torch.as_tensor(batch["observations"], dtype=torch.float32)
        latent_actions = torch.as_tensor(batch["latent_actions"], dtype=torch.float32)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32)
        old_values = torch.as_tensor(batch["values"], dtype=torch.float32)
        advantages, returns = self._advantages(batch)
        batch_size = len(observations)
        metrics = []

        # Reuse rollout data for clipped PPO updates
        for _ in range(self.epochs):
            permutation = torch.randperm(batch_size, generator=self.generator)
            for start in range(0, batch_size, self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                distribution = self.network.distribution(observations[indices])
                log_probs = _squashed_log_prob(distribution, latent_actions[indices])
                ratios = torch.exp(log_probs - old_log_probs[indices])
                unclipped = ratios * advantages[indices]
                clipped = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[
                    indices
                ]
                policy_loss = -torch.minimum(unclipped, clipped).mean()

                values = self.network.value(observations[indices])
                clipped_values = old_values[indices] + torch.clamp(
                    values - old_values[indices],
                    -self.clip_ratio,
                    self.clip_ratio,
                )
                value_loss = 0.5 * torch.maximum(
                    (values - returns[indices]).square(),
                    (clipped_values - returns[indices]).square(),
                ).mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                metrics.append((policy_loss.item(), value_loss.item(), entropy.item()))

        metric_array = np.asarray(metrics, dtype=float)
        return {
            "policy_loss": float(metric_array[:, 0].mean()),
            "value_loss": float(metric_array[:, 1].mean()),
            "action_entropy": float(metric_array[:, 2].mean()),
        }


class DDPGActor(nn.Module):
    def __init__(self, observation_size: int, config: dict) -> None:
        super().__init__()
        self.network = _mlp(observation_size, tuple(config["hidden_sizes"]), 2, config["activation"])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observations))


class DDPGCritic(nn.Module):
    def __init__(self, observation_size: int, config: dict) -> None:
        super().__init__()
        self.network = _mlp(observation_size + 2, tuple(config["hidden_sizes"]), 1, config["activation"])

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observations, actions), dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.storage = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.storage.append(
            (
                np.asarray(observation, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_observation, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        observations, actions, rewards, next_observations, dones = zip(*(self.storage[index] for index in indices))
        return (
            torch.as_tensor(np.stack(observations), dtype=torch.float32),
            torch.as_tensor(np.stack(actions), dtype=torch.float32),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.stack(next_observations), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
        )


class DDPGAgent:
    def __init__(self, config: dict, seed: int) -> None:
        torch.manual_seed(seed)
        algorithm = config["algorithm"]
        observation_size = int(config["observation"]["size"])
        self.actor = DDPGActor(observation_size, algorithm)
        self.critic = DDPGCritic(observation_size, algorithm)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=algorithm["actor_learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=algorithm["critic_learning_rate"])
        self.replay = ReplayBuffer(algorithm["replay_capacity"], seed)
        self.rng = np.random.default_rng(seed)
        self.gamma = float(algorithm["discount_factor"])
        self.target_update_rate = float(algorithm["target_update_rate"])
        self.minibatch_size = int(algorithm["minibatch_size"])
        self.learning_starts = int(algorithm["learning_starts"])
        self.exploration_noise_std = float(algorithm["exploration_noise_std"])
        self.gradient_clip_norm = float(config["training"]["gradient_clip_norm"])

    def act(self, observation: np.ndarray, explore: bool = True) -> np.ndarray:
        observations = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(observations).squeeze(0).numpy()
        if explore:
            action = action + self.rng.normal(0.0, self.exploration_noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def observe(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(observation, action, reward, next_observation, done)

    def update(self) -> dict | None:
        required_samples = max(self.learning_starts, self.minibatch_size)
        if len(self.replay) < required_samples:
            return None

        observations, actions, rewards, next_observations, dones = self.replay.sample(self.minibatch_size)
        # Build critic targets with target networks
        with torch.no_grad():
            next_actions = self.target_actor(next_observations)
            targets = rewards + self.gamma * (1.0 - dones) * self.target_critic(
                next_observations, next_actions
            )

        critic_loss = functional.mse_loss(self.critic(observations, actions), targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_norm)
        self.critic_optimizer.step()

        actor_loss = -self.critic(observations, self.actor(observations)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
        self.actor_optimizer.step()

        # Move target networks toward learned networks
        with torch.no_grad():
            for target, current in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.mul_(1.0 - self.target_update_rate)
                target.add_(self.target_update_rate * current)
            for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.mul_(1.0 - self.target_update_rate)
                target.add_(self.target_update_rate * current)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
        }


def create_agent(config: dict, seed: int) -> PPOAgent | DDPGAgent:
    if config["algorithm"]["name"] == "ppo":
        return PPOAgent(config=config, seed=seed)
    if config["algorithm"]["name"] == "ddpg":
        return DDPGAgent(config=config, seed=seed)
    raise ValueError(f"Unknown algorithm '{config['algorithm']['name']}'")
