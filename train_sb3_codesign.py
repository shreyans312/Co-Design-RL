import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from torch import nn

sys.path.insert(0, str(Path(__file__).with_name("stochastic-model")))

from config import get_default_config  # noqa: E402
from data import DEFAULT_YEAR, generate_dataset  # noqa: E402
from simulator import EnergySystemSimulator  # noqa: E402
from stochastic import generate_episode_scenario  # noqa: E402

from design_distributions import create_design_distribution  # noqa: E402
from rl_algorithms import StateEncoder  # noqa: E402
from rl_config import get_rl_config  # noqa: E402
from train_rl import build_system_config  # noqa: E402
from train_sb3_ddpg import DDPGSplitLR  # noqa: E402


class DesignLearner:
    """Design distribution p_phi(x) trained with REINFORCE + entropy, as in train_rl.py.

    One iteration = `episodes_per_iteration` episodes. Designs are drawn as a batch (the
    log-prob graph is kept), each finished episode reports its return, and once the batch
    is complete p_phi is updated and a new batch is drawn.
    """

    def __init__(self, rl_config: dict, seed: int, iterations: int, episodes_per_iteration: int,
                 warmup_iterations: int = 0) -> None:
        self.rl_config = rl_config
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.n = episodes_per_iteration
        self.mixture = create_design_distribution(rl_config, seed + 1)
        parameters = [p for p in self.mixture.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=rl_config["design_distribution"]["learning_rate"])
        self.validation_generator = torch.Generator().manual_seed(seed + 2)
        self.iteration = 0
        self.history: list[dict] = []
        self._draw()

    def _draw(self) -> None:
        self.batch = self.mixture.sample(self.n)
        self.designs = self.batch["designs"].numpy()
        self.returns = np.full(self.n, np.nan)
        self.next_index = 0
        self.recorded = 0

    def entropy_weight(self, iteration: int) -> float:
        distribution = self.rl_config["design_distribution"]
        weight = float(distribution["entropy_weight"])
        zero_fraction = float(distribution["entropy_zero_fraction"])
        if weight == 0.0 or zero_fraction == 0.0:
            return 0.0
        return weight * max(0.0, 1.0 - iteration / (self.iterations * zero_fraction))

    def next_design(self) -> tuple[int, np.ndarray]:
        index = self.next_index
        self.next_index += 1
        return index, self.designs[index]

    def record(self, index: int, episode_return: float) -> None:
        self.returns[index] = episode_return
        self.recorded += 1

    def ready(self) -> bool:
        return self.recorded >= self.n

    def validation_designs(self, count: int) -> np.ndarray:
        with torch.no_grad():
            return self.mixture.sample(count, generator=self.validation_generator)["designs"].numpy()

    def mode(self) -> np.ndarray:
        return self.mixture.mode().numpy()

    def update(self) -> None:
        returns = torch.as_tensor(self.returns, dtype=torch.float32)
        if self.rl_config["training"]["normalize_design_returns"]:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        log_prob = self.batch["log_prob"]
        entropy_weight = self.entropy_weight(self.iteration)
        loss = -(log_prob * returns.detach()).mean() + entropy_weight * log_prob.mean()
        if self.iteration >= self.warmup_iterations:  # design stays frozen during warmup
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.mixture.parameters(), self.rl_config["training"]["gradient_clip_norm"])
            self.optimizer.step()

        mode = self.mode()
        self.iteration += 1
        self.history.append(
            {
                "iteration": self.iteration,
                "mean_reward_chf": float(np.mean(self.returns)),
                "std_reward_chf": float(np.std(self.returns)),
                "mean_pv_capacity_kwp": float(self.designs[:, 0].mean()),
                "mean_battery_capacity_kwh": float(self.designs[:, 1].mean()),
                "mode_pv_capacity_kwp": float(mode[0]),
                "mode_battery_capacity_kwh": float(mode[1]),
                "design_loss": float(loss.item()),
                "design_entropy": float(-log_prob.detach().mean().item()),
                "design_entropy_weight": entropy_weight,
            }
        )
        self._draw()


class CoDesignEnv(gym.Env):
    """Each reset() draws a design x ~ p_phi; x is appended to the observation (as in the paper)."""

    metadata = {"render_modes": []}

    def __init__(self, learner: DesignLearner | None, rl_config: dict, data: pd.DataFrame, split: str, seed: int,
                 reward_scale: float = 1.0) -> None:
        super().__init__()
        self.reward_scale = reward_scale
        self.learner = learner
        self.data = data
        self.split = split
        self.base_config = get_default_config()
        self.encoder = StateEncoder(self.base_config, {"observation": rl_config["observation"]})
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.encoder.size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._design = np.zeros(2, dtype=np.float32)
        self._design_index = 0
        self._last_observation = np.zeros(self.encoder.size, dtype=np.float32)
        self.simulator = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options is not None and "design" in options:
            design, self._design_index = np.asarray(options["design"], dtype=np.float32), -1
        else:
            if self.learner.ready():
                self.learner.update()
            self._design_index, design = self.learner.next_design()
        self._design = np.asarray(design, dtype=np.float32)
        episode_seed = options["seed"] if options is not None and "seed" in options else int(self._rng.integers(0, 2**31 - 1))
        configured = build_system_config(self.base_config, self._design)
        scenario = generate_episode_scenario(data=self.data, config=configured, split=self.split, seed=episode_seed)
        self.simulator = EnergySystemSimulator(configured)
        state = self.simulator.reset(scenario)
        self._last_observation = self.encoder.encode(state, self._design)
        return self._last_observation, {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        next_state, reward, done, _ = self.simulator.step(action)
        if done:
            # Fixed 168 h horizon is truncation, not a terminal state: reuse the last
            # observation so the critic bootstraps from a real state.
            observation = self._last_observation
            if self._design_index >= 0:
                self.learner.record(self._design_index, float(self.simulator.cumulative_reward))
        else:
            observation = self.encoder.encode(next_state, self._design)
            self._last_observation = observation
        return observation, float(reward) * self.reward_scale, False, bool(done), {}


class IterationCallback(BaseCallback):
    """After each design-learning iteration: validate, log, checkpoint."""

    def __init__(self, learner, validation_env, output_dir: Path, validation_episodes: int,
                 validation_every: int, log_every: int, checkpoint_every: int, seed: int) -> None:
        super().__init__()
        self.learner = learner
        self.validation_env = validation_env
        self.output_dir = output_dir
        self.validation_episodes = validation_episodes
        self.validation_every = validation_every
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.seed = seed
        self.seen = 0

    def _validate(self, iteration: int) -> float:
        designs = self.learner.validation_designs(self.validation_episodes)
        rewards = []
        for episode, design in enumerate(designs):
            seed = self.seed + 1_000_000 + iteration * 10_000 + episode
            obs, _ = self.validation_env.reset(options={"design": design, "seed": seed})
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, _, truncated, _ = self.validation_env.step(action)
                done = truncated
            rewards.append(self.validation_env.simulator.cumulative_reward)
        return float(np.mean(rewards))

    def _on_step(self) -> bool:
        while self.seen < self.learner.iteration:
            self.seen += 1
            record = self.learner.history[self.seen - 1]
            if self.seen % self.validation_every == 0 or self.seen == 1:
                record["validation_mean_reward_chf"] = self._validate(self.seen)
            values = self.model.logger.name_to_value
            record["actor_loss"] = float(values.get("train/actor_loss", float("nan")))
            record["critic_loss"] = float(values.get("train/critic_loss", float("nan")))
            pd.DataFrame(self.learner.history).to_csv(self.output_dir / "training.csv", index=False)
            if self.seen % self.log_every == 0 or self.seen == 1:
                print(
                    f"sb3_codesign iteration {self.seen}/{self.learner.iterations} "
                    f"reward={record['mean_reward_chf']:.3f} "
                    f"validation={record.get('validation_mean_reward_chf', float('nan')):.3f} "
                    f"pv={record['mode_pv_capacity_kwp']:.3f} battery={record['mode_battery_capacity_kwh']:.3f} "
                    f"mean_pv={record['mean_pv_capacity_kwp']:.3f} mean_battery={record['mean_battery_capacity_kwh']:.3f} "
                    f"actor_loss={record['actor_loss']:.2f} critic_loss={record['critic_loss']:.3f}",
                    flush=True,
                )
            if self.seen % self.checkpoint_every == 0:
                self.model.save(str(self.output_dir / "model"))
                torch.save(self.learner.mixture.state_dict(), self.output_dir / "design_distribution.pt")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="SB3 DDPG + design distribution co-optimisation (Cauz et al.).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--episodes-per-iteration", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration-noise-std", type=float, default=0.20)
    parser.add_argument("--design-learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--init-log-mean-low", type=float, default=0.0)
    parser.add_argument("--init-log-mean-high", type=float, default=1.0)
    parser.add_argument("--init-log-std", type=float, default=1.0)
    parser.add_argument("--design-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--reward-scale", type=float, default=0.01, help="scales rewards seen by SB3 only")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sb3_ddpg_codesign_stable"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rl_config = get_rl_config("ddpg_lognormal")
    rl_config["training"]["iterations"] = args.iterations
    # Stability settings (paper/README-style narrow init, no upper cap on designs).
    rl_config["training"]["normalize_design_returns"] = True
    design = rl_config["design_distribution"]
    design["initial_parameter_low"] = (args.init_log_mean_low,) * 2
    design["initial_parameter_high"] = (args.init_log_mean_high,) * 2
    design["initial_std"] = args.init_log_std
    design["learning_rate"] = args.design_learning_rate
    design["entropy_weight"] = args.entropy_weight
    data = generate_dataset(year=DEFAULT_YEAR, days=365, seed=args.seed)

    warmup = int(round(args.iterations * args.design_warmup_fraction))
    learner = DesignLearner(rl_config, args.seed, args.iterations, args.episodes_per_iteration, warmup)
    train_env = Monitor(CoDesignEnv(learner, rl_config, data, "train", args.seed, args.reward_scale))
    validation_env = CoDesignEnv(None, rl_config, data, "validation", args.seed + 1)

    algorithm = rl_config["algorithm"]
    model = DDPGSplitLR(
        "MlpPolicy",
        train_env,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        buffer_size=algorithm["replay_capacity"],
        batch_size=algorithm["minibatch_size"],
        tau=algorithm["target_update_rate"],
        gamma=algorithm["discount_factor"],
        learning_starts=algorithm["learning_starts"],
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=args.exploration_noise_std * np.ones(2)),
        policy_kwargs={"net_arch": list(algorithm["hidden_sizes"])},
        seed=args.seed,
        verbose=0,
    )
    callback = IterationCallback(
        learner, validation_env, args.output_dir, args.validation_episodes,
        args.validation_every, args.log_every, args.checkpoint_every, args.seed,
    )
    episode_hours = get_default_config()["simulation"]["episode_hours"]
    total_timesteps = args.iterations * args.episodes_per_iteration * episode_hours
    print(f"Training {args.iterations} iterations x {args.episodes_per_iteration} episodes = {total_timesteps} timesteps", flush=True)
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=10**9)
    model.save(str(args.output_dir / "final_model"))
    torch.save(learner.mixture.state_dict(), args.output_dir / "design_distribution.pt")
    print("Done. Final mode design:", learner.mode(), flush=True)


if __name__ == "__main__":
    main()
