import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).with_name("stochastic-model")))

from config import get_default_config, validate_config  # noqa: E402
from cauz_model import generate_cauz_like_dataset  # noqa: E402
from data import DEFAULT_YEAR, generate_dataset, load_dataset  # noqa: E402
from simulator import EnergySystemSimulator  # noqa: E402
from stochastic import generate_episode_scenario  # noqa: E402

from rl_algorithms import StateEncoder  # noqa: E402
from rl_config import COMMON_CONFIG  # noqa: E402

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

CAUZ_DATASET_PATH = Path(__file__).with_name("stochastic-model") / "cauz_dataset.csv"


class EnergySystemEnv(gym.Env):
    """Gymnasium wrapper around EnergySystemSimulator for a fixed design.

    Stable-Baselines3 only learns a control policy for a single, fixed MDP, so
    this environment does not reproduce the repo's design-distribution
    co-optimisation loop (see CoDesignTrainer in train_rl.py) - it fixes the
    PV/battery design and trains SB3's DDPG purely as a control-policy
    baseline, comparable to the paper's "best two-step" scenario.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pv_capacity_kwp: float,
        battery_capacity_kwh: float,
        split: str = "train",
        seed: int = 0,
        dataset_days: int = 365,
        episode_seeds: list[int] | None = None,
        dataset: str = "synthetic",
    ) -> None:
        super().__init__()
        # When given, reset() cycles through these scenario seeds so repeated
        # evaluations score the policy on the same episodes.
        self.episode_seeds = episode_seeds
        self._episode_count = 0

        self.system_config = get_default_config()
        self.system_config["design"]["pv_capacity_kwp"] = float(pv_capacity_kwp)
        self.system_config["design"]["battery_capacity_kwh"] = float(battery_capacity_kwh)
        validate_config(self.system_config)

        self.split = split
        if dataset == "cauz":
            # Reconstructed real building data (see stochastic-model/prepare_cauz_dataset.py).
            self.data = load_dataset(CAUZ_DATASET_PATH)
        elif dataset == "cauz_model":
            # Generator fitted to that data (see stochastic-model/cauz_model.py); a new year per seed.
            self.data = generate_cauz_like_dataset(seed=seed)
        else:
            self.data = generate_dataset(year=DEFAULT_YEAR, days=dataset_days, seed=seed)
        self.design = np.array([pv_capacity_kwp, battery_capacity_kwh], dtype=np.float32)

        observation_config = deepcopy(COMMON_CONFIG["observation"])
        self.encoder = StateEncoder(self.system_config, {"observation": observation_config})

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.encoder.size,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.simulator = EnergySystemSimulator(self.system_config)
        self._last_observation = np.zeros(self.encoder.size, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.episode_seeds is not None:
            episode_seed = self.episode_seeds[self._episode_count % len(self.episode_seeds)]
            self._episode_count += 1
        else:
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        scenario = generate_episode_scenario(
            data=self.data, config=self.system_config, split=self.split, seed=episode_seed
        )
        state = self.simulator.reset(scenario)
        observation = self.encoder.encode(state, self.design)
        self._last_observation = observation
        return observation, {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        next_state, reward, done, transition = self.simulator.step(action)

        # The simulator only ever ends an episode via the fixed 168-hour
        # horizon, never a genuine terminal state - see EnergySystemSimulator.step.
        terminated = False
        truncated = bool(done)

        # There is no real observation beyond the horizon, so reuse the last one
        # as the bootstrap input (as train_rl.py does) instead of an all-zero
        # placeholder that the critic would otherwise bootstrap from.
        if done:
            observation = self._last_observation
        else:
            observation = self.encoder.encode(next_state, self.design)
            self._last_observation = observation

        info = {"cost_chf": transition["total_cost_chf"]}
        return observation, float(reward), terminated, truncated, info


class DDPGSplitLR(DDPG):
    """SB3 DDPG with separate actor/critic learning rates (the repo uses 1e-4 / 1e-3).

    With sign_preserving_noise the exploration noise can never flip the sign of an action
    (charge vs discharge): a noisy value that crosses zero is folded back, |a + noise|, keeping
    the sign of the actor's own output. Actions stay clipped to [-1, 1] as in SB3.
    """

    def __init__(
        self,
        *args,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-3,
        sign_preserving_noise: bool = False,
        **kwargs,
    ):
        self._actor_lr = actor_learning_rate
        self._critic_lr = critic_learning_rate
        self._sign_preserving_noise = sign_preserving_noise
        super().__init__(*args, learning_rate=critic_learning_rate, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if not self._sign_preserving_noise or action_noise is None:
            return super()._sample_action(learning_starts, action_noise, n_envs)
        action, buffer_action = super()._sample_action(learning_starts, None, n_envs)
        if self.num_timesteps < learning_starts:  # warm-up: uniform random actions, no noise
            return action, buffer_action
        sign = np.where(buffer_action >= 0.0, 1.0, -1.0)
        folded = np.clip(sign * np.abs(buffer_action + action_noise()), -1.0, 1.0)
        return self.policy.unscale_action(folded), folded

    def _update_learning_rate(self, optimizers) -> None:
        super()._update_learning_rate(optimizers)
        for group in self.actor.optimizer.param_groups:
            group["lr"] = self._actor_lr
        for group in self.critic.optimizer.param_groups:
            group["lr"] = self._critic_lr


def build_env(
    pv_capacity_kwp: float,
    battery_capacity_kwh: float,
    split: str,
    seed: int,
    dataset_days: int,
    episode_seeds: list[int] | None = None,
    dataset: str = "synthetic",
) -> Monitor:
    env = EnergySystemEnv(
        pv_capacity_kwp=pv_capacity_kwp,
        battery_capacity_kwh=battery_capacity_kwh,
        split=split,
        seed=seed,
        dataset_days=dataset_days,
        episode_seeds=episode_seeds,
        dataset=dataset,
    )
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DDPG (Stable-Baselines3) on the fixed-design energy system.")
    parser.add_argument("--pv-capacity-kwp", type=float, default=6.0)
    parser.add_argument("--battery-capacity-kwh", type=float, default=14.0)
    parser.add_argument("--dataset-days", type=int, default=365)
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "cauz", "cauz_model"),
        default="synthetic",
        help="'cauz' = reconstructed real building data; 'cauz_model' = generator fitted to it "
             "(one year per seed). Both ignore --dataset-days.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="500 iterations x 32 episodes x 168 hours (the repo's custom DDPG schedule) is 2688000.",
    )
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=512)
    parser.add_argument("--exploration-noise-std", type=float, default=0.20)
    parser.add_argument(
        "--sign-preserving-noise",
        action="store_true",
        help="Exploration noise may change an action's size but never its sign (charge/discharge).",
    )
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sb3_ddpg"))
    args = parser.parse_args()

    train_env = build_env(
        pv_capacity_kwp=args.pv_capacity_kwp,
        battery_capacity_kwh=args.battery_capacity_kwh,
        split="train",
        seed=args.seed,
        dataset_days=args.dataset_days,
        dataset=args.dataset,
    )
    eval_env = build_env(
        pv_capacity_kwp=args.pv_capacity_kwp,
        battery_capacity_kwh=args.battery_capacity_kwh,
        split="validation",
        seed=args.seed + 1,
        dataset_days=args.dataset_days,
        episode_seeds=[10_000 + args.seed * 1000 + i for i in range(args.n_eval_episodes)],
        dataset=args.dataset,
    )

    action_noise = NormalActionNoise(
        mean=np.zeros(2), sigma=args.exploration_noise_std * np.ones(2)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = DDPGSplitLR(
        "MlpPolicy",
        train_env,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        learning_starts=args.learning_starts,
        action_noise=action_noise,
        sign_preserving_noise=args.sign_preserving_noise,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.output_dir / "tensorboard"),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output_dir / "best_model"),
        log_path=str(args.output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(str(args.output_dir / "final_model"))


if __name__ == "__main__":
    main()
