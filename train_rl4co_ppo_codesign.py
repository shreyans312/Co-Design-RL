"""Design/control co-optimisation with RL4CO's PPO (Cauz et al. framework, PPO controller).

RL4CO's PPO treats one whole solution as a single-step MDP: the "action" is the full
168-hour control trajectory, the reward is the episode return, and the advantage is
(episode return - critic(instance)), with no GAE. Here the "instance" is the sampled design
x ~ p_phi, and the policy is a Gaussian MLP applied at every hour. The design distribution
p_phi is trained with the same REINFORCE + entropy update as train_rl.py / train_sb3_codesign.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from rl4co.data.dataset import TensorDictDataset
from rl4co.models.rl.ppo.ppo import PPO
from tensordict import TensorDict
from torch import nn

sys.path.insert(0, str(Path(__file__).with_name("stochastic-model")))

from config import get_default_config  # noqa: E402
from data import DEFAULT_YEAR, generate_dataset  # noqa: E402
from simulator import EnergySystemSimulator  # noqa: E402
from stochastic import generate_episode_scenario  # noqa: E402

from rl_algorithms import StateEncoder  # noqa: E402
from rl_config import get_rl_config  # noqa: E402
from train_rl import build_system_config  # noqa: E402
from train_sb3_codesign import DesignLearner  # noqa: E402

ACTION_DIM = 2


def mlp(sizes: list[int], activation=nn.Tanh) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class EnergyDesignEnv:
    """Minimal stand-in for an RL4CO env: it only has to produce datasets of (design, seed) instances."""

    name = "energy_codesign"
    dataset_cls = TensorDictDataset

    def __init__(self, learner: DesignLearner, seed: int) -> None:
        self.learner = learner
        self.seed = seed
        self.train_epoch = 0
        self.validation_epoch = 0

    def reset(self, td: TensorDict) -> TensorDict:
        return td

    def dataset(self, batch_size=None, phase: str = "train", filename=None) -> TensorDictDataset:
        count = int(batch_size)
        if phase == "train":
            assert count == self.learner.n, "train_data_size must equal episodes per iteration"
            designs = self.learner.designs
            seeds = self.seed + self.train_epoch * 10_000 + np.arange(count)
            self.train_epoch += 1
        else:
            designs = self.learner.validation_designs(count)
            seeds = self.seed + 1_000_000 + self.validation_epoch * 10_000 + np.arange(count)
            self.validation_epoch += 1
        td = TensorDict(
            {
                "design": torch.as_tensor(np.asarray(designs), dtype=torch.float32),
                "seed": torch.as_tensor(seeds, dtype=torch.long),
                "idx": torch.arange(count),
            },
            batch_size=[count],
        )
        return TensorDictDataset(td)


class GaussianControlPolicy(nn.Module):
    """Hourly Gaussian policy. forward() rolls whole episodes out (no actions given) or
    re-scores stored (action, observation) pairs. Observations are packed after the actions in
    the 'actions' tensor so PPO's replay of stored actions can recompute log-probs without
    re-simulating."""

    def __init__(self, rl_config: dict, data: pd.DataFrame, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.base_config = get_default_config()
        self.horizon = int(self.base_config["simulation"]["episode_hours"])
        self.encoder = StateEncoder(self.base_config, {"observation": rl_config["observation"]})
        self.obs_dim = self.encoder.size
        self.data = data
        self.net = mlp([self.obs_dim] + [hidden] * depth + [ACTION_DIM])
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))
        self.last_train_reward: np.ndarray | None = None
        self.last_train_idx: np.ndarray | None = None

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.net(obs), self.log_std.exp())

    def _rollout(self, td: TensorDict, deterministic: bool, split: str) -> dict:
        designs = td["design"].cpu().numpy()
        seeds = td["seed"].cpu().numpy()
        simulators, states = [], []
        for design, seed in zip(designs, seeds):
            configured = build_system_config(self.base_config, design)
            scenario = generate_episode_scenario(data=self.data, config=configured, split=split, seed=int(seed))
            simulator = EnergySystemSimulator(configured)
            states.append(simulator.reset(scenario))
            simulators.append(simulator)

        observations, actions, log_probs = [], [], []
        for _ in range(self.horizon):
            obs = torch.as_tensor(
                np.stack([self.encoder.encode(state, design) for state, design in zip(states, designs)]),
                dtype=torch.float32,
            )
            distribution = self._distribution(obs)
            action = distribution.mean if deterministic else distribution.sample()
            log_probs.append(distribution.log_prob(action).sum(-1))
            observations.append(obs)
            actions.append(action)
            env_actions = action.clamp(-1.0, 1.0).numpy()
            for i, simulator in enumerate(simulators):
                states[i], _, _, _ = simulator.step(env_actions[i])

        packed = torch.cat([torch.stack(actions, 1), torch.stack(observations, 1)], dim=-1)
        reward = torch.as_tensor([sim.cumulative_reward for sim in simulators], dtype=torch.float32)
        return {"packed": packed, "log_prob": torch.stack(log_probs, 1), "reward": reward}

    def forward(self, td, env=None, phase="train", actions=None, return_entropy=False,
                return_sum_log_likelihood=True, **kwargs) -> dict:
        if actions is None:
            split = "train" if phase == "train" else "validation"
            result = self._rollout(td, deterministic=phase != "train", split=split)
            if phase == "train":
                self.last_train_reward = result["reward"].numpy().copy()
                self.last_train_idx = td["idx"].cpu().numpy().copy()
            log_likelihood = result["log_prob"].sum(-1) if return_sum_log_likelihood else result["log_prob"]
            return {"actions": result["packed"], "log_likelihood": log_likelihood, "reward": result["reward"]}

        stored_actions, obs = actions[..., :ACTION_DIM], actions[..., ACTION_DIM:]
        distribution = self._distribution(obs)
        log_prob = distribution.log_prob(stored_actions).sum(-1)  # [batch, horizon]
        out = {"log_likelihood": log_prob.sum(-1) if return_sum_log_likelihood else log_prob}
        if return_entropy:
            out["entropy"] = distribution.entropy().sum(-1).sum(-1)  # per trajectory
        return out


class DesignCritic(nn.Module):
    """V(x): expected episode return for design x (the 'instance')."""

    def __init__(self, rl_config: dict, hidden: int = 128) -> None:
        super().__init__()
        observation = rl_config["observation"]
        self.register_buffer(
            "scale",
            torch.tensor([observation["pv_design_scale_kwp"], observation["battery_design_scale_kwh"]]),
        )
        self.net = mlp([2, hidden, hidden, 1])

    def forward(self, td: TensorDict) -> torch.Tensor:
        return self.net(td["design"] / self.scale)


class CoDesignPPO(PPO):
    def attach(self, learner: DesignLearner, output_dir: Path, log_every: int, checkpoint_every: int) -> "CoDesignPPO":
        # set after __init__: Lightning deep-copies init args, and the learner holds an autograd graph
        self.learner = learner
        self.output_dir = output_dir
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.last_validation = float("nan")
        return self

    def val_dataloader(self):
        # fresh designs from the current p_phi every validation round
        self.val_dataset = self.env.dataset(self.data_cfg["val_data_size"], phase="val")
        return super().val_dataloader()

    def on_validation_epoch_end(self):
        value = self.trainer.callback_metrics.get("val/reward")
        if value is not None:
            self.last_validation = float(value)

    def on_train_epoch_end(self):
        policy = self.policy
        order = np.argsort(policy.last_train_idx)
        self.learner.returns[:] = policy.last_train_reward[order]
        self.learner.recorded = self.learner.n
        self.learner.update()  # REINFORCE + entropy step on p_phi, then draws the next batch
        self.train_dataset = self.wrap_dataset(self.env.dataset(self.data_cfg["train_data_size"], "train"))

        metrics = self.trainer.callback_metrics
        record = self.learner.history[-1]
        record["validation_mean_reward_chf"] = self.last_validation
        for key in ("loss", "surrogate_loss", "value_loss", "entropy"):
            value = metrics.get(f"train/{key}")
            record[f"ppo_{key}"] = float(value) if value is not None else float("nan")
        iteration = self.learner.iteration
        pd.DataFrame(self.learner.history).to_csv(self.output_dir / "training.csv", index=False)
        if iteration % self.log_every == 0 or iteration == 1:
            print(
                f"rl4co_ppo_codesign iteration {iteration}/{self.learner.iterations} "
                f"reward={record['mean_reward_chf']:.3f} validation={self.last_validation:.3f} "
                f"pv={record['mode_pv_capacity_kwp']:.3f} battery={record['mode_battery_capacity_kwh']:.3f} "
                f"mean_pv={record['mean_pv_capacity_kwp']:.3f} mean_battery={record['mean_battery_capacity_kwh']:.3f} "
                f"surrogate={record['ppo_surrogate_loss']:.3f} value_loss={record['ppo_value_loss']:.3f}",
                flush=True,
            )
        if iteration % self.checkpoint_every == 0:
            torch.save(
                {"policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
                 "design_distribution": self.learner.mixture.state_dict()},
                self.output_dir / "checkpoint.pt",
            )
        super().on_train_epoch_end()


def main() -> None:
    parser = argparse.ArgumentParser(description="RL4CO PPO + design distribution co-optimisation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--episodes-per-iteration", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=8, help="episodes per PPO minibatch")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rl4co_ppo_codesign"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rl_config = get_rl_config("ddpg_lognormal")  # paper's 3-component log-normal design distribution
    rl_config["training"]["iterations"] = args.iterations
    data = generate_dataset(year=DEFAULT_YEAR, days=365, seed=args.seed)

    n = args.episodes_per_iteration
    learner = DesignLearner(rl_config, args.seed, args.iterations, n)
    env = EnergyDesignEnv(learner, args.seed)
    policy = GaussianControlPolicy(rl_config, data)
    critic = DesignCritic(rl_config)
    model = CoDesignPPO(
        env=env, policy=policy, critic=critic,
        clip_range=0.2, ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
        vf_lambda=0.5, entropy_lambda=0.0, normalize_adv=True, max_grad_norm=0.5,
        batch_size=n, val_batch_size=n, train_data_size=n, val_data_size=n, test_data_size=n,
        optimizer="Adam", optimizer_kwargs={"lr": args.learning_rate},
    )
    model.attach(learner, args.output_dir, args.log_every, args.checkpoint_every)
    trainer = Trainer(
        max_epochs=args.iterations, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, check_val_every_n_epoch=args.validation_every,
        reload_dataloaders_every_n_epochs=1,
    )
    total = args.iterations * n * policy.horizon
    print(f"Training {args.iterations} iterations x {n} episodes = {total} timesteps", flush=True)
    trainer.fit(model)
    torch.save(
        {"policy": policy.state_dict(), "critic": critic.state_dict(),
         "design_distribution": learner.mixture.state_dict()},
        args.output_dir / "final.pt",
    )
    print("Done. Final mode design:", learner.mode(), flush=True)


if __name__ == "__main__":
    main()
