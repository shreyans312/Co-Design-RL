import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).with_name("stochastic-model")))

from config import get_default_config, validate_config
from data import DEFAULT_YEAR, generate_dataset, validate_dataset
from design_distributions import create_design_distribution
from rl_algorithms import DDPGAgent, PPOAgent, StateEncoder, create_agent
from rl_config import VARIANTS, get_rl_config, validate_rl_config
from simulator import EnergySystemSimulator
from stochastic import generate_episode_scenario


def build_system_config(system_config: dict, design: np.ndarray) -> dict:
    design = np.asarray(design, dtype=float)
    if design.shape != (2,):
        raise ValueError("design must contain PV and battery capacities")
    if not np.isfinite(design).all():
        raise ValueError("design values must be finite")
    if (design < 0.0).any():
        raise ValueError("design values must be non-negative")

    configured = deepcopy(system_config)
    configured["design"]["pv_capacity_kwp"] = float(design[0])
    configured["design"]["battery_capacity_kwh"] = float(design[1])
    validate_config(configured)
    return configured


def _mean_metrics(metrics: list[dict]) -> dict:
    if not metrics:
        return {}
    keys = sorted(set().union(*(metric.keys() for metric in metrics)))
    return {
        key: float(np.mean([metric[key] for metric in metrics if key in metric]))
        for key in keys
    }


class CoDesignTrainer:
    def __init__(self, system_config: dict, data: pd.DataFrame, rl_config: dict, seed: int | None = None) -> None:
        validate_config(system_config)
        validate_dataset(data)
        validate_rl_config(rl_config)
        self.system_config = deepcopy(system_config)
        self.data = data.copy()
        self.rl_config = deepcopy(rl_config)
        self.seed = int(rl_config["seed"] if seed is None else seed)
        self.agent = create_agent(self.rl_config, self.seed)
        self.design_distribution = create_design_distribution(self.rl_config, self.seed + 1)
        parameters = [
            parameter for parameter in self.design_distribution.parameters() if parameter.requires_grad
        ]
        self.design_optimizer = torch.optim.Adam(
            parameters, lr=self.rl_config["design_distribution"]["learning_rate"]
        )
        self.encoder = StateEncoder(self.system_config, self.rl_config)
        self.validation_design_generator = torch.Generator().manual_seed(self.seed + 2)
        self.pruning_design_generator = torch.Generator().manual_seed(self.seed + 3)
        self.completed_iterations = 0
        self.distribution_history = []
        self.training_samples = []
        self._record_distribution(iteration=0)

    def _record_distribution(self, iteration: int) -> None:
        # Save mixture state for convergence plots
        parameters = self.design_distribution.component_parameters()
        parameter_space = "physical" if self.design_distribution.name == "gaussian" else "log"
        for component in range(self.design_distribution.components):
            self.distribution_history.append(
                {
                    "iteration": iteration,
                    "model": self.rl_config["model_name"],
                    "distribution": self.design_distribution.name,
                    "parameter_space": parameter_space,
                    "component": component,
                    "active": bool(parameters["active"][component]),
                    "weight": float(parameters["weights"][component]),
                    "pv_location": float(parameters["locations"][component, 0]),
                    "pv_scale": float(parameters["scales"][component, 0]),
                    "battery_location": float(parameters["locations"][component, 1]),
                    "battery_scale": float(parameters["scales"][component, 1]),
                }
            )

    def _set_learning_rates(self, iteration: int) -> None:
        if not self.rl_config["training"]["linear_learning_rate_decay"]:
            return
        iterations = self.rl_config["training"]["iterations"]
        fraction = max(0.0, 1.0 - iteration / iterations)
        design_learning_rate = self.rl_config["design_distribution"]["learning_rate"] * fraction
        for group in self.design_optimizer.param_groups:
            group["lr"] = design_learning_rate
        if isinstance(self.agent, PPOAgent):
            policy_learning_rate = self.rl_config["algorithm"]["learning_rate"] * fraction
            for group in self.agent.optimizer.param_groups:
                group["lr"] = policy_learning_rate

    def _episode_seed(self, iteration: int, episode: int, split: str, purpose: int = 0) -> int:
        split_offset = 0 if split == "train" else 1_000_000
        return self.seed + split_offset + purpose + iteration * 10_000 + episode

    def _run_episode(
        self,
        design: np.ndarray,
        split: str,
        seed: int,
        training: bool,
        stochastic_policy: bool,
    ) -> dict:
        # Run one scenario and collect learning data
        configured = build_system_config(self.system_config, design)
        scenario = generate_episode_scenario(data=self.data, config=configured, split=split, seed=seed)
        simulator = EnergySystemSimulator(configured)
        state = simulator.reset(scenario)
        rollout = {
            "observations": [],
            "latent_actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "next_values": [],
        }
        update_metrics = []
        max_power_balance_error_kw = 0.0

        while not simulator.done:
            observation = self.encoder.encode(state, design)
            if isinstance(self.agent, PPOAgent):
                decision = self.agent.act(observation, deterministic=not stochastic_policy)
                action = decision["action"]
            else:
                action = self.agent.act(observation, explore=training and stochastic_policy)

            next_state, reward, done, transition = simulator.step(action)
            max_power_balance_error_kw = max(max_power_balance_error_kw, abs(transition["power_balance_error_kw"]))
            next_observation = (
                np.zeros(self.encoder.size, dtype=np.float32)
                if done
                else self.encoder.encode(next_state, design)
            )

            if training and isinstance(self.agent, PPOAgent):
                rollout["observations"].append(observation)
                rollout["latent_actions"].append(decision["latent_action"])
                rollout["log_probs"].append(decision["log_prob"])
                rollout["values"].append(decision["value"])
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)
                rollout["next_values"].append(0.0 if done else self.agent.value(next_observation))
            elif training and isinstance(self.agent, DDPGAgent):
                self.agent.observe(observation, action, reward, next_observation, done)
                metrics = self.agent.update()
                if metrics is not None:
                    update_metrics.append(metrics)

            state = next_state

        return {
            "reward_chf": float(simulator.cumulative_reward),
            "cost_chf": float(simulator.cumulative_cost_chf),
            "max_power_balance_error_kw": float(max_power_balance_error_kw),
            "rollout": rollout,
            "update_metrics": _mean_metrics(update_metrics),
        }

    def _combine_rollouts(self, episodes: list[dict]) -> dict:
        keys = episodes[0]["rollout"].keys()
        return {
            key: np.asarray([value for episode in episodes for value in episode["rollout"][key]])
            for key in keys
        }

    def _schedule(self) -> dict:
        # Split training into warmup learning and fixed design phases
        training = self.rl_config["training"]
        iterations = int(training["iterations"])
        warmup = int(round(iterations * training["design_warmup_fraction"]))
        fixed = int(round(iterations * training["fixed_design_fraction"]))
        fixed_start = iterations - fixed
        prune_fraction = float(training["prune_interval_fraction"])
        prune_interval = int(round(iterations * prune_fraction)) if prune_fraction > 0.0 else 0
        return {
            "warmup": warmup,
            "fixed_start": fixed_start,
            "prune_interval": max(1, prune_interval) if prune_interval else 0,
        }

    def _entropy_weight(self, iteration: int) -> float:
        distribution = self.rl_config["design_distribution"]
        initial_weight = float(distribution["entropy_weight"])
        zero_fraction = float(distribution["entropy_zero_fraction"])
        if initial_weight == 0.0 or zero_fraction == 0.0:
            return 0.0
        zero_iteration = self.rl_config["training"]["iterations"] * zero_fraction
        return initial_weight * max(0.0, 1.0 - iteration / zero_iteration)

    def _update_design(self, design_batch: dict | None, returns: np.ndarray, iteration: int) -> dict:
        # Update design probabilities from episode returns
        schedule = self._schedule()
        if design_batch is None or iteration < schedule["warmup"] or iteration >= schedule["fixed_start"]:
            return {
                "design_loss": float("nan"),
                "design_entropy": float("nan"),
                "design_entropy_weight": self._entropy_weight(iteration),
            }

        design_returns = torch.as_tensor(returns, dtype=torch.float32)
        if self.rl_config["training"]["normalize_design_returns"]:
            design_returns = (design_returns - design_returns.mean()) / (design_returns.std(unbiased=False) + 1e-8)
        log_prob = design_batch["log_prob"]
        entropy_weight = self._entropy_weight(iteration)
        loss = -(log_prob * design_returns.detach()).mean()
        loss = loss + entropy_weight * log_prob.mean()

        self.design_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.design_distribution.parameters(), self.rl_config["training"]["gradient_clip_norm"]
        )
        self.design_optimizer.step()
        return {
            "design_loss": float(loss.item()),
            "design_entropy": float(-log_prob.detach().mean().item()),
            "design_entropy_weight": entropy_weight,
        }

    def _should_prune(self, iteration: int) -> bool:
        distribution = self.rl_config["design_distribution"]
        schedule = self._schedule()
        completed = iteration + 1
        return bool(
            distribution["prune_components"]
            and self.design_distribution.active_components > 1
            and schedule["prune_interval"]
            and completed > schedule["warmup"]
            and completed < schedule["fixed_start"]
            and completed % schedule["prune_interval"] == 0
        )

    def _prune(self, iteration: int) -> None:
        # Compare components before removing weak ones
        distribution = self.rl_config["design_distribution"]
        evaluations = int(distribution["prune_evaluations_per_component"])
        scores = torch.full((distribution["components"],), -torch.inf, dtype=torch.float32)
        active_indices = self.design_distribution.active_mask.nonzero(as_tuple=False).flatten()

        for component in active_indices.tolist():
            with torch.no_grad():
                batch = self.design_distribution.sample_component(
                    component, evaluations, generator=self.pruning_design_generator
                )
            rewards = []
            for sample_index, design in enumerate(batch["designs"].numpy()):
                result = self._run_episode(
                    design=design,
                    split="train",
                    seed=self._episode_seed(
                        iteration, sample_index, "train", purpose=100_000 + component * evaluations
                    ),
                    training=False,
                    stochastic_policy=True,
                )
                rewards.append(result["reward_chf"])
            scores[component] = float(np.mean(rewards))

        self.design_distribution.prune(scores)
        self.design_optimizer.state.clear()
        if isinstance(self.agent, PPOAgent):
            self.agent.optimizer.state.clear()

    def train(self, log_every: int | None = None) -> pd.DataFrame:
        if log_every is not None and log_every <= 0:
            raise ValueError("log_every must be positive")
        records = []
        training = self.rl_config["training"]
        episodes_per_iteration = int(training["episodes_per_iteration"])
        schedule = self._schedule()

        # Alternate controller and design learning
        for iteration in range(self.completed_iterations, int(training["iterations"])):
            self._set_learning_rates(iteration)
            fixed_design = iteration >= schedule["fixed_start"]
            if fixed_design:
                mode = self.design_distribution.mode().numpy()
                designs = np.repeat(mode[None, :], episodes_per_iteration, axis=0)
                design_batch = None
            else:
                design_batch = self.design_distribution.sample(episodes_per_iteration)
                designs = design_batch["designs"].numpy()

            episode_results = []
            for episode, design in enumerate(designs):
                episode_results.append(
                    self._run_episode(
                        design=design,
                        split="train",
                        seed=self._episode_seed(iteration, episode, "train"),
                        training=True,
                        stochastic_policy=True,
                    )
                )

            for episode, (design, result) in enumerate(zip(designs, episode_results)):
                self.training_samples.append(
                    {
                        "iteration": iteration + 1,
                        "episode": episode + 1,
                        "model": self.rl_config["model_name"],
                        "scenario_seed": self._episode_seed(iteration, episode, "train"),
                        "sampled_from_distribution": design_batch is not None,
                        "component": (
                            int(design_batch["components"][episode])
                            if design_batch is not None
                            else None
                        ),
                        "pv_capacity_kwp": float(design[0]),
                        "battery_capacity_kwh": float(design[1]),
                        "reward_chf": result["reward_chf"],
                        "cost_chf": result["cost_chf"],
                        "max_power_balance_error_kw": result["max_power_balance_error_kw"],
                    }
                )

            algorithm_metrics = {}
            if isinstance(self.agent, PPOAgent):
                algorithm_metrics = self.agent.update(self._combine_rollouts(episode_results))
            else:
                algorithm_metrics = _mean_metrics(
                    [
                        result["update_metrics"]
                        for result in episode_results
                        if result["update_metrics"]
                    ]
                )

            returns = np.asarray([result["reward_chf"] for result in episode_results], dtype=float)
            design_metrics = self._update_design(design_batch, returns, iteration)
            if self._should_prune(iteration):
                self._prune(iteration)

            self.completed_iterations = iteration + 1
            self._record_distribution(iteration=self.completed_iterations)
            validation = self.evaluate(episodes=int(training["validation_episodes"]), sample_designs=not fixed_design)
            mode = self.design_distribution.mode().numpy()
            record = {
                "iteration": iteration + 1,
                "model": self.rl_config["model_name"],
                "algorithm": self.rl_config["algorithm"]["name"],
                "design_distribution": self.rl_config["design_distribution"]["name"],
                "active_components": self.design_distribution.active_components,
                "mean_reward_chf": float(returns.mean()),
                "std_reward_chf": float(returns.std()),
                "mean_cost_chf": float(np.mean([result["cost_chf"] for result in episode_results])),
                "mean_pv_capacity_kwp": float(designs[:, 0].mean()),
                "mean_battery_capacity_kwh": float(designs[:, 1].mean()),
                "mode_pv_capacity_kwp": float(mode[0]),
                "mode_battery_capacity_kwh": float(mode[1]),
                "max_power_balance_error_kw": float(
                    max(result["max_power_balance_error_kw"] for result in episode_results)
                ),
                "validation_mean_reward_chf": validation["mean_reward_chf"],
                "validation_mean_cost_chf": validation["mean_cost_chf"],
                "validation_max_power_balance_error_kw": validation["max_power_balance_error_kw"],
                **design_metrics,
                **algorithm_metrics,
            }
            records.append(record)
            if log_every is not None and self.completed_iterations % log_every == 0:
                print(
                    f"{record['model']} iteration "
                    f"{self.completed_iterations}/{training['iterations']} "
                    f"reward={record['mean_reward_chf']:.3f} "
                    f"validation_cost={record['validation_mean_cost_chf']:.3f} "
                    f"pv={record['mode_pv_capacity_kwp']:.3f} "
                    f"battery={record['mode_battery_capacity_kwh']:.3f}",
                    flush=True,
                )

        return pd.DataFrame(records)

    def evaluate(self, episodes: int | None = None, sample_designs: bool = True) -> dict:
        # Evaluate on validation scenarios
        if episodes is None:
            episodes = int(self.rl_config["training"]["validation_episodes"])
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        if sample_designs:
            with torch.no_grad():
                designs = self.design_distribution.sample(episodes, generator=self.validation_design_generator)[
                    "designs"
                ].numpy()
        else:
            mode = self.design_distribution.mode().numpy()
            designs = np.repeat(mode[None, :], episodes, axis=0)

        iteration = self.completed_iterations
        results = [
            self._run_episode(
                design=design,
                split="validation",
                seed=self._episode_seed(iteration, episode, "validation"),
                training=False,
                stochastic_policy=False,
            )
            for episode, design in enumerate(designs)
        ]
        return {
            "model": self.rl_config["model_name"],
            "split": "validation",
            "episodes": episodes,
            "sampled_designs": bool(sample_designs),
            "mean_reward_chf": float(np.mean([result["reward_chf"] for result in results])),
            "std_reward_chf": float(np.std([result["reward_chf"] for result in results])),
            "mean_cost_chf": float(np.mean([result["cost_chf"] for result in results])),
            "mean_pv_capacity_kwp": float(designs[:, 0].mean()),
            "mean_battery_capacity_kwh": float(designs[:, 1].mean()),
            "max_power_balance_error_kw": float(
                max(result["max_power_balance_error_kw"] for result in results)
            ),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "rl_config": self.rl_config,
            "system_config": self.system_config,
            "seed": self.seed,
            "completed_iterations": self.completed_iterations,
            "design_distribution": self.design_distribution.state_dict(),
            "design_optimizer": self.design_optimizer.state_dict(),
        }
        if isinstance(self.agent, PPOAgent):
            checkpoint["agent"] = {
                "network": self.agent.network.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
            }
        else:
            checkpoint["agent"] = {
                "actor": self.agent.actor.state_dict(),
                "critic": self.agent.critic.state_dict(),
                "target_actor": self.agent.target_actor.state_dict(),
                "target_critic": self.agent.target_critic.state_dict(),
                "actor_optimizer": self.agent.actor_optimizer.state_dict(),
                "critic_optimizer": self.agent.critic_optimizer.state_dict(),
            }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)


def run_experiment(
    model_name: str,
    seed: int,
    episode_hours: int | None = None,
    iterations: int | None = None,
    episodes_per_iteration: int | None = None,
    validation_episodes: int | None = None,
    log_every: int | None = None,
) -> tuple[CoDesignTrainer, pd.DataFrame, dict]:
    system_config = get_default_config()
    rl_config = get_rl_config(model_name)
    if episode_hours is not None:
        system_config["simulation"]["episode_hours"] = episode_hours
    if iterations is not None:
        rl_config["training"]["iterations"] = iterations
    if episodes_per_iteration is not None:
        rl_config["training"]["episodes_per_iteration"] = episodes_per_iteration
    if validation_episodes is not None:
        rl_config["training"]["validation_episodes"] = validation_episodes
    validate_config(system_config)
    validate_rl_config(rl_config)
    data = generate_dataset(year=DEFAULT_YEAR, days=rl_config["training"]["dataset_days"], seed=seed)
    trainer = CoDesignTrainer(system_config, data, rl_config, seed=seed)
    history = trainer.train(log_every=log_every)
    validation = trainer.evaluate(episodes=rl_config["training"]["validation_episodes"], sample_designs=False)
    return trainer, history, validation


def _write_results(output_directory: Path, trainer: CoDesignTrainer, history: pd.DataFrame, validation: dict) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_directory / "training.csv", index=False)
    pd.DataFrame(trainer.training_samples).to_csv(
        output_directory / "training_samples.csv", index=False
    )
    pd.DataFrame(trainer.distribution_history).to_csv(
        output_directory / "design_distribution.csv", index=False
    )
    with (output_directory / "validation.json").open("w") as file:
        json.dump(validation, file, indent=2)
    with (output_directory / "configuration.json").open("w") as file:
        json.dump({"system": trainer.system_config, "rl": trainer.rl_config}, file, indent=2)
    trainer.save_checkpoint(output_directory / "checkpoint.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(VARIANTS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-hours", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--episodes-per-iteration", type=int)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    model_names = sorted(VARIANTS) if args.model == "all" else [args.model]
    summaries = []
    for model_name in model_names:
        trainer, history, validation = run_experiment(
            model_name=model_name,
            seed=args.seed,
            episode_hours=args.episode_hours,
            iterations=args.iterations,
            episodes_per_iteration=args.episodes_per_iteration,
            validation_episodes=args.validation_episodes,
            log_every=args.log_every,
        )
        summaries.append(validation)
        if args.output_directory is not None:
            _write_results(args.output_directory / model_name, trainer, history, validation)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
