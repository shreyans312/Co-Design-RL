import argparse
from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).with_name("stochastic-model")))

from config import get_default_config, validate_config
from data import DEFAULT_YEAR, generate_dataset, load_dataset, validate_dataset
from design_distributions import create_design_distribution
from rl_algorithms import DDPGAgent, PPOAgent, StateEncoder, create_agent
from rl_config import VARIANTS, get_rl_config, validate_rl_config
from simulator import EnergySystemSimulator
from stochastic import generate_episode_scenario


CAUZ_DATASET_PATH = Path(__file__).with_name("stochastic-model") / "cauz_dataset.csv"


def load_experiment_dataset(training: dict) -> pd.DataFrame:
    dataset = training.get("dataset", "synthetic")
    if dataset == "synthetic":
        return generate_dataset(year=DEFAULT_YEAR, days=training["dataset_days"], seed=training["dataset_seed"])
    if dataset != "cauz":
        raise ValueError(f"Unknown dataset: {dataset}")
    expected_hash = training.get("dataset_sha256")
    actual_hash = hashlib.sha256(CAUZ_DATASET_PATH.read_bytes()).hexdigest()
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("Cauz dataset does not match the dataset used for training")
    return load_dataset(CAUZ_DATASET_PATH)


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


def _design_returns(returns: np.ndarray, training: dict) -> torch.Tensor:
    values = torch.as_tensor(returns, dtype=torch.float32)
    if training["design_return_baseline"] == "leave_one_out":
        if len(values) == 1:
            return torch.zeros_like(values)
        values = values - (values.sum() - values) / (len(values) - 1)
    if training["normalize_design_returns"]:
        values = (values - values.mean()) / (values.std(unbiased=False) + 1e-8)
    return values


class CoDesignTrainer:
    def __init__(self, system_config: dict, data: pd.DataFrame, rl_config: dict, seed: int | None = None) -> None:
        validate_config(system_config)
        validate_dataset(data)
        validate_rl_config(rl_config)
        self.system_config = deepcopy(system_config)
        self.data = data.copy()
        self.rl_config = deepcopy(rl_config)
        if self.rl_config["training"].get("design_scoring", "training") == "frozen_actor":
            self.rl_config["provenance"]["paper_faithful"] = False
            self.rl_config["provenance"]["design_scoring"] = "post-controller frozen actor on fresh common training scenarios"
        fixed_design = self.rl_config["training"].get("fixed_design_values")
        self.fixed_design = (
            None if fixed_design is None else np.asarray(fixed_design, dtype=np.float32)
        )
        if self.fixed_design is not None:
            build_system_config(self.system_config, self.fixed_design)
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
        self.design_scores = []
        self.validation_records = []
        self.best_validation_cost = float("inf")
        self._record_distribution(iteration=0)

    def _design_mode(self) -> np.ndarray:
        if self.fixed_design is not None:
            return self.fixed_design.copy()
        return self.design_distribution.mode().numpy()

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

    def _set_exploration_noise(self, iteration: int) -> None:
        if not isinstance(self.agent, DDPGAgent):
            return
        algorithm = self.rl_config["algorithm"]
        initial = float(algorithm["exploration_noise_std"])
        final = float(algorithm.get("final_exploration_noise_std", initial))
        denominator = max(int(self.rl_config["training"]["iterations"]) - 1, 1)
        self.agent.exploration_noise_std = initial + (final - initial) * iteration / denominator

    def _episode_seed(self, iteration: int, episode: int, split: str, purpose: int = 0) -> int:
        split_offset = 0 if split == "train" else 1_000_000
        return self.seed + split_offset + purpose + iteration * 10_000 + episode

    def _bootstrap_observation(
        self,
        scenario: dict,
        simulator: EnergySystemSimulator,
        design: np.ndarray,
    ) -> np.ndarray | None:
        data = scenario["bootstrap_data"]
        ev = scenario["bootstrap_ev_schedule"]
        if data.empty:
            return None

        data_row = data.iloc[0]
        ev_row = ev.iloc[0]
        if ev_row["ev_arrival"]:
            ev_soc = float(ev_row["ev_arrival_soc_kwh"])
        elif ev_row["ev_present"]:
            ev_soc = simulator.ev_soc_kwh
        else:
            ev_soc = 0.0
        state = {
            "timestamp": data_row["timestamp"],
            "hour": int(data_row["hour"]),
            "battery_soc_kwh": simulator.battery_soc_kwh,
            "pv_power_kw": float(data_row["pv_per_kwp"]) * float(design[0]),
            "load_kw": float(data_row["load_kw"]),
            "import_price": float(data_row["import_price"]),
            "export_price": float(data_row["export_price"]),
            "ev_present": bool(ev_row["ev_present"]),
            "ev_soc_kwh": ev_soc,
        }
        return self.encoder.encode(state, design)

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
        bootstrap_hours = int(training and isinstance(self.agent, DDPGAgent))
        scenario = generate_episode_scenario(
            data=self.data,
            config=configured,
            split=split,
            seed=seed,
            bootstrap_hours=bootstrap_hours,
        )
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

            if training and isinstance(self.agent, PPOAgent):
                next_observation = (
                    np.zeros(self.encoder.size, dtype=np.float32)
                    if done
                    else self.encoder.encode(next_state, design)
                )
                rollout["observations"].append(observation)
                rollout["latent_actions"].append(decision["latent_action"])
                rollout["log_probs"].append(decision["log_prob"])
                rollout["values"].append(decision["value"])
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)
                rollout["next_values"].append(0.0 if done else self.agent.value(next_observation))
            elif training and isinstance(self.agent, DDPGAgent):
                if done:
                    next_observation = self._bootstrap_observation(scenario, simulator, design)
                    terminated = next_observation is None
                    if terminated:
                        next_observation = np.zeros(self.encoder.size, dtype=np.float32)
                else:
                    next_observation = self.encoder.encode(next_state, design)
                    terminated = False
                self.agent.observe(observation, action, reward, next_observation, terminated)
                metrics = self.agent.update()
                if metrics is not None:
                    update_metrics.append(metrics)

            state = next_state

        actions = pd.DataFrame(simulator.history)
        ev_actions = actions.loc[actions["ev_present"], "ev_action"].abs()
        diagnostics = {}
        for storage, frame in [("battery", actions), ("ev", actions[actions["ev_present"]])]:
            diagnostics[f"{storage}_action_mean"] = float(frame[f"{storage}_action"].mean()) if len(frame) else 0.0
            diagnostics[f"{storage}_action_std"] = float(frame[f"{storage}_action"].std(ddof=0)) if len(frame) else 0.0
            diagnostics[f"{storage}_abs_power_kw"] = float(frame[f"{storage}_power_kw"].abs().mean()) if len(frame) else 0.0
            diagnostics[f"{storage}_power_limited_fraction"] = float((
                (frame[f"requested_{storage}_power_kw"] - frame[f"{storage}_power_kw"]).abs() > 1e-6
            ).mean()) if len(frame) else 0.0
        return {
            "reward_chf": float(simulator.cumulative_reward),
            "cost_chf": float(simulator.cumulative_cost_chf),
            "max_power_balance_error_kw": float(max_power_balance_error_kw),
            "battery_action_saturation": float((actions["battery_action"].abs() >= 0.95).mean()),
            "ev_action_saturation": float((ev_actions >= 0.95).mean()) if len(ev_actions) else 0.0,
            "rollout": rollout,
            "update_metrics": _mean_metrics(update_metrics),
            "grid_cost_chf": float(actions["grid_energy_cost_chf"].sum()),
            "design_cost_chf": float(actions["design_cost_chf"].sum()),
            "ev_connected_hours": int(actions["ev_present"].sum()),
            **diagnostics,
        }

    def _score_designs(self, design_batch: dict, iteration: int) -> np.ndarray:
        if not isinstance(self.agent, DDPGAgent):
            raise ValueError("Frozen-actor design scoring requires DDPG")
        training = self.rl_config["training"]
        scenarios = int(training["scenarios_per_design"])
        # Place scoring seeds after the planned controller-training seed block
        seed_offset = 10_000 * int(training["iterations"])
        was_training = self.agent.actor.training
        self.agent.actor.eval()
        design_returns = []
        try:
            with torch.no_grad():
                for index, design in enumerate(design_batch["designs"].detach().numpy()):
                    returns = []
                    for scenario in range(scenarios):
                        seed = self._episode_seed(iteration, scenario, "train", purpose=seed_offset)
                        result = self._run_episode(design, "train", seed, False, False)
                        returns.append(result["reward_chf"])
                        self.design_scores.append({
                            "iteration": iteration + 1, "design_index": index + 1,
                            "scenario_index": scenario + 1, "scenario_seed": seed,
                            "component": int(design_batch["components"][index]),
                            "pv_capacity_kwp": float(design[0]), "battery_capacity_kwh": float(design[1]),
                            "actor_phase": "post_controller", "split": "train", "action_noise_std": 0.0,
                            "reward_chf": result["reward_chf"], "cost_chf": result["cost_chf"],
                            "grid_cost_chf": result["grid_cost_chf"], "design_cost_chf": result["design_cost_chf"],
                            "max_power_balance_error_kw": result["max_power_balance_error_kw"],
                        })
                    design_returns.append(np.mean(returns))
        finally:
            self.agent.actor.train(was_training)
        return np.asarray(design_returns, dtype=float)

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

        design_returns = _design_returns(returns, self.rl_config["training"])
        log_prob = design_batch["log_prob"]
        entropy_weight = self._entropy_weight(iteration)
        augmented_returns = design_returns - entropy_weight * log_prob.detach()
        loss = -(log_prob * augmented_returns.detach()).mean()

        self.design_optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.design_distribution.parameters(), self.rl_config["training"]["gradient_clip_norm"],
            error_if_nonfinite=True,
        )
        self.design_optimizer.step()
        self.design_distribution.constrain_parameters()
        return {
            "design_loss": float(loss.item()),
            "design_entropy": float(-log_prob.detach().mean().item()),
            "design_entropy_weight": entropy_weight,
            "design_gradient_norm": float(gradient_norm),
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

    def evaluate_matched(self, episodes: int | None = None) -> dict:
        training = self.rl_config["training"]
        count = int(training["validation_episodes"] if episodes is None else episodes)
        if count <= 0:
            raise ValueError("episodes must be positive")
        validation_seed = int(training.get("validation_seed", 1_000_042))
        generator = torch.Generator().manual_seed(validation_seed + 1_000_000)
        with torch.no_grad():
            sampled = self.design_distribution.sample(count, generator=generator)["designs"].numpy()
        groups = {"mode": np.repeat(self._design_mode()[None, :], count, axis=0), "distribution": sampled}
        references = training.get("reference_designs", [[4.0, 4.5], [6.6, 4.5], [9.0, 4.5]])
        for index, design in enumerate(references):
            groups[f"reference_{index}"] = np.repeat([design], count, axis=0)
        summary = {
            "model": self.rl_config["model_name"], "split": "validation", "episodes": count,
            "protocol": "matched", "sampled_designs": False,
            "iteration": self.completed_iterations, "validation_seed": validation_seed,
        }
        reference_costs = []
        max_error = 0.0
        for label, designs in groups.items():
            results = []
            for episode, design in enumerate(designs):
                scenario_seed = validation_seed + episode
                result = self._run_episode(design, "validation", scenario_seed, False, False)
                results.append(result)
                self.validation_records.append({
                    "iteration": self.completed_iterations, "evaluation": label,
                    "episode": episode + 1, "scenario_seed": scenario_seed,
                    "pv_capacity_kwp": float(design[0]), "battery_capacity_kwh": float(design[1]),
                    **{key: value for key, value in result.items() if key not in {"rollout", "update_metrics"}},
                })
            costs = np.array([result["cost_chf"] for result in results])
            max_error = max(max_error, max(result["max_power_balance_error_kw"] for result in results))
            summary[f"validation_{label}_cost_chf"] = float(costs.mean())
            summary[f"validation_{label}_cost_std_chf"] = float(costs.std())
            if label.startswith("reference_"):
                reference_costs.append(float(costs.mean()))
            if label == "mode":
                summary.update(mean_cost_chf=float(costs.mean()), mean_reward_chf=float(-costs.mean()),
                               std_reward_chf=float(costs.std()), mean_pv_capacity_kwp=float(designs[0, 0]),
                               mean_battery_capacity_kwh=float(designs[0, 1]))
        summary["validation_reference_cost_chf"] = float(np.mean(reference_costs))
        summary["max_power_balance_error_kw"] = max_error
        self.latest_matched_validation = summary
        return summary

    def _save_best(self, validation: dict, output_directory: Path | None) -> None:
        if validation["mean_cost_chf"] >= self.best_validation_cost:
            return
        self.best_validation_cost = validation["mean_cost_chf"]
        if output_directory is not None:
            self.save_checkpoint(output_directory / "best_checkpoint.pt")
            (output_directory / "best_validation.json").write_text(json.dumps(validation, indent=2))

    def train(
        self,
        log_every: int | None = None,
        output_directory: Path | None = None,
        save_every: int | None = None,
        keep_checkpoints: bool = False,
        validation_every: int = 1,
    ) -> pd.DataFrame:
        if log_every is not None and log_every <= 0:
            raise ValueError("log_every must be positive")
        if save_every is not None and save_every <= 0:
            raise ValueError("save_every must be positive")
        if validation_every <= 0:
            raise ValueError("validation_every must be positive")
        records = []
        training = self.rl_config["training"]
        episodes_per_iteration = int(training["episodes_per_iteration"])
        scenarios_per_design = int(training["scenarios_per_design"])
        designs_per_iteration = episodes_per_iteration // scenarios_per_design
        schedule = self._schedule()
        matched_validation = bool(training.get("matched_validation", False))
        if matched_validation and self.completed_iterations == 0:
            initial = self.evaluate_matched()
            if output_directory is not None:
                output_directory.mkdir(parents=True, exist_ok=True)
                (output_directory / "initial_validation.json").write_text(json.dumps(initial, indent=2))
                self.save_checkpoint(output_directory / "initial_checkpoint.pt")
            self._save_best(initial, output_directory)
            print(f"initial mode_cost={initial['mean_cost_chf']:.3f} "
                  f"reference_cost={initial['validation_reference_cost_chf']:.3f}", flush=True)

        # Alternate controller and design learning
        for iteration in range(self.completed_iterations, int(training["iterations"])):
            self._set_learning_rates(iteration)
            self._set_exploration_noise(iteration)
            fixed_design = self.fixed_design is not None or iteration >= schedule["fixed_start"]
            if fixed_design:
                mode = self._design_mode()
                sampled_designs = np.repeat(mode[None, :], designs_per_iteration, axis=0)
                design_batch = None
            else:
                design_batch = self.design_distribution.sample(designs_per_iteration)
                sampled_designs = design_batch["designs"].numpy()
            designs = np.repeat(sampled_designs, scenarios_per_design, axis=0)
            components = (
                np.repeat(design_batch["components"].numpy(), scenarios_per_design)
                if design_batch is not None
                else np.full(episodes_per_iteration, -1)
            )
            scenario_indices = (
                np.tile(np.arange(scenarios_per_design), designs_per_iteration)
                if scenarios_per_design > 1
                else np.arange(episodes_per_iteration)
            )
            scenario_seeds = [
                self._episode_seed(iteration, int(index), "train") for index in scenario_indices
            ]

            episode_results = []
            for design, seed in zip(designs, scenario_seeds):
                episode_results.append(
                    self._run_episode(
                        design=design,
                        split="train",
                        seed=seed,
                        training=True,
                        stochastic_policy=True,
                    )
                )

            for episode, (design, result) in enumerate(zip(designs, episode_results)):
                self.training_samples.append(
                    {
                        "iteration": iteration + 1,
                        "episode": episode + 1,
                        "design_index": episode // scenarios_per_design + 1,
                        "scenario_index": int(scenario_indices[episode]) + 1,
                        "model": self.rl_config["model_name"],
                        "scenario_seed": scenario_seeds[episode],
                        "sampled_from_distribution": design_batch is not None,
                        "component": int(components[episode]) if design_batch is not None else None,
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
            design_returns = returns.reshape(designs_per_iteration, scenarios_per_design).mean(axis=1)
            scoring_metrics = {}
            if training.get("design_scoring", "training") == "frozen_actor":
                scoring_metrics["design_scoring_episodes"] = 0
                if design_batch is not None and schedule["warmup"] <= iteration < schedule["fixed_start"]:
                    design_returns = self._score_designs(design_batch, iteration)
                    scoring_metrics.update(
                        design_scoring_episodes=episodes_per_iteration,
                        design_score_mean_cost_chf=float(-design_returns.mean()),
                        design_score_std_cost_chf=float(design_returns.std()),
                    )
            design_metrics = self._update_design(design_batch, design_returns, iteration)
            if self._should_prune(iteration):
                self._prune(iteration)

            self.completed_iterations = iteration + 1
            self._record_distribution(iteration=self.completed_iterations)
            final_iteration = self.completed_iterations == int(training["iterations"])
            if self.completed_iterations % validation_every == 0 or final_iteration or (matched_validation and iteration == 0):
                if matched_validation:
                    validation = self.evaluate_matched()
                    self._save_best(validation, output_directory)
                else:
                    validation = self.evaluate(
                        episodes=int(training["validation_episodes"]),
                        sample_designs=not fixed_design,
                    )
            else:
                validation = {
                    "mean_reward_chf": float("nan"),
                    "mean_cost_chf": float("nan"),
                    "max_power_balance_error_kw": float("nan"),
                }
            mode = self._design_mode()
            record = {
                "iteration": iteration + 1,
                "model": self.rl_config["model_name"],
                "algorithm": self.rl_config["algorithm"]["name"],
                "design_distribution": self.rl_config["design_distribution"]["name"],
                "active_components": self.design_distribution.active_components,
                "designs_per_iteration": designs_per_iteration,
                "scenarios_per_design": scenarios_per_design,
                "mean_reward_chf": float(returns.mean()),
                "std_reward_chf": float(returns.std()),
                "mean_cost_chf": float(np.mean([result["cost_chf"] for result in episode_results])),
                "mean_pv_capacity_kwp": float(designs[:, 0].mean()),
                "std_pv_capacity_kwp": float(designs[:, 0].std()),
                "mean_battery_capacity_kwh": float(designs[:, 1].mean()),
                "std_battery_capacity_kwh": float(designs[:, 1].std()),
                "mode_pv_capacity_kwp": float(mode[0]),
                "mode_battery_capacity_kwh": float(mode[1]),
                "max_power_balance_error_kw": float(
                    max(result["max_power_balance_error_kw"] for result in episode_results)
                ),
                "validation_mean_reward_chf": validation["mean_reward_chf"],
                "validation_mean_cost_chf": validation["mean_cost_chf"],
                "validation_max_power_balance_error_kw": validation["max_power_balance_error_kw"],
                "battery_action_saturation": float(
                    np.mean([result["battery_action_saturation"] for result in episode_results])
                ),
                "ev_action_saturation": float(
                    np.mean([result["ev_action_saturation"] for result in episode_results])
                ),
                **design_metrics,
                **scoring_metrics,
                **algorithm_metrics,
                **{key: value for key, value in validation.items() if key.startswith("validation_")},
            }
            if isinstance(self.agent, DDPGAgent):
                record["exploration_noise_std"] = self.agent.exploration_noise_std
            records.append(record)
            if log_every is not None and self.completed_iterations % log_every == 0:
                ddpg_metrics = ""
                if isinstance(self.agent, DDPGAgent):
                    ddpg_metrics = (
                        f" sampled_pv={record['mean_pv_capacity_kwp']:.2f}±{record['std_pv_capacity_kwp']:.2f}"
                        f" sampled_battery={record['mean_battery_capacity_kwh']:.2f}±{record['std_battery_capacity_kwh']:.2f}"
                        f" actor={record.get('actor_loss', float('nan')):.4g} critic={record.get('critic_loss', float('nan')):.4g}"
                        f" battery_sat={record['battery_action_saturation']:.2f}"
                    )
                if matched_validation:
                    ddpg_metrics += f" reference_cost={record.get('validation_reference_cost_chf', float('nan')):.3f}"
                if "design_score_mean_cost_chf" in record:
                    ddpg_metrics += f" design_score_cost={record['design_score_mean_cost_chf']:.3f}"
                print(
                    f"{record['model']} iteration "
                    f"{self.completed_iterations}/{training['iterations']} "
                    f"reward={record['mean_reward_chf']:.3f} "
                    f"validation_cost={record['validation_mean_cost_chf']:.3f} "
                    f"pv={record['mode_pv_capacity_kwp']:.3f} "
                    f"battery={record['mode_battery_capacity_kwh']:.3f}"
                    f"{ddpg_metrics}",
                    flush=True,
                )
            if (
                output_directory is not None
                and save_every is not None
                and (self.completed_iterations % save_every == 0 or final_iteration)
            ):
                _write_progress(
                    output_directory,
                    self,
                    pd.DataFrame(records),
                    validation,
                    keep_checkpoint=keep_checkpoints,
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
            mode = self._design_mode()
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
            "exact_resume_supported": False,
            "checkpoint_purpose": "paired controller and design evaluation",
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
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(output_path)


def run_experiment(
    model_name: str,
    seed: int,
    episode_hours: int | None = None,
    iterations: int | None = None,
    episodes_per_iteration: int | None = None,
    validation_episodes: int | None = None,
    log_every: int | None = None,
    output_directory: Path | None = None,
    save_every: int | None = None,
    validation_every: int = 1,
    matched_validation: bool = False,
    data_seed: int | None = None,
    validation_seed: int = 1_000_042,
    exploration_noise_std: float | None = None,
    final_exploration_noise_std: float | None = None,
    design_warmup_fraction: float | None = None,
    design_scoring: str = "training",
    dataset: str = "synthetic",
) -> tuple[CoDesignTrainer, pd.DataFrame, dict]:
    system_config = get_default_config()
    rl_config = get_rl_config(model_name)
    rl_config["seed"] = seed
    training = rl_config["training"]
    if dataset not in {"synthetic", "cauz"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    training["dataset"] = dataset
    if dataset == "cauz":
        training["dataset_sha256"] = hashlib.sha256(CAUZ_DATASET_PATH.read_bytes()).hexdigest()
    training["dataset_seed"] = seed if data_seed is None else data_seed
    training["matched_validation"] = matched_validation
    training["validation_seed"] = validation_seed
    training["validation_every"] = validation_every
    training["design_scoring"] = design_scoring
    if matched_validation:
        training["reference_designs"] = [[4.0, 4.5], [6.6, 4.5], [9.0, 4.5]]
    if design_warmup_fraction is not None:
        if not 0 <= design_warmup_fraction < 1:
            raise ValueError("design_warmup_fraction must be in [0, 1)")
        training["design_warmup_fraction"] = design_warmup_fraction
    for key, value in [("exploration_noise_std", exploration_noise_std),
                       ("final_exploration_noise_std", final_exploration_noise_std)]:
        if value is not None:
            if rl_config["algorithm"]["name"] != "ddpg" or not np.isfinite(value) or value < 0:
                raise ValueError("Noise overrides require DDPG and finite non-negative values")
            rl_config["algorithm"][key] = value
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
    data = load_experiment_dataset(training)
    trainer = CoDesignTrainer(system_config, data, rl_config, seed=seed)
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_training_files(output_directory, trainer, pd.DataFrame())
    history = trainer.train(
        log_every=log_every,
        output_directory=output_directory,
        save_every=save_every,
        validation_every=validation_every,
    )
    validation = trainer.latest_matched_validation if matched_validation else trainer.evaluate(
        episodes=rl_config["training"]["validation_episodes"],
        sample_designs=rl_config["algorithm"]["name"] == "ddpg",
    )
    return trainer, history, validation


def _write_training_files(
    output_directory: Path,
    trainer: CoDesignTrainer,
    history: pd.DataFrame,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_directory / "training.csv", index=False)
    pd.DataFrame(trainer.training_samples).to_csv(
        output_directory / "training_samples.csv", index=False
    )
    pd.DataFrame(trainer.distribution_history).to_csv(
        output_directory / "design_distribution.csv", index=False
    )
    with (output_directory / "configuration.json").open("w") as file:
        json.dump({"system": trainer.system_config, "rl": trainer.rl_config,
                   "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": np.__version__, "pandas": pd.__version__,
                               "torch_threads": torch.get_num_threads()}}, file, indent=2)
    if trainer.validation_records:
        pd.DataFrame(trainer.validation_records).to_csv(output_directory / "validation_episodes.csv", index=False)
    if trainer.design_scores:
        pd.DataFrame(trainer.design_scores).to_csv(output_directory / "design_scores.csv", index=False)
    trainer.save_checkpoint(output_directory / "checkpoint.pt")


def _write_progress(
    output_directory: Path,
    trainer: CoDesignTrainer,
    history: pd.DataFrame,
    validation: dict,
    keep_checkpoint: bool = False,
) -> None:
    _write_training_files(output_directory, trainer, history)
    if keep_checkpoint:
        trainer.save_checkpoint(
            output_directory / "checkpoints" / f"iteration_{trainer.completed_iterations:04d}.pt"
        )
    with (output_directory / "latest_validation.json").open("w") as file:
        latest = getattr(trainer, "latest_matched_validation", validation)
        json.dump(latest, file, indent=2)


def _write_results(output_directory: Path, trainer: CoDesignTrainer, history: pd.DataFrame, validation: dict) -> None:
    _write_training_files(output_directory, trainer, history)
    with (output_directory / "validation.json").open("w") as file:
        json.dump(validation, file, indent=2)
    if isinstance(trainer.agent, DDPGAgent):
        generator = torch.Generator().manual_seed(trainer.seed + 4_000_000)
        with torch.no_grad():
            batch = trainer.design_distribution.sample(1000, generator=generator)
        designs = pd.DataFrame(
            batch["designs"].numpy(),
            columns=["pv_capacity_kwp", "battery_capacity_kwh"],
        )
        designs.insert(0, "component", batch["components"].numpy())
        designs.to_csv(output_directory / "final_design_samples.csv", index=False)
        summary = {"samples": len(designs), "seed": trainer.seed + 4_000_000}
        for column in ("pv_capacity_kwp", "battery_capacity_kwh"):
            summary[column] = {
                "mean": float(designs[column].mean()),
                "q25": float(designs[column].quantile(0.25)),
                "median": float(designs[column].median()),
                "q75": float(designs[column].quantile(0.75)),
            }
        with (output_directory / "design_summary.json").open("w") as file:
            json.dump(summary, file, indent=2)


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
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--matched-validation", action="store_true")
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--dataset", choices=["synthetic", "cauz"], default="synthetic")
    parser.add_argument("--validation-seed", type=int, default=1_000_042)
    parser.add_argument("--exploration-noise-std", type=float)
    parser.add_argument("--final-exploration-noise-std", type=float)
    parser.add_argument("--design-warmup-fraction", type=float)
    parser.add_argument("--design-scoring", choices=["training", "frozen_actor"], default="training")
    parser.add_argument("--torch-threads", type=int)
    args = parser.parse_args()
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            parser.error("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
    if args.save_every <= 0 or args.validation_every <= 0:
        parser.error("save and validation intervals must be positive")

    model_names = sorted(VARIANTS) if args.model == "all" else [args.model]
    if args.design_scoring == "frozen_actor" and any(VARIANTS[name]["algorithm"]["name"] != "ddpg" for name in model_names):
        parser.error("--design-scoring frozen_actor requires a single DDPG model")
    if args.output_directory is not None:
        for name in model_names:
            destination = args.output_directory / name
            if destination.exists() and any(destination.iterdir()):
                parser.error(f"Output directory is not empty: {destination}")
    summaries = []
    for model_name in model_names:
        model_output_directory = (
            args.output_directory / model_name if args.output_directory is not None else None
        )
        trainer, history, validation = run_experiment(
            model_name=model_name,
            seed=args.seed,
            episode_hours=args.episode_hours,
            iterations=args.iterations,
            episodes_per_iteration=args.episodes_per_iteration,
            validation_episodes=args.validation_episodes,
            log_every=args.log_every,
            output_directory=model_output_directory,
            save_every=args.save_every,
            validation_every=args.validation_every,
            matched_validation=args.matched_validation,
            data_seed=args.data_seed,
            validation_seed=args.validation_seed,
            exploration_noise_std=args.exploration_noise_std,
            final_exploration_noise_std=args.final_exploration_noise_std,
            design_warmup_fraction=args.design_warmup_fraction,
            design_scoring=args.design_scoring,
            dataset=args.dataset,
        )
        summaries.append(validation)
        if model_output_directory is not None:
            _write_results(model_output_directory, trainer, history, validation)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
