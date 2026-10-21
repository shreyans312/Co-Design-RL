import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rl import CoDesignTrainer, DEFAULT_YEAR, _design_returns, generate_dataset


def expected_weight_gradient(distribution, component_returns):
    logits = distribution.mixture_logits.detach().double()
    active = distribution.active_mask
    probabilities = torch.softmax(logits.masked_fill(~active, -torch.inf), dim=0)
    returns = torch.as_tensor(component_returns, dtype=torch.float64)
    mass = 1 - distribution.minimum_component_weight * int(active.sum())
    return (mass * probabilities * (returns - (probabilities * returns).sum())).numpy()


def frozen_returns(trainer, batch, scenario_seed, scenarios):
    records, returns = [], []
    for index, design in enumerate(batch["designs"].detach().numpy()):
        costs = []
        for scenario in range(scenarios):
            seed = scenario_seed + scenario
            result = trainer._run_episode(design, "train", seed, False, False)
            costs.append(result["cost_chf"])
            records.append({
                "design_index": index, "component": int(batch["components"][index]),
                "scenario_seed": seed, "pv_capacity_kwp": float(design[0]),
                "battery_capacity_kwh": float(design[1]), "cost_chf": result["cost_chf"],
            })
        returns.append(-np.mean(costs))
    return np.array(returns), pd.DataFrame(records)


def score_gradients(trainer, batch, returns, iteration):
    distribution = trainer.design_distribution
    advantages = _design_returns(returns, trainer.rl_config["training"])
    alpha = trainer._entropy_weight(iteration)
    signal = advantages - alpha * batch["log_prob"].detach()
    objective = (batch["log_prob"] * signal.detach()).mean()
    gradients = torch.autograd.grad(
        objective,
        [distribution.mixture_logits, distribution.component_means, distribution.component_log_stds],
        retain_graph=True,
    )
    return [gradient.detach().numpy() for gradient in gradients]


def perfect_signal_progress(distribution, learning_rate, updates, preferred_component):
    model = deepcopy(distribution)
    with torch.no_grad():
        model.mixture_logits.zero_()
    optimizer = torch.optim.Adam([model.mixture_logits], lr=learning_rate)
    rows = []
    for step in range(updates + 1):
        weights = model._log_weights().exp()
        if step in {0, 50, 100, updates}:
            rows.append({"update": step, "preferred_weight": float(weights[preferred_component].detach())})
        if step < updates:
            optimizer.zero_grad()
            (-weights[preferred_component]).backward()
            optimizer.step()
    return rows


def probe_design_updates(trainer, batches, scenarios, scenario_seed, generator, iteration, compare_sampling=False):
    distribution = trainer.design_distribution
    base_state = deepcopy(distribution.state_dict())
    base_optimizer = deepcopy(trainer.design_optimizer.state_dict())
    base_sampling = distribution.stratified_sampling
    generator_state = generator.get_state().clone()
    initial_weights = distribution.component_parameters()["weights"]
    training = trainer.rl_config["training"]
    batch_size = int(training["episodes_per_iteration"] // training["scenarios_per_design"])
    sampling_modes = [True, False] if compare_sampling else [base_sampling]
    gradients, episode_rows = [], []
    try:
        for stratified in sampling_modes:
            sampler = "stratified" if stratified else "weighted_random"
            distribution.stratified_sampling = stratified
            generator.set_state(generator_state)
            for index in range(batches):
                distribution.load_state_dict(base_state)
                trainer.design_optimizer.load_state_dict(deepcopy(base_optimizer))
                batch = distribution.sample(batch_size, generator=generator)
                returns, records = frozen_returns(trainer, batch, scenario_seed, scenarios)
                logit_gradient, location_gradient, scale_gradient = score_gradients(trainer, batch, returns, iteration)
                # Reset each probe so sampler differences cannot accumulate into different models
                trainer._update_design(batch, returns, iteration)
                after_weights = distribution.component_parameters()["weights"]
                logit_change = (distribution.mixture_logits.detach() - base_state["mixture_logits"]).numpy()
                for component in range(distribution.components):
                    gradients.append({
                        "sampler": sampler, "batch": index, "component": component,
                        "sample_count": int((batch["components"] == component).sum()),
                        "logit_ascent_gradient": float(logit_gradient[component]),
                        "pv_location_ascent_gradient": float(location_gradient[component, 0]),
                        "battery_location_ascent_gradient": float(location_gradient[component, 1]),
                        "pv_log_scale_ascent_gradient": float(scale_gradient[component, 0]),
                        "battery_log_scale_ascent_gradient": float(scale_gradient[component, 1]),
                        "adam_weight_change": float(after_weights[component] - initial_weights[component]),
                        "adam_logit_change": float(logit_change[component]),
                    })
                episode_rows.append(records.assign(phase="batch_gradient", sampler=sampler, batch=index))
                if (index + 1) % 8 == 0 or index == batches - 1:
                    print(f"{sampler}: gradient batch {index + 1}/{batches} complete", flush=True)
    finally:
        distribution.load_state_dict(base_state)
        distribution.stratified_sampling = base_sampling
        trainer.design_optimizer.load_state_dict(base_optimizer)
        trainer.design_optimizer.zero_grad(set_to_none=True)
        generator.set_state(generator_state)
    return pd.DataFrame(gradients), episode_rows


def sampling_metrics(frame, component_rewards, reference):
    rows = []
    for (sampler, batch), group in frame.groupby(["sampler", "batch"], sort=False):
        group = group.sort_values("component")
        gradient = group.logit_ascent_gradient.to_numpy()
        norm = np.linalg.norm(gradient) * np.linalg.norm(reference)
        projection = float(gradient @ reference)
        rows.append({
            "sampler": sampler, "batch": int(batch),
            "represented_components": int((group.sample_count > 0).sum()),
            "gradient_reference_dot_product": projection,
            "gradient_reference_cosine": projection / norm if norm > 0 else np.nan,
            "weight_only_predicted_saving_chf": float(group.adam_weight_change.to_numpy() @ component_rewards),
        })
    metrics = pd.DataFrame(rows)
    summaries = []
    for sampler, group in metrics.groupby("sampler", sort=False):
        gradients = frame[frame.sampler == sampler].pivot(index="batch", columns="component", values="logit_ascent_gradient")
        savings = group.weight_only_predicted_saving_chf
        summaries.append({
            "sampler": sampler, "batches": len(group),
            "all_components_present_fraction": float((group.represented_components == len(reference)).mean()),
            "mean_represented_components": float(group.represented_components.mean()),
            "logit_gradient_variance_trace": float(gradients.var(ddof=1).sum()),
            "mean_gradient_reference_cosine": float(group.gradient_reference_cosine.mean()),
            "positive_reference_projection_fraction": float((group.gradient_reference_dot_product > 0).mean()),
            "mean_weight_only_predicted_saving_chf": float(savings.mean()),
            "weight_only_predicted_saving_se_chf": float(savings.std(ddof=1) / np.sqrt(len(group))),
        })
    return metrics, pd.DataFrame(summaries)


def run_audit(checkpoint, output, designs_per_component=32, batches=16, scenarios=4,
              scenario_seed=40_000_000, design_seed=801, compare_sampling=False):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    payload = checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = saved["rl_config"]
    if config["algorithm"]["name"] != "ddpg" or not config["design_distribution"]["trainable_weights"]:
        raise ValueError("This audit requires a DDPG checkpoint with trainable mixture weights")
    training = config["training"]
    data_seed = training.get("dataset_seed", saved["seed"])
    data = generate_dataset(year=DEFAULT_YEAR, days=training["dataset_days"], seed=data_seed)
    trainer = CoDesignTrainer(saved["system_config"], data, config, seed=saved["seed"])
    trainer.agent.actor.load_state_dict(saved["agent"]["actor"])
    trainer.agent.actor.eval()
    trainer.design_distribution.load_state_dict(saved["design_distribution"])
    trainer.design_optimizer.load_state_dict(saved["design_optimizer"])
    trainer.completed_iterations = int(saved["completed_iterations"])
    iteration = max(trainer.completed_iterations - 1, 0)
    schedule = trainer._schedule()
    if not schedule["warmup"] <= iteration < schedule["fixed_start"]:
        raise ValueError("Use a checkpoint from the design-learning phase")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_snapshot.pt").write_bytes(payload)
    distribution = trainer.design_distribution
    generator = torch.Generator().manual_seed(design_seed)
    initial_weights = distribution.component_parameters()["weights"]
    component_rows, episode_rows = [], []
    component_rewards = np.zeros(distribution.components)
    for component in range(distribution.components):
        if not distribution.active_mask[component]:
            continue
        batch = distribution.sample_component(component, designs_per_component, generator=generator)
        returns, records = frozen_returns(trainer, batch, scenario_seed, scenarios)
        component_rewards[component] = returns.mean()
        component_rows.append({
            "component": component, "weight": float(initial_weights[component]),
            "mean_cost_chf": float(-returns.mean()), "design_cost_std_chf": float(returns.std(ddof=1)),
            "conditional_design_mean_se_chf": float(returns.std(ddof=1) / np.sqrt(len(returns))),
            "mean_pv_kwp": float(batch["designs"][:, 0].mean()),
            "mean_battery_kwh": float(batch["designs"][:, 1].mean()),
        })
        episode_rows.append(records.assign(phase="component_quality", batch=component))
        print(f"Component {component}: weight={initial_weights[component]:.3f}, mean cost={-returns.mean():.3f} CHF", flush=True)
    reference = expected_weight_gradient(distribution, component_rewards)
    for row in component_rows:
        row["reference_logit_ascent_gradient"] = float(reference[row["component"]])
    pd.DataFrame(component_rows).to_csv(output / "component_quality.csv", index=False)
    batch_size = int(training["episodes_per_iteration"] // training["scenarios_per_design"])
    frame, batch_episodes = probe_design_updates(
        trainer, batches, scenarios, scenario_seed, generator, iteration, compare_sampling,
    )
    episode_rows.extend(batch_episodes)
    assert len(trainer.agent.replay) == 0
    for key, value in trainer.agent.actor.state_dict().items():
        torch.testing.assert_close(value, saved["agent"]["actor"][key], rtol=0, atol=0)
    frame.to_csv(output / "batch_gradients.csv", index=False)
    pd.concat(episode_rows, ignore_index=True).to_csv(output / "episode_costs.csv", index=False)
    summaries = []
    for (sampler, component), rows in frame.groupby(["sampler", "component"], sort=False):
        values = rows.logit_ascent_gradient
        summaries.append({
            "sampler": sampler, "component": int(component),
            "reference_logit_ascent_gradient": float(reference[component]),
            "mean_logit_ascent_gradient": float(values.mean()), "gradient_std": float(values.std(ddof=1)),
            "gradient_mean_se": float(values.std(ddof=1) / np.sqrt(len(values))),
            "positive_gradient_fraction": float((values > 0).mean()),
            "mean_sample_count": float(rows.sample_count.mean()),
            "sample_count_std": float(rows.sample_count.std(ddof=1)),
            "absent_fraction": float((rows.sample_count == 0).mean()),
            "mean_adam_weight_change": float(rows.adam_weight_change.mean()),
            "mean_adam_logit_change": float(rows.adam_logit_change.mean()),
            "positive_adam_weight_change_fraction": float((rows.adam_weight_change > 0).mean()),
        })
    pd.DataFrame(summaries).to_csv(output / "gradient_summary.csv", index=False)
    batch_metrics, sampler_summary = sampling_metrics(frame, component_rewards, reference)
    batch_metrics.to_csv(output / "batch_metrics.csv", index=False)
    sampler_summary.to_csv(output / "sampler_summary.csv", index=False)
    updates = max(0, schedule["fixed_start"] - schedule["warmup"])
    preferred = int(min(component_rows, key=lambda row: row["mean_cost_chf"])["component"])
    progress = perfect_signal_progress(distribution, config["design_distribution"]["learning_rate"], updates, preferred)
    pd.DataFrame(progress).to_csv(output / "perfect_signal_progress.csv", index=False)
    metadata = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "iteration": trainer.completed_iterations, "dataset_seed": data_seed, "dataset_year": DEFAULT_YEAR,
        "design_seed": design_seed, "scenario_seed": scenario_seed, "scenarios_per_design": scenarios,
        "designs_per_component": designs_per_component, "gradient_batches": batches, "designs_per_batch": batch_size,
        "samplers": frame.sampler.unique().tolist(),
        "sampling_comparison": "Same starting RNG state per sampler, not identical sampled designs; same fixed actor, scenarios and optimizer state per probe",
        "weight_only_saving": "Estimated return change from weights alone using fixed component mean returns; excludes location/scale changes and is not measured post-update simulator improvement",
        "split": "train", "actor_frozen": True, "action_noise": 0.0,
        "reference": "Exact weight-only gradient conditional on sampled component mean returns",
        "gradient_comparison": "Reference uses raw returns; batch gradients use existing baseline, normalisation and entropy",
        "uncertainty": "Component SE is over design draws conditional on the common scenarios, not a population or multi-seed SE",
        "perfect_signal": "Weight-only Adam from uniform weights, reward 1 for preferred component and 0 for others, no entropy or controller",
        "limitations": "One checkpoint and a small common training-scenario set; one-step probes do not measure subsequent training improvement",
        "rl_config": config, "system_config": saved["system_config"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(component_rows).round(5).to_string(index=False))
    print(pd.DataFrame(summaries).round(6).to_string(index=False))
    print(sampler_summary.round(6).to_string(index=False))
    print("Perfect known-signal weight-only check:", progress)
    return component_rows, summaries, progress


def main():
    parser = argparse.ArgumentParser(description="Audit frozen-controller component quality and design gradients")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--designs-per-component", type=int, default=32)
    parser.add_argument("--gradient-batches", type=int, default=16)
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--scenario-seed", type=int, default=40_000_000)
    parser.add_argument("--design-seed", type=int, default=801)
    parser.add_argument("--compare-sampling", action="store_true")
    args = parser.parse_args()
    if min(args.designs_per_component, args.gradient_batches) < 2 or args.scenarios < 1:
        parser.error("Use at least two design draws and gradient batches and at least one scenario")
    if min(args.scenario_seed, args.design_seed) < 0:
        parser.error("Seeds must be non-negative")
    torch.set_num_threads(1)
    run_audit(args.checkpoint, args.output_directory, args.designs_per_component, args.gradient_batches,
              args.scenarios, args.scenario_seed, args.design_seed, args.compare_sampling)


if __name__ == "__main__":
    main()
