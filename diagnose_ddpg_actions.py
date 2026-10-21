import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rl import (
    CoDesignTrainer, DEFAULT_YEAR, EnergySystemSimulator, build_system_config,
    generate_dataset, generate_episode_scenario,
)
from simulator import apply_storage_power


def probe_battery(action, soc, config, noise_std, standard_noise):
    battery = config["battery"]
    capacity = config["design"]["battery_capacity_kwh"]
    maximum_power = capacity * battery["c_rate"]
    raw_actions = action + noise_std * standard_noise
    requested = np.clip(raw_actions, -1.0, 1.0) * maximum_power
    applied = np.array([
        apply_storage_power(
            float(power), soc, capacity, capacity * battery["minimum_soc_fraction"],
            maximum_power, battery["efficiency"], config["simulation"]["timestep_hours"],
        )[0]
        for power in requested
    ])
    return {
        "requested_std_kw": float(requested.std()),
        "applied_std_kw": float(applied.std()),
        "action_clipped_fraction": float((np.abs(raw_actions) > 1.0).mean()),
        "power_limited_fraction": float((np.abs(requested - applied) > 1e-6).mean()),
        "blocked_request_fraction": float(((np.abs(requested) > 1e-6) & (np.abs(applied) <= 1e-6)).mean()),
    }


def collect_episode(trainer, design, scenario_seed, noise_levels, probe_draws=16, probe_seed=701):
    config = build_system_config(trainer.system_config, design)
    scenario = generate_episode_scenario(trainer.data, config, split="validation", seed=scenario_seed)
    simulator = EnergySystemSimulator(config)
    state = simulator.reset(scenario)
    rng = np.random.default_rng(np.random.SeedSequence([probe_seed, scenario_seed]))
    capacity = float(design[1])
    minimum = capacity * config["battery"]["minimum_soc_fraction"]
    rows, probes = [], []
    while not simulator.done:
        observation = trainer.encoder.encode(state, design)
        action = trainer.agent.act(observation, explore=False)
        before_soc = float(state["battery_soc_kwh"])
        # Compare noise at identical states without advancing the simulator
        standard_noise = rng.normal(size=probe_draws)
        for noise in noise_levels:
            probes.append({
                "step": simulator.current_step, "noise_std": noise,
                **probe_battery(float(action[0]), before_soc, config, noise, standard_noise),
            })
        state, _, _, transition = simulator.step(action)
        requested = transition["requested_battery_power_kw"]
        applied = transition["battery_power_kw"]
        rows.append({
            **transition,
            "battery_soc_before_kwh": before_soc,
            "battery_soc_before_fraction": before_soc / capacity if capacity else 0.0,
            "battery_at_minimum": before_soc <= minimum + 1e-6,
            "battery_at_capacity": before_soc >= capacity - 1e-6,
            "battery_power_limited": abs(requested - applied) > 1e-6,
            "blocked_discharge": requested > 1e-6 and abs(applied) <= 1e-6,
            "blocked_charge": requested < -1e-6 and abs(applied) <= 1e-6,
        })
    return pd.DataFrame(rows), pd.DataFrame(probes)


def make_plots(steps, summary, probes, output, iteration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode = steps[steps.design == "mode"]
    example = mode[mode.scenario_seed == mode.scenario_seed.min()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(example.step, example.requested_battery_power_kw, label="Requested", alpha=0.7)
    axes[0, 0].plot(example.step, example.battery_power_kw, label="Applied", linewidth=1.2)
    axes[0, 0].axhline(0, color="gray", linewidth=0.5)
    axes[0, 0].set(title="First matched week at selected design", xlabel="Hour", ylabel="Battery power (kW)")
    axes[0, 0].legend()
    axes[1, 0].plot(example.step, example.battery_soc_before_fraction * 100, color="tab:green")
    axes[1, 0].set(xlabel="Hour", ylabel="Battery charge before action (%)", ylim=(-2, 102))
    points = axes[0, 1].scatter(
        mode.requested_battery_power_kw, mode.battery_power_kw,
        c=mode.battery_soc_before_fraction * 100, cmap="viridis", s=8, alpha=0.35, rasterized=True,
    )
    bound = max(mode.requested_battery_power_kw.abs().max(), 0.1)
    axes[0, 1].plot([-bound, bound], [-bound, bound], "k--", linewidth=0.8)
    axes[0, 1].set(title="All matched weeks at selected design", xlabel="Requested power (kW)", ylabel="Applied power (kW)")
    fig.colorbar(points, ax=axes[0, 1], label="Charge before action (%)")
    x = np.arange(len(summary))
    axes[1, 1].bar(x - 0.18, summary.battery_power_limited * 100, 0.36, label="Power request limited")
    blocked = summary.blocked_discharge + summary.blocked_charge
    axes[1, 1].bar(x + 0.18, blocked * 100, 0.36, label="Nonzero request applied as zero")
    axes[1, 1].set(xticks=x, xticklabels=summary.design, ylabel="Steps (%)", ylim=(0, 100))
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"DDPG checkpoint {iteration}: deterministic validation, positive power = discharge")
    fig.savefig(output / "battery_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = list(summary.design)
    levels = sorted(probes.noise_std.unique())
    width = 0.8 / len(levels)
    for index, noise in enumerate(levels):
        values = probes[probes.noise_std == noise].set_index("design").loc[labels]
        positions = np.arange(len(labels)) + (index - (len(levels) - 1) / 2) * width
        axes[0].bar(positions, values.applied_std_kw, width, label=f"Noise SD {noise:.3f}")
        axes[1].bar(positions, values.blocked_request_fraction * 100, width)
    axes[0].set(ylabel="Mean within-state applied-power SD (kW)", title="Physical variation caused by noise")
    axes[1].set(ylabel="Probes (%)", title="Nonzero requests applied as zero", ylim=(0, 100))
    for axis in axes:
        axis.set_xticks(np.arange(len(labels)), labels)
    axes[0].legend()
    fig.suptitle("Same-state battery probes: not retraining or noisy trajectory evaluation")
    fig.savefig(output / "noise_probe.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replay a local DDPG checkpoint without training")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--probe-draws", type=int, default=16)
    parser.add_argument("--probe-seed", type=int, default=701)
    parser.add_argument("--comparison-noise-std", type=float, default=0.40)
    args = parser.parse_args()
    if args.episodes <= 0 or args.probe_draws < 2 or args.probe_seed < 0:
        parser.error("Episodes must be positive, probe draws at least 2 and probe seed non-negative")
    if not np.isfinite(args.comparison_noise_std) or args.comparison_noise_std < 0:
        parser.error("Comparison noise must be finite and non-negative")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("Output directory must be empty")
    torch.set_num_threads(1)
    # Read once so a new training checkpoint cannot change this evaluation
    payload = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = checkpoint["rl_config"]
    if config["algorithm"]["name"] != "ddpg":
        parser.error("This diagnostic requires a DDPG checkpoint")
    training = config["training"]
    data_seed = training.get("dataset_seed", checkpoint["seed"])
    data = generate_dataset(year=DEFAULT_YEAR, days=training["dataset_days"], seed=data_seed)
    trainer = CoDesignTrainer(checkpoint["system_config"], data, config, seed=checkpoint["seed"])
    trainer.agent.actor.load_state_dict(checkpoint["agent"]["actor"])
    trainer.agent.actor.eval()
    trainer.design_distribution.load_state_dict(checkpoint["design_distribution"])
    trainer.completed_iterations = int(checkpoint["completed_iterations"])
    trainer._set_exploration_noise(max(trainer.completed_iterations - 1, 0))
    noise_levels = sorted(set([0.0, trainer.agent.exploration_noise_std, args.comparison_noise_std]))
    scenario_seed = int(training.get("validation_seed", 1_000_042))
    designs = {"mode": trainer._design_mode()}
    for index, design in enumerate(training.get("reference_designs", [[4, 4.5], [6.6, 4.5], [9, 4.5]])):
        designs[f"reference_{index}"] = np.asarray(design, dtype=np.float32)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "checkpoint_snapshot.pt").write_bytes(payload)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "iteration": trainer.completed_iterations, "episodes_per_design": args.episodes,
        "scenario_seed": scenario_seed, "dataset_seed": data_seed, "dataset_year": DEFAULT_YEAR,
        "noise_levels": noise_levels, "probe_draws": args.probe_draws, "probe_seed": args.probe_seed,
        "designs": {key: list(map(float, value)) for key, value in designs.items()},
        "protocol": "Deterministic validation with paired same-state battery noise probes, no learning",
        "probe_limitation": "Probes do not advance state, perturb EV actions or establish training benefit",
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (args.output_directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    all_steps, all_probes = [], []
    for label, design in designs.items():
        for episode in range(args.episodes):
            seed = scenario_seed + episode
            steps, probes = collect_episode(trainer, design, seed, noise_levels, args.probe_draws, args.probe_seed)
            identifiers = dict(design=label, scenario_seed=seed, pv_capacity_kwp=design[0], battery_capacity_kwh=design[1])
            all_steps.append(steps.assign(**identifiers))
            all_probes.append(probes.assign(**identifiers))
        print(f"Evaluated {label}: {args.episodes} matched weeks at checkpoint {trainer.completed_iterations}", flush=True)
    steps = pd.concat(all_steps, ignore_index=True)
    probes = pd.concat(all_probes, ignore_index=True)
    summary = steps.groupby("design", sort=False)[[
        "battery_power_limited", "blocked_discharge", "blocked_charge", "battery_at_minimum", "battery_at_capacity",
    ]].mean()
    summary["mean_weekly_cost_chf"] = steps.groupby(["design", "scenario_seed"]).total_cost_chf.sum().groupby("design").mean()
    summary = summary.reset_index()
    probe_summary = probes.groupby(["design", "noise_std"], sort=False)[[
        "requested_std_kw", "applied_std_kw", "action_clipped_fraction", "power_limited_fraction", "blocked_request_fraction",
    ]].mean().reset_index()
    steps.to_csv(args.output_directory / "action_traces.csv", index=False)
    probes.to_csv(args.output_directory / "noise_probes.csv", index=False)
    summary.to_csv(args.output_directory / "summary.csv", index=False)
    probe_summary.to_csv(args.output_directory / "noise_summary.csv", index=False)
    make_plots(steps, summary, probe_summary, args.output_directory, trainer.completed_iterations)
    print(summary.round(4).to_string(index=False))
    print(probe_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
