import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rl import CoDesignTrainer, load_experiment_dataset


def component_candidates(trainer):
    distribution = trainer.design_distribution
    with torch.no_grad():
        scales = torch.exp(distribution.component_log_stds)
        candidates = distribution.component_means
        if distribution.name == "lognormal":
            candidates = torch.exp(candidates - scales.square())
        candidates = torch.maximum(candidates, distribution.minimum)
        candidates = torch.minimum(candidates, distribution.maximum).numpy()
    return {f"component_{index}": design for index, design in enumerate(candidates) if distribution.active_mask[index]}


def evaluate_design(trainer, design, count, seed):
    return pd.DataFrame([
        {"scenario_seed": seed + index, "cost_chf": trainer._run_episode(
            design, "validation", seed + index, False, False,
        )["cost_chf"]}
        for index in range(count)
    ])


def main():
    parser = argparse.ArgumentParser(description="Compare component-design candidates without retraining DDPG")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--assessment-episodes", type=int, default=320)
    parser.add_argument("--assessment-seed", type=int, default=31_000_000)
    args = parser.parse_args()
    if args.validation_episodes < 1 or args.assessment_episodes < 2 or args.assessment_seed < 0:
        parser.error("Use positive validation episodes, at least two assessment episodes and a non-negative seed")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        parser.error("Output directory must be empty")
    torch.set_num_threads(1)
    payload = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = checkpoint["rl_config"]
    if config["algorithm"]["name"] != "ddpg":
        parser.error("A DDPG checkpoint is required")
    training = config["training"]
    validation_seed = int(training.get("validation_seed", 1_000_042))
    if max(validation_seed, args.assessment_seed) < min(
        validation_seed + max(args.validation_episodes, training["validation_episodes"]),
        args.assessment_seed + args.assessment_episodes,
    ):
        parser.error("Assessment seeds must not overlap selection or original validation seeds")
    data_seed = training.get("dataset_seed", checkpoint["seed"])
    training.setdefault("dataset_seed", data_seed)
    data = load_experiment_dataset(training)
    trainer = CoDesignTrainer(checkpoint["system_config"], data, config, seed=checkpoint["seed"])
    trainer.agent.actor.load_state_dict(checkpoint["agent"]["actor"])
    trainer.agent.actor.eval()
    trainer.design_distribution.load_state_dict(checkpoint["design_distribution"])
    original = trainer._design_mode()
    candidates = component_candidates(trainer)
    candidates["original_mode"] = original
    args.output_directory.mkdir(parents=True, exist_ok=True)
    cache, selection, ranking = {}, [], []
    for label, design in candidates.items():
        key = tuple(map(float, design))
        if key not in cache:
            cache[key] = evaluate_design(trainer, design, args.validation_episodes, validation_seed)
        rows = cache[key]
        selection.append(rows.assign(candidate=label, pv_capacity_kwp=key[0], battery_capacity_kwh=key[1]))
        ranking.append({"candidate": label, "pv_capacity_kwp": key[0], "battery_capacity_kwh": key[1],
                        "validation_cost_chf": float(rows.cost_chf.mean())})
        print(f"{label}: PV={key[0]:.3f}, battery={key[1]:.3f}, validation cost={rows.cost_chf.mean():.3f}", flush=True)
    ranking = pd.DataFrame(ranking).sort_values("validation_cost_chf", kind="stable")
    selected_label = str(ranking.iloc[0].candidate)
    selected = candidates[selected_label]
    pd.concat(selection, ignore_index=True).to_csv(args.output_directory / "selection_episodes.csv", index=False)
    ranking.to_csv(args.output_directory / "candidate_ranking.csv", index=False)
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "iteration": int(checkpoint["completed_iterations"]), "dataset": training.get("dataset", "synthetic"),
        "dataset_sha256": training.get("dataset_sha256"), "dataset_seed": data_seed,
        "dataset_year": int(data["timestamp"].iloc[0].year),
        "validation_seed": validation_seed, "validation_episodes": args.validation_episodes,
        "assessment_seed": args.assessment_seed, "assessment_episodes": args.assessment_episodes,
        "selected_candidate": selected_label, "selected_design": selected.tolist(), "original_design": original.tolist(),
        "protocol": "Select component candidates on validation, freeze choice, assess on disjoint scenario seeds",
        "scope": "Same trained actor and same annual dataset and held-out weeks, no retraining or cross-seed training comparison",
        "system_config": checkpoint["system_config"], "rl_config": config,
    }
    # Freeze the selection before generating assessment results
    (args.output_directory / "selection.json").write_text(json.dumps(result, indent=2))
    print(f"Selected {selected_label}; assessing on {args.assessment_episodes} new matched scenarios", flush=True)
    original_rows = evaluate_design(trainer, original, args.assessment_episodes, args.assessment_seed)
    selected_rows = original_rows if np.array_equal(original, selected) else evaluate_design(
        trainer, selected, args.assessment_episodes, args.assessment_seed,
    )
    difference = original_rows.cost_chf.to_numpy() - selected_rows.cost_chf.to_numpy()
    result.update(
        original_assessment_cost_chf=float(original_rows.cost_chf.mean()),
        selected_assessment_cost_chf=float(selected_rows.cost_chf.mean()),
        mean_paired_saving_chf=float(difference.mean()),
        paired_saving_standard_error_chf=float(difference.std(ddof=1) / np.sqrt(len(difference))),
        identical_designs=bool(np.array_equal(original, selected)),
    )
    pd.concat([
        original_rows.assign(design="original_mode"), selected_rows.assign(design="validation_selected"),
    ], ignore_index=True).to_csv(args.output_directory / "assessment_episodes.csv", index=False)
    (args.output_directory / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Original mode: {result['original_assessment_cost_chf']:.3f} CHF")
    print(f"Validation-selected design: {result['selected_assessment_cost_chf']:.3f} CHF")
    print(f"Paired saving: {result['mean_paired_saving_chf']:.3f} CHF, scenario SE={result['paired_saving_standard_error_chf']:.3f}")
    print(f"Saved results to {args.output_directory}")


if __name__ == "__main__":
    main()
