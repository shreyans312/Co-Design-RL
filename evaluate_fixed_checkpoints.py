import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_rl import CoDesignTrainer, load_experiment_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--scenario-seed", type=int, default=30_000_000)
    args = parser.parse_args()

    with (args.run_directory / "configuration.json").open() as file:
        configuration = json.load(file)
    rl_config = configuration["rl"]
    training = rl_config["training"]
    training.setdefault("dataset_seed", rl_config["seed"])
    data = load_experiment_dataset(training)
    trainer = CoDesignTrainer(
        configuration["system"], data, rl_config, seed=rl_config["seed"]
    )
    design = np.asarray(rl_config["training"]["fixed_design_values"], dtype=np.float32)
    records = []

    for iteration in args.iterations:
        checkpoint_path = (
            args.run_directory / "checkpoints" / f"iteration_{iteration:04d}.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        trainer.agent.network.load_state_dict(checkpoint["agent"]["network"])
        for episode in range(args.episodes):
            result = trainer._run_episode(
                design=design,
                split="validation",
                seed=args.scenario_seed + episode,
                training=False,
                stochastic_policy=False,
            )
            records.append({
                "training_seed": rl_config["seed"],
                "iteration": iteration,
                "episode": episode + 1,
                "scenario_seed": args.scenario_seed + episode,
                "cost_chf": result["cost_chf"],
                "battery_action_saturation": result["battery_action_saturation"],
                "ev_action_saturation": result["ev_action_saturation"],
            })

    pd.DataFrame(records).to_csv(args.run_directory / "checkpoint_test.csv", index=False)


if __name__ == "__main__":
    main()
