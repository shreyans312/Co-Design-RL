import argparse
import json
from pathlib import Path

from rl_config import get_rl_config, validate_rl_config
from train_rl import (
    DEFAULT_YEAR,
    CoDesignTrainer,
    _write_results,
    generate_dataset,
    get_default_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("ppo_gmm", "ppo_lognormal"), default="ppo_gmm")
    parser.add_argument("--pv-capacity", type=float, required=True)
    parser.add_argument("--battery-capacity", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--episodes-per-iteration", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--final-validation-episodes", type=int, default=320)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.entropy_coefficient is not None and args.entropy_coefficient < 0.0:
        parser.error("--entropy-coefficient must be non-negative")

    system_config = get_default_config()
    rl_config = get_rl_config(args.model)
    rl_config["seed"] = args.seed
    if args.entropy_coefficient is not None:
        rl_config["algorithm"]["entropy_coefficient"] = args.entropy_coefficient
    rl_config["training"].update(
        iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        validation_episodes=args.validation_episodes,
        design_warmup_fraction=0.0,
        fixed_design_fraction=1.0,
        prune_interval_fraction=0.0,
        fixed_design_values=[args.pv_capacity, args.battery_capacity],
        dataset_seed=args.data_seed,
    )
    validate_rl_config(rl_config)
    data = generate_dataset(
        year=DEFAULT_YEAR,
        days=rl_config["training"]["dataset_days"],
        seed=args.data_seed,
    )
    trainer = CoDesignTrainer(system_config, data, rl_config, seed=args.seed)
    history = trainer.train(
        log_every=args.log_every,
        output_directory=args.output_directory,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        validation_every=args.validation_every,
    )
    validation = trainer.evaluate(
        episodes=args.final_validation_episodes,
        sample_designs=False,
    )
    _write_results(args.output_directory, trainer, history, validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
