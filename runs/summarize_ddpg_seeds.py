import json
from pathlib import Path

import numpy as np

SEEDS = (42, 43, 44)
RUNS_DIR = Path(__file__).parent / "ddpg_lognormal_fixed"


def load_validation(seed: int) -> dict:
    path = RUNS_DIR / f"seed_{seed}" / "validation.json"
    with path.open() as file:
        return json.load(file)


def main() -> None:
    rows = []
    for seed in SEEDS:
        validation = load_validation(seed)
        rows.append(
            {
                "seed": seed,
                "mean_cost_chf": validation["mean_cost_chf"],
                "mean_pv_capacity_kwp": validation["mean_pv_capacity_kwp"],
                "mean_battery_capacity_kwh": validation["mean_battery_capacity_kwh"],
                "std_reward_chf": validation["std_reward_chf"],
            }
        )

    header = f"{'seed':>6} {'cost_chf':>12} {'pv_kwp':>10} {'battery_kwh':>13} {'reward_std':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['seed']:>6} {row['mean_cost_chf']:>12.3f} {row['mean_pv_capacity_kwp']:>10.3f} "
            f"{row['mean_battery_capacity_kwh']:>13.3f} {row['std_reward_chf']:>12.3f}"
        )

    print("-" * len(header))
    for key in ("mean_cost_chf", "mean_pv_capacity_kwp", "mean_battery_capacity_kwh", "std_reward_chf"):
        values = np.array([row[key] for row in rows])
        print(f"{key}: mean={values.mean():.3f}  std={values.std():.3f}")


if __name__ == "__main__":
    main()
