"""Convert the reconstructed Cauz et al. 2021 building data into this project's dataset format.

Usage: python prepare_cauz_dataset.py [path/to/cauz_building_2021_reconstructed.csv]

The source CSV has load and normalised PV only. Tariffs come from generate_tariffs (paper Table 4).
The source validation blocks start at 02:00/03:00 (figure digitisation / timezone offset), so each
block is snapped back to midnight and kept at 168 h, which the episode sampler requires.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import generate_tariffs, save_dataset

DEFAULT_SOURCE = Path.home() / "Downloads" / "cauz_building_2021_reconstructed.csv"
OUTPUT_PATH = Path(__file__).with_name("cauz_dataset.csv")
VALIDATION_HOURS = 168


def build_validation_mask(source: pd.DataFrame) -> np.ndarray:
    timestamps = source["timestamp"]
    is_validation = (source["split"] == "validation").to_numpy()
    block_ids = np.cumsum(np.diff(is_validation.astype(int), prepend=0) == 1)

    mask = np.zeros(len(source), dtype=bool)
    for block in np.unique(block_ids[is_validation]):
        first = np.flatnonzero(is_validation & (block_ids == block))[0]
        start = first - int(timestamps.iloc[first].hour)  # back to midnight
        mask[start:start + VALIDATION_HOURS] = True
    return mask


def main() -> None:
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    source = pd.read_csv(source_path)
    source["timestamp"] = pd.to_datetime(source["timestamp"], format="%d-%m-%Y %H:%M")

    timestamps = pd.DatetimeIndex(source["timestamp"])
    import_price, export_price = generate_tariffs(timestamps)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "day": timestamps.dayofyear.to_numpy() - 1,
        "hour": timestamps.hour.to_numpy(),
        "weekday": timestamps.dayofweek.to_numpy(),
        "pv_per_kwp": source["pv_normalised_kWh_per_kWp"].clip(0.0, 1.0).to_numpy(),
        "load_kw": source["load_kWh"].to_numpy(),
        "import_price": import_price,
        "export_price": export_price,
        "is_validation": build_validation_mask(source),
    })

    save_dataset(data, OUTPUT_PATH)
    print(f"Saved {len(data)} rows to '{OUTPUT_PATH}'.")
    print("Training hours:", int((~data["is_validation"]).sum()))
    print("Validation hours:", int(data["is_validation"].sum()))
    print("Average load:", round(data["load_kw"].mean(), 2), "kW")
    print("PV capacity factor:", round(data["pv_per_kwp"].mean(), 4))


if __name__ == "__main__":
    main()
