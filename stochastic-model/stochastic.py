import numpy as np
import pandas as pd

from config import get_default_config, validate_config
from data import (DEFAULT_YEAR, generate_dataset, validate_dataset,)

VALID_SPLITS = {"train", "validation", "all"}

def normalize_arrival_weights(config: dict) -> np.ndarray:
    weights = np.asarray(config["ev"]["arrival_weights"], dtype=float)
    
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("ev.arrival_weights must sum to a positive value")
    
    return weights / total_weight

def get_valid_episode_starts(data: pd.DataFrame, episode_hours: int, split: str) -> np.ndarray:
    validate_dataset(data)
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {VALID_SPLITS}.")
    
    if (not isinstance(episode_hours, int) or isinstance(episode_hours, bool)) or episode_hours <= 0:
        raise ValueError("episode_hours must be a positive integer")
    
    midnight_indices = np.flatnonzero(data["hour"].to_numpy() == 0)
    
    validation_mask = data["is_validation"].to_numpy(dtype=bool)
    
    valid_starts = []
    for start_index in midnight_indices:
        end_index = start_index + episode_hours
        
        if end_index > len(data):
            continue
        
        episode_validation_mask = validation_mask[start_index: end_index]
        
        if split == "train":
            if episode_validation_mask.any():
                continue
        elif split == "validation":
            if not episode_validation_mask.all():
                continue

        valid_starts.append(start_index)
    
    return np.asarray(valid_starts, dtype=int)
    
def sample_episode_start(data: pd.DataFrame, episode_hours: int, split: str, rng: np.random.Generator) -> int:
    valid_starts = get_valid_episode_starts(data=data, episode_hours=episode_hours, split=split)
    if len(valid_starts) == 0:
        raise ValueError(f"No valid episode starts found for split '{split}' with episode_hours={episode_hours}.")
    
    return int(rng.choice(valid_starts))

def sample_initial_battery_soc(config: dict, split: str, rng: np.random.Generator) -> float:
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {VALID_SPLITS}.")
    
    battery_capacity = config["design"]["battery_capacity_kwh"]
    minimum_soc_fraction = config["battery"]["minimum_soc_fraction"]
    
    minimum_soc_kwh = battery_capacity * minimum_soc_fraction
    
    if battery_capacity == 0: return 0.0
    
    if split == "validation":
        validation_soc = 0.5 * battery_capacity
        return float(max(validation_soc, minimum_soc_kwh))
    
    return float(rng.uniform(low = minimum_soc_kwh, high = battery_capacity))

def sample_ev_arrival_hour(config: dict, rng: np.random.Generator) -> int:
    arrival_hours = np.asarray(config["ev"]["arrival_hours"], dtype=int)
    
    arrival_probabilities = normalize_arrival_weights(config)
    
    return int(rng.choice(arrival_hours, p=arrival_probabilities))

def sample_ev_duration(config: dict, rng: np.random.Generator) -> int:
    minimum_duration = config["ev"]["minimum_duration_hours"]
    maximum_duration = config["ev"]["maximum_duration_hours"]
    
    return int(rng.integers(low=minimum_duration, high=maximum_duration + 1))

def sample_ev_arrival_soc(config: dict, rng: np.random.Generator) -> float:
    ev_capacity = config["ev"]["capacity_kwh"]
    minimum_soc_fraction = config["ev"]["minimum_soc_fraction"]
    
    minimum_soc_kwh = ev_capacity * minimum_soc_fraction
    
    return float(rng.uniform(low=minimum_soc_kwh, high=ev_capacity))

def generate_ev_schedule(episode_data: pd.DataFrame, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    validate_dataset(episode_data)
    validate_config(config)
    
    if episode_data.iloc[0]["hour"] != 0:
        raise ValueError("episode_data must begin at midnight")
    
    number_of_hours = len(episode_data)
    ev_present = np.zeros(number_of_hours, dtype=bool)
    
    ev_session_id = np.full(number_of_hours, fill_value=-1, dtype=int)
    ev_arrival = np.zeros(number_of_hours, dtype=bool)
    ev_arrival_soc_kwh = np.zeros(number_of_hours, dtype=float)
    
    session_id = 0
    for day_start_index in range(0, number_of_hours, 24):
        weekday = int(episode_data.iloc[day_start_index]["weekday"])
        
        is_weekday = weekday < 5
        
        if is_weekday:
            visit_probability = config["ev"]["weekday_visit_probability"]
        else:
            visit_probability = config["ev"]["weekend_visit_probability"]
            
        if rng.random() >= visit_probability: 
            continue
        
        arrival_hour = sample_ev_arrival_hour(config=config, rng=rng)
        duration_hours = sample_ev_duration(config=config, rng=rng)
        
        arrival_soc_kwh = sample_ev_arrival_soc(config=config, rng=rng)
        arrival_index = (day_start_index + arrival_hour)
        
        if arrival_index >= number_of_hours:
            continue
        
        departure_index = min(arrival_index + duration_hours, number_of_hours)
        
        ev_present[arrival_index: departure_index] = True
        ev_session_id[arrival_index: departure_index] = session_id
        ev_arrival[arrival_index] = True
        
        ev_arrival_soc_kwh[arrival_index] = arrival_soc_kwh
        session_id += 1
        
    return pd.DataFrame({
        "ev_present": ev_present,
        "ev_session_id": ev_session_id,
        "ev_arrival": ev_arrival,
        "ev_arrival_soc_kwh": ev_arrival_soc_kwh,
    })
    
def generate_episode_scenario(data: pd.DataFrame, config: dict, split: str = "train", seed: int = 42,) -> dict:
    validate_dataset(data)
    validate_config(config)
    
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
    
    rng = np.random.default_rng(seed)
    
    episode_hours = config["simulation"]["episode_hours"]
    
    start_index = sample_episode_start(data=data, episode_hours=episode_hours, split=split, rng=rng)
    end_index = (start_index + episode_hours)
    
    episode_data = (data.iloc[start_index: end_index].copy().reset_index(drop=True))
    
    initial_battery_soc_kwh = (sample_initial_battery_soc(config=config, split=split, rng=rng))
    
    ev_schedule = generate_ev_schedule(episode_data=episode_data, config=config, rng=rng)
    
    return {
        "seed": seed,
        "split": split,
        "start_index": start_index,
        "initial_battery_soc_kwh": initial_battery_soc_kwh,
        "episode_data": episode_data,
        "ev_schedule": ev_schedule,
    }
    
def validate_episode_scenario(scenario: dict, config: dict) -> None:
    required_keys = {
        "seed",
        "split",
        "start_index",
        "initial_battery_soc_kwh",
        "episode_data",
        "ev_schedule",
    }
    
    missing_keys = (required_keys - set(scenario.keys()))
    
    if missing_keys:
        raise ValueError(f"Scenario is missing keys: {sorted(missing_keys)}")
    
    episode_data = scenario["episode_data"]
    ev_schedule = scenario["ev_schedule"]
    
    expected_hours = config["simulation"]["episode_hours"]
    
    if len(episode_data) != expected_hours:
        raise ValueError("Episode data has the wrong length")
    
    if len(ev_schedule) != expected_hours:
        raise ValueError("EV schedule has the wrong length")
    
    if scenario["start_index"] % 24 != 0:
        raise ValueError("Episode must begin at midnight")
    
    battery_capacity = config["design"]["battery_capacity_kwh"]
    battery_minimum_soc = (
        battery_capacity * config["battery"]["minimum_soc_fraction"]
    )
    
    initial_soc = scenario["initial_battery_soc_kwh"]
    
    if not (battery_minimum_soc <= initial_soc <= battery_capacity):
        raise ValueError("Initial battery SOC is outside its limits")
    
    arrival_rows = ev_schedule["ev_arrival"]
    arrival_soc = ev_schedule.loc[arrival_rows, "ev_arrival_soc_kwh",]
    
    ev_minimum_soc = (config["ev"]["capacity_kwh"] * config["ev"]["minimum_soc_fraction"])
    
    ev_capacity = config["ev"]["capacity_kwh"]
    
    if not arrival_soc.between(ev_minimum_soc, ev_capacity).all():
        raise ValueError("An EV arrival SOC is outside its limits")
    
    non_arrival_soc = ev_schedule.loc[~arrival_rows, "ev_arrival_soc_kwh",]
    
    if not non_arrival_soc.eq(0.0).all():
        raise ValueError("EV arrival SOC must be zero outside arrivals")
    
if __name__ == "__main__":
    config = get_default_config()
    dataset = generate_dataset(
        year = DEFAULT_YEAR,
        days = 365,
        seed = 42,
    )
    
    scenario = generate_episode_scenario(
        data=dataset,
        config=config,
        split="train",
        seed=42,
    )
    
    validate_episode_scenario(scenario=scenario, config=config,)
    episode_data = scenario["episode_data"]
    ev_schedule = scenario["ev_schedule"]
    
    start_timestamp = episode_data.iloc[0]["timestamp"]
    
    print("Episode start:", start_timestamp)
    print("Initial battery SOC:", round(scenario["initial_battery_soc_kwh"], 2,), "kWh")
    
    print("EV visits:", int(ev_schedule["ev_arrival"].sum()))
    print()
    print(ev_schedule[ev_schedule["ev_present"]].head(20))
