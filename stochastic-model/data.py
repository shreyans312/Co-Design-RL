from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_YEAR = 2026

REQUIRED_COLUMNS = (
    "timestamp",
    "day",
    "hour",
    "weekday",
    "pv_per_kwp",
    "load_kw",
    "import_price",
    "export_price",
    "is_validation",
)

def create_hourly_index(year: int = DEFAULT_YEAR, days: int = 365) -> pd.DatetimeIndex:
    if not isinstance(year, int) or isinstance(year, bool):
        raise TypeError("year must be an integer")
    
    if not isinstance(days, int) or isinstance(days, bool):
        raise TypeError("days must be an integer")
    
    if days <= 0:
        raise ValueError("days must be a positive integer")
    
    return pd.date_range(start=f"{year}-01-01", periods=days * 24, freq="h")

def generate_pv_profile(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    hour = timestamps.hour.to_numpy(dtype=float)
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)
    
    daylight = np.sin(np.pi * (hour - 6.0) / 12.0)
    
    daylight_available = ((hour >= 6.0) & (hour <= 18.0))
    
    daylight = np.where(daylight_available, daylight, 0.0)
    
    daylight = np.clip(daylight, 0.0, 1.0)
    
    seasonal_factor = (
        0.75 + 0.25*np.sin(2.0*np.pi*(day_of_year - 80.0)/365.0)
    )
    
    cloud_factor = rng.uniform(low=0.5, high=1.0, size=len(timestamps))
    
    pv_per_kwp = daylight * seasonal_factor * cloud_factor
    
    return np.clip(pv_per_kwp, 0.0, 1.0)

def generate_load_profile(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    
    is_working_day = weekday < 5
    is_office_hour = (is_working_day) & (hour >= 8) & (hour < 18)
    is_morning_period = (is_working_day) & (hour >= 6) & (hour <= 9)
    base_load_kw = 1.5
    
    load_kw = np.full(len(timestamps), base_load_kw, dtype=float)
    load_kw += (is_office_hour.astype(float) * 1.8)
    load_kw += (is_morning_period.astype(float) * 0.4)
    random_noise = rng.normal(loc=0.0, scale=0.15, size=len(timestamps))
    load_kw += random_noise
    
    return np.clip(load_kw, 0.1, None)    

def generate_tariffs(timestamps: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    hour = timestamps.hour.to_numpy()
    
    morning_peak = (hour >= 6) & (hour <= 9)
    evening_peak = (hour >= 16) & (hour <= 21)
    
    peak_period = morning_peak | evening_peak
    
    import_price = np.where(peak_period, 0.50, 0.30).astype(float)
    export_price = np.zeros(len(timestamps), dtype=float)
    
    return import_price, export_price

def generate_validation_mask(timestamps: pd.DatetimeIndex) -> np.ndarray:
    validation_mask = np.zeros(len(timestamps), dtype=bool)
    validation_start_days = (84, 175, 266, 357)
    
    validation_hours = 168
    
    for start_day in validation_start_days:
        start_index = start_day * 24
        end_index = start_index + validation_hours
        
        if start_index >= len(timestamps):
            continue
        
        validation_mask[start_index:min(end_index, len(timestamps))] = True
    
    return validation_mask

def generate_dataset(year: int = DEFAULT_YEAR, days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    timestamps = create_hourly_index(year=year, days=days)
    
    pv_per_kwp = generate_pv_profile(timestamps, rng)
    load_kw = generate_load_profile(timestamps, rng)
    import_price, export_price = generate_tariffs(timestamps)
    validation_mask = generate_validation_mask(timestamps)
    
    data = pd.DataFrame({
        "timestamp": timestamps,
        "day": timestamps.dayofyear.to_numpy() - 1,
        "hour": timestamps.hour.to_numpy(),
        "weekday": timestamps.dayofweek.to_numpy(),
        "pv_per_kwp": pv_per_kwp,
        "load_kw": load_kw,
        "import_price": import_price,
        "export_price": export_price,
        "is_validation": validation_mask,
    })
    
    validate_dataset(data)
    
    return data

def validate_dataset(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        
    if data.empty:
        raise ValueError("dataset cannot be empty")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("dataset contains missing values")

    if data["timestamp"].duplicated().any():
        raise ValueError("dataset contains duplicate timestamps")

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be chronological")

    if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
        raise TypeError("timestamp must contain datetime values")

    timestamp_differences = data["timestamp"].diff().dropna()
    if not timestamp_differences.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("timestamps must be exactly one hour apart")

    numeric_columns = (
        "day",
        "hour",
        "weekday",
        "pv_per_kwp",
        "load_kw",
        "import_price",
        "export_price",
    )

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"{column} must contain numeric values")

    if not data["day"].between(0, 364).all():
        raise ValueError("day must be between 0 and 364")

    if not data["hour"].between(0, 23).all():
        raise ValueError("hour must be between 0 and 23")

    if not data["weekday"].between(0, 6).all():
        raise ValueError("weekday must be between 0 and 6")

    if not data["pv_per_kwp"].between(0, 1).all():
        raise ValueError("pv_per_kwp must be between 0 and 1")

    if (data["load_kw"] < 0).any():
        raise ValueError("load_kw cannot be negative")

    if (data[["import_price", "export_price"]] < 0).any().any():
        raise ValueError("electricity prices cannot be negative")

    if not pd.api.types.is_bool_dtype(data["is_validation"]):
        raise TypeError("is_validation must contain boolean values")
    
def save_dataset(data: pd.DataFrame, path: str | Path) -> None:
    validate_dataset(data)
    output_path = Path(path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    
def load_dataset(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")
    
    data = pd.read_csv(input_path, parse_dates=["timestamp"])
    
    validate_dataset(data)
    return data

if __name__ == "__main__":
    dataset = generate_dataset(year=DEFAULT_YEAR, days=365, seed=42)
    print(dataset.head())
    print()
    print("Total rows in the dataset:", len(dataset))
    print("Training hours:", (~dataset["is_validation"]).sum())
    print("Validation hours:", dataset["is_validation"].sum())
    print("Average load:", round(dataset["load_kw"].mean(), 2), "kW")
    
    output_path = Path(__file__).with_name("synthetic_dataset.csv")
    save_dataset(dataset, output_path)
    print(f"Synthetic dataset saved to '{output_path}'.")