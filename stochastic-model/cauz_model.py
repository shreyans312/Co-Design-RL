"""Stochastic generator fitted to the reconstructed Cauz et al. (2021) office-building data.

    python cauz_model.py fit     # fit parameters from cauz_building_2021_reconstructed.csv -> cauz_model_params.json
    python cauz_model.py check   # compare generated years against the source data (table + figure)

Load  = daily baseline + weekday activity (one of three fitted weekday archetypes, random daily amplitude)
        + residual, where the residual is an AR(1) Gaussian copula mapped through the empirical residual
        quantiles (keeps the heavy tail), plus rare low-load outages.
PV    = clear-sky curve for 47 N x daily clearness (seasonal empirical distribution, AR(1) copula for
        day-to-day persistence) x hourly cloud noise.

Only the points actually read off the figure (the *_point_in_figure flags) are used for noise, tails and
autocorrelation, because the remaining points are linear interpolations and look too smooth.
"""
import json
import math
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data import generate_tariffs, validate_dataset

PARAMS_PATH = Path(__file__).with_name("cauz_model_params.json")
SOURCE_PATH = Path(__file__).with_name("cauz_building_2021_reconstructed.csv")
FIGURE_PATH = Path(__file__).resolve().parent.parent / "runs" / "cauz_model_fit.png"

YEAR = 2021
VALIDATION_START_DAYS = (79, 171, 263, 354)  # 21 Mar, 21 Jun, 21 Sep, 21 Dec (paper Fig. 4)
VALIDATION_HOURS = 168
LOAD_FLOOR_KW = 0.3
PERIODS = {"night": (19, 6), "morning": (6, 10), "midday": (10, 14), "afternoon": (14, 19)}  # residual noise differs by period
QUANTILE_LEVELS = np.linspace(0.0, 1.0, 201)
N_WEEKDAY_ARCHETYPES = 8
DARK_K, CLEAR_K = 0.12, 0.55  # daily clearness thresholds for the weather states
MONTH_MIDDLE_DAYS = np.array([15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0, 288.5, 319.0, 349.5])
LATITUDE_DEG = 47.0

_inverse_normal = np.vectorize(statistics.NormalDist().inv_cdf)
_erf = np.vectorize(math.erf)


def _period_masks() -> dict[str, np.ndarray]:
    hours = np.arange(24)
    return {name: ((hours >= a) | (hours < b)) if a > b else ((hours >= a) & (hours < b)) for name, (a, b) in PERIODS.items()}


def _normal_cdf(z):
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def _normal_scores(x: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(x)) + 0.5
    return _inverse_normal(ranks / len(x))


def _quantile_table(x: np.ndarray) -> list[float]:
    return np.quantile(x, QUANTILE_LEVELS).tolist()


def _seasonal(values_by_month, doy: np.ndarray) -> np.ndarray:
    """Cyclic linear interpolation of 12 monthly values to day of year."""
    values = np.asarray(values_by_month, dtype=float)
    days = np.concatenate(([MONTH_MIDDLE_DAYS[-1] - 365.0], MONTH_MIDDLE_DAYS, [MONTH_MIDDLE_DAYS[0] + 365.0]))
    padded = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(doy, days, padded)


def _daylight_shape(doy: int, noon: float, scale: float, power: float) -> np.ndarray:
    """Normalised (sums to 1) clear-sky hourly shape for one day."""
    declination = np.deg2rad(23.44) * np.sin(2.0 * np.pi * (284 + doy + 1) / 365.0)
    cos_sunset = np.clip(-np.tan(np.deg2rad(LATITUDE_DEG)) * np.tan(declination), -1.0, 1.0)
    half_day = np.arccos(cos_sunset) * 12.0 / np.pi * scale
    x = (np.arange(24) + 0.5 - noon) / half_day
    shape = np.where(np.abs(x) < 1.0, np.cos(np.pi * x / 2.0), 0.0) ** power
    total = shape.sum()
    return shape / total if total > 0 else shape


def _load_source(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw["timestamp"], format="%d-%m-%Y %H:%M")
    return pd.DataFrame({
        "doy": ts.dt.dayofyear - 1,
        "hour": ts.dt.hour,
        "month": ts.dt.month,
        "weekend": ts.dt.dayofweek >= 5,
        "load": raw["load_kWh"],
        "pv": raw["pv_normalised_kWh_per_kWp"],
        "load_obs": raw["load_point_in_figure"],
        "pv_obs": raw["pv_point_in_figure"],
    })


# ---------------------------------------------------------------- fitting

def _kmeans(points: np.ndarray, k: int, iterations: int = 50) -> np.ndarray:
    """Deterministic Lloyd k-means (farthest-point initialisation)."""
    centres = [points[np.argmax(points.sum(axis=1))]]
    while len(centres) < k:
        distance = np.min([np.sum((points - c) ** 2, axis=1) for c in centres], axis=0)
        centres.append(points[np.argmax(distance)])
    centres = np.array(centres)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        labels = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
        for c in range(k):
            if np.any(labels == c):
                centres[c] = points[labels == c].mean(axis=0)
    return labels


def _fit_load(frame: pd.DataFrame) -> dict:
    n_days = 365
    load = frame["load"].to_numpy().reshape(n_days, 24)
    observed = frame["load_obs"].to_numpy().reshape(n_days, 24)
    weekend = frame["weekend"].to_numpy().reshape(n_days, 24)[:, 0]
    month = frame["month"].to_numpy().reshape(n_days, 24)[:, 0]
    hours = np.arange(24)

    baseline = np.median(load[:, :6], axis=1)
    excess = load - baseline[:, None]
    shape_weekend = np.clip(excess[weekend].mean(axis=0), 0.0, None)

    weekday_index = np.flatnonzero(~weekend)
    labels = _kmeans(excess[weekday_index], N_WEEKDAY_ARCHETYPES)
    archetypes = []
    shape_by_day = np.zeros((n_days, 24))
    amplitude = np.ones(n_days)
    for cluster in range(N_WEEKDAY_ARCHETYPES):
        members = weekday_index[labels == cluster]
        shape = np.clip(excess[members].mean(axis=0), 0.0, None)
        active = shape > 0.2 * shape.max()
        amplitude[members] = np.clip((excess[members][:, active] @ shape[active]) / float(shape[active] @ shape[active]), 0.05, None)
        shape_by_day[members] = shape
        archetypes.append({"probability": len(members) / len(weekday_index), "shape": shape.tolist()})

    amplitude_mean = np.array([amplitude[(~weekend) & (month == m)].mean() for m in range(1, 13)])
    baseline_mean = np.array([baseline[month == m].mean() for m in range(1, 13)])
    amplitude_log_std = float(np.std(np.log(amplitude[~weekend] / amplitude_mean[month[~weekend] - 1])))
    baseline_std = float(np.std(baseline - baseline_mean[month - 1]))

    fitted = baseline[:, None] + np.where(weekend[:, None], shape_weekend[None, :], amplitude[:, None] * shape_by_day)
    residual = load - fitted

    tables, scores = {}, np.full(load.shape, np.nan)
    for name, day_mask in (("weekday", ~weekend), ("weekend", weekend)):
        for period, hour_mask in _period_masks().items():
            selection = np.zeros(load.shape, dtype=bool)
            selection[np.ix_(np.flatnonzero(day_mask), np.flatnonzero(hour_mask))] = True
            selection &= observed
            values = residual[selection]
            tables[f"{name}_{period}"] = _quantile_table(values)
            scores[selection] = _normal_scores(values)

    flat_scores, flat_observed = scores.reshape(-1), observed.reshape(-1)
    pairs = flat_observed[1:] & flat_observed[:-1]
    ar_coefficient = float(np.corrcoef(flat_scores[1:][pairs], flat_scores[:-1][pairs])[0, 1])

    # outages: contiguous runs of very low load
    low = (load.reshape(-1) < 1.0).astype(int)
    edges = np.diff(np.concatenate(([0], low, [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends - starts

    return {
        "baseline_mean_by_month": baseline_mean.tolist(),
        "baseline_std": baseline_std,
        "amplitude_mean_by_month": amplitude_mean.tolist(),
        "amplitude_log_std": amplitude_log_std,
        "weekday_archetypes": archetypes,
        "shape_weekend": shape_weekend.tolist(),
        "residual_quantiles": tables,
        "residual_ar": ar_coefficient,
        "outage_per_day": len(lengths) / n_days,
        "outage_lengths": lengths.tolist(),
        "outage_level_range": [LOAD_FLOOR_KW, 1.2],
        "load_cap": float(1.05 * load.max()),
        "residual_scale": 1.0,
        "amplitude_sigma_scale": 1.0,
    }


def _fit_pv(frame: pd.DataFrame) -> dict:
    n_days = 365
    pv = frame["pv"].to_numpy().reshape(n_days, 24)
    observed = frame["pv_obs"].to_numpy().reshape(n_days, 24)
    month = frame["month"].to_numpy().reshape(n_days, 24)[:, 0]
    doy = np.arange(n_days)
    energy = pv.sum(axis=1)

    # clear-sky daily energy: smooth seasonal curve through the monthly 90th percentile
    monthly_p90 = np.array([np.quantile(energy[month == m], 0.90) for m in range(1, 13)])
    log_p90 = np.log(np.clip(monthly_p90, 0.05, None))
    angle = 2.0 * np.pi * MONTH_MIDDLE_DAYS / 365.0
    design = np.column_stack([np.ones(12), np.cos(angle), np.sin(angle), np.cos(2 * angle), np.sin(2 * angle)])
    harmonics = np.linalg.lstsq(design, log_p90, rcond=None)[0]

    def clear_energy(days):
        a = 2.0 * np.pi * np.asarray(days, dtype=float) / 365.0
        basis = np.column_stack([np.ones_like(a), np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)])
        return np.exp(basis @ harmonics)

    clearness = np.clip(energy / clear_energy(doy), 0.0, 1.2)

    # clear-sky hourly shape: fitted so the energy-weighted mean hourly profile matches the source
    source_profile = pv.sum(axis=0) / pv.sum()
    best = (np.inf, 11.5, 1.0, 1.2)
    for noon in np.arange(10.5, 13.01, 0.25):
        for scale in np.arange(0.8, 1.81, 0.1):
            for power in np.arange(0.8, 5.01, 0.3):
                model = sum(energy[d] * _daylight_shape(d, noon, scale, power) for d in range(0, n_days, 3))
                error = float(np.sum((model / model.sum() - source_profile) ** 2))
                if error < best[0]:
                    best = (error, noon, scale, power)
    _, noon, scale, power = best

    # daily clearness: empirical distribution per month (+- 1 month window) and day-to-day AR(1) copula
    clearness_by_month = []
    for m in range(1, 13):
        window = np.isin(month, [(m - 2) % 12 + 1, m, m % 12 + 1])
        clearness_by_month.append(_quantile_table(clearness[window]))
    clearness_scores = _normal_scores(clearness + 1e-6 * np.arange(n_days))
    clearness_ar = float(np.corrcoef(clearness_scores[1:], clearness_scores[:-1])[0, 1])
    state = np.where(clearness < DARK_K, 0, np.where(clearness < CLEAR_K, 1, 2))

    # hourly cloud noise from the points read off the figure
    log_ratio = np.full(pv.shape, np.nan)
    for day in range(n_days):
        base = clear_energy(day) * clearness[day] * _daylight_shape(day, noon, scale, power)
        usable = observed[day] & (base > 0.02) & (pv[day] > 0.0)
        log_ratio[day, usable] = np.log(pv[day, usable] / base[usable])
    noise_std = []
    for s in range(3):
        values = log_ratio[state == s]
        values = values[np.isfinite(values)]
        noise_std.append(float(min(np.std(values), 1.0)) if len(values) > 20 else 0.5)
    left, right = log_ratio[:, :-1].reshape(-1), log_ratio[:, 1:].reshape(-1)
    both = np.isfinite(left) & np.isfinite(right)
    noise_ar = float(np.clip(np.corrcoef(left[both], right[both])[0, 1], 0.0, 0.9)) if both.sum() > 50 else 0.5

    return {
        "clear_energy_harmonics": harmonics.tolist(),
        "shape": {"noon": float(noon), "scale": float(scale), "power": float(power)},
        "clearness_quantiles_by_month": clearness_by_month,
        "clearness_ar": clearness_ar,
        "pv_cap": float(1.05 * pv.max()),
        "noise_std_by_state": noise_std,
        "noise_ar": noise_ar,
        "noise_scale": 1.0,
    }


def _calendar() -> tuple[np.ndarray, np.ndarray]:
    days = pd.date_range(f"{YEAR}-01-01", periods=365, freq="D")
    return days.dayofweek.to_numpy() >= 5, days.month.to_numpy()


def _load_targets(load: np.ndarray, weekend_by_day: np.ndarray) -> np.ndarray:
    days = load.reshape(-1, 24)
    daily_max = days[~weekend_by_day].max(axis=1)
    centred = (days - np.where(weekend_by_day[:, None], days[weekend_by_day].mean(axis=0), days[~weekend_by_day].mean(axis=0))).reshape(-1)
    lag_one = float(np.corrcoef(centred[1:], centred[:-1])[0, 1])
    return np.array([load.std(), np.quantile(load, 0.99), np.median(daily_max), np.quantile(daily_max, 0.95), lag_one])


def _calibrate(params: dict, frame: pd.DataFrame, n_years: int = 12) -> None:
    """Scale the noise terms so whole-series statistics match the source.

    The points read off the figure are biased towards peaks, so noise measured only there is too large;
    the all-rows statistics (which include the smooth interpolated parts) bound it from below.
    """
    weekend_by_day, month_by_day = _calendar()
    real_load, real_pv = frame["load"].to_numpy(), frame["pv"].to_numpy()
    real_energy = real_pv.reshape(-1, 24).sum(axis=1)

    load_targets = _load_targets(real_load, weekend_by_day)
    best = (np.inf, 1.0, 1.0)
    for residual_scale in np.arange(0.3, 1.21, 0.1):
        for amplitude_scale in (0.0, 0.25, 0.5, 0.75, 1.0):
            trial = dict(params["load"], residual_scale=float(residual_scale), amplitude_sigma_scale=amplitude_scale)
            stats = [_load_targets(_generate_load(trial, np.random.default_rng(seed), weekend_by_day), weekend_by_day) for seed in range(n_years)]
            error = float(np.sum((np.mean(stats, axis=0) / load_targets - 1.0) ** 2))
            if error < best[0]:
                best = (error, float(residual_scale), float(amplitude_scale))
    params["load"]["residual_scale"], params["load"]["amplitude_sigma_scale"] = best[1], best[2]

    def clearness_lag_one(pv_year: np.ndarray) -> float:
        energy = pv_year.reshape(-1, 24).sum(axis=1)
        p95 = np.array([np.quantile(energy[month_by_day == m], 0.95) for m in month_by_day])
        return float(pd.Series(energy / p95).autocorr(1))

    best = (np.inf, params["pv"]["clearness_ar"])
    for clearness_ar in np.arange(0.0, 0.81, 0.1):
        trial = dict(params["pv"], clearness_ar=float(clearness_ar), noise_scale=0.1)
        lag = np.mean([clearness_lag_one(_generate_pv(trial, np.random.default_rng(seed), month_by_day)) for seed in range(n_years)])
        error = abs(lag - clearness_lag_one(real_pv))
        if error < best[0]:
            best = (error, float(clearness_ar))
    params["pv"]["clearness_ar"] = best[1]

    pv_targets = np.array([real_pv.max(), np.median(real_energy), np.quantile(real_energy, 0.9), real_pv.mean()])
    best = (np.inf, 1.0)
    for noise_scale in np.arange(0.1, 1.01, 0.1):
        trial = dict(params["pv"], noise_scale=float(noise_scale))
        stats = []
        for seed in range(n_years):
            pv = _generate_pv(trial, np.random.default_rng(seed), month_by_day)
            energy = pv.reshape(-1, 24).sum(axis=1)
            stats.append([pv.max(), np.median(energy), np.quantile(energy, 0.9), pv.mean()])
        error = float(np.sum((np.mean(stats, axis=0) / pv_targets - 1.0) ** 2))
        if error < best[0]:
            best = (error, float(noise_scale))
    params["pv"]["noise_scale"] = best[1]


def fit_cauz_model(source_path: Path = SOURCE_PATH, output_path: Path = PARAMS_PATH) -> dict:
    frame = _load_source(source_path)
    params = {"load": _fit_load(frame), "pv": _fit_pv(frame)}
    _calibrate(params, frame)
    output_path.write_text(json.dumps(params, indent=1))
    _read_params.cache_clear()
    return params


@lru_cache(maxsize=1)
def _read_params() -> dict:
    return json.loads(PARAMS_PATH.read_text())


# ---------------------------------------------------------------- generation

def _generate_load(params: dict, rng: np.random.Generator, weekend_by_day: np.ndarray) -> np.ndarray:
    n_days = len(weekend_by_day)
    doy = np.arange(n_days)
    baseline = _seasonal(params["baseline_mean_by_month"], doy) + rng.normal(0.0, params["baseline_std"], n_days)
    sigma = params["amplitude_log_std"] * params["amplitude_sigma_scale"]
    amplitude = _seasonal(params["amplitude_mean_by_month"], doy) * np.exp(rng.normal(0.0, sigma, n_days) - sigma ** 2 / 2)
    shape_weekend = np.asarray(params["shape_weekend"])
    archetype_shapes = np.array([a["shape"] for a in params["weekday_archetypes"]])
    chosen = rng.choice(len(archetype_shapes), size=n_days, p=[a["probability"] for a in params["weekday_archetypes"]])
    profile = baseline[:, None] + np.where(weekend_by_day[:, None], shape_weekend[None, :], amplitude[:, None] * archetype_shapes[chosen])

    phi = params["residual_ar"]
    z = np.empty(n_days * 24)
    z[0] = rng.normal()
    shocks = rng.normal(size=n_days * 24)
    for t in range(1, len(z)):
        z[t] = phi * z[t - 1] + math.sqrt(1.0 - phi ** 2) * shocks[t]
    u = _normal_cdf(z).reshape(n_days, 24)

    residual = np.empty((n_days, 24))
    for name, day_mask in (("weekday", ~weekend_by_day), ("weekend", weekend_by_day)):
        for period, hour_mask in _period_masks().items():
            table = np.asarray(params["residual_quantiles"][f"{name}_{period}"])
            rows, cols = np.flatnonzero(day_mask), np.flatnonzero(hour_mask)
            residual[np.ix_(rows, cols)] = params["residual_scale"] * np.interp(u[np.ix_(rows, cols)], QUANTILE_LEVELS, table)

    load = (profile + residual).reshape(-1)

    level_low, level_high = params["outage_level_range"]
    for day in range(n_days):
        if rng.random() < params["outage_per_day"]:
            start = day * 24 + int(rng.integers(0, 24))
            length = int(rng.choice(params["outage_lengths"]))
            end = min(start + length, len(load))
            load[start:end] = np.linspace(level_low, level_high, end - start) + rng.normal(0.0, 0.08, end - start)
    return np.clip(load, LOAD_FLOOR_KW, params["load_cap"])


def _generate_pv(params: dict, rng: np.random.Generator, month_by_day: np.ndarray) -> np.ndarray:
    n_days = len(month_by_day)
    harmonics = np.asarray(params["clear_energy_harmonics"])
    shape = params["shape"]
    angle = 2.0 * np.pi * np.arange(n_days) / 365.0
    clear_energy = np.exp(np.column_stack([np.ones(n_days), np.cos(angle), np.sin(angle), np.cos(2 * angle), np.sin(2 * angle)]) @ harmonics)

    quantiles = [np.asarray(q) for q in params["clearness_quantiles_by_month"]]
    noise_std, phi = params["noise_std_by_state"], params["noise_ar"]

    clearness_phi = params["clearness_ar"]
    day_z = np.empty(n_days)
    day_z[0] = rng.normal()
    for day in range(1, n_days):
        day_z[day] = clearness_phi * day_z[day - 1] + math.sqrt(1.0 - clearness_phi ** 2) * rng.normal()
    day_u = _normal_cdf(day_z)

    pv = np.zeros((n_days, 24))
    for day in range(n_days):
        clearness = float(np.interp(day_u[day], QUANTILE_LEVELS, quantiles[month_by_day[day] - 1]))
        state = 0 if clearness < DARK_K else (1 if clearness < CLEAR_K else 2)
        sigma = noise_std[state] * params["noise_scale"]
        z = np.empty(24)
        z[0] = rng.normal()
        for h in range(1, 24):
            z[h] = phi * z[h - 1] + math.sqrt(1.0 - phi ** 2) * rng.normal()
        cloud = np.exp(sigma * z - sigma ** 2 / 2)
        pv[day] = clear_energy[day] * clearness * _daylight_shape(day, shape["noon"], shape["scale"], shape["power"]) * cloud
    return np.clip(pv.reshape(-1), 0.0, params["pv_cap"])


def generate_cauz_like_dataset(seed: int = 42) -> pd.DataFrame:
    """One synthetic year (2021 calendar) statistically matching the reconstructed Cauz data."""
    params = _read_params()
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=f"{YEAR}-01-01", periods=365 * 24, freq="h")
    weekend_by_day = timestamps[::24].dayofweek.to_numpy() >= 5
    month_by_day = timestamps[::24].month.to_numpy()

    load_kw = _generate_load(params["load"], rng, weekend_by_day)
    pv_per_kwp = _generate_pv(params["pv"], rng, month_by_day)
    import_price, export_price = generate_tariffs(timestamps)

    validation_mask = np.zeros(len(timestamps), dtype=bool)
    for start_day in VALIDATION_START_DAYS:
        validation_mask[start_day * 24:start_day * 24 + VALIDATION_HOURS] = True

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


# ---------------------------------------------------------------- checking

def _summary(load: np.ndarray, pv: np.ndarray, weekend_by_day: np.ndarray, month_by_day: np.ndarray) -> dict:
    load_days, pv_days = load.reshape(-1, 24), pv.reshape(-1, 24)
    energy = pv_days.sum(axis=1)
    centred = load_days - np.where(weekend_by_day[:, None], load_days[weekend_by_day].mean(axis=0), load_days[~weekend_by_day].mean(axis=0))
    flat = centred.reshape(-1)
    k = energy / np.array([np.quantile(energy[month_by_day == m], 0.95) for m in month_by_day])
    return {
        "load mean (kW)": load.mean(),
        "load std": load.std(),
        "load p95": np.quantile(load, 0.95),
        "load p99": np.quantile(load, 0.99),
        "load max": load.max(),
        "weekday mean": load_days[~weekend_by_day].mean(),
        "weekend mean": load_days[weekend_by_day].mean(),
        "weekday 08:00 mean": load_days[~weekend_by_day, 8].mean(),
        "weekday night (0-5) mean": load_days[~weekend_by_day, :6].mean(),
        "weekday daily-max median": np.median(load_days[~weekend_by_day].max(axis=1)),
        "weekday daily-max p95": np.quantile(load_days[~weekend_by_day].max(axis=1), 0.95),
        "load lag-1 autocorr": float(np.corrcoef(flat[1:], flat[:-1])[0, 1]),
        "hours load<1 kW (%)": 100.0 * (load < 1.0).mean(),
        "PV capacity factor": pv.mean(),
        "PV kWh/kWp/year": pv.sum(),
        "PV max": pv.max(),
        "PV daily energy median": np.median(energy),
        "PV daily energy p90": np.quantile(energy, 0.9),
        "days with PV<0.05 (%)": 100.0 * (energy < 0.05).mean(),
        "PV May mean": pv_days[month_by_day == 5].mean(),
        "PV Dec mean": pv_days[month_by_day == 12].mean(),
        "PV noon (11-12h) mean": pv_days[:, 11:13].mean(),
        "daily clearness lag-1": float(pd.Series(k).autocorr(1)),
    }


def check(n_years: int = 100) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = _load_source(SOURCE_PATH)
    real_load, real_pv = frame["load"].to_numpy(), frame["pv"].to_numpy()
    ts = pd.date_range(f"{YEAR}-01-01", periods=365 * 24, freq="h")
    weekend_by_day, month_by_day = ts[::24].dayofweek.to_numpy() >= 5, ts[::24].month.to_numpy()
    real = _summary(real_load, real_pv, weekend_by_day, month_by_day)

    years = [generate_cauz_like_dataset(seed) for seed in range(n_years)]
    sims = [_summary(y["load_kw"].to_numpy(), y["pv_per_kwp"].to_numpy(), weekend_by_day, month_by_day) for y in years]

    print(f"{'statistic':28s} {'source':>9s} {'model mean':>11s} {'model 5-95%':>16s}")
    for key, value in real.items():
        sim = np.array([s[key] for s in sims])
        print(f"{key:28s} {value:9.3f} {sim.mean():11.3f} {np.quantile(sim, 0.05):8.3f}-{np.quantile(sim, 0.95):<7.3f}")

    sample = years[0]
    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.1, 1.1, 1, 1], hspace=0.55, wspace=0.3)
    day_axis = np.arange(365 * 24) / 24.0
    for row, (title, load, pv) in enumerate((("Source: reconstructed paper data (Fig. 4)", real_load, real_pv),
                                              ("Model: one generated year (seed 0)", sample["load_kw"].to_numpy(), sample["pv_per_kwp"].to_numpy()))):
        axis = fig.add_subplot(grid[row, :])
        axis.plot(day_axis, load, lw=0.35, color="#2b6cb0", label="Load (kW)")
        axis.plot(day_axis, pv, lw=0.35, color="#ed8936", label="PV per kWp")
        for start in VALIDATION_START_DAYS:
            axis.axvspan(start, start + 7, color="grey", alpha=0.25, lw=0)
        axis.set_xlim(0, 365); axis.set_ylim(0, 9); axis.set_title(title, fontsize=10, loc="left")
        axis.set_xlabel("Day of year"); axis.legend(loc="upper right", ncol=2, fontsize=8)

    hours = np.arange(24)
    sim_days = np.stack([y["load_kw"].to_numpy().reshape(-1, 24) for y in years])
    axis = fig.add_subplot(grid[2, 0])
    real_days = real_load.reshape(-1, 24)
    for mask, name, color in ((~weekend_by_day, "weekday", "#2b6cb0"), (weekend_by_day, "weekend", "#718096")):
        axis.plot(hours, real_days[mask].mean(axis=0), color=color, lw=2, label=f"source {name}")
        axis.plot(hours, sim_days[:, mask].mean(axis=(0, 1)), color=color, lw=1.5, ls="--", label=f"model {name}")
    axis.set_title("Mean load by hour", fontsize=10, loc="left"); axis.set_xlabel("Hour"); axis.set_ylabel("kW"); axis.legend(fontsize=7)

    axis = fig.add_subplot(grid[2, 1])
    levels = np.linspace(0, 8.5, 120)
    sim_load = np.concatenate([y["load_kw"].to_numpy() for y in years])
    axis.semilogy(levels, [(real_load > x).mean() for x in levels], color="#2b6cb0", lw=2, label="source")
    axis.semilogy(levels, [(sim_load > x).mean() + 1e-9 for x in levels], color="#2b6cb0", lw=1.5, ls="--", label="model")
    axis.set_ylim(1e-4, 1); axis.set_title("Load exceedance (tail)", fontsize=10, loc="left"); axis.set_xlabel("kW"); axis.legend(fontsize=7)

    axis = fig.add_subplot(grid[2, 2])
    real_energy = real_pv.reshape(-1, 24).sum(axis=1)
    sim_energy = np.concatenate([y["pv_per_kwp"].to_numpy().reshape(-1, 24).sum(axis=1) for y in years])
    grid_e = np.linspace(0, 6.5, 100)
    axis.plot(grid_e, [(real_energy <= x).mean() for x in grid_e], color="#ed8936", lw=2, label="source")
    axis.plot(grid_e, [(sim_energy <= x).mean() for x in grid_e], color="#ed8936", lw=1.5, ls="--", label="model")
    axis.set_title("Daily PV energy CDF", fontsize=10, loc="left"); axis.set_xlabel("kWh/kWp/day"); axis.legend(fontsize=7)

    axis = fig.add_subplot(grid[3, 0])
    months = np.arange(1, 13)
    sim_pv_month = np.array([[y["pv_per_kwp"].to_numpy().reshape(-1, 24)[month_by_day == m].mean() for m in months] for y in years])
    axis.plot(months, [real_pv.reshape(-1, 24)[month_by_day == m].mean() for m in months], color="#ed8936", lw=2, label="source")
    axis.errorbar(months, sim_pv_month.mean(axis=0), yerr=sim_pv_month.std(axis=0), color="#ed8936", lw=1.5, ls="--", label="model ±1σ")
    axis.set_title("Mean PV by month", fontsize=10, loc="left"); axis.set_xlabel("Month"); axis.legend(fontsize=7)

    axis = fig.add_subplot(grid[3, 1])
    sim_pv_hour = np.stack([y["pv_per_kwp"].to_numpy().reshape(-1, 24).mean(axis=0) for y in years])
    axis.plot(hours, real_pv.reshape(-1, 24).mean(axis=0), color="#ed8936", lw=2, label="source")
    axis.plot(hours, sim_pv_hour.mean(axis=0), color="#ed8936", lw=1.5, ls="--", label="model")
    axis.set_title("Mean PV by hour", fontsize=10, loc="left"); axis.set_xlabel("Hour"); axis.legend(fontsize=7)

    axis = fig.add_subplot(grid[3, 2])
    week = slice(19 * 24, 26 * 24)
    axis.plot(np.arange(168), real_load[week], color="#2b6cb0", lw=1.2, label="source load")
    axis.plot(np.arange(168), sample["load_kw"].to_numpy()[week], color="#2b6cb0", lw=1.2, ls="--", label="model load")
    axis.set_title("One week (same calendar days)", fontsize=10, loc="left"); axis.set_xlabel("Hour of week"); axis.legend(fontsize=7)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=110, bbox_inches="tight")
    print("figure:", FIGURE_PATH)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "fit":
        fitted = fit_cauz_model()
        print("saved", PARAMS_PATH)
        print("load AR(1):", round(fitted["load"]["residual_ar"], 3), "| outages/day:", round(fitted["load"]["outage_per_day"], 4))
        print("calibration: residual_scale", round(fitted["load"]["residual_scale"], 2), "| amplitude_sigma_scale", round(fitted["load"]["amplitude_sigma_scale"], 2), "| PV noise_scale", round(fitted["pv"]["noise_scale"], 2))
        print("PV shape:", fitted["pv"]["shape"], "| noise std by state:", [round(x, 2) for x in fitted["pv"]["noise_std_by_state"]], "| noise AR:", round(fitted["pv"]["noise_ar"], 2))
    elif command == "check":
        check()
    else:
        raise SystemExit("usage: python cauz_model.py [fit|check]")
