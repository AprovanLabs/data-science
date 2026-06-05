"""USGS GLM depth-resolved lake temperature profiles.

Loads pre-processed USGS General Lake Model (GLM) temperature data
stored as per-lake parquet files in ``data/usgs_glm/``.  Falls back
gracefully when data is unavailable.

Each parquet file is named by DOW ID (e.g. ``86025200.parquet``) and
contains columns: ``date``, ``depth_m``, ``temp_c``.

Public API
----------
get_temperature_profile  - depth vs temperature for a given date
get_thermocline_depth    - thermocline depth in metres
get_climatological_profile - day-of-year average profile
get_species_depth_zones  - map species preferences to depth ranges
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "usgs_glm"

_THERMOCLINE_GRADIENT_THRESHOLD = 1.0


def _parquet_path(dow_id: str) -> Path:
    return _DATA_DIR / f"{dow_id}.parquet"


def _load_lake_data(dow_id: str) -> Optional[pd.DataFrame]:
    path = _parquet_path(dow_id)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        required = {"date", "depth_m", "temp_c"}
        if not required.issubset(df.columns):
            logger.warning("Parquet for %s missing columns %s", dow_id, required - set(df.columns))
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        logger.warning("Failed to read parquet for %s", dow_id, exc_info=True)
        return None


def available_lakes() -> List[str]:
    """Return DOW IDs that have pre-processed USGS GLM data."""
    if not _DATA_DIR.exists():
        return []
    return sorted(
        p.stem for p in _DATA_DIR.glob("*.parquet")
    )


def has_data(dow_id: str) -> bool:
    return _parquet_path(dow_id).exists()


def get_temperature_profile(
    dow_id: str,
    query_date: date,
) -> Optional[pd.DataFrame]:
    """Return depth-resolved temperature profile for *query_date*.

    If the exact date is not in the dataset (e.g. post-2022), the
    climatological average for that day-of-year is used instead.

    Returns a DataFrame with columns ``depth_m``, ``temp_c`` sorted by
    depth ascending, or ``None`` if the lake has no USGS data.
    """
    df = _load_lake_data(dow_id)
    if df is None:
        return None

    exact = df[df["date"].dt.date == query_date]
    if not exact.empty:
        profile = (
            exact.groupby("depth_m")["temp_c"]
            .mean()
            .reset_index()
            .sort_values("depth_m")
        )
        return profile

    doy = query_date.timetuple().tm_yday
    return get_climatological_profile(dow_id, doy)


def get_climatological_profile(
    dow_id: str,
    day_of_year: int,
) -> Optional[pd.DataFrame]:
    """Day-of-year climatological mean profile from 1980-2021 data.

    Uses a +/-7 day window around the target day-of-year to smooth
    interannual variability.
    """
    df = _load_lake_data(dow_id)
    if df is None:
        return None

    df["doy"] = df["date"].dt.dayofyear

    window_low = max(day_of_year - 7, 1)
    window_high = min(day_of_year + 7, 366)

    if window_low <= window_high:
        subset = df[(df["doy"] >= window_low) & (df["doy"] <= window_high)]
    else:
        subset = pd.concat([
            df[df["doy"] >= window_low],
            df[df["doy"] <= window_high],
        ])

    if subset.empty:
        return None

    profile = (
        subset.groupby("depth_m")["temp_c"]
        .mean()
        .reset_index()
        .sort_values("depth_m")
    )
    return profile


def get_thermocline_depth(
    dow_id: str,
    query_date: date,
) -> Optional[float]:
    """Compute thermocline depth (metres) for *query_date*.

    The thermocline is defined as the depth of maximum vertical
    temperature gradient, provided that gradient exceeds
    ``_THERMOCLINE_GRADIENT_THRESHOLD`` (1 °C/m).  Returns ``None``
    if the lake has no data or no thermocline is detectable (e.g.
    isothermal or ice-covered conditions).
    """
    profile = get_temperature_profile(dow_id, query_date)
    if profile is None or len(profile) < 3:
        return None

    depths = profile["depth_m"].values
    temps = profile["temp_c"].values

    gradients = np.abs(np.diff(temps)) / np.diff(depths)
    if gradients.size == 0:
        return None

    max_idx = int(np.argmax(gradients))
    max_gradient = gradients[max_idx]

    if max_gradient < _THERMOCLINE_GRADIENT_THRESHOLD:
        return None

    thermo_depth = 0.5 * (depths[max_idx] + depths[max_idx + 1])
    return float(thermo_depth)


def _f_to_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def _c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def _parse_optimum_range(optimum_str: str) -> Tuple[float, float]:
    """Parse optimum temperature string to (low_c, high_c) in Celsius."""
    try:
        if "-" in optimum_str:
            parts = optimum_str.split("-")
            low_f, high_f = float(parts[0]), float(parts[1])
        else:
            low_f = high_f = float(optimum_str)
        return _f_to_c(low_f), _f_to_c(high_f)
    except (ValueError, AttributeError):
        return (0.0, 100.0)


def get_species_depth_zones(
    dow_id: str,
    query_date: date,
    species_prefs: List[dict],
) -> Optional[List[Dict]]:
    """Map species temperature preferences to depth zones.

    Each entry in *species_prefs* must have keys ``species``, ``optimum``
    (a temperature string in °F, possibly a range like ``"64-70"``),
    ``lower_avoidance`` (°F or None), and ``upper_avoidance`` (°F or
    None).

    Returns a list of dicts:
    ``species, opt_temp_c, opt_depth_low_m, opt_depth_high_m,
    thermo_depth_m`` or ``None`` if the lake has no USGS data.
    """
    profile = get_temperature_profile(dow_id, query_date)
    if profile is None:
        return None

    thermo = get_thermocline_depth(dow_id, query_date)
    depths = profile["depth_m"].values
    temps = profile["temp_c"].values

    zones: List[Dict] = []
    for pref in species_prefs:
        species = pref.get("species", "")
        opt_low_c, opt_high_c = _parse_optimum_range(str(pref.get("optimum", "")))

        mask = (temps >= opt_low_c) & (temps <= opt_high_c)
        if not mask.any():
            zones.append({
                "species": species,
                "opt_temp_c": (opt_low_c + opt_high_c) / 2,
                "opt_depth_low_m": None,
                "opt_depth_high_m": None,
                "thermo_depth_m": thermo,
            })
            continue

        matching_depths = depths[mask]
        zones.append({
            "species": species,
            "opt_temp_c": (opt_low_c + opt_high_c) / 2,
            "opt_depth_low_m": float(matching_depths.min()),
            "opt_depth_high_m": float(matching_depths.max()),
            "thermo_depth_m": thermo,
        })

    return zones


def generate_sample_data(
    dow_id: str,
    lake_name: str,
    max_depth_m: float,
    lat: float,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate synthetic temperature data for testing / demo.

    Produces a physically plausible depth-resolved temperature
    climatology for a dimictic Minnesota lake.  NOT a substitute for
    real USGS data — only for development and testing.
    """
    import warnings

    warnings.warn(
        "generate_sample_data produces synthetic data for testing only",
        stacklevel=2,
    )

    out_dir = output_dir or _DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    doy_range = np.arange(1, 367)
    depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                        6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0,
                        20.0, 25.0, 30.0, 35.0, 40.0])
    depths = depths[depths <= max_depth_m]

    rows = []
    year = 2021
    for doy in doy_range:
        d = date(year, 1, 1) + timedelta(days=int(doy) - 1)

        surface_temp = _dimictic_surface_temp(doy, lat)
        for depth in depths:
            temp = _temperature_at_depth(surface_temp, depth, doy, lat)
            if temp is not None and not np.isnan(temp):
                rows.append({
                    "date": d.isoformat(),
                    "depth_m": round(float(depth), 2),
                    "temp_c": round(float(temp), 2),
                })

    df = pd.DataFrame(rows)
    out_path = out_dir / f"{dow_id}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote synthetic data for %s (%s) → %s", lake_name, dow_id, out_path)
    return out_path


def _dimictic_surface_temp(doy: int, lat: float) -> float:
    """Approximate surface temperature for a dimictic lake by day-of-year."""
    peak_doy = 200
    amplitude = 24.0
    mean_annual = 8.0

    if doy < 90 or doy > 355:
        return 1.0
    if doy < 120:
        return 1.0 + (doy - 90) * 0.07
    if doy > 330:
        return 1.0 + (355 - doy) * 0.1

    day_angle = 2.0 * np.pi * (doy - peak_doy) / 365.0
    return mean_annual + amplitude * np.cos(day_angle) * 0.5


def _temperature_at_depth(
    surface_temp: float,
    depth: float,
    doy: int,
    lat: float,
) -> float:
    """Estimate temperature at a given depth from surface temp and season."""
    if surface_temp <= 2.0:
        return max(surface_temp - 0.1 * depth, 0.5)

    is_stratified = surface_temp > 15.0 and 120 < doy < 300

    if is_stratified:
        thermo_depth = 5.0 + 2.0 * np.sin(2.0 * np.pi * (doy - 180) / 365.0)
        hypolimnion_temp = 5.0 + 1.5 * np.sin(2.0 * np.pi * (doy - 200) / 365.0)

        if depth <= thermo_depth:
            return surface_temp - (surface_temp - hypolimnion_temp) * (depth / thermo_depth) ** 1.5
        else:
            return hypolimnion_temp + 0.3 * (depth - thermo_depth) * 0.01
    else:
        cooling_rate = 0.15 if surface_temp > 8.0 else 0.05
        return surface_temp - cooling_rate * depth
