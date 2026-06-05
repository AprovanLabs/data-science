"""Open-Meteo weather integration for fishing intelligence.

Provides real hourly weather data (air temp, barometric pressure, wind, cloud cover)
for a given lake's coordinates and date. Routes to the forecast API for future dates
and the historical archive API for past dates.

Main entry point:
    get_weather_for_window(lat, lon, date, start_hour, end_hour) -> WeatherResult

Falls back to monthly averages if the API is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

_US_CENTRAL = timezone(timedelta(hours=-5))

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_HOURLY_VARS = [
    "temperature_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]

_MONTHLY_WATER_TEMP: Dict[int, float] = {
    1: 33, 2: 33, 3: 35, 4: 44,
    5: 56, 6: 66, 7: 74, 8: 72,
    9: 63, 10: 51, 11: 39, 12: 34,
}

_MONTHLY_CLOUD_COVER: Dict[int, float] = {
    1: 65, 2: 60, 3: 55, 4: 50,
    5: 48, 6: 40, 7: 32, 8: 35,
    9: 42, 10: 50, 11: 60, 12: 65,
}


class PressureTrend(Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    RISING_AFTER_DROP = "rising_after_drop"
    HIGH_STEADY = "high_steady"


@dataclass
class PressureInterpretation:
    trend: PressureTrend
    label: str
    fishing_note: str


@dataclass
class WeatherResult:
    air_temp_f: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pressure_inhg: Optional[float] = None
    wind_speed_mph: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    pressure_trend: Optional[PressureTrend] = None
    pressure_interpretation: Optional[PressureInterpretation] = None
    source: str = "open-meteo"
    fallback_used: bool = False


def _wind_direction_label(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


wind_direction_label = _wind_direction_label


def _estimate_monthly_water_temp(query_date: date) -> float:
    month = query_date.month
    base = _MONTHLY_WATER_TEMP[month]
    next_month = (month % 12) + 1
    next_base = _MONTHLY_WATER_TEMP[next_month]
    day_fraction = (query_date.day - 1) / 30.0
    return base + day_fraction * (next_base - base)


def _estimate_monthly_cloud_cover(query_date: date) -> float:
    month = query_date.month
    base = _MONTHLY_CLOUD_COVER[month]
    next_month = (month % 12) + 1
    next_base = _MONTHLY_CLOUD_COVER[next_month]
    day_fraction = (query_date.day - 1) / 30.0
    return base + day_fraction * (next_base - base)


def _hpa_to_inhg(hpa: float) -> float:
    return hpa * 0.02953


def _compute_pressure_trend(
    current: float,
    pressure_6h_ago: Optional[float],
    pressure_12h_ago: Optional[float],
) -> PressureInterpretation:
    if pressure_6h_ago is None and pressure_12h_ago is None:
        return PressureInterpretation(
            trend=PressureTrend.STABLE,
            label="Stable",
            fishing_note="Normal fish activity expected.",
        )

    delta_6h = (current - pressure_6h_ago) if pressure_6h_ago is not None else 0.0
    delta_12h = (current - pressure_12h_ago) if pressure_12h_ago is not None else 0.0

    threshold_hpa = 1.0

    if delta_6h < -threshold_hpa:
        return PressureInterpretation(
            trend=PressureTrend.FALLING,
            label="Falling rapidly",
            fishing_note="Fish feed aggressively before a front — best bite window!",
        )

    if delta_6h > threshold_hpa and delta_12h < -threshold_hpa:
        return PressureInterpretation(
            trend=PressureTrend.RISING_AFTER_DROP,
            label="Rising after a drop",
            fishing_note="Fish resume feeding after pressure stabilizes — good action.",
        )

    if delta_6h > threshold_hpa:
        return PressureInterpretation(
            trend=PressureTrend.RISING,
            label="Rising",
            fishing_note="Fish may be less active — try deeper water and slower presentations.",
        )

    if current > 1020.0 and abs(delta_6h) < threshold_hpa:
        return PressureInterpretation(
            trend=PressureTrend.HIGH_STEADY,
            label="High and steady",
            fishing_note="High pressure = slower fishing. Focus on deep structure and live bait.",
        )

    return PressureInterpretation(
        trend=PressureTrend.STABLE,
        label="Stable",
        fishing_note="Normal fish activity expected.",
    )


def _fetch_hourly_data(
    lat: float,
    lon: float,
    query_date: date,
    logger: logging.Logger,
) -> Optional[Dict]:
    today = date.today()
    is_forecast = query_date >= today

    if is_forecast:
        url = FORECAST_URL
        start = today.isoformat()
        end = (today + timedelta(days=6)).isoformat()
    else:
        url = ARCHIVE_URL
        start = query_date.isoformat()
        end = query_date.isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_HOURLY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "pressure_unit": "hPa",
        "timezone": "America/Chicago",
        "start_date": start,
        "end_date": end,
    }

    logger.debug("Open-Meteo request: %s params=%s", url, params)
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.warning("Connection error fetching weather from Open-Meteo")
        return None
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching weather from Open-Meteo")
        return None
    except requests.exceptions.HTTPError:
        logger.warning("HTTP error from Open-Meteo: %s", resp.status_code)
        return None
    except Exception:
        logger.warning("Unexpected error fetching weather from Open-Meteo")
        return None

    try:
        data = resp.json()
    except Exception:
        logger.warning("JSON decode error from Open-Meteo response")
        return None

    hourly = data.get("hourly", {})
    if not hourly.get("time"):
        logger.warning("No hourly time data in Open-Meteo response")
        return None

    return hourly


def _extract_window_values(
    hourly: Dict,
    query_date: date,
    start_hour: int,
    end_hour: int,
) -> Dict[str, List[Optional[float]]]:
    times: List[str] = hourly.get("time", [])
    temps: List[Optional[float]] = hourly.get("temperature_2m", [])
    pressures: List[Optional[float]] = hourly.get("pressure_msl", [])
    wind_speeds: List[Optional[float]] = hourly.get("wind_speed_10m", [])
    wind_dirs: List[Optional[float]] = hourly.get("wind_direction_10m", [])
    clouds: List[Optional[float]] = hourly.get("cloud_cover", [])

    window_temps = []
    window_pressures = []
    window_winds = []
    window_wind_dirs = []
    window_clouds = []

    date_str = query_date.isoformat()
    for i, t in enumerate(times):
        if not t.startswith(date_str):
            continue
        hour = int(t.split("T")[1].split(":")[0])
        if start_hour <= hour <= end_hour:
            window_temps.append(temps[i] if i < len(temps) else None)
            window_pressures.append(pressures[i] if i < len(pressures) else None)
            window_winds.append(wind_speeds[i] if i < len(wind_speeds) else None)
            window_wind_dirs.append(wind_dirs[i] if i < len(wind_dirs) else None)
            window_clouds.append(clouds[i] if i < len(clouds) else None)

    return {
        "temps": window_temps,
        "pressures": window_pressures,
        "wind_speeds": window_winds,
        "wind_dirs": window_wind_dirs,
        "clouds": window_clouds,
    }


def _extract_pressure_at_hour(
    hourly: Dict,
    query_date: date,
    hour: int,
) -> Optional[float]:
    times: List[str] = hourly.get("time", [])
    pressures: List[Optional[float]] = hourly.get("pressure_msl", [])
    target = f"{query_date.isoformat()}T{hour:02d}:00"
    for i, t in enumerate(times):
        if t == target and i < len(pressures):
            val = pressures[i]
            return val if val is not None else None
    return None


def _avg(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def get_weather_for_window(
    lat: float,
    lon: float,
    query_date: date,
    start_hour: int = 17,
    end_hour: int = 21,
    logger: Optional[logging.Logger] = None,
) -> WeatherResult:
    """Fetch real weather for the evening fishing window.

    For future dates uses the Open-Meteo forecast API; for past dates uses the
    historical archive API. Falls back to monthly averages on any failure.

    Args:
        lat: Latitude of the lake.
        lon: Longitude of the lake.
        query_date: The date to fetch weather for.
        start_hour: Start of the fishing window (24h, default 17 = 5 PM CDT).
        end_hour: End of the fishing window (24h, default 21 = 9 PM CDT).
        logger: Optional logger.

    Returns:
        WeatherResult with averaged window values and pressure trend.
    """
    log = logger or logging.getLogger(__name__)

    hourly = _fetch_hourly_data(lat, lon, query_date, log)
    if hourly is None:
        return _fallback_result(query_date, start_hour, end_hour)

    window = _extract_window_values(hourly, query_date, start_hour, end_hour)

    avg_temp = _avg(window["temps"])
    avg_pressure = _avg(window["pressures"])
    avg_wind = _avg(window["wind_speeds"])
    avg_wind_dir = _avg(window["wind_dirs"])
    avg_cloud = _avg(window["clouds"])

    if avg_temp is None and avg_pressure is None:
        return _fallback_result(query_date, start_hour, end_hour)

    pressure_trend_interp = None
    if avg_pressure is not None:
        current_pressure = avg_pressure
        p_6h = _extract_pressure_at_hour(hourly, query_date, start_hour - 6) if start_hour >= 6 else None
        prev_day = query_date - timedelta(days=1)
        p_12h = _extract_pressure_at_hour(hourly, prev_day, end_hour - 12 + 24) if hourly.get("time") else None
        if p_12h is None and start_hour >= 12:
            p_12h = _extract_pressure_at_hour(hourly, query_date, start_hour - 12)
        pressure_trend_interp = _compute_pressure_trend(current_pressure, p_6h, p_12h)

    return WeatherResult(
        air_temp_f=avg_temp,
        pressure_hpa=avg_pressure,
        pressure_inhg=_hpa_to_inhg(avg_pressure) if avg_pressure is not None else None,
        wind_speed_mph=avg_wind,
        wind_direction_deg=avg_wind_dir,
        cloud_cover_pct=avg_cloud,
        pressure_trend=pressure_trend_interp.trend if pressure_trend_interp else None,
        pressure_interpretation=pressure_trend_interp,
        source="open-meteo",
        fallback_used=False,
    )


def _fallback_result(query_date: date, start_hour: int, end_hour: int) -> WeatherResult:
    avg_temp = _estimate_monthly_water_temp(query_date)
    avg_cloud = _estimate_monthly_cloud_cover(query_date)
    return WeatherResult(
        air_temp_f=avg_temp,
        cloud_cover_pct=avg_cloud,
        source="monthly_averages",
        fallback_used=True,
    )
