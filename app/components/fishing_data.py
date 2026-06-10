"""Shared fishing intelligence constants, utilities, and cached data loaders.

Single source of truth for data shared between the standalone entry point
(app/app.py) and the multi-page hub page (app/pages/fishing.py).

Drifted variants (different dash characters in _TECHNIQUES, emoji differences
in _TIME_OF_DAY_ADVICE) are resolved here in favour of the fishing.py style:
plain hyphens in depth ranges, no emoji labels, double-dash notation.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from onkia.dnr_client import MnDnrLakeTopographyService
from onkia.models import WaterTempPreference
from onkia.weather import get_weather_for_window

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SPECIES = {
    "Largemouth Bass": "LMB",
    "Northern Pike": "NOP",
    "Sunfish": "GSF",
    "Black Crappie": "BLC",
    "Walleye": "WAE",
}

SPECIES_CODE_MAP: Dict[str, str] = {
    "WAE": "Walleye",
    "NOP": "Northern Pike",
    "LMB": "Largemouth Bass",
    "SMB": "Smallmouth Bass",
    "BLC": "Black Crappie",
    "BLG": "Bluegill",
    "GSF": "Green Sunfish",
    "YEP": "Yellow Perch",
    "RKB": "Rock Bass",
    "PMK": "Pumpkinseed",
    "BRB": "Brown Bullhead",
    "BLB": "Black Bullhead",
    "CAP": "Common Carp",
    "WTS": "White Sucker",
    "TLC": "Tiger Muskellunge",
    "GOS": "Golden Shiner",
    "HSF": "Hybrid Sunfish",
    "BOF": "Buffalo",
}

SPECIES_TO_CODE: Dict[str, List[str]] = {
    "Walleye": ["WAE"],
    "Northern Pike": ["NOP"],
    "Largemouth Bass": ["LMB"],
    "Black Crappie": ["BLC"],
    "Sunfish": ["BLG", "GSF", "HSF", "PMK"],
}

SPECIES_COLORS = {
    "Walleye": "#f4a261",
    "Northern Pike": "#2a9d8f",
    "Largemouth Bass": "#264653",
    "Black Crappie": "#e76f51",
    "Sunfish": "#e9c46a",
}

_MONTHLY_WATER_TEMP = {
    1: 33, 2: 33, 3: 35, 4: 44,
    5: 56, 6: 66, 7: 74, 8: 72,
    9: 63, 10: 51, 11: 39, 12: 34,
}

_MONTHLY_CLOUD_COVER = {
    1: 65, 2: 60, 3: 55, 4: 50,
    5: 48, 6: 40, 7: 32, 8: 35,
    9: 42, 10: 50, 11: 60, 12: 65,
}

_TECHNIQUES: Dict[str, Dict] = {
    "Largemouth Bass": {
        "cold": {
            "lures": ["Jig with paddle tail", "Drop shot with finesse worm"],
            "depth": "12-20 ft near structure",
            "time": "Midday (warmest water)",
        },
        "optimal": {
            "lures": ["Topwater frog/popper", "Spinnerbait", "Soft plastic Texas rig"],
            "depth": "Shallow flats, weed edges (4-10 ft)",
            "time": "Early morning / late evening",
        },
        "warm": {
            "lures": ["Deep-diving crankbait", "Carolina rig", "Swimbait"],
            "depth": "Thermocline break (15-25 ft)",
            "time": "Night fishing or dawn",
        },
    },
    "Northern Pike": {
        "cold": {
            "lures": ["Jigging spoon", "Large blade bait"],
            "depth": "10-20 ft over weed flats",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Spinnerbait", "Inline spinner (Mepps)", "Sucker minnow on bobber"],
            "depth": "Weed edges 6-12 ft",
            "time": "Morning and evening",
        },
        "warm": {
            "lures": ["Large glide bait", "Musky-style bucktail"],
            "depth": "Deep weed beds or cool tributaries",
            "time": "Early morning",
        },
    },
    "Sunfish": {
        "cold": {
            "lures": ["Tiny jig (1/32 oz)", "Ice-fishing style teardrops"],
            "depth": "6-12 ft near structure",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Wax worm under bobber", "Beetle spin", "Small popper"],
            "depth": "Shallow weeds 2-6 ft",
            "time": "Morning",
        },
        "warm": {
            "lures": ["Night crawler pieces", "Small spinner"],
            "depth": "Slightly deeper 4-8 ft in shade",
            "time": "Early morning / evening",
        },
    },
    "Black Crappie": {
        "cold": {
            "lures": ["Tube jig 1/16 oz", "Marabou jig"],
            "depth": "15-25 ft suspended",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Small jig under bobber", "Minnow on hook", "Crappie tube"],
            "depth": "Brush piles / timber 8-15 ft",
            "time": "Dawn and dusk",
        },
        "warm": {
            "lures": ["Tiny swimbaits", "Inline spinner 1/8 oz"],
            "depth": "Shaded docks / deep structure",
            "time": "Night (crappie are active nocturnally in summer)",
        },
    },
    "Walleye": {
        "cold": {
            "lures": ["Jigging rap / jigging spoon", "Live sucker minnow"],
            "depth": "20-35 ft on hard bottom",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Lindy rig with night crawler", "Jig + minnow", "Crankbait troll"],
            "depth": "Weed edges and rock bars 8-18 ft",
            "time": "Dusk into darkness",
        },
        "warm": {
            "lures": ["Deep-troll crankbait", "Live minnow on slip sinker"],
            "depth": "Below thermocline 20-30 ft",
            "time": "Night",
        },
    },
}

_TIME_OF_DAY_ADVICE: Dict[str, Dict[str, str]] = {
    "dawn": {"label": "Dawn (5-7 AM)", "note": "Low light -- topwater and shallow patterns excel. Walleye bite windows."},
    "morning": {"label": "Morning (7-11 AM)", "note": "Sun rising -- fish move to cover. Try weed edges and submerged structure."},
    "midday": {"label": "Midday (11 AM-3 PM)", "note": "Bright sun -- fish go deep. Slow presentations at depth."},
    "evening": {"label": "Evening (3-7 PM)", "note": "Light fading -- fish feed actively. Crankbaits and spinners near weeds."},
    "night": {"label": "Night (7 PM-12 AM)", "note": "Walleye and crappie peak. Use glow jigs, slow troll minnows."},
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _estimate_water_temp(query_date: date) -> float:
    month = query_date.month
    base = _MONTHLY_WATER_TEMP[month]
    next_month = (month % 12) + 1
    next_base = _MONTHLY_WATER_TEMP[next_month]
    day_fraction = (query_date.day - 1) / 30.0
    return base + day_fraction * (next_base - base)


def _estimate_cloud_cover(query_date: date) -> float:
    month = query_date.month
    base = _MONTHLY_CLOUD_COVER[month]
    next_month = (month % 12) + 1
    next_base = _MONTHLY_CLOUD_COVER[next_month]
    day_fraction = (query_date.day - 1) / 30.0
    return base + day_fraction * (next_base - base)


def _temp_condition(temp_f: float, pref: WaterTempPreference) -> str:
    lower = pref.lower_avoidance
    upper = pref.upper_avoidance
    optimum_str = pref.optimum
    try:
        if "-" in optimum_str:
            opt_low, opt_high = (float(x) for x in optimum_str.split("-"))
        else:
            opt_low = opt_high = float(optimum_str)
    except (ValueError, AttributeError):
        return "unknown"
    if lower is not None and temp_f < lower:
        return "cold"
    if upper is not None and temp_f > upper:
        return "warm"
    if opt_low <= temp_f <= opt_high:
        return "optimal"
    if temp_f < opt_low:
        return "cold"
    return "warm"


def _parse_stocking(xml_text: str) -> List[Dict]:
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    records = []
    for item in root.iter():
        if item.tag in ("stocking", "record", "row"):
            record = {child.tag: child.text for child in item}
            if record:
                records.append(record)
    return records


def _build_depth_profile(max_depth: float, mean_depth: float, area_acres: float) -> plt.Figure:
    depths = np.linspace(0, max_depth, 50)
    fraction = 1.0 - (depths / max_depth) ** 0.6
    cumulative_area = area_acres * fraction
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_betweenx(depths, 0, cumulative_area, alpha=0.3, color="#457b9d")
    ax.plot(cumulative_area, depths, color="#1d3557", linewidth=2)
    ax.axhline(y=mean_depth, color="#e63946", linestyle="--", linewidth=1.5, label=f"Mean depth ({mean_depth:.0f} ft)")
    ax.set_xlabel("Cumulative Area (acres)")
    ax.set_ylabel("Depth (ft)")
    ax.set_title("Lake Depth Profile")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _build_water_temp_year_chart() -> plt.Figure:
    months = list(range(1, 13))
    temps = [_MONTHLY_WATER_TEMP[m] for m in months]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(months, temps, marker="o", color="#e63946", linewidth=2, markersize=6)
    ax.fill_between(months, temps, alpha=0.15, color="#e63946")
    ax.set_xlabel("Month")
    ax.set_ylabel("Water Temp (F)")
    ax.set_title("Seasonal Water Temperature -- Central MN Lakes")
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylim(25, 85)
    plt.tight_layout()
    return fig


def _build_cloud_cover_year_chart() -> plt.Figure:
    months = list(range(1, 13))
    cover = [_MONTHLY_CLOUD_COVER[m] for m in months]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(months, cover, color="#a8dadc", edgecolor="#457b9d")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cloud Cover (%)")
    ax.set_title("Average Cloud Cover -- Central MN")
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Searching DNR database...")
def _search_lake_cached(name: str, county_id: Optional[int] = 86) -> Optional[dict]:
    svc = MnDnrLakeTopographyService()
    # county_id=None searches statewide.
    lake = svc.get_lake(name, county_id=county_id)
    # by_alias keeps keys like "epsg:4326" that the map component expects.
    return lake.model_dump(by_alias=True) if lake else None


@st.cache_data(show_spinner="Fetching bathymetry contours from DNR...")
def _fetch_contours_cached(dow: str, lake_name: str) -> Optional[dict]:
    """Runtime contour fetch for lakes without pre-processed bathymetry files."""
    from onkia.bathymetry import fetch_contours_from_dnr

    return fetch_contours_from_dnr(lake_name, dow)


@st.cache_data(show_spinner="Loading survey data from DNR...")
def _load_survey(lake_id: str) -> Optional[dict]:
    try:
        svc = MnDnrLakeTopographyService()
        survey = svc.get_survey(lake_id)
        return survey.model_dump() if survey else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner="Fetching weather from Open-Meteo...")
def _get_weather_cached(lat: float, lon: float, query_date_str: str, start_hour: int = 17, end_hour: int = 21):
    qd = date.fromisoformat(query_date_str)
    return get_weather_for_window(lat, lon, qd, start_hour, end_hour)


@st.cache_data(show_spinner="Generating evening plan...")
def _generate_plan_cached(
    _weather_json: str,
    water_temp_f: float,
    _prefs_json: str,
    _survey_json: str,
    wind_dir_deg: Optional[float],
    start_hour: int,
    end_hour: int,
):
    import json as _json
    from onkia.plan_generator import generate_evening_plan as _gp
    from onkia.weather import WeatherResult as _WR, PressureTrend as _PT, PressureInterpretation as _PI
    from onkia.models import WaterTempPreference as _WTP

    _TREND_DEFAULTS = {
        _PT.FALLING: ("Falling rapidly", "Fish feed aggressively before a front — best bite window!"),
        _PT.RISING: ("Rising", "Fish may be less active — try deeper water and slower presentations."),
        _PT.RISING_AFTER_DROP: ("Rising after a drop", "Fish resume feeding after pressure stabilizes — good action."),
        _PT.HIGH_STEADY: ("High and steady", "High pressure = slower fishing. Focus on deep structure and live bait."),
        _PT.STABLE: ("Stable", "Normal fish activity expected."),
    }

    raw = _json.loads(_weather_json)
    if raw.get("pressure_trend"):
        trend = _PT(raw["pressure_trend"])
        raw["pressure_trend"] = trend
        label, note = _TREND_DEFAULTS.get(trend, (trend.value, ""))
        raw["pressure_interpretation"] = _PI(trend=trend, label=label, fishing_note=note)

    weather = _WR(**raw)
    pref_objs = [_WTP(**p) for p in _json.loads(_prefs_json)]
    survey = _json.loads(_survey_json) if _survey_json else None
    return _gp(weather, water_temp_f, pref_objs, survey, wind_dir_deg, start_hour, end_hour)


@st.cache_data(show_spinner="Loading stocking data from DNR...")
def _load_stocking_xml(lake_id: str) -> str:
    import requests as _req
    try:
        resp = _req.get(
            "https://files.dnr.state.mn.us/cgi-bin/lk_stocking.cgi",
            params={"downum": lake_id},
            timeout=15,
        )
        return resp.text
    except Exception:
        return ""


@st.cache_data(show_spinner="Loading survey data for {lake_name}...")
def _load_survey_for_lake(lake_id: str, lake_name: str) -> Optional[List[Dict]]:
    try:
        svc = MnDnrLakeTopographyService()
        overview = svc.get_survey(lake_id)
        if not overview:
            return None
    except Exception:
        return None
    records = []
    for sv in overview.surveys:
        survey_date = sv.survey_date
        for catch in sv.fish_catch_summaries:
            try:
                cpue = float(catch.cpue)
            except (ValueError, TypeError):
                cpue = None
            try:
                avg_w = float(catch.average_weight)
            except (ValueError, TypeError):
                avg_w = None
            records.append({
                "lake": lake_name,
                "lake_id": lake_id,
                "survey_date": survey_date,
                "species_code": catch.species,
                "species": SPECIES_CODE_MAP.get(catch.species, catch.species),
                "total_catch": catch.total_catch,
                "cpue": cpue,
                "average_weight": avg_w,
                "gear": catch.gear,
            })
    return records
