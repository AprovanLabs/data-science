"""AprovanLabs Data Science — Streamlit entry point.

Single-page app for Wright County fishing intelligence.
Run with:

    streamlit run app/app.py
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Make src/ importable
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from components.dnr_map import WRIGHT_COUNTY_LAKES, build_lake_map  # noqa: E402
from onkia.dnr_client import MnDnrLakeTopographyService  # noqa: E402
from onkia.water_temp import WATER_TEMP_PREFERENCES  # noqa: E402
from onkia.models import WaterTempPreference  # noqa: E402
from onkia.weather import get_weather_for_window, wind_direction_label  # noqa: E402
from onkia.plan_generator import generate_evening_plan  # noqa: E402

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
            "depth": "12–20 ft near structure",
            "time": "Midday (warmest water)",
        },
        "optimal": {
            "lures": ["Topwater frog/popper", "Spinnerbait", "Soft plastic Texas rig"],
            "depth": "Shallow flats, weed edges (4–10 ft)",
            "time": "Early morning / late evening",
        },
        "warm": {
            "lures": ["Deep-diving crankbait", "Carolina rig", "Swimbait"],
            "depth": "Thermocline break (15–25 ft)",
            "time": "Night fishing or dawn",
        },
    },
    "Northern Pike": {
        "cold": {
            "lures": ["Jigging spoon", "Large blade bait"],
            "depth": "10–20 ft over weed flats",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Spinnerbait", "Inline spinner (Mepps)", "Sucker minnow on bobber"],
            "depth": "Weed edges 6–12 ft",
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
            "depth": "6–12 ft near structure",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Wax worm under bobber", "Beetle spin", "Small popper"],
            "depth": "Shallow weeds 2–6 ft",
            "time": "Morning",
        },
        "warm": {
            "lures": ["Night crawler pieces", "Small spinner"],
            "depth": "Slightly deeper 4–8 ft in shade",
            "time": "Early morning / evening",
        },
    },
    "Black Crappie": {
        "cold": {
            "lures": ["Tube jig 1/16 oz", "Marabou jig"],
            "depth": "15–25 ft suspended",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Small jig under bobber", "Minnow on hook", "Crappie tube"],
            "depth": "Brush piles / timber 8–15 ft",
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
            "depth": "20–35 ft on hard bottom",
            "time": "Midday",
        },
        "optimal": {
            "lures": ["Lindy rig with night crawler", "Jig + minnow", "Crankbait troll"],
            "depth": "Weed edges and rock bars 8–18 ft",
            "time": "Dusk into darkness",
        },
        "warm": {
            "lures": ["Deep-troll crankbait", "Live minnow on slip sinker"],
            "depth": "Below thermocline 20–30 ft",
            "time": "Night",
        },
    },
}

_TIME_OF_DAY_ADVICE: Dict[str, Dict[str, str]] = {
    "dawn": {"label": "🌅 Dawn (5–7 AM)", "note": "Low light — topwater and shallow patterns excel. Walleye bite windows."},
    "morning": {"label": "☀️ Morning (7–11 AM)", "note": "Sun rising — fish move to cover. Try weed edges and submerged structure."},
    "midday": {"label": "🌤️ Midday (11 AM–3 PM)", "note": "Bright sun — fish go deep. Slow presentations at depth."},
    "evening": {"label": "🌇 Evening (3–7 PM)", "note": "Light fading — fish feed actively. Crankbaits and spinners near weeds."},
    "night": {"label": "🌙 Night (7 PM–12 AM)", "note": "Walleye and crappie peak. Use glow jigs, slow troll minnows."},
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AprovanLabs — Wright County Fishing",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "selected_lake_name" not in st.session_state:
    st.session_state["selected_lake_name"] = None
if "selected_lake_id" not in st.session_state:
    st.session_state["selected_lake_id"] = None
if "dnr_search_results" not in st.session_state:
    st.session_state["dnr_search_results"] = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## AprovanLabs 🎣")
    st.caption("Wright County, MN — Fishing Intelligence")
    st.divider()

    if st.button("🔄 Refresh Data", key="btn_refresh_data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared — data will reload on next action.")

    st.divider()

    # --- Lake selector in sidebar ---
    st.markdown("**Select a Lake**")
    lake_names = sorted(WRIGHT_COUNTY_LAKES.keys())
    cols = st.columns(3)
    for i, name in enumerate(lake_names):
        with cols[i % 3]:
            is_sel = name == st.session_state["selected_lake_name"]
            label = f"{'✓ ' if is_sel else ''}{name}"
            if st.button(label, key=f"pick_{name}", use_container_width=True):
                st.session_state["selected_lake_name"] = name
                st.session_state["selected_lake_id"] = WRIGHT_COUNTY_LAKES[name][2]

    st.divider()

    col_search, col_btn = st.columns([3, 1])
    with col_search:
        search_query = st.text_input(
            "Search DNR by name",
            placeholder="e.g. Clearwater",
            label_visibility="collapsed",
            key="sidebar_search",
        )
    with col_btn:
        do_search = st.button("Search", key="btn_search_dnr", use_container_width=True)

    if do_search and search_query:
        result = _search_lake_cached(search_query.strip())
        if result:
            st.session_state["dnr_search_results"] = [result]
            st.success(f"Found: **{result['name']}** (DOW: {result['id']})")
        else:
            st.session_state["dnr_search_results"] = []
            st.warning(f"No lake named '{search_query}' found in Wright County.")

    st.divider()
    if st.session_state["selected_lake_name"]:
        st.info(f"Selected: **{st.session_state['selected_lake_name']}**")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Searching DNR database…")
def _search_lake_cached(name: str) -> Optional[dict]:
    try:
        svc = MnDnrLakeTopographyService()
        lake = svc.get_lake(name, county_id=86)
        return lake.model_dump() if lake else None
    except Exception:
        return None


@st.cache_data(show_spinner="Loading survey data from DNR…")
def _load_survey(lake_id: str) -> Optional[dict]:
    try:
        svc = MnDnrLakeTopographyService()
        survey = svc.get_survey(lake_id)
        return survey.model_dump() if survey else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner="Fetching weather from Open-Meteo…")
def _get_weather_cached(lat: float, lon: float, query_date_str: str, start_hour: int = 17, end_hour: int = 21):
    from onkia.weather import get_weather_for_window as _gw
    qd = date.fromisoformat(query_date_str)
    return _gw(lat, lon, qd, start_hour, end_hour)


@st.cache_data(show_spinner="Generating evening plan…")
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
    from onkia.weather import WeatherResult as _WR
    weather = _WR(**_json.loads(_weather_json))
    prefs = [_json.loads(p) for p in _json.loads(_prefs_json)]
    from onkia.models import WaterTempPreference as _WTP
    pref_objs = [_WTP(**p) for p in prefs]
    survey = _json.loads(_survey_json) if _survey_json else None
    return _gp(weather, water_temp_f, pref_objs, survey, wind_dir_deg, start_hour, end_hour)


@st.cache_data(show_spinner="Loading stocking data from DNR…")
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


@st.cache_data(show_spinner="Loading survey data for {lake_name}…")
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


# ---------------------------------------------------------------------------
# Helpers
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
    ax.set_ylabel("Water Temp (°F)")
    ax.set_title("Seasonal Water Temperature — Central MN Lakes")
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
    ax.set_title("Average Cloud Cover — Central MN")
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("🎣 Wright County Fishing Intelligence")

# --- Lake map ---
st.subheader("🗺️ Lake Map")
fmap = build_lake_map(
    selected_lake=st.session_state["selected_lake_name"],
    search_results=st.session_state["dnr_search_results"],
)
map_data = st_folium(fmap, width="100%", height=400, returned_objects=["last_object_clicked"])

if map_data and map_data.get("last_object_clicked"):
    clicked = map_data["last_object_clicked"]
    clicked_lat = clicked.get("lat")
    clicked_lng = clicked.get("lng")
    if clicked_lat is not None:
        best_name, best_dist = None, float("inf")
        for name, (lat, lon, dow) in WRIGHT_COUNTY_LAKES.items():
            dist = (lat - clicked_lat) ** 2 + (lon - clicked_lng) ** 2
            if dist < best_dist:
                best_dist, best_name = dist, name
        for r in st.session_state["dnr_search_results"]:
            coords = r.get("point", {}).get("epsg:4326", [])
            if len(coords) >= 2:
                dist = (float(coords[1]) - clicked_lat) ** 2 + (float(coords[0]) - clicked_lng) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_name = r["name"]
                    st.session_state["selected_lake_id"] = r["id"]
        if best_name and best_dist < 0.001:
            if best_name in WRIGHT_COUNTY_LAKES:
                st.session_state["selected_lake_id"] = WRIGHT_COUNTY_LAKES[best_name][2]
            st.session_state["selected_lake_name"] = best_name

lake_name = st.session_state["selected_lake_name"]
lake_id = st.session_state["selected_lake_id"]

if not lake_name or not lake_id:
    st.info("Select a lake from the sidebar or click a marker on the map.")
    st.stop()

st.success(f"Selected: **{lake_name}** (DOW: {lake_id})")

# --- Date & time selectors ---
col_date, col_time = st.columns([1, 1])
with col_date:
    selected_date = st.date_input(
        "Date",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today() + timedelta(days=6),
    )
with col_time:
    time_of_day = st.selectbox(
        "Time of Day",
        options=list(_TIME_OF_DAY_ADVICE.keys()),
        format_func=lambda k: _TIME_OF_DAY_ADVICE[k]["label"],
    )

st.divider()

# --- Weather Conditions ---
st.subheader("🌤️ Evening Weather Conditions")

lake_coords = WRIGHT_COUNTY_LAKES.get(lake_name)
if lake_coords:
    lat, lon = lake_coords[0], lake_coords[1]
else:
    lat, lon = 45.17, -94.05

weather = _get_weather_cached(lat, lon, selected_date.isoformat())

col_temp, col_pressure, col_wind, col_cloud = st.columns(4)
with col_temp:
    temp_val = weather.air_temp_f
    if temp_val is not None:
        st.metric("Air Temp (5–9 PM)", f"{temp_val:.0f}°F")
    else:
        st.metric("Air Temp", "N/A")
with col_pressure:
    if weather.pressure_inhg is not None:
        pi = weather.pressure_interpretation
        trend_label = pi.label if pi else "—"
        st.metric("Barometric Pressure", f"{weather.pressure_inhg:.2f} inHg", delta=trend_label)
    else:
        st.metric("Barometric Pressure", "N/A")
with col_wind:
    if weather.wind_speed_mph is not None and weather.wind_direction_deg is not None:
        wdir = wind_direction_label(weather.wind_direction_deg)
        st.metric("Wind", f"{weather.wind_speed_mph:.1f} mph {wdir}")
    elif weather.wind_speed_mph is not None:
        st.metric("Wind Speed", f"{weather.wind_speed_mph:.1f} mph")
    else:
        st.metric("Wind", "N/A")
with col_cloud:
    if weather.cloud_cover_pct is not None:
        st.metric("Cloud Cover", f"{weather.cloud_cover_pct:.0f}%")
    else:
        st.metric("Cloud Cover", "N/A")

if weather.pressure_interpretation:
    st.info(f"**Pressure Trend:** {weather.pressure_interpretation.label} — {weather.pressure_interpretation.fishing_note}")

if weather.fallback_used:
    st.caption("⚠️ Using seasonal averages — Open-Meteo API unavailable. Pressure and wind data not available from fallback.")
else:
    st.caption(f"Data source: Open-Meteo ({weather.source})")

water_temp = weather.air_temp_f if weather.air_temp_f is not None else _estimate_water_temp(selected_date)
cloud_cover = weather.cloud_cover_pct if weather.cloud_cover_pct is not None else _estimate_cloud_cover(selected_date)

tod = _TIME_OF_DAY_ADVICE[time_of_day]
st.info(f"**{tod['label']}** — {tod['note']}")

# Historical charts
with st.expander("📈 Seasonal Temperature & Cloud Cover Charts"):
    fig_temp = _build_water_temp_year_chart()
    st.pyplot(fig_temp)
    plt.close(fig_temp)

    fig_cloud = _build_cloud_cover_year_chart()
    st.pyplot(fig_cloud)
    plt.close(fig_cloud)

    month = selected_date.month
    fig_combo, ax1 = plt.subplots(figsize=(8, 3))
    ax2 = ax1.twinx()
    months = list(range(1, 13))
    temps = [_MONTHLY_WATER_TEMP[m] for m in months]
    covers = [_MONTHLY_CLOUD_COVER[m] for m in months]
    ax1.plot(months, temps, marker="o", color="#e63946", linewidth=2, label="Water Temp (°F)")
    ax2.bar(months, covers, alpha=0.3, color="#457b9d", label="Cloud Cover (%)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Water Temp (°F)", color="#e63946")
    ax2.set_ylabel("Cloud Cover (%)", color="#457b9d")
    ax1.axvline(x=month, color="#264653", linestyle=":", linewidth=1.5, label="Selected month")
    ax1.set_xticks(months)
    ax1.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax1.set_title("Water Temp & Cloud Cover — Central MN")
    plt.tight_layout()
    st.pyplot(fig_combo)
    plt.close(fig_combo)

# --- Species Suitability ---
st.subheader("🐟 Species Suitability")

target_prefs = [p for p in WATER_TEMP_PREFERENCES if p.species in TARGET_SPECIES]
rows = []
for pref in WATER_TEMP_PREFERENCES:
    if pref.species not in TARGET_SPECIES:
        continue
    condition = _temp_condition(water_temp, pref)
    condition_label = {"cold": "❄️ Cold", "optimal": "✅ Optimal", "warm": "🔥 Warm"}.get(condition, "—")
    rows.append({
        "Species": pref.species,
        "Condition": condition_label,
        "Lower Avoidance (°F)": pref.lower_avoidance or "—",
        "Optimum (°F)": pref.optimum,
        "Upper Avoidance (°F)": pref.upper_avoidance or "—",
    })

if rows:
    df_species = pd.DataFrame(rows)
    st.dataframe(df_species, use_container_width=True, hide_index=True)

optimal_species = [r["Species"] for r in rows if "Optimal" in r["Condition"]]
acceptable_species = [
    r["Species"] for r in rows
    if "Optimal" not in r["Condition"] and "Warm" not in r["Condition"]
]

if optimal_species:
    st.success(f"**Best targets today:** {', '.join(optimal_species)}")
elif acceptable_species:
    st.info(f"**Acceptable targets (not optimal):** {', '.join(acceptable_species)}")
else:
    st.warning("Water temperature outside preferred ranges for all target species — fishing may be slow.")

# --- Technique Suggestions ---
st.subheader("🪝 Fishing Technique Suggestions")

for pref in target_prefs:
    condition = _temp_condition(water_temp, pref)
    tech = _TECHNIQUES.get(pref.species, {}).get(condition)
    if not tech:
        continue
    with st.expander(f"{pref.species} — {condition.title()} conditions"):
        st.markdown(f"**Lures / Baits:** {', '.join(tech['lures'])}")
        st.markdown(f"**Target Depth:** {tech['depth']}")
        st.markdown(f"**Best Time of Day:** {tech['time']}")
        if time_of_day in ("dawn", "evening", "night"):
            st.markdown(f"🌙 **{time_of_day.title()} tip:** Low-light conditions favor active feeding — use noisy/scented presentations.")
        elif time_of_day == "midday":
            st.markdown("☀️ **Midday tip:** Fish deeper and slower. Try vertical jigging or live bait under a slip bobber.")

st.divider()

# --- Evening Fishing Plan ---
st.subheader("📅 Evening Fishing Plan (5:30–9:00 PM)")

col_start, col_end = st.columns(2)
with col_start:
    plan_start_hour = st.number_input("Start hour (24h)", min_value=0, max_value=23, value=17, key="plan_start")
with col_end:
    plan_end_hour = st.number_input("End hour (24h)", min_value=0, max_value=23, value=21, key="plan_end")

import json as _json

survey_records_for_plan = _load_survey_for_lake(lake_id, lake_name)
_weather_json = _json.dumps({
    "air_temp_f": weather.air_temp_f,
    "pressure_hpa": weather.pressure_hpa,
    "pressure_inhg": weather.pressure_inhg,
    "wind_speed_mph": weather.wind_speed_mph,
    "wind_direction_deg": weather.wind_direction_deg,
    "cloud_cover_pct": weather.cloud_cover_pct,
    "pressure_trend": weather.pressure_trend.value if weather.pressure_trend else None,
    "source": weather.source,
    "fallback_used": weather.fallback_used,
})
_prefs_json = _json.dumps([
    {"species": p.species, "lower_avoidance": p.lower_avoidance, "optimum": p.optimum, "upper_avoidance": p.upper_avoidance}
    for p in target_prefs
])
_survey_json = _json.dumps(survey_records_for_plan) if survey_records_for_plan else ""

plan = _generate_plan_cached(
    _weather_json, water_temp, _prefs_json, _survey_json,
    weather.wind_direction_deg, plan_start_hour, plan_end_hour,
)

if plan.blocks:
    st.markdown(f"**Conditions:** {plan.conditions_summary}")
    st.markdown("---")

    for i, block in enumerate(plan.blocks):
        col_time_block, col_detail = st.columns([1, 3])
        with col_time_block:
            st.markdown(f"#### {block.time_start} – {block.time_end}")
            st.markdown(f"**{block.species}**")
        with col_detail:
            st.markdown(f"📍 **Location:** {block.location}")
            st.markdown(f"🪝 **Technique:** {block.technique}")
            if block.lures:
                st.markdown(f"**Lures:** {', '.join(block.lures)}")
            if block.depth:
                st.markdown(f"**Depth:** {block.depth}")
            with st.expander("Supporting evidence"):
                for ev in block.evidence:
                    st.markdown(f"- {ev}")

        if i < len(plan.blocks) - 1:
            st.markdown("---")

    with st.expander("📊 Data sources"):
        for ds in plan.data_sources:
            st.markdown(f"- {ds}")
else:
    st.info("No plan blocks generated — adjust time window or check species preferences.")

st.divider()

# --- DNR Survey Data ---
st.subheader("📋 DNR Survey Data")

try:
    survey_data = _load_survey(lake_id)
except Exception:
    survey_data = None

if survey_data:
    surveys = survey_data.get("surveys", [])
    if surveys:
        surveys_sorted = sorted(surveys, key=lambda s: s.get("survey_date", ""), reverse=True)
        most_recent = surveys_sorted[0]
        survey_date_str = most_recent.get("survey_date", "Unknown")
        survey_type = most_recent.get("survey_type", "")

        st.caption(f"Most recent survey: **{survey_type}** on {survey_date_str} ({len(surveys)} surveys total)")

        catch_summaries = most_recent.get("fish_catch_summaries", [])
        if catch_summaries:
            catch_rows = []
            for c in catch_summaries:
                species_code = c.get("species", "")
                species_name = SPECIES_CODE_MAP.get(species_code, species_code)
                try:
                    cpue = float(c.get("cpue", 0))
                except (ValueError, TypeError):
                    cpue = 0.0
                catch_rows.append({
                    "Species": species_name,
                    "Code": species_code,
                    "Total Catch": c.get("total_catch", 0),
                    "CPUE": f"{cpue:.2f}",
                    "Avg Weight (lbs)": c.get("average_weight", "—"),
                    "Gear": c.get("gear", "—"),
                })
            df_catch = pd.DataFrame(catch_rows).sort_values("Total Catch", ascending=False)
            st.dataframe(df_catch, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(8, 3))
            species_labels = [r["Species"] for r in catch_rows]
            catches = [r["Total Catch"] for r in catch_rows]
            ax.barh(species_labels, catches, color="#457b9d")
            ax.set_xlabel("Total Catch")
            ax.set_title(f"Catch by Species — {survey_date_str}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No fish catch summaries in the most recent survey.")

        # Water clarity trend
        clarity_data = survey_data.get("water_clarity", [])
        if len(clarity_data) > 1:
            try:
                clarity_df = pd.DataFrame(clarity_data, columns=["date", "clarity_ft"])
                clarity_df["clarity_ft"] = pd.to_numeric(clarity_df["clarity_ft"], errors="coerce")
                clarity_df["date"] = pd.to_datetime(clarity_df["date"], errors="coerce")
                clarity_df = clarity_df.dropna()
                if not clarity_df.empty:
                    fig2, ax2 = plt.subplots(figsize=(8, 2.5))
                    ax2.plot(clarity_df["date"], clarity_df["clarity_ft"], marker="o", color="#1d3557")
                    ax2.set_ylabel("Clarity (ft)")
                    ax2.set_title("Water Clarity Over Time")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
            except Exception:
                pass

        # Lake morphology + depth profile
        with st.expander("🏞️ Lake Details & Depth Profile"):
            area = survey_data.get("area_acres", 0)
            max_d = survey_data.get("max_depth_feet", 0)
            mean_d = survey_data.get("mean_depth_feet", 0)
            st.markdown(f"- **Area:** {area} acres")
            st.markdown(f"- **Max Depth:** {max_d} ft")
            st.markdown(f"- **Mean Depth:** {mean_d} ft")
            st.markdown(f"- **Shore Length:** {survey_data.get('shore_length_miles', '—')} miles")
            avg_clarity = survey_data.get("average_water_clarity", "—")
            st.markdown(f"- **Avg Water Clarity:** {avg_clarity} ft")

            if max_d and area:
                fig_depth = _build_depth_profile(float(max_d), float(mean_d), float(area))
                st.pyplot(fig_depth)
                plt.close(fig_depth)
    else:
        st.info("No survey data available for this lake.")
else:
    st.warning("Could not load survey data from DNR — API may be unavailable. The fishing recommendations above still work based on seasonal estimates.")

st.divider()

# --- Stocking History ---
st.subheader("🐠 Stocking History")

try:
    stocking_xml = _load_stocking_xml(lake_id)
    stocking_records = _parse_stocking(stocking_xml)
except Exception:
    stocking_records = []

if stocking_records:
    df_stock = pd.DataFrame(stocking_records)
    st.dataframe(df_stock, use_container_width=True, hide_index=True)
else:
    st.info("No stocking records retrieved. This may mean no recent stocking, or the DNR API is unavailable.")

st.divider()

# --- Species Dashboard (cross-lake trends) ---
st.subheader("📊 Cross-Lake Species Trends")

selected_lakes = st.multiselect(
    "Include lakes",
    options=sorted(WRIGHT_COUNTY_LAKES.keys()),
    default=[lake_name] if lake_name in WRIGHT_COUNTY_LAKES else ["Clearwater"],
)
selected_species = st.multiselect(
    "Filter species",
    options=list(TARGET_SPECIES.keys()),
    default=list(TARGET_SPECIES.keys()),
)

if selected_lakes and selected_species:
    all_records: List[Dict] = []
    load_errors = []
    for name in selected_lakes:
        lid = WRIGHT_COUNTY_LAKES[name][2]
        try:
            records = _load_survey_for_lake(lid, name)
            if records:
                all_records.extend(records)
        except Exception:
            load_errors.append(name)

    if load_errors:
        st.caption(f"⚠️ Could not load data for: {', '.join(load_errors)}")

    if all_records:
        df = pd.DataFrame(all_records)
        df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
        df = df.dropna(subset=["survey_date"])
        df["year"] = df["survey_date"].dt.year

        target_codes = []
        for sp in selected_species:
            target_codes.extend(SPECIES_TO_CODE.get(sp, []))
        df_filtered = df[df["species_code"].isin(target_codes)].copy()
        df_filtered["species_display"] = df_filtered["species_code"].apply(
            lambda c: SPECIES_CODE_MAP.get(c, c)
        )

        if not df_filtered.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records", len(df_filtered))
            with col2:
                st.metric("Lakes", df_filtered["lake"].nunique())
            with col3:
                st.metric("Species", df_filtered["species_display"].nunique())
            with col4:
                year_range = f"{int(df_filtered['year'].min())}–{int(df_filtered['year'].max())}"
                st.metric("Year Range", year_range)

            # CPUE trend
            cpue_df = (
                df_filtered.dropna(subset=["cpue"])
                .groupby(["year", "species_display"])["cpue"]
                .mean()
                .reset_index()
            )
            if not cpue_df.empty:
                fig_cpue, ax_cpue = plt.subplots(figsize=(10, 4))
                for sp in cpue_df["species_display"].unique():
                    sp_data = cpue_df[cpue_df["species_display"] == sp].sort_values("year")
                    color = SPECIES_COLORS.get(sp, None)
                    ax_cpue.plot(sp_data["year"], sp_data["cpue"], marker="o", label=sp, color=color)
                ax_cpue.set_xlabel("Year")
                ax_cpue.set_ylabel("Mean CPUE")
                ax_cpue.set_title("Mean CPUE by Species and Year")
                ax_cpue.legend(loc="upper left", fontsize=8)
                ax_cpue.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                plt.tight_layout()
                st.pyplot(fig_cpue)
                plt.close(fig_cpue)

            # Total catch by lake
            catch_pivot = (
                df_filtered.groupby(["lake", "species_display"])["total_catch"]
                .sum()
                .unstack(fill_value=0)
            )
            if not catch_pivot.empty:
                fig_catch, ax_catch = plt.subplots(figsize=(10, 4))
                catch_pivot.plot(kind="bar", ax=ax_catch, stacked=True, colormap="tab10")
                ax_catch.set_xlabel("Lake")
                ax_catch.set_ylabel("Total Catch")
                ax_catch.set_title("Total Catch by Lake")
                ax_catch.legend(loc="upper right", fontsize=8, title="Species")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig_catch)
                plt.close(fig_catch)

            # Average weight
            weight_df = (
                df_filtered.dropna(subset=["average_weight"])
                .groupby("species_display")["average_weight"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            if not weight_df.empty:
                fig_w, ax_w = plt.subplots(figsize=(7, 3))
                ax_w.barh(weight_df["species_display"], weight_df["average_weight"], color="#1d3557")
                ax_w.set_xlabel("Avg Weight (lbs)")
                ax_w.set_title("Average Weight per Fish by Species")
                plt.tight_layout()
                st.pyplot(fig_w)
                plt.close(fig_w)

            with st.expander("Show raw data"):
                display_cols = [
                    "lake", "survey_date", "species_display", "total_catch",
                    "cpue", "average_weight", "gear",
                ]
                display_cols = [c for c in display_cols if c in df_filtered.columns]
                st.dataframe(
                    df_filtered[display_cols].sort_values("survey_date", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No records match your filter combination.")
    else:
        st.info("No survey data returned. The DNR API may be unavailable.")
else:
    st.info("Select at least one lake and one species above.")
