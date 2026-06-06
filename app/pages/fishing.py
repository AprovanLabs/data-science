"""Fishing Intelligence — Wright County, MN.

Loaded as a page inside the AprovanLabs Data Science hub.
Requires onkia package installed (pip install -e .) or PYTHONPATH=src.
Run the hub with:

    streamlit run app/app.py
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, time, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from components.dnr_map import WRIGHT_COUNTY_LAKES, build_lake_map
from onkia.analysis import LakeAnalysis, TrendStatus, analyze_lake, parse_stocking_events  # noqa: F401
from onkia.bathymetry import (  # noqa: F401
    available_lakes as bathymetry_available_lakes,
    contour_color,
    load_contours,
    load_depth_profile,
    species_depth_zone,
)
from onkia.dnr_client import MnDnrLakeTopographyService
from onkia.water_temp import WATER_TEMP_PREFERENCES
from onkia.models import WaterTempPreference
from onkia.usgs_glm import (  # noqa: F401
    get_temperature_profile,
    get_thermocline_depth,
    get_species_depth_zones,
    has_data,
    available_lakes,
)
from onkia.satellite_lst import (  # noqa: F401
    HEATMAP_THRESHOLD_ACRES,
    LAKE_ACRES,
    LST_USEFUL_THRESHOLD_ACRES,
    get_latest_lst,
    get_lst_history,
    get_lst_heatmap_url,
    get_ndci,
    lake_radius_m,
)
from onkia.weather import get_weather_for_window, wind_direction_label  # noqa: F401
from onkia.plan_generator import generate_evening_plan  # noqa: F401

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

if "selected_lake_name" not in st.session_state:
    st.session_state["selected_lake_name"] = None
if "selected_lake_id" not in st.session_state:
    st.session_state["selected_lake_id"] = None
if "dnr_search_results" not in st.session_state:
    st.session_state["dnr_search_results"] = []

with st.sidebar:
    st.markdown("**Fishing Intelligence**")
    st.caption("Wright County, MN")
    st.divider()

    if st.button("Refresh Data", key="btn_refresh_data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared -- data will reload on next action.")

    st.divider()

    st.markdown("**Select a Lake**")
    lake_names = sorted(WRIGHT_COUNTY_LAKES.keys())
    cols = st.columns(3)
    for i, name in enumerate(lake_names):
        with cols[i % 3]:
            is_sel = name == st.session_state["selected_lake_name"]
            label = f"{'>> ' if is_sel else ''}{name}"
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

    st.divider()
    st.markdown("**Map Overlays**")
    show_bathymetry = st.checkbox("Show Depth Contours", value=False, key="chk_bathymetry")
    show_species_zones = st.checkbox("Show Species Zones", value=False, key="chk_species_zones")

    if show_species_zones:
        zone_species = st.multiselect(
            "Species to highlight",
            options=list(TARGET_SPECIES.keys()),
            default=list(TARGET_SPECIES.keys()),
            key="sel_zone_species",
        )
    else:
        zone_species = []

    wind_dir = st.selectbox("Wind Direction", options=["N", "NE", "E", "SE", "S", "SW", "W", "NW"], key="sel_wind_dir")


@st.cache_data(show_spinner="Searching DNR database...")
def _search_lake_cached(name: str) -> Optional[dict]:
    try:
        svc = MnDnrLakeTopographyService()
        lake = svc.get_lake(name, county_id=86)
        return lake.model_dump() if lake else None
    except Exception:
        return None


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


@st.cache_data(ttl=3600, show_spinner="Fetching satellite surface temperature...")
def _get_satellite_lst_cached(lake_name: str, lat: float, lon: float):
    return get_latest_lst(lake_name, lat, lon)


@st.cache_data(ttl=3600, show_spinner="Loading satellite LST history...")
def _get_satellite_lst_history_cached(lake_name: str, lat: float, lon: float):
    return get_lst_history(lake_name, lat, lon, days_back=180)


@st.cache_data(ttl=3600, show_spinner="Fetching chlorophyll index...")
def _get_ndci_cached(lake_name: str, lat: float, lon: float):
    return get_ndci(lake_name, lat, lon)


@st.cache_data(ttl=3600, show_spinner="Generating LST heatmap...")
def _get_heatmap_url_cached(lat: float, lon: float, radius_m: float):
    return get_lst_heatmap_url(lat, lon, radius_m)


@st.cache_data(show_spinner="Computing trend analysis...")
def _compute_trend_analysis(lake_id: str, lake_name: str) -> Optional[dict]:
    """Run analyze_lake and return a JSON-serialisable dict for caching."""
    try:
        svc = MnDnrLakeTopographyService()
        overview = svc.get_survey(lake_id)
        if not overview:
            return None
        import requests as _req

        try:
            resp = _req.get(
                "https://files.dnr.state.mn.us/cgi-bin/lk_stocking.cgi",
                params={"downum": lake_id},
                timeout=15,
            )
            xml_text = resp.text
        except Exception:
            xml_text = ""

        stocking_records: List[Dict] = []
        if xml_text.strip():
            try:
                root = ET.fromstring(xml_text)
                for item in root.iter():
                    if item.tag in ("stocking", "record", "row"):
                        record = {child.tag: child.text for child in item}
                        if record:
                            stocking_records.append(record)
            except ET.ParseError:
                pass

        analysis = analyze_lake(lake_name, overview, stocking_records=stocking_records)
        return {
            "lake_name": analysis.lake_name,
            "species_trends": [
                {
                    "species": t.species,
                    "status": t.status.value,
                    "evidence": t.evidence,
                    "latest_cpue": t.latest_cpue,
                    "earliest_cpue": t.earliest_cpue,
                    "survey_years": t.survey_years,
                    "avg_weight_lbs": t.avg_weight_lbs,
                    "avg_weight_trend": t.avg_weight_trend.value,
                    "best_gear": t.best_gear,
                }
                for t in analysis.species_trends
            ],
            "water_clarity": (
                {
                    "status": analysis.water_clarity.status.value,
                    "evidence": analysis.water_clarity.evidence,
                    "recent_clarity_ft": analysis.water_clarity.recent_clarity_ft,
                    "earliest_clarity_ft": analysis.water_clarity.earliest_clarity_ft,
                }
                if analysis.water_clarity
                else None
            ),
            "stocking_events": [
                {
                    "species": e.species,
                    "year": e.year,
                    "quantity": e.quantity,
                    "life_stage": e.life_stage,
                }
                for e in analysis.stocking_events
            ],
        }
    except Exception:
        return None


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
    from onkia.weather import WeatherResult as _WR
    weather = _WR(**_json.loads(_weather_json))
    from onkia.models import WaterTempPreference as _WTP
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


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _m_to_ft(m: float) -> float:
    return m * 3.28084


def _build_depth_temp_chart(
    profile: pd.DataFrame,
    thermo_depth_m: Optional[float],
    species_zones: Optional[List[Dict]],
    lake_name: str,
) -> plt.Figure:
    """Depth-temperature profile chart with species preference bands."""
    fig, ax = plt.subplots(figsize=(7, 6))

    depths_ft = _m_to_ft(profile["depth_m"].values)
    temps_f = _c_to_f(profile["temp_c"].values)

    ax.plot(temps_f, depths_ft, color="#1d3557", linewidth=2.5, label="Temperature")

    if thermo_depth_m is not None:
        thermo_ft = _m_to_ft(thermo_depth_m)
        ax.axhline(y=thermo_ft, color="#e63946", linestyle="--", linewidth=1.5,
                   label=f"Thermocline ({thermo_ft:.1f} ft)")

    if species_zones:
        for zone in species_zones:
            species = zone["species"]
            color = SPECIES_COLORS.get(species, "#999999")
            opt_low = zone.get("opt_depth_low_m")
            opt_high = zone.get("opt_depth_high_m")
            if opt_low is not None and opt_high is not None:
                low_ft = _m_to_ft(opt_low)
                high_ft = _m_to_ft(opt_high)
                ax.axhspan(low_ft, high_ft, alpha=0.15, color=color,
                           label=f"{species} zone ({low_ft:.0f}-{high_ft:.0f} ft)")

    ax.set_xlabel("Temperature (F)")
    ax.set_ylabel("Depth (ft)")
    ax.set_title(f"Depth-Temperature Profile -- {lake_name}")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


st.title("Wright County Fishing Intelligence")

st.subheader("Lake Map")

lake_name_for_zones = st.session_state["selected_lake_name"]
species_zones = []
if lake_name_for_zones and show_species_zones and zone_species:
    for sp in zone_species:
        pref = next((p for p in WATER_TEMP_PREFERENCES if p.species == sp), None)
        zone = species_depth_zone(
            species=sp,
            water_temp_f=65.0,
            time_of_day="evening",
            wind_direction=wind_dir,
            water_temp_pref=pref,
        )
        if zone:
            species_zones.append({
                "species": sp,
                "depth_range": zone,
                "color": SPECIES_COLORS.get(sp, "#457b9d"),
            })

fmap = build_lake_map(
    selected_lake=st.session_state["selected_lake_name"],
    search_results=st.session_state["dnr_search_results"],
    show_bathymetry=show_bathymetry,
    species_zones=species_zones if species_zones else None,
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

st.subheader("Evening Weather Conditions")

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
        st.metric("Air Temp (5-9 PM)", f"{temp_val:.0f} F")
    else:
        st.metric("Air Temp", "N/A")
with col_pressure:
    if weather.pressure_inhg is not None:
        pi = weather.pressure_interpretation
        trend_label = pi.label if pi else "--"
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
    st.info(f"**Pressure Trend:** {weather.pressure_interpretation.label} -- {weather.pressure_interpretation.fishing_note}")

if weather.fallback_used:
    st.caption("Using seasonal averages -- Open-Meteo API unavailable. Pressure and wind data not available from fallback.")
else:
    st.caption(f"Data source: Open-Meteo ({weather.source})")

water_temp = weather.air_temp_f if weather.air_temp_f is not None else _estimate_water_temp(selected_date)
cloud_cover = weather.cloud_cover_pct if weather.cloud_cover_pct is not None else _estimate_cloud_cover(selected_date)

_usgs_lakes_check = available_lakes()
_usgs_profile_check = None
if _usgs_lakes_check and lake_id in _usgs_lakes_check:
    _usgs_profile_check = get_temperature_profile(lake_id, selected_date)
    if _usgs_profile_check is not None and len(_usgs_profile_check) > 0:
        water_temp = _c_to_f(_usgs_profile_check.iloc[0]["temp_c"])

tod = _TIME_OF_DAY_ADVICE[time_of_day]
st.info(f"**{tod['label']}** -- {tod['note']}")

with st.expander("Seasonal Temperature & Cloud Cover Charts"):
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
    ax1.plot(months, temps, marker="o", color="#e63946", linewidth=2, label="Water Temp (F)")
    ax2.bar(months, covers, alpha=0.3, color="#457b9d", label="Cloud Cover (%)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Water Temp (F)", color="#e63946")
    ax2.set_ylabel("Cloud Cover (%)", color="#457b9d")
    ax1.axvline(x=month, color="#264653", linestyle=":", linewidth=1.5, label="Selected month")
    ax1.set_xticks(months)
    ax1.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax1.set_title("Water Temp & Cloud Cover -- Central MN")
    plt.tight_layout()
    st.pyplot(fig_combo)
    plt.close(fig_combo)

# --- USGS GLM Depth-Temperature Profile ---
_usgs_lakes = available_lakes()
if _usgs_lakes and lake_id in _usgs_lakes:
    st.subheader("Depth-Temperature Profile (USGS GLM)")

    usgs_profile = get_temperature_profile(lake_id, selected_date)
    usgs_thermo = get_thermocline_depth(lake_id, selected_date)

    species_pref_dicts = [
        {
            "species": p.species,
            "optimum": p.optimum,
            "lower_avoidance": p.lower_avoidance,
            "upper_avoidance": p.upper_avoidance,
        }
        for p in WATER_TEMP_PREFERENCES
        if p.species in TARGET_SPECIES
    ]
    usgs_zones = get_species_depth_zones(lake_id, selected_date, species_pref_dicts)

    if usgs_profile is not None:
        col_depth_info, col_depth_chart = st.columns([1, 2])
        with col_depth_info:
            surface_c = usgs_profile.iloc[0]["temp_c"]
            surface_f = _c_to_f(surface_c)
            bottom_c = usgs_profile.iloc[-1]["temp_c"]
            bottom_f = _c_to_f(bottom_c)
            st.metric("Surface Temp", f"{surface_f:.1f} F")
            st.metric("Bottom Temp", f"{bottom_f:.1f} F")
            if usgs_thermo is not None:
                thermo_ft = _m_to_ft(usgs_thermo)
                st.metric("Thermocline", f"{thermo_ft:.1f} ft ({usgs_thermo:.1f} m)")
            else:
                st.info("No thermocline detected (isothermal or ice-covered)")

            if usgs_zones:
                zone_rows = []
                for z in usgs_zones:
                    low = z.get("opt_depth_low_m")
                    high = z.get("opt_depth_high_m")
                    zone_rows.append({
                        "Species": z["species"],
                        "Optimum Depth": f"{_m_to_ft(low):.0f}-{_m_to_ft(high):.0f} ft" if low and high else "N/A",
                    })
                st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, hide_index=True)

        with col_depth_chart:
            fig_depth_temp = _build_depth_temp_chart(
                usgs_profile, usgs_thermo, usgs_zones, lake_name,
            )
            st.pyplot(fig_depth_temp)
            plt.close(fig_depth_temp)

        if selected_date.year > 2021:
            st.caption("USGS GLM data ends in 2021. Showing climatological average for this day of year.")
    else:
        st.info("No depth-temperature profile available for this date.")
elif _usgs_lakes and lake_id not in _usgs_lakes:
    with st.expander("USGS Depth-Temperature Profile"):
        st.info("This lake is not in the USGS GLM dataset. Depth-resolved temperature is unavailable.")
elif not _usgs_lakes:
    with st.expander("USGS Depth-Temperature Profile"):
        st.caption("No USGS GLM data loaded. Run `python scripts/download_usgs_glm.py --sample` to generate test data.")

# --- Bathymetry Contours ---
st.subheader("Bathymetry & Depth Contours")

_bathy_lakes = bathymetry_available_lakes()
_has_bathy = any(l.lower() == lake_name.lower() for l in _bathy_lakes)

if _has_bathy:
    contour_data = load_contours(lake_name)
    profile_data = load_depth_profile(lake_name)

    if contour_data:
        num_contours = len(contour_data.get("features", []))
        st.success(f"Depth contours available: **{num_contours}** contour lines for {lake_name}")
        st.caption("Toggle 'Show Depth Contours' in the sidebar to visualize on the map.")

    if profile_data:
        with st.expander("Depth Profile"):
            st.markdown(f"- **Max Depth:** {profile_data.get('max_depth', 'N/A')} ft")
            depths = profile_data.get("depths", [])
            if depths:
                st.markdown(f"- **Contour Depths:** {', '.join(str(int(d)) for d in depths)} ft")

                fig_bathy, ax_bathy = plt.subplots(figsize=(8, 3))
                ax_bathy.barh(
                    [f"{d} ft" for d in depths],
                    [1] * len(depths),
                    color=[contour_color(d) for d in depths],
                    edgecolor="#1d3557",
                    linewidth=0.5,
                )
                ax_bathy.set_xlabel("Contour")
                ax_bathy.set_title(f"Depth Contours -- {lake_name}")
                ax_bathy.xaxis.set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_bathy)
                plt.close(fig_bathy)
    else:
        st.info("Depth profile data not available for this lake.")
else:
    if _bathy_lakes:
        st.info(f"No bathymetry data for **{lake_name}**. Available: {', '.join(_bathy_lakes)}")
    else:
        st.caption("No bathymetry contour data loaded. Run `python scripts/prepare_bathymetry.py --sample` to generate test data.")

if species_zones:
    st.markdown("**Species Depth Zone Summary** (depth contour overlay):")
    zone_rows = []
    for z in species_zones:
        pref = next((p for p in WATER_TEMP_PREFERENCES if p.species == z["species"]), None)
        cond = "Optimal"
        if pref and water_temp:
            from onkia.bathymetry import _temp_condition as _bathy_temp_cond
            cond = _bathy_temp_cond(water_temp, pref).title()
        zone_rows.append({
            "Species": z["species"],
            "Preferred Depth": f"{z['depth_range'][0]:.0f}-{z['depth_range'][1]:.0f} ft",
            "Condition": cond,
        })
    if zone_rows:
        st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, hide_index=True)

st.divider()

# --- Satellite Surface Temperature ---
st.subheader("Satellite Surface Temperature")

_lake_acres = LAKE_ACRES.get(lake_name, 0.0)
_sat_lst = None

if _lake_acres >= LST_USEFUL_THRESHOLD_ACRES:
    _sat_lst = _get_satellite_lst_cached(lake_name, lat, lon)

    if _sat_lst and not _sat_lst.fallback_used and _sat_lst.temp_fahrenheit is not None:
        col_sat1, col_sat2, col_sat3 = st.columns(3)
        with col_sat1:
            st.metric(
                "Satellite LST",
                f"{_sat_lst.temp_fahrenheit:.1f} F ({_sat_lst.temp_celsius:.1f} C)",
            )
        with col_sat2:
            obs_label = _sat_lst.observation_date.strftime("%b %d, %Y") if _sat_lst.observation_date else "--"
            st.metric("Observation Date", obs_label)
        with col_sat3:
            st.metric("Scenes Composited", _sat_lst.scene_count)

        water_temp = _sat_lst.temp_fahrenheit
        st.caption(
            f"Surface temperature from {_sat_lst.satellite} thermal imagery. "
            "Replaces seasonal estimate for species suitability."
        )

        _ndci = _get_ndci_cached(lake_name, lat, lon)
        if _ndci and not _ndci.fallback_used and _ndci.ndci_value is not None:
            _cat_labels = {
                "low": "Low (clear)",
                "moderate": "Moderate",
                "high": "High (algae risk)",
            }
            _cat_label = _cat_labels.get(_ndci.chlorophyll_category or "", "--")
            _ndci_obs = _ndci.observation_date.strftime("%b %d, %Y") if _ndci.observation_date else "--"
            st.info(
                f"**Chlorophyll-a (NDCI):** {_ndci.ndci_value:.3f} -- {_cat_label}  "
                f"(Sentinel-2, {_ndci_obs})"
            )

        with st.expander("Historical LST Trend (last 6 months)"):
            _history = _get_satellite_lst_history_cached(lake_name, lat, lon)
            if _history:
                _hist_dates = [p.observation_date for p in _history]
                _hist_temps = [p.temp_fahrenheit for p in _history]
                fig_lst, ax_lst = plt.subplots(figsize=(10, 3))
                ax_lst.plot(_hist_dates, _hist_temps, marker="o", color="#e63946", linewidth=2, markersize=5)
                ax_lst.fill_between(_hist_dates, _hist_temps, alpha=0.15, color="#e63946")
                ax_lst.set_ylabel("Surface Temp (F)")
                ax_lst.set_title(f"Satellite LST -- {lake_name} (last 6 months)")
                ax_lst.set_ylim(25, 90)
                plt.tight_layout()
                st.pyplot(fig_lst)
                plt.close(fig_lst)
            else:
                st.info("No historical Landsat data found for this lake in the last 6 months.")

        if _lake_acres >= HEATMAP_THRESHOLD_ACRES:
            with st.expander("Spatial Temperature Map"):
                _heatmap_url = _get_heatmap_url_cached(lat, lon, lake_radius_m(lake_name))
                if _heatmap_url:
                    st.image(
                        _heatmap_url,
                        caption=f"LST spatial heatmap -- {lake_name} (blue=cold, red=warm)",
                        use_column_width=True,
                    )
                else:
                    st.info("Spatial heatmap unavailable -- no recent cloud-free Landsat coverage.")
    else:
        _err = getattr(_sat_lst, "error_msg", None) if _sat_lst else None
        if _err and "not initialised" not in _err:
            st.caption(f"Satellite LST unavailable: {_err}")
        else:
            st.info(
                "Satellite surface temperature requires Google Earth Engine credentials "
                "(set `EE_SERVICE_ACCOUNT_EMAIL` and `EE_SERVICE_ACCOUNT_KEY_PATH`). "
                "Using seasonal estimate below."
            )
else:
    st.caption(
        f"Satellite LST not shown -- {lake_name} ({int(_lake_acres)} acres) is below "
        f"the {int(LST_USEFUL_THRESHOLD_ACRES)}-acre threshold for meaningful thermal pixels."
    )

st.divider()

st.subheader("Species Suitability")

target_prefs = [p for p in WATER_TEMP_PREFERENCES if p.species in TARGET_SPECIES]
rows = []
for pref in WATER_TEMP_PREFERENCES:
    if pref.species not in TARGET_SPECIES:
        continue
    condition = _temp_condition(water_temp, pref)
    condition_label = {"cold": "Cold", "optimal": "Optimal", "warm": "Warm"}.get(condition, "--")
    rows.append({
        "Species": pref.species,
        "Condition": condition_label,
        "Lower Avoidance (F)": pref.lower_avoidance or "--",
        "Optimum (F)": pref.optimum,
        "Upper Avoidance (F)": pref.upper_avoidance or "--",
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
    st.warning("Water temperature outside preferred ranges for all target species -- fishing may be slow.")

st.subheader("Fishing Technique Suggestions")

for pref in target_prefs:
    condition = _temp_condition(water_temp, pref)
    tech = _TECHNIQUES.get(pref.species, {}).get(condition)
    if not tech:
        continue
    with st.expander(f"{pref.species} -- {condition.title()} conditions"):
        st.markdown(f"**Lures / Baits:** {', '.join(tech['lures'])}")
        st.markdown(f"**Target Depth:** {tech['depth']}")
        st.markdown(f"**Best Time of Day:** {tech['time']}")
        if time_of_day in ("dawn", "evening", "night"):
            st.markdown(f"**{time_of_day.title()} tip:** Low-light conditions favor active feeding -- use noisy/scented presentations.")
        elif time_of_day == "midday":
            st.markdown("**Midday tip:** Fish deeper and slower. Try vertical jigging or live bait under a slip bobber.")

st.divider()

# --- Evening Fishing Plan ---
st.subheader("Evening Fishing Plan (5:30--9:00 PM)")

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
            st.markdown(f"#### {block.time_start} -- {block.time_end}")
            st.markdown(f"**{block.species}**")
        with col_detail:
            st.markdown(f"**Location:** {block.location}")
            st.markdown(f"**Technique:** {block.technique}")
            if block.lures:
                st.markdown(f"**Lures:** {', '.join(block.lures)}")
            if block.depth:
                st.markdown(f"**Depth:** {block.depth}")
            with st.expander("Supporting evidence"):
                for ev in block.evidence:
                    st.markdown(f"- {ev}")

        if i < len(plan.blocks) - 1:
            st.markdown("---")

    with st.expander("Data sources"):
        for ds in plan.data_sources:
            st.markdown(f"- {ds}")
else:
    st.info("No plan blocks generated -- adjust time window or check species preferences.")

st.divider()

st.subheader("DNR Survey Data")

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
                    "Avg Weight (lbs)": c.get("average_weight", "--"),
                    "Gear": c.get("gear", "--"),
                })
            df_catch = pd.DataFrame(catch_rows).sort_values("Total Catch", ascending=False)
            st.dataframe(df_catch, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(8, 3))
            species_labels = [r["Species"] for r in catch_rows]
            catches = [r["Total Catch"] for r in catch_rows]
            ax.barh(species_labels, catches, color="#457b9d")
            ax.set_xlabel("Total Catch")
            ax.set_title(f"Catch by Species -- {survey_date_str}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No fish catch summaries in the most recent survey.")

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

        with st.expander("Lake Details & Depth Profile"):
            area = survey_data.get("area_acres", 0)
            max_d = survey_data.get("max_depth_feet", 0)
            mean_d = survey_data.get("mean_depth_feet", 0)
            st.markdown(f"- **Area:** {area} acres")
            st.markdown(f"- **Max Depth:** {max_d} ft")
            st.markdown(f"- **Mean Depth:** {mean_d} ft")
            st.markdown(f"- **Shore Length:** {survey_data.get('shore_length_miles', '--')} miles")
            avg_clarity = survey_data.get("average_water_clarity", "--")
            st.markdown(f"- **Avg Water Clarity:** {avg_clarity} ft")

            if max_d and area:
                fig_depth = _build_depth_profile(float(max_d), float(mean_d), float(area))
                st.pyplot(fig_depth)
                plt.close(fig_depth)
    else:
        st.info("No survey data available for this lake.")
else:
    st.warning("Could not load survey data from DNR -- API may be unavailable. The fishing recommendations above still work based on seasonal estimates.")

st.divider()

st.subheader("Species Trend Analysis")
st.caption("Population trends with traceable evidence from historical DNR surveys.")

_trend_data = _compute_trend_analysis(lake_id, lake_name)

if _trend_data and _trend_data.get("species_trends"):
    _STATUS_ICON = {
        "INCREASING": "Increasing",
        "DECREASING": "Declining",
        "STABLE": "Stable",
        "INSUFFICIENT_DATA": "Insufficient data",
    }

    for trend in _trend_data["species_trends"]:
        label = _STATUS_ICON.get(trend["status"], trend["status"])
        with st.expander(f"**{trend['species']}** -- {label}", expanded=True):
            st.markdown(f"**Population trend:** {label}")
            st.markdown(f"**Evidence:** {trend['evidence']}")

            col_cpue, col_weight, col_gear = st.columns(3)
            with col_cpue:
                if trend["latest_cpue"] is not None:
                    st.metric("Latest CPUE", f"{trend['latest_cpue']:.2f}")
            with col_weight:
                if trend["avg_weight_lbs"] is not None:
                    wt_label = _STATUS_ICON.get(trend["avg_weight_trend"], trend["avg_weight_trend"])
                    st.metric("Avg Weight (lbs)", f"{trend['avg_weight_lbs']:.2f}", delta=f"{wt_label} weight")
            with col_gear:
                if trend["best_gear"]:
                    st.metric("Best Gear", trend["best_gear"])

            if trend["survey_years"]:
                years_str = ", ".join(str(y) for y in trend["survey_years"])
                st.caption(f"Survey years with data: {years_str}")

    if _trend_data.get("water_clarity"):
        clarity = _trend_data["water_clarity"]
        clarity_label = _STATUS_ICON.get(clarity["status"], clarity["status"])
        st.markdown(f"**Water Clarity:** {clarity_label}")
        st.markdown(f"_{clarity['evidence']}_")
else:
    st.info("Trend analysis unavailable -- DNR survey data could not be loaded.")

st.divider()

st.subheader("Stocking History")

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

st.subheader("Cross-Lake Species Trends")

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
        st.caption(f"Could not load data for: {', '.join(load_errors)}")

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
                year_range = f"{int(df_filtered['year'].min())}-{int(df_filtered['year'].max())}"
                st.metric("Year Range", year_range)

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
