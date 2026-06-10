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
from components.fishing_data import (
    TARGET_SPECIES,
    SPECIES_CODE_MAP,
    SPECIES_TO_CODE,
    SPECIES_COLORS,
    _MONTHLY_WATER_TEMP,
    _MONTHLY_CLOUD_COVER,
    _TECHNIQUES,
    _TIME_OF_DAY_ADVICE,
    _estimate_water_temp,
    _estimate_cloud_cover,
    _temp_condition,
    _parse_stocking,
    _build_depth_profile,
    _build_water_temp_year_chart,
    _build_cloud_cover_year_chart,
    _fetch_contours_cached,
    _search_lake_cached,
    _load_survey,
    _get_weather_cached,
    _generate_plan_cached,
    _load_stocking_xml,
    _load_survey_for_lake,
)
from onkia.analysis import LakeAnalysis, TrendStatus, analyze_lake, parse_stocking_events  # noqa: F401
from onkia.bathymetry import (  # noqa: F401
    contour_color,
    contours_to_profile,
    depth_area_profile,
    load_contours,
    load_depth_profile,
    species_depth_zone,
)
from onkia.dnr_client import DnrApiUnavailableError, MN_COUNTIES, MnDnrLakeTopographyService
from onkia.water_temp import WATER_TEMP_PREFERENCES
from onkia.models import WaterTempPreference
from onkia.usgs_glm import (  # noqa: F401
    get_temperature_profile,
    get_thermocline_depth,
    get_species_depth_zones,
    has_data,
    available_lakes,
)
from onkia.satellite_lst_stac import (  # noqa: F401
    HEATMAP_THRESHOLD_ACRES,
    LAKE_ACRES,
    LST_USEFUL_THRESHOLD_ACRES,
    get_latest_lst,
    get_lst_history,
    get_lst_heatmap,
    get_ndci,
    get_overlay_grids,
    grid_bounds,
    lake_radius_m,
)
from onkia.weather import get_weather_for_window, wind_direction_label  # noqa: F401
from onkia.plan_generator import generate_evening_plan  # noqa: F401


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
def _get_heatmap_cached(lat: float, lon: float, radius_m: float):
    return get_lst_heatmap(lat, lon, radius_m)


@st.cache_data(ttl=3600, show_spinner="Fetching satellite overlay imagery...")
def _get_overlay_grids_cached(lake_name: str, lat: float, lon: float, radius_m: float):
    return get_overlay_grids(lake_name, lat, lon, radius_m=radius_m)


if "selected_lake_name" not in st.session_state:
    st.session_state["selected_lake_name"] = None
if "selected_lake_id" not in st.session_state:
    st.session_state["selected_lake_id"] = None
if "dnr_search_results" not in st.session_state:
    st.session_state["dnr_search_results"] = []
if "custom_lakes" not in st.session_state:
    # Lakes added via DNR search, in the same {name: (lat, lon, dow)} shape
    # as WRIGHT_COUNTY_LAKES so they behave identically to roster lakes.
    st.session_state["custom_lakes"] = {}

with st.sidebar:
    st.markdown("**Fishing Intelligence**")
    st.caption("Wright County, MN")
    st.divider()

    if st.button("Refresh Data", key="btn_refresh_data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared -- data will reload on next action.")

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


def _parse_optimum_band(optimum: str) -> Optional[Tuple[float, float]]:
    """Parse a WATER_TEMP_PREFERENCES optimum string like "64-70" into (lo, hi)."""
    try:
        if "-" in optimum:
            lo, hi = (float(x) for x in optimum.split("-"))
        else:
            lo = hi = float(optimum)
        return lo, hi
    except (ValueError, AttributeError, TypeError):
        return None


def _grid_to_rgba(values: np.ndarray, vmin: float, vmax: float, cmap_name: str, alpha: float) -> np.ndarray:
    """Map a 2-D grid to an RGBA image; NaN pixels become fully transparent."""
    norm = (values - vmin) / max(vmax - vmin, 1e-9)
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(norm)
    rgba[..., 3] = np.where(np.isfinite(values), alpha, 0.0)
    return rgba


def _mask_to_rgba(mask: np.ndarray, hex_color: str, alpha: float) -> np.ndarray:
    """Render a boolean mask as a single-colour RGBA image (transparent elsewhere)."""
    from matplotlib.colors import to_rgba

    r, g, b, _a = to_rgba(hex_color)
    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


st.title("Wright County Fishing Intelligence")

# --- DNR lake search (found lakes are added to the roster below) ---
_ALL_MN = "All Minnesota"
_county_options = [_ALL_MN] + sorted(MN_COUNTIES.values())
_county_name_to_id = {v: k for k, v in MN_COUNTIES.items()}

col_search, col_county, col_btn = st.columns([2.2, 1.4, 0.8])
with col_search:
    search_query = st.text_input(
        "Search DNR by name",
        placeholder="e.g. Clearwater",
        label_visibility="collapsed",
        key="lake_search",
    )
with col_county:
    search_county = st.selectbox(
        "County",
        options=_county_options,
        index=_county_options.index("Wright"),
        label_visibility="collapsed",
        key="lake_search_county",
    )
with col_btn:
    do_search = st.button("Search", key="btn_search_dnr", use_container_width=True)

if do_search and search_query:
    # None searches statewide.
    search_county_id = _county_name_to_id.get(search_county)
    try:
        result = _search_lake_cached(search_query.strip(), search_county_id)
    except DnrApiUnavailableError:
        st.session_state["dnr_search_results"] = []
        st.error("DNR API is unavailable. Please try again later.")
    else:
        if result:
            st.session_state["dnr_search_results"] = [result]
            _rname = result["name"]
            _rid = str(result["id"])
            _coords = result.get("point", {}).get("epsg:4326", [])
            if _rname not in WRIGHT_COUNTY_LAKES and len(_coords) >= 2:
                # Same shape as WRIGHT_COUNTY_LAKES so searched lakes behave
                # exactly like roster lakes (markers, buttons, contours).
                st.session_state["custom_lakes"][_rname] = (
                    float(_coords[1]),
                    float(_coords[0]),
                    _rid,
                )
            st.session_state["selected_lake_name"] = _rname
            st.session_state["selected_lake_id"] = _rid
            _county = result.get("county", "")
            _county_note = f", {_county} County" if _county else ""
            st.success(
                f"Found and selected: **{_rname}** (DOW: {_rid}{_county_note}). "
                "It now appears in the lake roster below."
            )
        else:
            st.session_state["dnr_search_results"] = []
            _scope = "Minnesota" if search_county == _ALL_MN else f"{search_county} County"
            st.warning(
                f"No lake named '{search_query}' found in {_scope}. "
                "Try a different county or 'All Minnesota'."
            )

st.markdown("**Select a Lake**")
# Wright County roster plus any lakes added via DNR search -- all rendered
# and selectable identically.
_all_lakes: Dict[str, Tuple[float, float, str]] = {
    **WRIGHT_COUNTY_LAKES,
    **st.session_state["custom_lakes"],
}
lake_names = sorted(_all_lakes.keys())
cols = st.columns(3)
for i, name in enumerate(lake_names):
    with cols[i % 3]:
        is_sel = name == st.session_state["selected_lake_name"]
        label = f"{'>> ' if is_sel else ''}{name}"
        if st.button(label, key=f"pick_{name}", use_container_width=True):
            st.session_state["selected_lake_name"] = name
            st.session_state["selected_lake_id"] = _all_lakes[name][2]

st.divider()

# --- Map overlay controls (top of the app) ---
st.markdown("**Map Overlays**")
ctl_layers, ctl_sat, ctl_species = st.columns([1, 1.2, 1.8])
with ctl_layers:
    show_bathymetry = st.checkbox("Depth contours", value=False, key="chk_bathymetry")
    show_species_zones = st.checkbox("Species depth zones", value=False, key="chk_species_zones")
with ctl_sat:
    show_temp_overlay = st.checkbox("Satellite surface temp", value=False, key="chk_sat_temp")
    show_veg_overlay = st.checkbox("Vegetation / algae (NDCI)", value=False, key="chk_sat_veg")
    show_hotspots = st.checkbox("Species hotspots (temp + weeds)", value=False, key="chk_hotspots")
with ctl_species:
    zone_species = st.multiselect(
        "Species to highlight",
        options=list(TARGET_SPECIES.keys()),
        default=list(TARGET_SPECIES.keys()),
        key="sel_zone_species",
        help="Used by the species depth zone and species hotspot overlays.",
    )

st.subheader("Lake Map")

_selected_name = st.session_state["selected_lake_name"]
_selected_id = st.session_state["selected_lake_id"]

species_zones = []
if _selected_name and show_species_zones and zone_species:
    for sp in zone_species:
        pref = next((p for p in WATER_TEMP_PREFERENCES if p.species == sp), None)
        zone = species_depth_zone(
            species=sp,
            water_temp_f=65.0,
            time_of_day="evening",
            water_temp_pref=pref,
        )
        if zone:
            species_zones.append({
                "species": sp,
                "depth_range": zone,
                "color": SPECIES_COLORS.get(sp, "#457b9d"),
            })

# Bathymetry contours: pre-processed Wright County files when available,
# otherwise fetched on demand from the DNR ArcGIS service (works for any
# MN lake DOW, so searched lakes get contours too).
_contour_data = None
if _selected_name:
    _contour_data = load_contours(_selected_name)
    if _contour_data is None and _selected_id:
        _contour_data = _fetch_contours_cached(str(_selected_id), _selected_name)

# --- Satellite raster overlays (LST / NDCI / species hotspots) ---
_image_overlays: List[Dict] = []
_overlay_notes: List[str] = []
if _selected_name and _selected_name in _all_lakes and (
    show_temp_overlay or show_veg_overlay or show_hotspots
):
    _sel_lat, _sel_lon = _all_lakes[_selected_name][0], _all_lakes[_selected_name][1]
    _sat_grids = _get_overlay_grids_cached(
        _selected_name, _sel_lat, _sel_lon, lake_radius_m(_selected_name)
    )
    _lst_da = _sat_grids.lst_grid
    _ndci_da = _sat_grids.ndci_grid

    if show_temp_overlay:
        if _lst_da is not None:
            _temp_f_vals = _lst_da.values * 9.0 / 5.0 + 32.0
            _vmin = float(np.nanpercentile(_temp_f_vals, 5))
            _vmax = float(np.nanpercentile(_temp_f_vals, 95))
            if _vmax - _vmin < 2.0:
                _vmin, _vmax = _vmin - 1.0, _vmax + 1.0
            _image_overlays.append({
                "name": "Satellite Surface Temp",
                "image": _grid_to_rgba(_temp_f_vals, _vmin, _vmax, "turbo", alpha=0.65),
                "bounds": grid_bounds(_lst_da),
            })
            _overlay_notes.append(
                f"Surface temp overlay (Landsat, {_sat_grids.lst_date}): "
                f"{_vmin:.0f}-{_vmax:.0f} F, blue = cooler, red = warmer."
            )
        else:
            _overlay_notes.append(
                "Surface temp overlay unavailable -- no recent cloud-free Landsat scene."
            )

    if show_veg_overlay:
        if _ndci_da is not None:
            _image_overlays.append({
                "name": "Vegetation / Algae (NDCI)",
                "image": _grid_to_rgba(_ndci_da.values, -0.1, 0.3, "YlGn", alpha=0.6),
                "bounds": grid_bounds(_ndci_da),
            })
            _overlay_notes.append(
                f"NDCI overlay (Sentinel-2, {_sat_grids.ndci_date}): "
                "darker green = denser vegetation / algae (weed proxy)."
            )
        else:
            _overlay_notes.append(
                "Vegetation overlay unavailable -- no recent cloud-free Sentinel-2 scene."
            )

    if show_hotspots:
        if _lst_da is not None and zone_species:
            _temp_f_grid = _lst_da.values * 9.0 / 5.0 + 32.0
            _ndci_on_lst = None
            if _ndci_da is not None:
                try:
                    _ndci_on_lst = _ndci_da.interp_like(_lst_da, method="nearest").values
                except Exception:
                    _ndci_on_lst = None
            _hotspot_count = 0
            for sp in zone_species:
                pref = next((p for p in WATER_TEMP_PREFERENCES if p.species == sp), None)
                band = _parse_optimum_band(pref.optimum) if pref else None
                if band is None:
                    continue
                mask = np.isfinite(_temp_f_grid) & (_temp_f_grid >= band[0]) & (_temp_f_grid <= band[1])
                label = f"{sp} hotspot ({band[0]:.0f}-{band[1]:.0f} F"
                if _ndci_on_lst is not None:
                    mask &= np.nan_to_num(_ndci_on_lst, nan=-1.0) > 0.0
                    label += " + weeds"
                label += ")"
                if not mask.any():
                    continue
                _hotspot_count += 1
                _image_overlays.append({
                    "name": label,
                    "image": _mask_to_rgba(mask, SPECIES_COLORS.get(sp, "#457b9d"), alpha=0.55),
                    "bounds": grid_bounds(_lst_da),
                })
            if _ndci_on_lst is None:
                _overlay_notes.append(
                    "Species hotspots use surface temperature only -- no Sentinel-2 "
                    "vegetation data available to correlate weed cover."
                )
            elif _hotspot_count:
                _overlay_notes.append(
                    "Species hotspots: pixels where satellite surface temp falls in the "
                    "species' optimum band AND NDCI indicates vegetation/algae (weed cover)."
                )
            else:
                _overlay_notes.append(
                    "No species hotspots found -- surface temp is outside the optimum "
                    "band (or no weed cover) for the selected species."
                )
        elif _lst_da is None:
            _overlay_notes.append(
                "Species hotspots unavailable -- no recent cloud-free Landsat scene."
            )

# Coordinate overrides for roster lakes found via search (live DNR centroids).
_dnr_coord_overrides = {}
for _r in st.session_state["dnr_search_results"]:
    _rname = _r.get("name", "")
    _coords = _r.get("point", {}).get("epsg:4326", [])
    if len(_coords) >= 2 and _rname in WRIGHT_COUNTY_LAKES:
        _dnr_coord_overrides[_rname] = (float(_coords[1]), float(_coords[0]))

# Center the map on lakes outside the default Wright County view.
_map_view_kwargs = {}
if _selected_name in st.session_state["custom_lakes"]:
    _c_lat, _c_lon, _ = st.session_state["custom_lakes"][_selected_name]
    _map_view_kwargs = {"center": (_c_lat, _c_lon), "zoom": 12}

fmap = build_lake_map(
    selected_lake=_selected_name,
    show_bathymetry=show_bathymetry,
    species_zones=species_zones if species_zones else None,
    lake_coord_overrides=_dnr_coord_overrides if _dnr_coord_overrides else None,
    extra_lakes=st.session_state["custom_lakes"] or None,
    contour_data=_contour_data,
    image_overlays=_image_overlays or None,
    **_map_view_kwargs,
)
map_data = st_folium(fmap, width="100%", height=400, returned_objects=["last_object_clicked"])

for _note in _overlay_notes:
    st.caption(_note)

if map_data and map_data.get("last_object_clicked"):
    clicked = map_data["last_object_clicked"]
    clicked_lat = clicked.get("lat")
    clicked_lng = clicked.get("lng")
    if clicked_lat is not None:
        best_name, best_dist = None, float("inf")
        for name, (lat, lon, dow) in _all_lakes.items():
            dist = (lat - clicked_lat) ** 2 + (lon - clicked_lng) ** 2
            if dist < best_dist:
                best_dist, best_name = dist, name
        if best_name and best_dist < 0.001:
            st.session_state["selected_lake_name"] = best_name
            st.session_state["selected_lake_id"] = _all_lakes[best_name][2]

lake_name = st.session_state["selected_lake_name"]
lake_id = st.session_state["selected_lake_id"]

if not lake_name or not lake_id:
    st.info("Select a lake above or click a marker on the map.")
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

lake_coords = _all_lakes.get(lake_name)
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
        st.info(
            "Depth-temperature profile data is not yet available. "
            "Click below to fetch it from the USGS ScienceBase dataset "
            "(first load may take several minutes)."
        )
        if st.button("Load USGS Temperature Data", key="btn_load_usgs_glm"):
            with st.spinner("Downloading from USGS ScienceBase…"):
                from onkia.usgs_glm import fetch_from_sciencebase
                _usgs_ok, _usgs_msg = fetch_from_sciencebase()
            if _usgs_ok:
                st.success(_usgs_msg)
                st.rerun()
            else:
                st.error(_usgs_msg)

# --- Bathymetry Contours ---
st.subheader("Bathymetry & Depth Contours")

# _contour_data was resolved above (local file or on-demand DNR fetch).
contour_data = _contour_data

if contour_data:
    profile_data = load_depth_profile(lake_name) or contours_to_profile(
        lake_name, lake_id, contour_data
    )

    num_contours = len(contour_data.get("features", []))
    st.success(f"Depth contours available: **{num_contours}** contour lines for {lake_name}")
    st.caption("Toggle 'Depth contours' above the map to visualize them.")

    if profile_data:
        with st.expander("Depth Profile"):
            st.markdown(f"- **Max Depth:** {profile_data.get('max_depth', 'N/A')} ft")
            depths = profile_data.get("depths", [])
            if depths:
                st.markdown(f"- **Contour Depths:** {', '.join(str(int(d)) for d in depths)} ft")

                area_profile = depth_area_profile(lake_name, contours=contour_data)
                if area_profile:
                    profile_depths = [d for d, _ in area_profile]
                    profile_acres = [a for _, a in area_profile]

                    fig_bathy, ax_bathy = plt.subplots(figsize=(8, 3.5))
                    ax_bathy.fill_betweenx(
                        profile_depths,
                        0,
                        profile_acres,
                        color="#4db8ff",
                        alpha=0.35,
                    )
                    ax_bathy.plot(profile_acres, profile_depths, color="#1d3557", linewidth=1.5)
                    ax_bathy.scatter(
                        profile_acres,
                        profile_depths,
                        c=[contour_color(d) for d in profile_depths],
                        edgecolors="#1d3557",
                        linewidths=0.5,
                        zorder=3,
                    )
                    ax_bathy.invert_yaxis()
                    ax_bathy.set_xlim(left=0)
                    ax_bathy.set_xlabel("Water area enclosed by contour (acres)")
                    ax_bathy.set_ylabel("Depth (ft)")
                    ax_bathy.set_title(f"Depth-Area Profile -- {lake_name}")
                    ax_bathy.grid(True, linestyle=":", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig_bathy)
                    plt.close(fig_bathy)
                    st.caption(
                        "Hypsographic curve: surface area enclosed by each DNR depth "
                        "contour. Steep drops indicate sharp breaks; gradual slopes "
                        "indicate flats."
                    )
    else:
        st.info("Depth profile data not available for this lake.")
else:
    st.info(
        f"No DNR bathymetry contours available for **{lake_name}** -- the lake "
        "may not have been surveyed, or the DNR contour service is unavailable."
    )

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
if not _lake_acres:
    # Searched lakes are not in the Wright County table -- use the DNR
    # survey morphology so they get the same satellite treatment.
    try:
        _sv_for_acres = _load_survey(lake_id)
        _lake_acres = float(_sv_for_acres.get("area_acres") or 0.0) if _sv_for_acres else 0.0
    except Exception:
        _lake_acres = 0.0
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
                _heatmap_png = _get_heatmap_cached(lat, lon, lake_radius_m(lake_name, acres=_lake_acres or None))
                if _heatmap_png:
                    st.image(
                        _heatmap_png,
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

st.subheader("Species Suitability & Technique Quick-Reference")

target_prefs = [p for p in WATER_TEMP_PREFERENCES if p.species in TARGET_SPECIES]

# Best targets banner
_cond_map_simple = {}
for _p in WATER_TEMP_PREFERENCES:
    if _p.species in TARGET_SPECIES:
        _cond_map_simple[_p.species] = _temp_condition(water_temp, _p)
optimal_species = [s for s, c in _cond_map_simple.items() if c == "optimal"]
acceptable_species = [s for s, c in _cond_map_simple.items() if c == "cold"]
if optimal_species:
    st.success(f"**Best targets today:** {', '.join(optimal_species)}")
elif acceptable_species:
    st.info(f"**Acceptable targets (not optimal):** {', '.join(acceptable_species)}")
else:
    st.warning("Water temperature outside preferred ranges for all target species -- fishing may be slow.")

_COND_BADGE = {"cold": "🔵 Cold", "optimal": "🟢 Optimal", "warm": "🔴 Warm"}
_tech_rows = []
for pref in WATER_TEMP_PREFERENCES:
    if pref.species not in TARGET_SPECIES:
        continue
    _cond = _temp_condition(water_temp, pref)
    _tech = _TECHNIQUES.get(pref.species, {}).get(_cond)
    if not _tech:
        continue
    _tech_rows.append({
        "Species": pref.species,
        "Condition": _COND_BADGE.get(_cond, _cond),
        "Optimum (°F)": pref.optimum,
        "Lures": " · ".join(_tech["lures"][:2]),
        "Depth": _tech["depth"],
        "Best Time": _tech["time"],
    })

if _tech_rows:
    st.dataframe(pd.DataFrame(_tech_rows), use_container_width=True, hide_index=True)
    if time_of_day in ("dawn", "evening", "night"):
        st.caption("Low-light window: noisy/scented presentations excel; topwater and glow jigs active.")
    elif time_of_day == "midday":
        st.caption("Midday: fish deeper and slower — try vertical jigging or live bait under a slip bobber.")

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
    # Compact conditions bar — parse "Key: Value | Key: Value" into metrics
    cond_parts = plan.conditions_summary.split(" | ")
    cond_cols = st.columns(len(cond_parts)) if len(cond_parts) <= 6 else st.columns(6)
    for _i, _part in enumerate(cond_parts[:6]):
        with cond_cols[_i]:
            if ":" in _part:
                _lbl, _val = _part.split(":", 1)
                st.metric(_lbl.strip(), _val.strip())
            else:
                st.caption(_part)

    # Compact plan table — one row per time block
    _plan_rows = []
    for _block in plan.blocks:
        _plan_rows.append({
            "Time": f"{_block.time_start} – {_block.time_end}",
            "Species": _block.species,
            "Depth": _block.depth or "--",
            "Lures": " · ".join(_block.lures[:2]) if _block.lures else "--",
            "Location": (_block.location[:70] + "…") if len(_block.location) > 70 else _block.location,
            "Technique": (_block.technique[:90] + "…") if len(_block.technique) > 90 else _block.technique,
        })
    st.dataframe(pd.DataFrame(_plan_rows), use_container_width=True, hide_index=True)

    with st.expander("Evidence & data sources"):
        for _block in plan.blocks:
            st.markdown(f"**{_block.time_start} – {_block.time_end} — {_block.species}**")
            for _ev in _block.evidence:
                st.markdown(f"  - {_ev}")
        st.markdown("**Data sources:**")
        for _ds in plan.data_sources:
            st.markdown(f"  - {_ds}")
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
            avg_clarity = survey_data.get("average_water_clarity", "--")
            _ld_cols = st.columns(5)
            _ld_cols[0].metric("Area (acres)", area or "--")
            _ld_cols[1].metric("Max Depth (ft)", max_d or "--")
            _ld_cols[2].metric("Mean Depth (ft)", mean_d or "--")
            _ld_cols[3].metric("Shore (mi)", survey_data.get("shore_length_miles", "--"))
            _ld_cols[4].metric("Avg Clarity (ft)", avg_clarity)

            if max_d and area:
                _md_safe = float(mean_d) if mean_d else float(max_d) / 2
                fig_depth = _build_depth_profile(float(max_d), _md_safe, float(area))
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
    _STATUS_ARROW = {
        "INCREASING": "↑ Increasing",
        "DECREASING": "↓ Declining",
        "STABLE": "→ Stable",
        "INSUFFICIENT_DATA": "? Insufficient",
    }
    _WT_ARROW = {
        "INCREASING": "↑", "DECREASING": "↓", "STABLE": "→", "INSUFFICIENT_DATA": "?",
    }

    _trend_rows = []
    for _t in _trend_data["species_trends"]:
        _trend_rows.append({
            "Species": _t["species"],
            "Population": _STATUS_ARROW.get(_t["status"], _t["status"]),
            "Latest CPUE": f"{_t['latest_cpue']:.2f}" if _t["latest_cpue"] is not None else "--",
            "Avg Wt (lbs)": f"{_t['avg_weight_lbs']:.2f}" if _t["avg_weight_lbs"] is not None else "--",
            "Wt Trend": _WT_ARROW.get(_t["avg_weight_trend"], "?"),
            "Best Gear": _t["best_gear"] or "--",
            "Survey Years": ", ".join(str(y) for y in _t["survey_years"]) if _t["survey_years"] else "--",
        })
    st.dataframe(pd.DataFrame(_trend_rows), use_container_width=True, hide_index=True)

    with st.expander("Evidence detail"):
        for _t in _trend_data["species_trends"]:
            st.markdown(f"**{_t['species']}:** {_t['evidence']}")

    if _trend_data.get("water_clarity"):
        _cl = _trend_data["water_clarity"]
        _cl_label = _STATUS_ARROW.get(_cl["status"], _cl["status"])
        _cl_cols = st.columns([1, 3])
        with _cl_cols[0]:
            if _cl.get("recent_clarity_ft") is not None:
                st.metric("Water Clarity", f"{_cl['recent_clarity_ft']:.1f} ft", delta=_cl_label)
            else:
                st.metric("Water Clarity", _cl_label)
        with _cl_cols[1]:
            st.caption(_cl["evidence"])
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
    # Attempt to build a year × species stocked chart
    _yr_col = next((c for c in df_stock.columns if "year" in c.lower()), None)
    _sp_col = next((c for c in df_stock.columns if "spec" in c.lower() or "fish" in c.lower()), None)
    _qty_col = next((c for c in df_stock.columns if "quan" in c.lower() or "numb" in c.lower() or "count" in c.lower()), None)
    if _yr_col and _sp_col and _qty_col:
        df_stock[_qty_col] = pd.to_numeric(df_stock[_qty_col], errors="coerce").fillna(0)
        _pivot = df_stock.pivot_table(
            index=_yr_col, columns=_sp_col, values=_qty_col, aggfunc="sum", fill_value=0
        )
        _fig_st, _ax_st = plt.subplots(figsize=(10, 4))
        _pivot.plot(kind="bar", ax=_ax_st, stacked=True, colormap="tab10")
        _ax_st.set_xlabel("Year")
        _ax_st.set_ylabel("Fish Stocked")
        _ax_st.set_title(f"Stocking History — {lake_name}")
        _ax_st.legend(loc="upper right", fontsize=8, title="Species")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(_fig_st)
        plt.close(_fig_st)
        with st.expander("Raw stocking records"):
            st.dataframe(df_stock, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_stock, use_container_width=True, hide_index=True)
else:
    st.info("No stocking records retrieved. This may mean no recent stocking, or the DNR API is unavailable.")

st.divider()

st.subheader("Cross-Lake Species Trends")

selected_lakes = st.multiselect(
    "Include lakes",
    options=sorted(_all_lakes.keys()),
    default=[lake_name] if lake_name in _all_lakes else ["Clearwater"],
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
        lid = _all_lakes[name][2]
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
