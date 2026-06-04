"""Fishing Day page — per-lake, per-date fishing recommendations."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ensure src/ is on path
_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from onkia import MnDnrLakeTopographyService, WATER_TEMP_PREFERENCES  # noqa: E402
from onkia.models import FishCatchSummary, SurveyOverview, WaterTempPreference  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target species for Wright County (scope from APR-64)
TARGET_SPECIES = {
    "Largemouth Bass": "LMB",
    "Northern Pike": "NOP",
    "Sunfish": "GSF",  # general sunfish / bluegill
    "Black Crappie": "BLC",
    "Walleye": "WAE",
}

# Monthly surface-water temperature estimates for central Minnesota (°F)
# Source: MN DNR / NOAA climatological averages
_MONTHLY_WATER_TEMP = {
    1: 33, 2: 33, 3: 35, 4: 44,
    5: 56, 6: 66, 7: 74, 8: 72,
    9: 63, 10: 51, 11: 39, 12: 34,
}

# Species code → common name mapping for DNR survey data
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

# Technique lookup: (species, conditions) -> suggestions
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_water_temp(query_date: date) -> float:
    """Return estimated surface water temperature (°F) for central MN lakes."""
    month = query_date.month
    base = _MONTHLY_WATER_TEMP[month]
    # Linear interpolation toward next month for day-of-month
    next_month = (month % 12) + 1
    next_base = _MONTHLY_WATER_TEMP[next_month]
    day_fraction = (query_date.day - 1) / 30.0
    return base + day_fraction * (next_base - base)


def _temp_condition(temp_f: float, pref: WaterTempPreference) -> str:
    """Return 'cold', 'optimal', or 'warm' based on water temp vs species preference."""
    lower = pref.lower_avoidance
    upper = pref.upper_avoidance
    optimum_str = pref.optimum
    # Parse optimum range (e.g. "64-70" or "65")
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


@st.cache_data(show_spinner="Loading survey data from DNR…")
def _load_survey(lake_id: str) -> Optional[dict]:
    svc = MnDnrLakeTopographyService()
    survey = svc.get_survey(lake_id)
    return survey.model_dump() if survey else None


@st.cache_data(show_spinner="Loading stocking data from DNR…")
def _load_stocking_xml(lake_id: str) -> str:
    """Return raw XML text for stocking report."""
    import requests
    try:
        resp = requests.get(
            "https://files.dnr.state.mn.us/cgi-bin/lk_stocking.cgi",
            params={"downum": lake_id},
            timeout=10,
        )
        return resp.text
    except Exception:
        return ""


def _parse_stocking(xml_text: str) -> List[Dict]:
    """Parse stocking XML into list of dicts."""
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


def _run_ge_validation(lake_id: str, survey_data: Optional[dict]) -> Optional[dict]:
    """Run Great Expectations validation on survey data if available."""
    if not survey_data:
        return None
    try:
        import great_expectations as gx

        records = []
        for sv in survey_data.get("surveys", []):
            for catch in sv.get("fish_catch_summaries", []):
                records.append({
                    "species": catch.get("species"),
                    "total_catch": catch.get("total_catch"),
                    "total_weight": catch.get("total_weight"),
                    "gear": catch.get("gear"),
                    "cpue": catch.get("cpue"),
                    "survey_date": sv.get("survey_date"),
                    "survey_type": sv.get("survey_type"),
                })
        if not records:
            return None

        df = pd.DataFrame(records)
        context = gx.get_context(mode="ephemeral")
        suite = context.suites.add(gx.ExpectationSuite(name="fish_survey_live"))
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToNotBeNull,
        )
        for col in ["species", "total_catch", "gear"]:
            suite.add_expectation(ExpectColumnToExist(column=col))
            suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

        ds = context.data_sources.add_pandas(name="ds_live")
        asset = ds.add_dataframe_asset(name="asset_live")
        batch_def = asset.add_batch_definition_whole_dataframe(name="batch_live")
        vd = gx.ValidationDefinition(name="vd_live", data=batch_def, suite=suite)
        result = vd.run(batch_parameters={"dataframe": df})
        return {
            "success": result.success,
            "evaluated": result.statistics["evaluated_expectations"],
            "successful": result.statistics["successful_expectations"],
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🎣 Fishing Day")

lake_name = st.session_state.get("selected_lake_name")
lake_id = st.session_state.get("selected_lake_id")

if not lake_name or not lake_id:
    st.warning("No lake selected. Go to **Lake Finder** to pick a lake first.")
    if st.button("← Go to Lake Finder"):
        st.switch_page("pages/lake_finder.py")
    st.stop()

# Date picker
col_lake, col_date = st.columns([2, 1])
with col_lake:
    st.subheader(f"Lake: {lake_name}")
    st.caption(f"DOW number: {lake_id}")
with col_date:
    selected_date = st.date_input(
        "Date",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )

st.divider()

# --- Water Temperature ---
st.subheader("🌡️ Water Temperature Estimate")
water_temp = _estimate_water_temp(selected_date)
is_today = selected_date == date.today()

col_temp, col_note = st.columns([1, 3])
with col_temp:
    st.metric(
        "Estimated Surface Temp",
        f"{water_temp:.0f}°F",
        help="Climatological estimate based on month — actual may vary ±5°F",
    )
with col_note:
    if is_today:
        st.info(
            "Today's estimate uses monthly averages for central Minnesota. "
            "For real-time data, check MN DNR LakeFinder or a local weather station."
        )
    else:
        st.info(f"Historical estimate for {selected_date.strftime('%B %d, %Y')}.")

# --- Species Temperature Analysis ---
st.subheader("🐟 Species Suitability")

target_prefs = [p for p in WATER_TEMP_PREFERENCES if p.species in TARGET_SPECIES]
rows = []
for pref in WATER_TEMP_PREFERENCES:
    if pref.species not in TARGET_SPECIES:
        continue
    condition = _temp_condition(water_temp, pref)
    condition_label = {"cold": "❄️ Cold", "optimal": "✅ Optimal", "warm": "🔥 Warm"}.get(
        condition, "—"
    )
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

# Recommend best species (those in optimal range)
optimal_species = [
    r["Species"] for r in rows if "Optimal" in r["Condition"]
]
acceptable_species = [
    r["Species"] for r in rows
    if "Optimal" not in r["Condition"] and "Warm" not in r["Condition"]
]

if optimal_species:
    st.success(f"**Best targets today:** {', '.join(optimal_species)}")
elif acceptable_species:
    st.info(f"**Acceptable targets (not optimal):** {', '.join(acceptable_species)}")
else:
    st.warning("Water temperature is outside preferred ranges for all target species — fishing may be slow.")

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

st.divider()

# --- DNR Survey Data ---
st.subheader("📋 Most Recent DNR Survey Data")

survey_data = _load_survey(lake_id)
ge_result = None

if survey_data:
    ge_result = _run_ge_validation(lake_id, survey_data)

    surveys = survey_data.get("surveys", [])
    if surveys:
        # Sort surveys by date descending
        surveys_sorted = sorted(
            surveys,
            key=lambda s: s.get("survey_date", ""),
            reverse=True,
        )
        most_recent = surveys_sorted[0]
        survey_date_str = most_recent.get("survey_date", "Unknown")
        survey_type = most_recent.get("survey_type", "")

        st.caption(
            f"Most recent survey: **{survey_type}** on {survey_date_str} "
            f"({len(surveys)} surveys total)"
        )

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
                    "Quartile Count": c.get("quartile_count", "—"),
                })
            df_catch = pd.DataFrame(catch_rows).sort_values("Total Catch", ascending=False)
            st.dataframe(df_catch, use_container_width=True, hide_index=True)

            # Bar chart: catch by species
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

        # Lake morphology
        with st.expander("Lake Details"):
            st.markdown(f"- **Area:** {survey_data.get('area_acres', '—')} acres")
            st.markdown(f"- **Max Depth:** {survey_data.get('max_depth_feet', '—')} ft")
            st.markdown(f"- **Mean Depth:** {survey_data.get('mean_depth_feet', '—')} ft")
            st.markdown(f"- **Shore Length:** {survey_data.get('shore_length_miles', '—')} miles")
            avg_clarity = survey_data.get("average_water_clarity", "—")
            st.markdown(f"- **Avg Water Clarity:** {avg_clarity} ft")
    else:
        st.info("No survey data available for this lake.")
else:
    st.warning("Could not load survey data from DNR — API may be unavailable.")

# GE validation badge
if ge_result:
    if ge_result.get("error"):
        st.caption(f"⚠️ Data validation skipped: {ge_result['error']}")
    elif ge_result.get("success"):
        st.caption(
            f"✅ Data validation passed "
            f"({ge_result['successful']}/{ge_result['evaluated']} checks)"
        )
    else:
        st.caption(
            f"⚠️ Data validation: {ge_result.get('successful', 0)}/{ge_result.get('evaluated', 0)} "
            "checks passed"
        )

st.divider()

# --- Stocking History ---
st.subheader("🐠 Stocking History")

stocking_xml = _load_stocking_xml(lake_id)
stocking_records = _parse_stocking(stocking_xml)

if stocking_records:
    df_stock = pd.DataFrame(stocking_records)
    st.dataframe(df_stock, use_container_width=True, hide_index=True)
else:
    st.info(
        "No stocking records retrieved. This may mean no recent stocking, "
        "or the DNR API did not return parseable data."
    )
    if stocking_xml:
        with st.expander("Raw stocking response"):
            st.code(stocking_xml[:2000], language="xml")
