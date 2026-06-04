"""Species Dashboard — historical catch trends for target species across Wright County."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

# Ensure src/ and app/ are on path (never import via `app.*` — it re-executes app.py)
_app = Path(__file__).resolve().parent.parent
_src = _app.parent / "src"
for _p in (_app, _src):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onkia import MnDnrLakeTopographyService  # noqa: E402
from components.dnr_map import WRIGHT_COUNTY_LAKES  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SPECIES = ["Largemouth Bass", "Northern Pike", "Sunfish", "Black Crappie", "Walleye"]

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

# Reverse map for filtering
SPECIES_TO_CODE: Dict[str, List[str]] = {
    "Walleye": ["WAE"],
    "Northern Pike": ["NOP"],
    "Largemouth Bass": ["LMB"],
    "Black Crappie": ["BLC"],
    "Sunfish": ["BLG", "GSF", "HSF", "PMK"],  # various sunfish species
}

_COLORS = {
    "Walleye": "#f4a261",
    "Northern Pike": "#2a9d8f",
    "Largemouth Bass": "#264653",
    "Black Crappie": "#e76f51",
    "Sunfish": "#e9c46a",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading survey data for {lake_name}…")
def _load_survey_for_lake(lake_id: str, lake_name: str) -> Optional[List[Dict]]:
    """Return raw fish_catch_summaries records with survey date attached."""
    svc = MnDnrLakeTopographyService()
    overview = svc.get_survey(lake_id)
    if not overview:
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


def _load_all_lakes(selected_lakes: List[str]) -> pd.DataFrame:
    """Aggregate survey records for selected lakes."""
    all_records: List[Dict] = []
    for name in selected_lakes:
        lake_id = WRIGHT_COUNTY_LAKES[name][2]
        records = _load_survey_for_lake(lake_id, name)
        if records:
            all_records.extend(records)
    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame(all_records)
    df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
    df = df.dropna(subset=["survey_date"])
    df["year"] = df["survey_date"].dt.year
    return df


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("📊 Species Dashboard")
st.caption("Historical catch trends for target species across Wright County lakes.")

# Lake selector
all_lake_names = sorted(WRIGHT_COUNTY_LAKES.keys())
selected_lakes = st.multiselect(
    "Select lakes to include",
    options=all_lake_names,
    default=["Clearwater"],
    help="Shift-click to select multiple lakes. Data is fetched live from MN DNR.",
)

if not selected_lakes:
    st.info("Select at least one lake above to load data.")
    st.stop()

# Species filter
selected_species = st.multiselect(
    "Filter species",
    options=TARGET_SPECIES,
    default=TARGET_SPECIES,
)

if not selected_species:
    st.info("Select at least one species.")
    st.stop()

# Gear filter
gear_options = [
    "All gears",
    "Standard gill nets",
    "Standard trap nets",
    "Electroshockers (all)",
]
selected_gear = st.selectbox("Filter by gear type", gear_options)

with st.spinner("Loading catch data from DNR…"):
    df = _load_all_lakes(selected_lakes)

if df.empty:
    st.warning("No survey data returned for the selected lakes. The DNR API may be unavailable.")
    st.stop()

# Apply species filter
target_codes: List[str] = []
for sp in selected_species:
    target_codes.extend(SPECIES_TO_CODE.get(sp, []))

df_filtered = df[df["species_code"].isin(target_codes)].copy()

# Apply gear filter
if selected_gear != "All gears":
    df_filtered = df_filtered[df_filtered["gear"].str.contains(selected_gear, case=False, na=False)]

if df_filtered.empty:
    st.warning("No records match your current filter combination.")
    st.stop()

# Map species codes to display names
df_filtered["species_display"] = df_filtered["species_code"].apply(
    lambda c: SPECIES_CODE_MAP.get(c, c)
)

st.divider()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Survey Records", len(df_filtered))
with col2:
    st.metric("Lakes", df_filtered["lake"].nunique())
with col3:
    st.metric("Species", df_filtered["species_display"].nunique())
with col4:
    year_range = f"{int(df_filtered['year'].min())}–{int(df_filtered['year'].max())}"
    st.metric("Year Range", year_range)

st.divider()

# ---------------------------------------------------------------------------
# CPUE trend over time
# ---------------------------------------------------------------------------
st.subheader("Catch-Per-Unit-Effort (CPUE) Over Time")
st.caption("CPUE = fish caught per net-night (or electrofishing run). Higher is more abundant.")

cpue_df = (
    df_filtered.dropna(subset=["cpue"])
    .groupby(["year", "species_display"])["cpue"]
    .mean()
    .reset_index()
)

if not cpue_df.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    for sp in cpue_df["species_display"].unique():
        sp_data = cpue_df[cpue_df["species_display"] == sp].sort_values("year")
        color = _COLORS.get(sp, None)
        ax.plot(sp_data["year"], sp_data["cpue"], marker="o", label=sp, color=color)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean CPUE")
    ax.set_title("Mean CPUE by Species and Year")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No CPUE data available for the current selection.")

# ---------------------------------------------------------------------------
# Total catch by lake and species (stacked bar)
# ---------------------------------------------------------------------------
st.subheader("Total Catch by Lake and Species")

catch_pivot = (
    df_filtered.groupby(["lake", "species_display"])["total_catch"]
    .sum()
    .unstack(fill_value=0)
)

if not catch_pivot.empty:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    catch_pivot.plot(kind="bar", ax=ax2, stacked=True, colormap="tab10")
    ax2.set_xlabel("Lake")
    ax2.set_ylabel("Total Catch")
    ax2.set_title("Total Catch by Lake")
    ax2.legend(loc="upper right", fontsize=8, title="Species")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ---------------------------------------------------------------------------
# Size distribution (average weight by species)
# ---------------------------------------------------------------------------
st.subheader("Average Weight by Species")

weight_df = (
    df_filtered.dropna(subset=["average_weight"])
    .groupby("species_display")["average_weight"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

if not weight_df.empty:
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.barh(weight_df["species_display"], weight_df["average_weight"], color="#1d3557")
    ax3.set_xlabel("Avg Weight (lbs)")
    ax3.set_title("Average Weight per Fish by Species")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
else:
    st.info("No weight data available for the current selection.")

# ---------------------------------------------------------------------------
# Raw data table
# ---------------------------------------------------------------------------
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
