"""Lake Finder page — interactive Folium map of Wright County lakes."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit_folium import st_folium

# Ensure src/ and app/ are on path (never import via `app.*` — it re-executes app.py)
_app = Path(__file__).resolve().parent.parent
_src = _app.parent / "src"
for _p in (_app, _src):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from components.dnr_map import WRIGHT_COUNTY_LAKES, build_lake_map  # noqa: E402
from onkia import MnDnrLakeTopographyService  # noqa: E402

st.title("🗺️ Lake Finder")
st.caption("Wright County, MN — Select a lake to get fishing recommendations.")

# Initialise session state
if "selected_lake_name" not in st.session_state:
    st.session_state["selected_lake_name"] = None
if "selected_lake_id" not in st.session_state:
    st.session_state["selected_lake_id"] = None
if "dnr_search_results" not in st.session_state:
    st.session_state["dnr_search_results"] = []


@st.cache_data(show_spinner="Searching DNR database…")
def _search_lake(name: str) -> Optional[dict]:
    svc = MnDnrLakeTopographyService()
    lake = svc.get_lake(name, county_id=86)
    return lake.model_dump() if lake else None


# --- Search bar ---
col_search, col_btn = st.columns([3, 1])
with col_search:
    search_query = st.text_input(
        "Search lake by name",
        placeholder="e.g. Clearwater",
        label_visibility="collapsed",
    )
with col_btn:
    do_search = st.button("Search DNR", use_container_width=True)

if do_search and search_query:
    result = _search_lake(search_query.strip())
    if result:
        st.session_state["dnr_search_results"] = [result]
        st.success(f"Found: **{result['name']}** (DOW: {result['id']})")
    else:
        st.session_state["dnr_search_results"] = []
        st.warning(f"No lake named '{search_query}' found in Wright County (county 86).")

# --- Curated lake quick-select ---
st.markdown("**Or pick from known Wright County lakes:**")
lake_names = sorted(WRIGHT_COUNTY_LAKES.keys())
cols = st.columns(4)
for i, name in enumerate(lake_names):
    with cols[i % 4]:
        label = f"{'✓ ' if name == st.session_state['selected_lake_name'] else ''}{name}"
        if st.button(label, key=f"pick_{name}", use_container_width=True):
            st.session_state["selected_lake_name"] = name
            st.session_state["selected_lake_id"] = WRIGHT_COUNTY_LAKES[name][2]

st.divider()

# --- Map ---
fmap = build_lake_map(
    selected_lake=st.session_state["selected_lake_name"],
    search_results=st.session_state["dnr_search_results"],
)

map_data = st_folium(fmap, width="100%", height=500, returned_objects=["last_object_clicked"])

# Handle map marker click — match to nearest curated lake or search result
if map_data and map_data.get("last_object_clicked"):
    clicked = map_data["last_object_clicked"]
    clicked_lat = clicked.get("lat")
    clicked_lng = clicked.get("lng")
    if clicked_lat is not None:
        # Find closest curated lake
        best_name, best_dist = None, float("inf")
        for name, (lat, lon, dow) in WRIGHT_COUNTY_LAKES.items():
            dist = (lat - clicked_lat) ** 2 + (lon - clicked_lng) ** 2
            if dist < best_dist:
                best_dist, best_name = dist, name
        # Also check DNR search results
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

# --- Selection summary ---
if st.session_state["selected_lake_name"]:
    st.success(
        f"Selected: **{st.session_state['selected_lake_name']}** "
        f"(DOW: {st.session_state['selected_lake_id']})"
    )
    if st.button("Go to Fishing Day →", type="primary", use_container_width=True):
        st.switch_page("pages/fishing_day.py")
else:
    st.info("Click a marker on the map or use the buttons above to select a lake.")
