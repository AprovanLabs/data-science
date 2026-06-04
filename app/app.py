"""AprovanLabs Data Science — Streamlit entry point.

General-purpose entry point. Each application is organised as a
section in the sidebar.  Run with:

    streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Make src/onkia importable when running from repo root
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

st.set_page_config(
    page_title="AprovanLabs Data Science",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Pages (grouped by section) ----

lake_finder_page = st.Page(
    "pages/lake_finder.py",
    title="Lake Finder",
    icon="🗺️",
)
fishing_day_page = st.Page(
    "pages/fishing_day.py",
    title="Fishing Day",
    icon="🎣",
)
species_dashboard_page = st.Page(
    "pages/species_dashboard.py",
    title="Species Dashboard",
    icon="📊",
)

pg = st.navigation(
    {
        "Wright County Fishing": [
            lake_finder_page,
            fishing_day_page,
            species_dashboard_page,
        ],
    },
    position="sidebar",
)

# ---- Sidebar (runs on every page load) ----

with st.sidebar:
    st.markdown("## AprovanLabs 🔬")
    st.caption("Data Science Applications")
    st.divider()

    if st.button("🔄 Refresh Data", key="btn_refresh_data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared — data will reload on next action.")

    st.divider()
    if "selected_lake_name" in st.session_state and st.session_state["selected_lake_name"]:
        st.info(f"Selected lake: **{st.session_state['selected_lake_name']}**")

pg.run()
