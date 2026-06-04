"""Onkia — Wright County Fishing Day Recommendation App.

Entry point. Run with:
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
    page_title="Onkia — Wright County Fishing",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    [lake_finder_page, fishing_day_page, species_dashboard_page],
    position="sidebar",
)

with st.sidebar:
    st.markdown("## Onkia 🎣")
    st.caption("Wright County, MN — Fishing Intelligence")
    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared — data will reload on next action.")

    st.divider()
    if "selected_lake_name" in st.session_state and st.session_state["selected_lake_name"]:
        st.info(f"Selected lake: **{st.session_state['selected_lake_name']}**")

pg.run()
