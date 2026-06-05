"""AprovanLabs Data Science Hub -- Streamlit entry point.

Multi-page app with a generic landing page and pluggable modules.
Run with:

    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

st.set_page_config(
    page_title="AprovanLabs",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none; }
    .hub-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .hub-card:hover {
        border-color: #457b9d;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


def home():
    st.markdown("# AprovanLabs Data Science")
    st.markdown("Intelligence tools and analytics for field operations.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="hub-card">', unsafe_allow_html=True)
        st.markdown("### Fishing Intelligence")
        st.caption("Wright County, MN")
        st.markdown("Real-time fishing recommendations, DNR survey data, and cross-lake species trends.")
        if st.button("Open Fishing Intelligence", key="btn_go_fishing", use_container_width=True):
            st.switch_page("pages/fishing.py")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="hub-card">', unsafe_allow_html=True)
        st.markdown("### More Apps")
        st.caption("Coming soon")
        st.markdown("Additional data science modules will appear here.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="hub-card">', unsafe_allow_html=True)
        st.markdown("### More Apps")
        st.caption("Coming soon")
        st.markdown("Additional data science modules will appear here.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("AprovanLabs -- Built for field intelligence.")


page = st.navigation({
    "Hub": [
        st.Page(home, title="Home", url_path="", default=True),
    ],
    "Apps": [
        st.Page("pages/fishing.py", title="Fishing Intelligence", icon=":material/fishing:", url_path="fishing"),
    ],
})

page.run()
