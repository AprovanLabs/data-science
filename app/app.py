"""AprovanLabs Data Science — Streamlit hub entry point.

Multi-page hub using st.navigation (requires streamlit>=1.36).
Run with:

    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make src/ and app/ importable for all pages
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
_app_dir = Path(__file__).resolve().parent

for _p in (_src, _app_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

st.set_page_config(
    page_title="AprovanLabs Data Science",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_pages = [
    st.Page("pages/fishing.py", title="Fishing Intelligence", icon="🎣"),
]

pg = st.navigation(_pages)
pg.run()