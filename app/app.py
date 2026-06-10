"""AprovanLabs Data Science — Streamlit hub entry point.

Multi-page hub using st.navigation (requires streamlit>=1.36).
Run with:

    streamlit run app/app.py

The fishing intelligence page is at app/pages/fishing.py.
Shared constants, utilities, and cached loaders live in
app/components/fishing_data.py.
"""
from __future__ import annotations

import os
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


def _purge_local_modules() -> None:
    """Evict all repo-local modules from sys.modules on every run.

    Streamlit Cloud pulls new commits into a *running* interpreter: page
    scripts are re-executed from disk, but modules under src/ (onkia) and
    app/components stay cached in sys.modules, causing ImportErrors for
    newly added names after every deploy. Unconditionally dropping them
    here forces each run to re-import fresh code straight from disk.
    (They are small pure-Python modules, and st.cache_data results are
    keyed by code hash, so caches survive the re-import.)
    """
    root = str(_repo_root) + os.sep
    for name, mod in list(sys.modules.items()):
        # Streamlit registers the running script itself as __main__ (its
        # __file__ is this repo's app.py); evicting it breaks inspect-based
        # introspection inside Streamlit (KeyError: '__main__').
        if name == "__main__":
            continue
        fpath = getattr(mod, "__file__", None)
        if fpath and str(fpath).startswith(root):
            sys.modules.pop(name, None)


_purge_local_modules()

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