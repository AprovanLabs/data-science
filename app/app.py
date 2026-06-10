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
import time
from pathlib import Path

import streamlit as st

# Make src/ and app/ importable for all pages
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
_app_dir = Path(__file__).resolve().parent

for _p in (_src, _app_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@st.cache_resource(show_spinner=False)
def _local_module_mtime_state() -> dict:
    """Per-interpreter record of repo module file mtimes (survives reruns)."""
    return {"created": time.time(), "mtimes": {}}


def _purge_stale_local_modules() -> None:
    """Evict repo modules from sys.modules when their source files change.

    Streamlit Cloud pulls new commits into a *running* interpreter: page
    scripts are re-executed and modules under the main-script folder are
    re-watched, but modules under src/ (onkia) stay cached in sys.modules,
    causing ImportErrors for newly added names after every deploy. Detect
    files modified since they were loaded and drop ALL repo-local modules
    (src/ and app/) so the next page run re-imports fresh code.
    """
    state = _local_module_mtime_state()
    root = str(_repo_root) + os.sep
    local_modules = {
        name: getattr(mod, "__file__", None)
        for name, mod in list(sys.modules.items())
        if getattr(mod, "__file__", None) and str(getattr(mod, "__file__", "")).startswith(root)
    }
    stale = False
    for name, fpath in local_modules.items():
        try:
            mtime = os.path.getmtime(fpath)
        except OSError:
            continue
        prev = state["mtimes"].get(name)
        if (prev is not None and mtime > prev) or (prev is None and mtime > state["created"]):
            stale = True
        state["mtimes"][name] = mtime
    if stale:
        for name in local_modules:
            sys.modules.pop(name, None)


_purge_stale_local_modules()

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