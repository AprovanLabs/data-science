"""Smoke tests for the Streamlit app — import validation and structural checks.

These run without a Streamlit server. They verify that all app modules
can be imported without error and that no page uses the `from app.*`
pattern that triggers the StreamlitDuplicateElementId crash on
Streamlit Community Cloud.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"
PAGES_DIR = APP_DIR / "pages"
COMPONENTS_DIR = APP_DIR / "components"


# ---------------------------------------------------------------------------
# 1. No `from app.*` imports in sub-pages (causes DuplicateElementId)
# ---------------------------------------------------------------------------

def _python_files_under(directory: Path) -> List[Path]:
    return sorted(p for p in directory.rglob("*.py") if p.is_file())


def _collect_imports(source: str) -> List[Tuple[str, int]]:
    """Return (module_name, line_number) for all import-from statements."""
    tree = ast.parse(source)
    imports: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, node.lineno))
    return imports


@pytest.mark.parametrize(
    "filepath",
    _python_files_under(PAGES_DIR),
    ids=[p.name for p in _python_files_under(PAGES_DIR)],
)
def test_no_app_prefix_import(filepath: Path) -> None:
    """Sub-pages must NOT import via `from app.*` — it re-executes app.py
    and raises StreamlitDuplicateElementId."""
    source = filepath.read_text()
    for module, lineno in _collect_imports(source):
        assert not module.startswith("app."), (
            f"{filepath.name}:{lineno} — `from {module} import ...` "
            "will re-execute app.py on Streamlit Cloud. "
            "Add app/ to sys.path and import from `components.*` instead."
        )


# ---------------------------------------------------------------------------
# 2. All app modules can be imported without Streamlit running
# ---------------------------------------------------------------------------

def test_dnr_map_importable() -> None:
    sys.path.insert(0, str(APP_DIR))
    try:
        from components.dnr_map import WRIGHT_COUNTY_LAKES, build_lake_map
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing runtime dep: {exc.name}")
    assert len(WRIGHT_COUNTY_LAKES) > 0
    assert callable(build_lake_map)


def test_onkia_importable() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from onkia import MnDnrLakeTopographyService, WATER_TEMP_PREFERENCES
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing runtime dep: {exc.name}")
    assert callable(MnDnrLakeTopographyService)
    assert len(WATER_TEMP_PREFERENCES) > 0


def test_onkia_models_importable() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from onkia.models import (
            FishCatchSummary,
            SurveyOverview,
            WaterTempPreference,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing runtime dep: {exc.name}")
    assert FishCatchSummary is not None
    assert SurveyOverview is not None
    assert WaterTempPreference is not None


# ---------------------------------------------------------------------------
# 3. Page files exist and are syntactically valid
# ---------------------------------------------------------------------------

REQUIRED_PAGES = ["lake_finder.py", "fishing_day.py", "species_dashboard.py"]


@pytest.mark.parametrize("page_name", REQUIRED_PAGES)
def test_page_file_exists(page_name: str) -> None:
    assert (PAGES_DIR / page_name).is_file()


@pytest.mark.parametrize("page_name", REQUIRED_PAGES)
def test_page_file_parseable(page_name: str) -> None:
    source = (PAGES_DIR / page_name).read_text()
    ast.parse(source)


def test_app_py_parseable() -> None:
    source = (APP_DIR / "app.py").read_text()
    ast.parse(source)


# ---------------------------------------------------------------------------
# 4. app/app.py uses section-based navigation (no flat list)
# ---------------------------------------------------------------------------

def test_app_uses_section_navigation() -> None:
    source = (APP_DIR / "app.py").read_text()
    assert "st.navigation" in source, "app.py must call st.navigation()"
    assert isinstance(ast.parse(source).body, list), "app.py must be valid Python"


# ---------------------------------------------------------------------------
# 5. requirements.txt exists and lists core deps
# ---------------------------------------------------------------------------

def test_requirements_txt_exists() -> None:
    assert (REPO_ROOT / "requirements.txt").is_file()


def test_requirements_includes_streamlit() -> None:
    content = (REPO_ROOT / "requirements.txt").read_text()
    for dep in ("streamlit", "pandas", "pydantic", "requests"):
        assert dep in content.lower(), f"requirements.txt missing: {dep}"


# ---------------------------------------------------------------------------
# 6. No duplicate Streamlit element keys in app.py sidebar
# ---------------------------------------------------------------------------

def test_app_sidebar_keys_explicit() -> None:
    source = (APP_DIR / "app.py").read_text()
    button_calls = re.findall(r"st\.button\([^)]*key\s*=\s*['\"]([^'\"]+)['\"]", source)
    assert len(button_calls) == len(set(button_calls)), (
        f"Duplicate button keys in app.py: {button_calls}"
    )
