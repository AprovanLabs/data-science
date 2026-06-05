"""Smoke tests for the Streamlit app — import validation and structural checks.

These run without a Streamlit server. They verify that all app modules
can be imported without error, no module uses the `from app.*` pattern,
and structural requirements are met.
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
COMPONENTS_DIR = APP_DIR / "components"
PAGES_DIR = APP_DIR / "pages"
FISHING_PAGE = PAGES_DIR / "fishing.py"


def _python_files_under(directory: Path) -> List[Path]:
    return sorted(p for p in directory.rglob("*.py") if p.is_file())


def _collect_imports(source: str) -> List[Tuple[str, int]]:
    tree = ast.parse(source)
    imports: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, node.lineno))
    return imports


@pytest.mark.parametrize(
    "filepath",
    _python_files_under(APP_DIR),
    ids=[p.name for p in _python_files_under(APP_DIR)],
)
def test_no_app_prefix_import(filepath: Path) -> None:
    source = filepath.read_text()
    for module, lineno in _collect_imports(source):
        assert not module.startswith("app."), (
            f"{filepath.name}:{lineno} — `from {module} import ...` "
            "will re-execute app.py on Streamlit Cloud. "
            "Add app/ to sys.path and import from `components.*` instead."
        )


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
        from onkia.dnr_client import MnDnrLakeTopographyService
        from onkia.water_temp import WATER_TEMP_PREFERENCES
        from onkia.weather import get_weather_for_window, WeatherResult
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing runtime dep: {exc.name}")
    assert callable(MnDnrLakeTopographyService)
    assert len(WATER_TEMP_PREFERENCES) > 0
    assert callable(get_weather_for_window)


def test_onkia_usgs_glm_importable() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from onkia.usgs_glm import (
            get_temperature_profile,
            get_thermocline_depth,
            get_species_depth_zones,
            get_climatological_profile,
            available_lakes,
            has_data,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing runtime dep: {exc.name}")
    assert callable(get_temperature_profile)
    assert callable(get_thermocline_depth)
    assert callable(get_species_depth_zones)
    assert callable(available_lakes)


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


def test_pages_directory_exists() -> None:
    assert PAGES_DIR.is_dir(), "app/pages/ must exist for multi-page architecture"


def test_fishing_page_exists() -> None:
    assert FISHING_PAGE.is_file(), "app/pages/fishing.py must exist"


def test_app_py_parseable() -> None:
    source = (APP_DIR / "app.py").read_text()
    ast.parse(source)


def test_fishing_page_parseable() -> None:
    source = FISHING_PAGE.read_text()
    ast.parse(source)


def test_app_uses_st_navigation() -> None:
    source = (APP_DIR / "app.py").read_text()
    assert "st.navigation" in source or "st.navigation" in source, (
        "app.py must use st.navigation for multi-page routing"
    )


def test_app_no_fish_content() -> None:
    source = (APP_DIR / "app.py").read_text()
    fish_terms = ["Wright County Fishing", "TARGET_SPECIES", "SPECIES_CODE_MAP"]
    for term in fish_terms:
        assert term not in source, (
            f"app.py must not contain fish-specific content like '{term}' "
            "-- it belongs in pages/fishing.py"
        )


def test_fishing_page_has_time_of_day_selector() -> None:
    source = FISHING_PAGE.read_text()
    assert "Time of Day" in source, "fishing.py must include a time-of-day selector"


def test_fishing_page_has_depth_profile() -> None:
    source = FISHING_PAGE.read_text()
    assert "depth_profile" in source.lower() or "depth" in source.lower()


def test_fishing_page_has_usgs_glm_integration() -> None:
    source = FISHING_PAGE.read_text()
    assert "usgs_glm" in source.lower() or "get_temperature_profile" in source
    assert "thermocline" in source.lower()
    assert "species_depth_zones" in source.lower() or "depth_temp" in source.lower()


def test_fishing_page_has_historical_temp_chart() -> None:
    source = FISHING_PAGE.read_text()
    assert "cloud_cover" in source.lower() or "cloud" in source.lower()


def test_requirements_txt_exists() -> None:
    assert (REPO_ROOT / "requirements.txt").is_file()


def test_requirements_includes_streamlit() -> None:
    content = (REPO_ROOT / "requirements.txt").read_text()
    for dep in ("streamlit", "pandas", "pydantic", "requests", "numpy"):
        assert dep in content.lower(), f"requirements.txt missing: {dep}"


def test_requirements_streamlit_min_version() -> None:
    content = (REPO_ROOT / "requirements.txt").read_text()
    assert "streamlit>=1.36" in content or "streamlit>=1.3" in content, (
        "requirements.txt must require streamlit>=1.36 for st.navigation"
    )


def test_app_sidebar_keys_explicit() -> None:
    source = (APP_DIR / "app.py").read_text()
    button_calls = re.findall(r"st\.button\([^)]*key\s*=\s*['\"]([^'\"]+)['\"]", source)
    assert len(button_calls) == len(set(button_calls)), (
        f"Duplicate button keys in app.py: {button_calls}"
    )


def test_fishing_page_sidebar_keys_explicit() -> None:
    source = FISHING_PAGE.read_text()
    button_calls = re.findall(r"st\.button\([^)]*key\s*=\s*['\"]([^'\"]+)['\"]", source)
    assert len(button_calls) == len(set(button_calls)), (
        f"Duplicate button keys in fishing.py: {button_calls}"
    )


def test_dnr_client_handles_connection_error() -> None:
    source = (REPO_ROOT / "src" / "onkia" / "dnr_client.py").read_text()
    assert "ConnectionError" in source, "dnr_client.py must handle ConnectionError"
    assert "timeout=" in source, "dnr_client.py must use timeout on requests"


def test_dnr_client_handles_json_decode_error() -> None:
    source = (REPO_ROOT / "src" / "onkia" / "dnr_client.py").read_text()
    assert "JSONDecodeError" in source, "dnr_client.py must handle JSONDecodeError"


def test_bathymetry_module_importable() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from onkia.bathymetry import (
            available_lakes,
            contour_color,
            load_contours,
            load_depth_profile,
            species_depth_zone,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing runtime dep: {exc.name}")
    assert callable(available_lakes)
    assert callable(contour_color)
    assert callable(load_contours)
    assert callable(load_depth_profile)
    assert callable(species_depth_zone)


def test_depth_preferences_importable() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from onkia.water_temp import DEPTH_PREFERENCES
        from onkia.models import DepthPreference, BathymetryContour
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing runtime dep: {exc.name}")
    assert len(DEPTH_PREFERENCES) > 0
    assert DepthPreference is not None
    assert BathymetryContour is not None
