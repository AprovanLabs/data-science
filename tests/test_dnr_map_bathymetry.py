"""Tests for dnr_map.py bathymetry overlay features."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    import folium
    from components.dnr_map import (
        _depth_to_color,
        add_depth_contours,
        add_species_zone_overlay,
        build_lake_map,
        load_contours_geojson,
        WRIGHT_COUNTY_LAKES,
    )
except ImportError:
    pytest.skip("folium not available", allow_module_level=True)


@pytest.fixture
def sample_contour_geojson(tmp_path: Path) -> Path:
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"depth_ft": 10.0, "lake": "Clearwater", "dow": "86025200"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[[-94.12, 45.30], [-94.12, 45.31], [-94.11, 45.31], [-94.11, 45.30], [-94.12, 45.30]]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"depth_ft": 25.0, "lake": "Clearwater", "dow": "86025200"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[[-94.12, 45.305], [-94.12, 45.308], [-94.115, 45.308], [-94.115, 45.305], [-94.12, 45.305]]]],
                },
            },
        ],
    }
    path = tmp_path / "clearwater_contours.geojson"
    with open(path, "w") as f:
        json.dump(data, f)
    return tmp_path


class TestLoadContoursGeojson:
    def test_loads_existing(self, sample_contour_geojson):
        result = load_contours_geojson("Clearwater", bathymetry_dir=sample_contour_geojson)
        assert result is not None
        assert len(result["features"]) == 2

    def test_returns_none_for_missing(self, tmp_path):
        assert load_contours_geojson("Nonexistent Lake", bathymetry_dir=tmp_path) is None


class TestAddDepthContours:
    def test_adds_contour_layer(self, sample_contour_geojson):
        m = folium.Map(location=[45.3, -94.1], zoom_start=10)
        fg = add_depth_contours(m, "Clearwater", bathymetry_dir=sample_contour_geojson)
        assert fg is not None

    def test_returns_none_when_no_data(self, tmp_path):
        m = folium.Map(location=[45.3, -94.1], zoom_start=10)
        fg = add_depth_contours(m, "Nonexistent Lake", bathymetry_dir=tmp_path)
        assert fg is None

    def test_style_function_reads_depth_from_properties(self, sample_contour_geojson):
        """style_function must extract depth_ft from feature properties, producing distinct
        colors for different depths — not a static style that ignores feature properties."""
        m = folium.Map(location=[45.3, -94.1], zoom_start=10)
        add_depth_contours(m, "Clearwater", bathymetry_dir=sample_contour_geojson)
        html = m._repr_html_()
        color_10ft = _depth_to_color(10.0)
        color_25ft = _depth_to_color(25.0)
        assert color_10ft != color_25ft, "test requires two distinct depth colors"
        assert color_10ft in html, f"expected color for 10 ft ({color_10ft}) not found in rendered HTML"
        assert color_25ft in html, f"expected color for 25 ft ({color_25ft}) not found in rendered HTML"


class TestAddSpeciesZoneOverlay:
    def test_adds_zone_for_matching_depths(self, sample_contour_geojson):
        m = folium.Map(location=[45.3, -94.1], zoom_start=10)
        fg = add_species_zone_overlay(
            m, "Clearwater", "Walleye",
            depth_range=(8.0, 18.0), zone_color="#f4a261",
            bathymetry_dir=sample_contour_geojson,
        )
        assert fg is not None

    def test_returns_none_for_no_matching_depths(self, sample_contour_geojson):
        m = folium.Map(location=[45.3, -94.1], zoom_start=10)
        fg = add_species_zone_overlay(
            m, "Clearwater", "Walleye",
            depth_range=(50.0, 70.0), zone_color="#f4a261",
            bathymetry_dir=sample_contour_geojson,
        )
        assert fg is None


class TestBuildLakeMapWithBathymetry:
    def test_map_with_bathymetry(self, sample_contour_geojson):
        m = build_lake_map(selected_lake="Clearwater", show_bathymetry=True, bathymetry_dir=sample_contour_geojson)
        assert isinstance(m, folium.Map)

    def test_map_with_species_zones(self, sample_contour_geojson):
        zones = [{"species": "Walleye", "depth_range": (8.0, 18.0), "color": "#f4a261"}]
        m = build_lake_map(selected_lake="Clearwater", show_bathymetry=True, species_zones=zones, bathymetry_dir=sample_contour_geojson)
        assert isinstance(m, folium.Map)

    def test_map_without_bathymetry(self):
        m = build_lake_map(selected_lake="Clearwater", show_bathymetry=False)
        assert isinstance(m, folium.Map)

    def test_lake_markers_still_present(self):
        m = build_lake_map()
        html = m._repr_html_()
        assert "Clearwater" in html or "Wright" in html


class TestLakeCoordOverrides:
    def test_override_replaces_hardcoded_coordinates(self):
        """Coordinate override must appear in the rendered HTML; hardcoded default must not."""
        # Clearwater hardcoded: (45.3052, -94.1184)
        override_lat, override_lon = 45.9999, -94.7777
        m = build_lake_map(
            lake_coord_overrides={"Clearwater": (override_lat, override_lon)},
        )
        html = m._repr_html_()
        assert str(override_lat) in html
        assert str(override_lon) in html
        # The default hardcoded value should NOT appear (it was replaced)
        assert "45.3052" not in html

    def test_non_overridden_lakes_keep_hardcoded_coords(self):
        """Lakes without an override must still use their hardcoded coordinates."""
        # Clearwater (overridden) and Howard Lake (not overridden, lat=45.0618)
        m = build_lake_map(
            lake_coord_overrides={"Clearwater": (45.9999, -94.7777)},
        )
        html = m._repr_html_()
        assert "45.0618" in html

    def test_none_override_uses_hardcoded_coords(self):
        """Passing lake_coord_overrides=None must not replace any hardcoded coordinates."""
        m = build_lake_map(lake_coord_overrides=None)
        html = m._repr_html_()
        # Clearwater hardcoded coordinates must still appear
        assert "45.3052" in html
        assert "-94.1184" in html

    def test_empty_override_dict_uses_hardcoded_coords(self):
        """An empty dict override must leave all hardcoded coords unchanged."""
        m = build_lake_map(lake_coord_overrides={})
        html = m._repr_html_()
        assert "45.3052" in html
        assert "-94.1184" in html
