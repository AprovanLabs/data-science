"""Tests for the onkia.bathymetry module."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from onkia.bathymetry import (
    SPECIES_DEPTH_PREFS,
    _blend_hex,
    _ellipse_coords,
    depth_contour_color,
    generate_depth_contours_geojson,
    get_species_zone_depths,
    get_wind_shore_note,
    load_bathymetry_geojson,
    save_bathymetry_geojson,
)


class TestDepthContourColor:
    def test_shallow_is_light(self):
        light = depth_contour_color(1.0, 50.0)
        deep = depth_contour_color(50.0, 50.0)
        # Shallow should have a higher red + green value (lighter colour)
        assert light != deep

    def test_zero_depth_returns_first_colour(self):
        c = depth_contour_color(0.0, 50.0)
        assert c.startswith("#")
        assert len(c) == 7

    def test_at_max_depth_returns_dark(self):
        c = depth_contour_color(50.0, 50.0)
        assert c == "#1a365d"

    def test_invalid_max_depth_returns_first(self):
        c = depth_contour_color(5.0, 0.0)
        assert c == "#c8e8f8"

    def test_clamps_above_max(self):
        c1 = depth_contour_color(50.0, 50.0)
        c2 = depth_contour_color(100.0, 50.0)
        assert c1 == c2


class TestBlendHex:
    def test_zero_t_returns_first(self):
        assert _blend_hex("#000000", "#ffffff", 0.0) == "#000000"

    def test_one_t_returns_second(self):
        assert _blend_hex("#000000", "#ffffff", 1.0) == "#ffffff"

    def test_midpoint(self):
        c = _blend_hex("#000000", "#ffffff", 0.5)
        assert c == "#7f7f7f"


class TestEllipseCoords:
    def test_returns_closed_ring(self):
        coords = _ellipse_coords(45.0, -94.0, 500.0)
        assert coords[0] == coords[-1], "Ring must be closed (first == last)"

    def test_length_is_n_plus_one(self):
        coords = _ellipse_coords(45.0, -94.0, 500.0, n=32)
        assert len(coords) == 33

    def test_centred_near_origin(self):
        lat, lon, r = 45.3, -94.1, 300.0
        coords = _ellipse_coords(lat, lon, r)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        # Bounding box should be centred approximately on (lat, lon)
        assert min(lons) < lon < max(lons)
        assert min(lats) < lat < max(lats)

    def test_larger_radius_produces_wider_ring(self):
        small = _ellipse_coords(45.0, -94.0, 100.0)
        large = _ellipse_coords(45.0, -94.0, 1000.0)
        small_span = max(c[0] for c in small) - min(c[0] for c in small)
        large_span = max(c[0] for c in large) - min(c[0] for c in large)
        assert large_span > small_span


class TestGenerateDepthContoursGeoJSON:
    def test_returns_feature_collection(self):
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, 40.0, 500.0)
        assert gj["type"] == "FeatureCollection"
        assert "features" in gj

    def test_features_have_depth_property(self):
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, 40.0, 500.0)
        for feat in gj["features"]:
            assert "depth_ft" in feat["properties"]

    def test_depths_increase_monotonically(self):
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, 40.0, 500.0, num_contours=6)
        depths = [f["properties"]["depth_ft"] for f in gj["features"]]
        assert depths == sorted(depths)

    def test_max_depth_contour_at_or_near_max(self):
        max_d = 40.0
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, max_d, 500.0)
        deepest = max(f["properties"]["depth_ft"] for f in gj["features"])
        assert deepest <= max_d

    def test_zero_max_depth_returns_empty(self):
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, 0.0, 500.0)
        assert gj["features"] == []

    def test_each_feature_is_polygon(self):
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, 30.0, 300.0)
        for feat in gj["features"]:
            assert feat["geometry"]["type"] == "Polygon"

    def test_colour_property_is_hex(self):
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, 30.0, 300.0)
        for feat in gj["features"]:
            color = feat["properties"]["color"]
            assert color.startswith("#"), f"Expected hex color, got {color}"
            assert len(color) == 7

    def test_lake_name_in_properties(self):
        gj = generate_depth_contours_geojson("Clearwater", 45.3052, -94.1184, 73.0, 3186.9)
        for feat in gj["features"]:
            assert feat["properties"]["lake"] == "Clearwater"


class TestGeoJSONPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        gj = generate_depth_contours_geojson("Test Lake", 45.0, -94.0, 30.0, 300.0)
        with patch("onkia.bathymetry._BATHYMETRY_DIR", tmp_path):
            from onkia import bathymetry as bm
            bm._BATHYMETRY_DIR = tmp_path
            path = save_bathymetry_geojson("Test Lake", gj)
            loaded = load_bathymetry_geojson("Test Lake")
        assert loaded is not None
        assert loaded["type"] == "FeatureCollection"
        assert len(loaded["features"]) == len(gj["features"])

    def test_load_nonexistent_returns_none(self, tmp_path):
        with patch("onkia.bathymetry._BATHYMETRY_DIR", tmp_path):
            from onkia import bathymetry as bm
            bm._BATHYMETRY_DIR = tmp_path
            result = load_bathymetry_geojson("Nonexistent Lake")
        assert result is None


class TestGetSpeciesZoneDepths:
    def test_known_species_returns_tuple(self):
        result = get_species_zone_depths("Walleye", "evening", "optimal", 30.0)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unknown_species_returns_none(self):
        result = get_species_zone_depths("Dragon Fish", "morning", "optimal", 30.0)
        assert result is None

    def test_warm_pushes_deeper(self):
        optimal = get_species_zone_depths("Largemouth Bass", "morning", "optimal", 40.0)
        warm = get_species_zone_depths("Largemouth Bass", "morning", "warm", 40.0)
        assert warm[0] >= optimal[0]
        assert warm[1] >= optimal[1]

    def test_cold_pushes_shallower(self):
        optimal = get_species_zone_depths("Walleye", "morning", "optimal", 40.0)
        cold = get_species_zone_depths("Walleye", "morning", "cold", 40.0)
        assert cold[0] <= optimal[0]

    def test_max_depth_clamps_result(self):
        result = get_species_zone_depths("Walleye", "midday", "warm", 10.0)
        assert result[1] <= 10.0

    def test_min_depth_never_negative(self):
        result = get_species_zone_depths("Sunfish", "dawn", "cold", 30.0)
        assert result[0] >= 0.0

    def test_different_times_return_different_zones(self):
        dawn = get_species_zone_depths("Walleye", "dawn", "optimal", 30.0)
        midday = get_species_zone_depths("Walleye", "midday", "optimal", 30.0)
        assert dawn != midday

    @pytest.mark.parametrize("species", list(SPECIES_DEPTH_PREFS.keys()))
    def test_all_species_return_valid_tuple(self, species):
        result = get_species_zone_depths(species, "evening", "optimal", 40.0)
        assert result is not None
        assert result[0] < result[1]


class TestGetWindShoreNote:
    def test_returns_string_for_valid_degrees(self):
        note = get_wind_shore_note(180.0)
        assert isinstance(note, str)
        assert len(note) > 0

    def test_returns_none_for_none(self):
        assert get_wind_shore_note(None) is None

    def test_north_wind_mentions_south(self):
        note = get_wind_shore_note(0.0)
        assert note is not None
        assert "S" in note

    def test_east_wind_mentions_west(self):
        note = get_wind_shore_note(90.0)
        assert note is not None
        assert "W" in note
