"""Tests for onkia.bathymetry module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

from onkia.bathymetry import (
    available_lakes,
    contour_color,
    load_contours,
    load_depth_profile,
    species_depth_zone,
)
from onkia.models import WaterTempPreference


@pytest.fixture
def sample_geojson(tmp_path: Path) -> Dict:
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"depth_ft": 5.0, "lake": "Test Lake", "dow": "86000000"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[[-94.0, 45.3], [-94.0, 45.31], [-93.99, 45.31], [-93.99, 45.3], [-94.0, 45.3]]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"depth_ft": 15.0, "lake": "Test Lake", "dow": "86000000"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[[-94.0, 45.3], [-94.0, 45.305], [-93.995, 45.305], [-93.995, 45.3], [-94.0, 45.3]]]],
                },
            },
        ],
    }
    geojson_path = tmp_path / "test_lake_contours.geojson"
    with open(geojson_path, "w") as f:
        json.dump(data, f)
    return {"dir": tmp_path, "data": data}


@pytest.fixture
def sample_profile(tmp_path: Path) -> Dict:
    profile = {
        "lake": "Test Lake",
        "dow": "86000000",
        "max_depth": 40.0,
        "min_depth": 5.0,
        "contour_count": 8,
        "depths": [5, 10, 15, 20, 25, 30, 35, 40],
    }
    profile_path = tmp_path / "test_lake_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f)
    return {"dir": tmp_path, "profile": profile}


class TestContourColor:
    def test_shallow(self):
        assert contour_color(0) == "#b3d9ff"

    def test_medium(self):
        assert contour_color(15) == "#1aa3ff"

    def test_deep(self):
        assert contour_color(50) == "#001a33"

    def test_very_deep(self):
        assert contour_color(100) == "#001a33"

    def test_step_boundary(self):
        assert contour_color(5) == "#80caff"
        assert contour_color(10) == "#4db8ff"


class TestLoadContours:
    def test_loads_existing(self, sample_geojson):
        result = load_contours("Test Lake", bathymetry_dir=sample_geojson["dir"])
        assert result is not None
        assert len(result["features"]) == 2
        assert result["features"][0]["properties"]["depth_ft"] == 5.0

    def test_returns_none_for_missing(self, tmp_path):
        assert load_contours("Nonexistent Lake", bathymetry_dir=tmp_path) is None

    def test_handles_corrupt_json(self, tmp_path):
        corrupt_path = tmp_path / "bad_lake_contours.geojson"
        corrupt_path.write_text("not valid json{{{")
        assert load_contours("Bad Lake", bathymetry_dir=tmp_path) is None


class TestLoadDepthProfile:
    def test_loads_existing(self, sample_profile):
        result = load_depth_profile("Test Lake", bathymetry_dir=sample_profile["dir"])
        assert result is not None
        assert result["max_depth"] == 40.0
        assert len(result["depths"]) == 8

    def test_returns_none_for_missing(self, tmp_path):
        assert load_depth_profile("Nonexistent Lake", bathymetry_dir=tmp_path) is None


class TestSpeciesDepthZone:
    @pytest.fixture
    def walleye_pref(self):
        return WaterTempPreference(
            species="Walleye",
            lower_avoidance=50,
            optimum="64-70",
            upper_avoidance=76,
        )

    def test_optimal_conditions(self, walleye_pref):
        zone = species_depth_zone(
            species="Walleye",
            water_temp_f=67.0,
            time_of_day="evening",
            water_temp_pref=walleye_pref,
        )
        assert zone is not None
        assert zone[0] >= 0.0
        assert zone[1] > zone[0]

    def test_cold_conditions(self, walleye_pref):
        zone = species_depth_zone(
            species="Walleye",
            water_temp_f=40.0,
            time_of_day="midday",
            water_temp_pref=walleye_pref,
        )
        assert zone is not None
        assert zone[0] >= 0.0

    def test_warm_conditions(self, walleye_pref):
        zone = species_depth_zone(
            species="Walleye",
            water_temp_f=80.0,
            time_of_day="night",
            water_temp_pref=walleye_pref,
        )
        assert zone is not None

    def test_unknown_species(self):
        assert species_depth_zone(species="Sturgeon", water_temp_f=65.0, time_of_day="morning") is None

    def test_time_of_day_shift(self, walleye_pref):
        zone_dawn = species_depth_zone("Walleye", 67.0, "dawn", water_temp_pref=walleye_pref)
        zone_midday = species_depth_zone("Walleye", 67.0, "midday", water_temp_pref=walleye_pref)
        assert zone_dawn is not None
        assert zone_midday is not None
        assert zone_dawn[0] < zone_midday[0]

    def test_no_negative_depth(self):
        zone = species_depth_zone(species="Sunfish", water_temp_f=67.0, time_of_day="dawn")
        if zone is not None:
            assert zone[0] >= 0.0

    def test_wind_direction_accepted(self, walleye_pref):
        zone = species_depth_zone(
            species="Walleye", water_temp_f=67.0, time_of_day="evening",
            wind_direction="NW", water_temp_pref=walleye_pref,
        )
        assert zone is not None


class TestAvailableLakes:
    def test_with_data(self, sample_geojson):
        lakes = available_lakes(bathymetry_dir=sample_geojson["dir"])
        assert "Test Lake" in lakes

    def test_empty_dir(self, tmp_path):
        assert available_lakes(bathymetry_dir=tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert available_lakes(bathymetry_dir=tmp_path / "nonexistent") == []
