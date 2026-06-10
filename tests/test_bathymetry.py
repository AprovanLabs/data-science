"""Tests for onkia.bathymetry module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

from onkia.bathymetry import (
    _keep_basin,
    available_lakes,
    contour_color,
    contours_bbox,
    contours_to_profile,
    depth_area_profile,
    fetch_contours_from_dnr,
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
                    "coordinates": [[[-94.0, 45.3], [-94.0, 45.31], [-93.99, 45.31], [-93.99, 45.3], [-94.0, 45.3]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"depth_ft": 15.0, "lake": "Test Lake", "dow": "86000000"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-94.0, 45.3], [-94.0, 45.305], [-93.995, 45.305], [-93.995, 45.3], [-94.0, 45.3]]],
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


class TestKeepBasin:
    def test_exact_match(self):
        assert _keep_basin("77002301", "77002301")

    def test_parent_dow_matches_sub_basins(self):
        assert _keep_basin("86025201", "86025200")
        assert _keep_basin("86025202", "86025200")

    def test_specific_basin_does_not_match_siblings(self):
        assert not _keep_basin("86025202", "86025201")

    def test_other_lake_rejected(self):
        assert not _keep_basin("86099901", "86025200")


class TestFetchContoursFromDnr:
    def _mock_response(self, features, exceeded=False):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.json.return_value = {
            "type": "FeatureCollection",
            "features": features,
            "exceededTransferLimit": exceeded,
        }
        resp.raise_for_status.return_value = None
        return resp

    def _service_feature(self, dowlknum="77002300", abs_depth=10.0, closed=True):
        coords = [[-94.79, 45.77], [-94.79, 45.78], [-94.78, 45.78], [-94.79, 45.77]]
        if closed:
            coords = [[-94.79, 45.77], [-94.79, 45.78], [-94.78, 45.78], [-94.79, 45.77]]
        return {
            "type": "Feature",
            "properties": {"dowlknum": dowlknum, "abs_depth": abs_depth, "depth": -abs_depth},
            "geometry": {"type": "LineString", "coordinates": coords},
        }

    def test_fetch_converts_closed_lines_to_polygons(self):
        features = [self._service_feature(abs_depth=10.0), self._service_feature(abs_depth=20.0)]
        with patch("requests.get", return_value=self._mock_response(features)) as mock_get:
            result = fetch_contours_from_dnr("Big Swan", "77002300")
        assert result is not None
        assert len(result["features"]) == 2
        feat = result["features"][0]
        assert feat["properties"]["depth_ft"] == 10.0
        assert feat["properties"]["lake"] == "Big Swan"
        assert feat["geometry"]["type"] == "Polygon"
        # Query must use the 6-digit DOW prefix for sub-basin matching.
        params = mock_get.call_args.kwargs["params"]
        assert "770023" in params["where"]

    def test_fetch_filters_other_lakes(self):
        features = [
            self._service_feature(dowlknum="77002300", abs_depth=10.0),
            # Different 6-digit DOW prefix = a different lake entirely.
            self._service_feature(dowlknum="77009900", abs_depth=15.0),
        ]
        with patch("requests.get", return_value=self._mock_response(features)):
            result = fetch_contours_from_dnr("Big Swan", "77002300")
        assert result is not None
        assert len(result["features"]) == 1

    def test_fetch_returns_none_on_network_error(self):
        with patch("requests.get", side_effect=ConnectionError("boom")):
            assert fetch_contours_from_dnr("Big Swan", "77002300") is None

    def test_fetch_returns_none_for_no_contours(self):
        with patch("requests.get", return_value=self._mock_response([])):
            assert fetch_contours_from_dnr("Big Swan", "77002300") is None


class TestContoursToProfile:
    def test_profile_from_contours(self, sample_geojson):
        profile = contours_to_profile("Test Lake", "86000000", sample_geojson["data"])
        assert profile is not None
        assert profile["max_depth"] == 15.0
        assert profile["min_depth"] == 5.0
        assert profile["depths"] == [5, 15]
        assert profile["contour_count"] == 2

    def test_empty_contours(self):
        assert contours_to_profile("X", "1", {"type": "FeatureCollection", "features": []}) is None


class TestContoursBbox:
    def test_bbox_with_padding(self, sample_geojson):
        bbox = contours_bbox(sample_geojson["data"], pad_frac=0.0)
        assert bbox == (-94.0, 45.3, -93.99, 45.31)

    def test_padding_expands_bbox(self, sample_geojson):
        inner = contours_bbox(sample_geojson["data"], pad_frac=0.0)
        padded = contours_bbox(sample_geojson["data"], pad_frac=0.15)
        assert padded[0] < inner[0] and padded[1] < inner[1]
        assert padded[2] > inner[2] and padded[3] > inner[3]

    def test_empty_features(self):
        assert contours_bbox({"type": "FeatureCollection", "features": []}) is None


class TestDepthAreaProfileWithPreloadedContours:
    def test_preloaded_contours_used_without_disk(self, sample_geojson):
        # No bathymetry_dir given — only the preloaded dict should be used.
        profile = depth_area_profile("Some Other Lake", contours=sample_geojson["data"])
        assert len(profile) == 2
        depths = [d for d, _ in profile]
        acres = [a for _, a in profile]
        assert depths == [5.0, 15.0]
        # Hypsographic: shallower contour encloses at least as much area.
        assert acres[0] >= acres[1] > 0


class TestAvailableLakes:
    def test_with_data(self, sample_geojson):
        lakes = available_lakes(bathymetry_dir=sample_geojson["dir"])
        assert "Test Lake" in lakes

    def test_empty_dir(self, tmp_path):
        assert available_lakes(bathymetry_dir=tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert available_lakes(bathymetry_dir=tmp_path / "nonexistent") == []
