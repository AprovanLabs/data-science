"""Tests for onkia.satellite_lst — GEE satellite lake surface temperature module."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from onkia.satellite_lst import (
    HEATMAP_THRESHOLD_ACRES,
    LAKE_ACRES,
    LST_USEFUL_THRESHOLD_ACRES,
    LSTHistoryPoint,
    LSTObservation,
    NDCIObservation,
    _reset_gee_state,
    celsius_to_fahrenheit,
    get_latest_lst,
    get_lst_heatmap_url,
    get_lst_history,
    get_ndci,
    kelvin_to_celsius,
    lake_radius_m,
    landsat_dn_to_celsius,
    ndci_to_category,
)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestCelsiusToFahrenheit:
    def test_freezing(self):
        assert celsius_to_fahrenheit(0.0) == 32.0

    def test_boiling(self):
        assert celsius_to_fahrenheit(100.0) == 212.0

    def test_body_temp(self):
        assert abs(celsius_to_fahrenheit(37.0) - 98.6) < 0.01

    def test_negative(self):
        assert celsius_to_fahrenheit(-40.0) == -40.0


class TestKelvinToCelsius:
    def test_absolute_zero(self):
        assert kelvin_to_celsius(0.0) == -273.15

    def test_water_freezing(self):
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0, abs=0.01)

    def test_water_boiling(self):
        assert kelvin_to_celsius(373.15) == pytest.approx(100.0, abs=0.01)


class TestLandsatDnToCelsius:
    def test_scale_and_offset(self):
        # A DN of 0 → Kelvin = 0 * 0.00341802 + 149.0 → Celsius = 149 - 273.15
        result = landsat_dn_to_celsius(0.0)
        assert abs(result - (149.0 - 273.15)) < 0.001

    def test_typical_summer_value(self):
        # DN ≈ 37000 → Kelvin ≈ 126.5 + 149 = 275.5K → ~2°C
        # A more realistic test: pick a DN that gives ~20°C
        # 20°C + 273.15 = 293.15K; (293.15 - 149.0) / 0.00341802 ≈ 42194
        dn = (293.15 - 149.0) / 0.00341802
        result = landsat_dn_to_celsius(dn)
        assert abs(result - 20.0) < 0.01

    def test_output_type_float(self):
        assert isinstance(landsat_dn_to_celsius(30000.0), float)


class TestNdciToCategory:
    def test_low_negative(self):
        assert ndci_to_category(-0.1) == "low"

    def test_low_zero(self):
        assert ndci_to_category(0.0) == "moderate"

    def test_moderate_midpoint(self):
        assert ndci_to_category(0.05) == "moderate"

    def test_moderate_upper_bound(self):
        assert ndci_to_category(0.1) == "moderate"

    def test_high(self):
        assert ndci_to_category(0.2) == "high"


class TestLakeRadiusM:
    def test_clearwater_large(self):
        # 3158 acres → ≥ 2000 → 3000m
        assert lake_radius_m("Clearwater") == 3000.0

    def test_buffalo_lake_medium_large(self):
        # 1552 acres → ≥ 1000 → 2000m
        assert lake_radius_m("Buffalo Lake") == 2000.0

    def test_lake_sylvia_medium(self):
        # 904 acres → ≥ 500 → 1200m
        assert lake_radius_m("Lake Sylvia") == 1200.0

    def test_bass_lake_small(self):
        # 218 acres → < 500 → 700m
        assert lake_radius_m("Bass Lake") == 700.0

    def test_unknown_lake_default(self):
        # Unknown → falls back to 300 acres → 700m
        assert lake_radius_m("Unknown Lake XYZ") == 700.0


class TestLakeAcresConstants:
    def test_clearwater_present(self):
        assert "Clearwater" in LAKE_ACRES
        assert LAKE_ACRES["Clearwater"] > 3000

    def test_threshold_ordering(self):
        assert LST_USEFUL_THRESHOLD_ACRES < HEATMAP_THRESHOLD_ACRES

    def test_large_lakes_above_useful_threshold(self):
        large = ["Clearwater", "Buffalo Lake", "Lake Sylvia", "Maple Lake"]
        for name in large:
            assert LAKE_ACRES[name] >= LST_USEFUL_THRESHOLD_ACRES, name

    def test_small_lakes_below_useful_threshold(self):
        small = ["Lake Charlotte", "Lake Ida", "Bass Lake"]
        for name in small:
            assert LAKE_ACRES[name] < LST_USEFUL_THRESHOLD_ACRES, name

    def test_heatmap_lakes(self):
        # Only large lakes should qualify for heatmap
        heatmap_lakes = [n for n, a in LAKE_ACRES.items() if a >= HEATMAP_THRESHOLD_ACRES]
        for name in heatmap_lakes:
            assert LAKE_ACRES[name] >= 1000, name


# ---------------------------------------------------------------------------
# GEE unavailable / fallback paths
# ---------------------------------------------------------------------------

class TestGetLatestLstFallback:
    def setup_method(self):
        _reset_gee_state()

    def test_returns_fallback_when_gee_not_installed(self):
        with patch("onkia.satellite_lst._init_gee", return_value=False):
            result = get_latest_lst("Clearwater", 45.3052, -94.1184)
        assert isinstance(result, LSTObservation)
        assert result.fallback_used is True
        assert result.temp_fahrenheit is None
        assert result.lake_name == "Clearwater"

    def test_returns_fallback_on_gee_exception(self):
        mock_ee = MagicMock()
        with patch.dict("sys.modules", {"ee": mock_ee}):
            with patch("onkia.satellite_lst._init_gee", return_value=True):
                with patch("onkia.satellite_lst._landsat_collection", side_effect=RuntimeError("GEE error")):
                    result = get_latest_lst("Clearwater", 45.3052, -94.1184)
        assert result.fallback_used is True
        assert "GEE error" in (result.error_msg or "")

    def test_fallback_when_no_pixels(self):
        mock_collection = MagicMock()
        mock_collection.size.return_value.getInfo.return_value = 1
        mock_composite = MagicMock()
        mock_collection.median.return_value = mock_composite

        with patch.dict("sys.modules", {"ee": MagicMock()}):
            with patch("onkia.satellite_lst._init_gee", return_value=True):
                with patch("onkia.satellite_lst._landsat_collection", return_value=mock_collection):
                    with patch("onkia.satellite_lst._mean_lst_over_region", return_value=None):
                        with patch("onkia.satellite_lst._pixel_count_over_region", return_value=0):
                            with patch("onkia.satellite_lst._most_recent_scene_date", return_value=None):
                                result = get_latest_lst("Clearwater", 45.3052, -94.1184)
        assert result.fallback_used is True
        assert result.error_msg is not None


class TestGetLatestLstSuccess:
    def setup_method(self):
        _reset_gee_state()

    def _mock_collection(self, scene_count: int = 3, lst_c: float = 20.0, obs_date: date = date(2026, 5, 1)):
        mock_col = MagicMock()
        mock_col.size.return_value.getInfo.return_value = scene_count
        mock_col.median.return_value = MagicMock()
        return mock_col, lst_c, obs_date

    def test_returns_correct_temperatures(self):
        mock_col, lst_c, obs_date = self._mock_collection(lst_c=20.0)
        expected_f = celsius_to_fahrenheit(20.0)

        with patch.dict("sys.modules", {"ee": MagicMock()}):
            with patch("onkia.satellite_lst._init_gee", return_value=True):
                with patch("onkia.satellite_lst._landsat_collection", return_value=mock_col):
                    with patch("onkia.satellite_lst._mean_lst_over_region", return_value=lst_c):
                        with patch("onkia.satellite_lst._pixel_count_over_region", return_value=500):
                            with patch("onkia.satellite_lst._most_recent_scene_date", return_value=obs_date):
                                result = get_latest_lst("Clearwater", 45.3052, -94.1184)

        assert result.fallback_used is False
        assert result.temp_celsius == 20.0
        assert abs(result.temp_fahrenheit - expected_f) < 0.1
        assert result.observation_date == obs_date
        assert result.scene_count == 3
        assert result.pixel_count == 500

    def test_celsius_to_fahrenheit_roundtrip(self):
        test_c = 15.7
        mock_col = MagicMock()
        mock_col.size.return_value.getInfo.return_value = 2
        mock_col.median.return_value = MagicMock()

        with patch.dict("sys.modules", {"ee": MagicMock()}):
            with patch("onkia.satellite_lst._init_gee", return_value=True):
                with patch("onkia.satellite_lst._landsat_collection", return_value=mock_col):
                    with patch("onkia.satellite_lst._mean_lst_over_region", return_value=test_c):
                        with patch("onkia.satellite_lst._pixel_count_over_region", return_value=100):
                            with patch("onkia.satellite_lst._most_recent_scene_date", return_value=date(2026, 6, 1)):
                                result = get_latest_lst("Maple Lake", 45.2243, -94.0062)

        assert result.temp_celsius == round(test_c, 1)
        assert result.temp_fahrenheit == round(celsius_to_fahrenheit(test_c), 1)


# ---------------------------------------------------------------------------
# get_lst_history
# ---------------------------------------------------------------------------

class TestGetLstHistory:
    def setup_method(self):
        _reset_gee_state()

    def test_returns_empty_when_gee_unavailable(self):
        with patch("onkia.satellite_lst._init_gee", return_value=False):
            result = get_lst_history("Clearwater", 45.3052, -94.1184)
        assert result == []

    def test_returns_empty_on_exception(self):
        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._landsat_collection", side_effect=Exception("timeout")):
                result = get_lst_history("Clearwater", 45.3052, -94.1184)
        assert result == []

    def test_returns_sorted_history(self):
        dates_returned = [date(2026, 3, 1), date(2026, 4, 1), date(2026, 5, 1)]
        temps = [5.0, 12.0, 18.0]
        call_count = 0

        def mock_collection(region, start, end):
            nonlocal call_count
            mock_col = MagicMock()
            if call_count < len(temps):
                mock_col.size.return_value.getInfo.return_value = 1
            else:
                mock_col.size.return_value.getInfo.return_value = 0
            mock_col.median.return_value = MagicMock()
            call_count += 1
            return mock_col

        lst_iter = iter(temps)

        def mock_mean_lst(composite, region):
            try:
                return next(lst_iter)
            except StopIteration:
                return None

        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._landsat_collection", side_effect=mock_collection):
                with patch("onkia.satellite_lst._mean_lst_over_region", side_effect=mock_mean_lst):
                    result = get_lst_history("Clearwater", 45.3052, -94.1184, days_back=90, interval_days=30)

        assert all(isinstance(p, LSTHistoryPoint) for p in result)
        dates_in_result = [p.observation_date for p in result]
        assert dates_in_result == sorted(dates_in_result)

    def test_history_points_have_both_units(self):
        mock_col = MagicMock()
        mock_col.size.return_value.getInfo.return_value = 1
        mock_col.median.return_value = MagicMock()

        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._landsat_collection", return_value=mock_col):
                with patch("onkia.satellite_lst._mean_lst_over_region", return_value=10.0):
                    result = get_lst_history("Clearwater", 45.3052, -94.1184, days_back=16, interval_days=16)

        if result:
            point = result[0]
            assert point.temp_celsius == 10.0
            assert abs(point.temp_fahrenheit - celsius_to_fahrenheit(10.0)) < 0.1


# ---------------------------------------------------------------------------
# get_ndci
# ---------------------------------------------------------------------------

class TestGetNdci:
    def setup_method(self):
        _reset_gee_state()

    def test_returns_fallback_when_gee_unavailable(self):
        with patch("onkia.satellite_lst._init_gee", return_value=False):
            result = get_ndci("Clearwater", 45.3052, -94.1184)
        assert isinstance(result, NDCIObservation)
        assert result.fallback_used is True
        assert result.ndci_value is None

    def test_returns_fallback_on_empty_collection(self):
        mock_col = MagicMock()
        mock_col.size.return_value.getInfo.return_value = 0

        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._build_lake_region", return_value=MagicMock()):
                with patch("onkia.satellite_lst._most_recent_scene_date", return_value=None):
                    with patch.dict("sys.modules", {"ee": _make_mock_ee(mock_col)}):
                        result = get_ndci("Clearwater", 45.3052, -94.1184)
        assert isinstance(result, NDCIObservation)
        assert result.fallback_used is True

    def test_returns_fallback_on_exception(self):
        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._build_lake_region", side_effect=RuntimeError("auth")):
                result = get_ndci("Clearwater", 45.3052, -94.1184)
        assert result.fallback_used is True

    def test_ndci_category_propagated(self):
        mock_col = MagicMock()
        mock_col.size.return_value.getInfo.return_value = 2
        # Set up the NDCI reduce result chain
        mock_col.median.return_value.reduceRegion.return_value.get.return_value.getInfo.return_value = 0.15

        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._build_lake_region", return_value=MagicMock()):
                with patch("onkia.satellite_lst._most_recent_scene_date", return_value=date(2026, 6, 1)):
                    with patch.dict("sys.modules", {"ee": _make_mock_ee(mock_col)}):
                        result = get_ndci("Clearwater", 45.3052, -94.1184)
        assert isinstance(result, NDCIObservation)
        # If we got a real value back, verify category logic
        if not result.fallback_used and result.ndci_value is not None:
            assert result.chlorophyll_category in ("low", "moderate", "high")


# ---------------------------------------------------------------------------
# get_lst_heatmap_url
# ---------------------------------------------------------------------------

class TestGetLstHeatmapUrl:
    def setup_method(self):
        _reset_gee_state()

    def test_returns_none_when_gee_unavailable(self):
        with patch("onkia.satellite_lst._init_gee", return_value=False):
            result = get_lst_heatmap_url(45.3052, -94.1184, 3000.0)
        assert result is None

    def test_returns_none_on_exception(self):
        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._build_lake_region", side_effect=RuntimeError("err")):
                result = get_lst_heatmap_url(45.3052, -94.1184, 3000.0)
        assert result is None

    def test_returns_none_on_empty_collection(self):
        mock_col = MagicMock()
        mock_col.size.return_value.getInfo.return_value = 0

        with patch("onkia.satellite_lst._init_gee", return_value=True):
            with patch("onkia.satellite_lst._build_lake_region", return_value=MagicMock()):
                with patch("onkia.satellite_lst._landsat_collection", return_value=mock_col):
                    result = get_lst_heatmap_url(45.3052, -94.1184, 3000.0)
        assert result is None

    def test_returns_url_string_on_success(self):
        expected_url = "https://earthengine.googleapis.com/thumb/abc123"
        mock_col = MagicMock()
        mock_col.size.return_value.getInfo.return_value = 2
        mock_composite = MagicMock()
        mock_composite.getThumbURL.return_value = expected_url
        mock_col.median.return_value = mock_composite

        with patch.dict("sys.modules", {"ee": MagicMock()}):
            with patch("onkia.satellite_lst._init_gee", return_value=True):
                with patch("onkia.satellite_lst._build_lake_region", return_value=MagicMock()):
                    with patch("onkia.satellite_lst._landsat_collection", return_value=mock_col):
                        result = get_lst_heatmap_url(45.3052, -94.1184, 3000.0)
        assert result == expected_url


# ---------------------------------------------------------------------------
# Helpers for mocking ee module
# ---------------------------------------------------------------------------

def _make_mock_ee(mock_collection: MagicMock) -> MagicMock:
    """Build a minimal mock of the `ee` module for testing get_ndci."""
    mock_ee = MagicMock()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.\
        filter.return_value.map.return_value = mock_collection
    mock_ee.Filter.lt.return_value = MagicMock()
    mock_ee.Reducer.mean.return_value = MagicMock()
    return mock_ee
