"""Tests for onkia.satellite_lst_stac — STAC-based satellite LST module."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from onkia.satellite_lst import (
    LSTHistoryPoint,
    LSTObservation,
    NDCIObservation,
    celsius_to_fahrenheit,
    ndci_to_category,
)
from onkia.satellite_lst_stac import (
    _cloud_mask_landsat,
    _point_to_bbox,
    _s2_band_keys,
    get_latest_lst,
    get_lst_heatmap,
    get_lst_history,
    get_ndci,
)


class TestPointToBbox:
    def test_returns_four_element_list(self):
        bbox = _point_to_bbox(45.0, -94.0, 1000.0)
        assert len(bbox) == 4

    def test_bbox_contains_point(self):
        lat, lon = 45.3052, -94.1184
        bbox = _point_to_bbox(lat, lon, 1000.0)
        assert bbox[0] < lon < bbox[2]
        assert bbox[1] < lat < bbox[3]

    def test_bbox_is_symmetric(self):
        lat, lon = 45.0, -94.0
        bbox = _point_to_bbox(lat, lon, 2000.0)
        assert abs((lon - bbox[0]) - (bbox[2] - lon)) < 0.0001
        assert abs((lat - bbox[1]) - (bbox[3] - lat)) < 0.0001

    def test_larger_radius_produces_larger_bbox(self):
        small = _point_to_bbox(45.0, -94.0, 1000.0)
        large = _point_to_bbox(45.0, -94.0, 3000.0)
        assert (large[2] - large[0]) > (small[2] - small[0])


class TestCloudMaskLandsat:
    def test_clear_pixel_passes(self):
        qa = np.array([0], dtype=np.uint16)
        mask = _cloud_mask_landsat(qa)
        assert mask[0] is np.True_

    def test_cloud_bit_fails(self):
        qa = np.array([1 << 3], dtype=np.uint16)
        mask = _cloud_mask_landsat(qa)
        assert mask[0] is np.False_

    def test_shadow_bit_fails(self):
        qa = np.array([1 << 4], dtype=np.uint16)
        mask = _cloud_mask_landsat(qa)
        assert mask[0] is np.False_

    def test_both_cloud_and_shadow_fails(self):
        qa = np.array([(1 << 3) | (1 << 4)], dtype=np.uint16)
        mask = _cloud_mask_landsat(qa)
        assert mask[0] is np.False_

    def test_other_bits_pass(self):
        qa = np.array([1 << 1], dtype=np.uint16)
        mask = _cloud_mask_landsat(qa)
        assert mask[0] is np.True_

    def test_2d_array(self):
        qa = np.array([[0, 1 << 3], [1 << 4, 0]], dtype=np.uint16)
        mask = _cloud_mask_landsat(qa)
        assert mask.shape == (2, 2)
        assert mask[0, 0] is np.True_
        assert mask[0, 1] is np.False_
        assert mask[1, 0] is np.False_
        assert mask[1, 1] is np.True_


class TestS2BandKeys:
    def test_planetary_computer_style(self):
        item = MagicMock()
        item.assets = {"B04": MagicMock(), "B05": MagicMock(), "SCL": MagicMock()}
        b04, b05, scl = _s2_band_keys(item)
        assert b04 == "B04"
        assert b05 == "B05"
        assert scl == "SCL"

    def test_copernicus_style(self):
        item = MagicMock()
        item.assets = {"B04_20m": MagicMock(), "B05_20m": MagicMock(), "SCL_20m": MagicMock()}
        b04, b05, scl = _s2_band_keys(item)
        assert b04 == "B04_20m"
        assert b05 == "B05_20m"
        assert scl == "SCL_20m"

    def test_no_matching_keys(self):
        item = MagicMock()
        item.assets = {"B01": MagicMock()}
        b04, b05, scl = _s2_band_keys(item)
        assert b04 is None


class TestGetLatestLstFallback:
    def test_returns_fallback_when_no_items(self):
        with patch("onkia.satellite_lst_stac._fetch_landsat_items", return_value=[]):
            result = get_latest_lst("Clearwater", 45.3052, -94.1184)
        assert isinstance(result, LSTObservation)
        assert result.fallback_used is True
        assert result.lake_name == "Clearwater"

    def test_returns_fallback_on_exception(self):
        with patch(
            "onkia.satellite_lst_stac._fetch_landsat_items",
            side_effect=RuntimeError("STAC error"),
        ):
            result = get_latest_lst("Clearwater", 45.3052, -94.1184)
        assert result.fallback_used is True
        assert "STAC error" in (result.error_msg or "")


class TestGetLatestLstSuccess:
    def test_returns_observation_with_temp(self):
        mock_item = MagicMock()
        mock_item.datetime = MagicMock()
        mock_item.datetime.date.return_value = date(2026, 5, 30)
        mock_item.id = "LC09_L2SP_028029_20260530_02_T1"
        mock_item.assets = {
            "lwir11": MagicMock(href="https://example.com/thermal.TIF"),
            "qa_pixel": MagicMock(href="https://example.com/qa.TIF"),
        }

        thermal_data = np.ma.array(
            np.full((10, 10), 20000.0), dtype=np.float64
        )
        qa_data = np.ma.array(
            np.zeros((10, 10), dtype=np.uint16)
        )

        with patch(
            "onkia.satellite_lst_stac._fetch_landsat_items",
            return_value=[mock_item],
        ):
            with patch(
                "onkia.satellite_lst_stac._read_windowed",
                side_effect=lambda href, lat, lon, rm, **kw: (
                    thermal_data if "thermal" in href else qa_data
                ),
            ):
                result = get_latest_lst("Clearwater", 45.3052, -94.1184)

        assert result.fallback_used is False
        assert result.temp_celsius is not None
        assert result.observation_date == date(2026, 5, 30)


class TestGetLstHistory:
    def test_returns_empty_when_no_items(self):
        with patch("onkia.satellite_lst_stac._fetch_landsat_items", return_value=[]):
            result = get_lst_history("Clearwater", 45.3052, -94.1184)
        assert result == []

    def test_returns_empty_on_exception(self):
        with patch(
            "onkia.satellite_lst_stac._fetch_landsat_items",
            side_effect=Exception("timeout"),
        ):
            result = get_lst_history("Clearwater", 45.3052, -94.1184)
        assert result == []


class TestGetNdciFallback:
    def test_returns_fallback_when_no_items(self):
        with patch("onkia.satellite_lst_stac._fetch_sentinel2_items", return_value=[]):
            result = get_ndci("Clearwater", 45.3052, -94.1184)
        assert isinstance(result, NDCIObservation)
        assert result.fallback_used is True

    def test_returns_fallback_on_exception(self):
        with patch(
            "onkia.satellite_lst_stac._fetch_sentinel2_items",
            side_effect=RuntimeError("auth error"),
        ):
            result = get_ndci("Clearwater", 45.3052, -94.1184)
        assert result.fallback_used is True


class TestGetLstHeatmap:
    def test_returns_none_when_no_items(self):
        with patch("onkia.satellite_lst_stac._fetch_landsat_items", return_value=[]):
            result = get_lst_heatmap(45.3052, -94.1184, 3000.0)
        assert result is None

    def test_returns_none_on_exception(self):
        with patch(
            "onkia.satellite_lst_stac._fetch_landsat_items",
            side_effect=RuntimeError("err"),
        ):
            result = get_lst_heatmap(45.3052, -94.1184, 3000.0)
        assert result is None


class TestStacToGeeCompatibility:
    def test_lst_observation_shape_matches_gee(self):
        result = LSTObservation(
            lake_name="Clearwater",
            temp_celsius=20.0,
            temp_fahrenheit=68.0,
            observation_date=date(2026, 5, 30),
            scene_count=3,
            pixel_count=100,
            fallback_used=False,
        )
        assert result.lake_name == "Clearwater"
        assert result.temp_celsius == 20.0
        assert result.temp_fahrenheit == 68.0
        assert result.observation_date == date(2026, 5, 30)
        assert result.scene_count == 3
        assert result.pixel_count == 100
        assert result.fallback_used is False

    def test_ndci_observation_shape_matches_gee(self):
        result = NDCIObservation(
            lake_name="Clearwater",
            ndci_value=0.085,
            chlorophyll_category="moderate",
            observation_date=date(2026, 5, 25),
            fallback_used=False,
        )
        assert result.ndci_value == 0.085
        assert result.chlorophyll_category == "moderate"
