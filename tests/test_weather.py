"""Tests for onkia.weather — Open-Meteo integration module."""
from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from onkia.weather import (
    PressureTrend,
    WeatherResult,
    _compute_pressure_trend,
    _estimate_monthly_cloud_cover,
    _estimate_monthly_water_temp,
    _extract_window_values,
    _hpa_to_inhg,
    _wind_direction_label,
    get_weather_for_window,
    wind_direction_label,
)


class TestWindDirectionLabel:
    def test_north(self):
        assert _wind_direction_label(0.0) == "N"
        assert _wind_direction_label(359.0) == "N"

    def test_east(self):
        assert _wind_direction_label(90.0) == "E"

    def test_south(self):
        assert _wind_direction_label(180.0) == "S"

    def test_west(self):
        assert _wind_direction_label(270.0) == "W"

    def test_northeast(self):
        assert _wind_direction_label(45.0) == "NE"

    def test_public_alias(self):
        assert wind_direction_label(90.0) == "E"


class TestPressureConversion:
    def test_standard_pressure(self):
        result = _hpa_to_inhg(1013.25)
        assert abs(result - 29.92) < 0.05

    def test_zero(self):
        assert _hpa_to_inhg(0.0) == 0.0


class TestMonthlyEstimates:
    def test_water_temp_mid_months(self):
        result = _estimate_monthly_water_temp(date(2024, 7, 15))
        assert 73.0 < result < 75.0

    def test_cloud_cover_mid_months(self):
        result = _estimate_monthly_cloud_cover(date(2024, 7, 15))
        assert 30.0 < result < 35.0

    def test_water_temp_jan_start(self):
        assert _estimate_monthly_water_temp(date(2024, 1, 1)) == 33.0

    def test_cloud_cover_jan_start(self):
        assert _estimate_monthly_cloud_cover(date(2024, 1, 1)) == 65.0


class TestPressureTrend:
    def test_falling_rapidly(self):
        interp = _compute_pressure_trend(1010.0, 1015.0, 1018.0)
        assert interp.trend == PressureTrend.FALLING
        assert "aggressively" in interp.fishing_note.lower() or "bite" in interp.fishing_note.lower()

    def test_stable(self):
        interp = _compute_pressure_trend(1013.0, 1013.0, 1013.0)
        assert interp.trend == PressureTrend.STABLE
        assert "normal" in interp.fishing_note.lower()

    def test_rising(self):
        interp = _compute_pressure_trend(1018.0, 1012.0, 1010.0)
        assert interp.trend == PressureTrend.RISING

    def test_rising_after_drop(self):
        interp = _compute_pressure_trend(1015.0, 1010.0, 1018.0)
        assert interp.trend == PressureTrend.RISING_AFTER_DROP
        assert "resume" in interp.fishing_note.lower() or "feeding" in interp.fishing_note.lower()

    def test_high_steady(self):
        interp = _compute_pressure_trend(1025.0, 1025.0, 1025.0)
        assert interp.trend == PressureTrend.HIGH_STEADY
        assert "slower" in interp.fishing_note.lower()

    def test_no_prior_data(self):
        interp = _compute_pressure_trend(1013.0, None, None)
        assert interp.trend == PressureTrend.STABLE

    def test_only_6h_available(self):
        interp = _compute_pressure_trend(1010.0, 1012.0, None)
        assert interp.trend == PressureTrend.FALLING


class TestExtractWindowValues:
    def _make_hourly(self, query_date: date) -> dict:
        times = []
        temps = []
        pressures = []
        wind_speeds = []
        wind_dirs = []
        clouds = []
        for h in range(24):
            times.append(f"{query_date.isoformat()}T{h:02d}:00")
            temps.append(50.0 + h)
            pressures.append(1013.0 + h * 0.1)
            wind_speeds.append(5.0 + h * 0.5)
            wind_dirs.append(float(h * 15))
            clouds.append(40.0 + h)
        return {
            "time": times,
            "temperature_2m": temps,
            "pressure_msl": pressures,
            "wind_speed_10m": wind_speeds,
            "wind_direction_10m": wind_dirs,
            "cloud_cover": clouds,
        }

    def test_extracts_evening_window(self):
        hourly = self._make_hourly(date(2024, 6, 15))
        window = _extract_window_values(hourly, date(2024, 6, 15), 17, 21)
        assert len(window["temps"]) == 5
        assert window["temps"][0] == 67.0
        assert window["temps"][-1] == 71.0

    def test_empty_for_wrong_date(self):
        hourly = self._make_hourly(date(2024, 6, 15))
        window = _extract_window_values(hourly, date(2024, 6, 16), 17, 21)
        assert len(window["temps"]) == 0


class TestGetWeatherForWindow:
    @patch("onkia.weather._fetch_hourly_data")
    def test_returns_real_data(self, mock_fetch):
        mock_fetch.return_value = {
            "time": [f"2024-06-15T{h:02d}:00" for h in range(24)],
            "temperature_2m": [50.0 + h for h in range(24)],
            "pressure_msl": [1013.0 for h in range(24)],
            "wind_speed_10m": [8.0 for h in range(24)],
            "wind_direction_10m": [180.0 for h in range(24)],
            "cloud_cover": [40.0 for h in range(24)],
        }
        result = get_weather_for_window(45.3, -94.1, date(2024, 6, 15))
        assert result.fallback_used is False
        assert result.air_temp_f is not None
        assert result.pressure_hpa is not None
        assert result.wind_speed_mph == 8.0
        assert result.cloud_cover_pct == 40.0
        assert result.pressure_trend is not None

    @patch("onkia.weather._fetch_hourly_data")
    def test_fallback_on_api_failure(self, mock_fetch):
        mock_fetch.return_value = None
        result = get_weather_for_window(45.3, -94.1, date(2024, 6, 15))
        assert result.fallback_used is True
        assert result.air_temp_f is not None
        assert result.cloud_cover_pct is not None
        assert result.pressure_hpa is None
        assert result.wind_speed_mph is None

    @patch("onkia.weather._fetch_hourly_data")
    def test_fallback_on_empty_window(self, mock_fetch):
        mock_fetch.return_value = {
            "time": [],
            "temperature_2m": [],
            "pressure_msl": [],
            "wind_speed_10m": [],
            "wind_direction_10m": [],
            "cloud_cover": [],
        }
        result = get_weather_for_window(45.3, -94.1, date(2024, 6, 15))
        assert result.fallback_used is True

    @patch("onkia.weather._fetch_hourly_data")
    def test_pressure_inhg_computed(self, mock_fetch):
        mock_fetch.return_value = {
            "time": [f"2024-07-10T{h:02d}:00" for h in range(24)],
            "temperature_2m": [75.0 for h in range(24)],
            "pressure_msl": [1013.25 for h in range(24)],
            "wind_speed_10m": [5.0 for h in range(24)],
            "wind_direction_10m": [270.0 for h in range(24)],
            "cloud_cover": [30.0 for h in range(24)],
        }
        result = get_weather_for_window(45.3, -94.1, date(2024, 7, 10))
        assert result.pressure_inhg is not None
        assert abs(result.pressure_inhg - 29.92) < 0.1

    @patch("onkia.weather._fetch_hourly_data")
    def test_forecast_for_future_date(self, mock_fetch):
        future = date.today() + timedelta(days=2)
        mock_fetch.return_value = {
            "time": [f"{future.isoformat()}T{h:02d}:00" for h in range(24)],
            "temperature_2m": [70.0 for h in range(24)],
            "pressure_msl": [1015.0 for h in range(24)],
            "wind_speed_10m": [6.0 for h in range(24)],
            "wind_direction_10m": [200.0 for h in range(24)],
            "cloud_cover": [35.0 for h in range(24)],
        }
        result = get_weather_for_window(45.3, -94.1, future)
        assert result.fallback_used is False
        assert result.air_temp_f == 70.0


class TestFetchHourlyData:
    @patch("onkia.weather.requests.get")
    def test_connection_error_returns_none(self, mock_get):
        import requests as _req
        mock_get.side_effect = _req.exceptions.ConnectionError()
        from onkia.weather import _fetch_hourly_data
        result = _fetch_hourly_data(45.3, -94.1, date(2024, 6, 15), MagicMock())
        assert result is None

    @patch("onkia.weather.requests.get")
    def test_timeout_returns_none(self, mock_get):
        import requests as _req
        mock_get.side_effect = _req.exceptions.Timeout()
        from onkia.weather import _fetch_hourly_data
        result = _fetch_hourly_data(45.3, -94.1, date(2024, 6, 15), MagicMock())
        assert result is None

    @patch("onkia.weather.requests.get")
    def test_http_error_returns_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("500")
        mock_get.return_value = mock_resp
        from onkia.weather import _fetch_hourly_data
        result = _fetch_hourly_data(45.3, -94.1, date(2024, 6, 15), MagicMock())
        assert result is None

    @patch("onkia.weather.requests.get")
    def test_valid_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2024-06-15T00:00"],
                "temperature_2m": [50.0],
            }
        }
        mock_get.return_value = mock_resp
        from onkia.weather import _fetch_hourly_data
        result = _fetch_hourly_data(45.3, -94.1, date(2024, 6, 15), MagicMock())
        assert result is not None
        assert "time" in result
