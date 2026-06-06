"""Tests for onkia.usgs_glm — USGS GLM depth-resolved temperature profiles."""
from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from onkia.usgs_glm import (
    available_lakes,
    generate_sample_data,
    get_climatological_profile,
    get_species_depth_zones,
    get_temperature_profile,
    get_thermocline_depth,
    has_data,
    _c_to_f,
    _f_to_c,
    _parse_optimum_range,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "usgs_glm"
    d.mkdir()
    return d


@pytest.fixture
def sample_lake_parquet(tmp_data_dir):
    """Write a small parquet file mimicking USGS GLM output."""
    rows = []
    base = date(2021, 6, 15)
    for day_offset in range(-14, 15):
        d = date(2021, 6, 15) + __import__("datetime").timedelta(days=day_offset)
        for depth_m in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]:
            surface = 22.0
            if depth_m <= 6.0:
                temp = surface - 1.5 * depth_m
            else:
                temp = 5.0
            if day_offset > 0:
                surface += 0.3 * day_offset
            rows.append({
                "date": d.isoformat(),
                "depth_m": depth_m,
                "temp_c": round(temp, 2),
            })
    df = pd.DataFrame(rows)
    out = tmp_data_dir / "86025200.parquet"
    df.to_parquet(out, index=False)
    return out


@pytest.fixture
def patched_data_dir(sample_lake_parquet, monkeypatch):
    import onkia.usgs_glm as mod
    monkeypatch.setattr(mod, "_DATA_DIR", sample_lake_parquet.parent)


class TestUnitConversions:
    def test_c_to_f(self):
        assert _c_to_f(0.0) == 32.0
        assert _c_to_f(100.0) == 212.0
        assert abs(_c_to_f(20.0) - 68.0) < 0.01

    def test_f_to_c(self):
        assert _f_to_c(32.0) == 0.0
        assert _f_to_c(212.0) == 100.0
        assert abs(_f_to_c(68.0) - 20.0) < 0.01

    def test_round_trip(self):
        for t_c in [-10.0, 0.0, 15.0, 30.0]:
            assert abs(_c_to_f(_f_to_c(t_c)) - t_c) < 0.001

    def test_parse_optimum_single(self):
        low, high = _parse_optimum_range("64")
        assert low == pytest.approx(_f_to_c(64.0))
        assert high == pytest.approx(_f_to_c(64.0))

    def test_parse_optimum_range(self):
        low, high = _parse_optimum_range("64-70")
        assert low == pytest.approx(_f_to_c(64.0))
        assert high == pytest.approx(_f_to_c(70.0))


class TestAvailableLakes:
    def test_available_lakes_empty(self, tmp_data_dir, monkeypatch):
        import onkia.usgs_glm as mod
        monkeypatch.setattr(mod, "_DATA_DIR", tmp_data_dir)
        empty = tmp_data_dir / "empty_sub"
        empty.mkdir()
        monkeypatch.setattr(mod, "_DATA_DIR", empty)
        assert available_lakes() == []

    def test_available_lakes_with_data(self, patched_data_dir):
        lakes = available_lakes()
        assert "86025200" in lakes

    def test_has_data(self, patched_data_dir):
        assert has_data("86025200") is True
        assert has_data("99999999") is False


class TestGetTemperatureProfile:
    def test_exact_date(self, patched_data_dir):
        profile = get_temperature_profile("86025200", date(2021, 6, 15))
        assert profile is not None
        assert "depth_m" in profile.columns
        assert "temp_c" in profile.columns
        assert len(profile) > 0
        assert profile["depth_m"].is_monotonic_increasing

    def test_post_2022_uses_climatology(self, patched_data_dir):
        profile = get_temperature_profile("86025200", date(2025, 6, 15))
        assert profile is not None
        assert len(profile) > 0

    def test_no_data_lake(self, patched_data_dir):
        profile = get_temperature_profile("99999999", date(2021, 6, 15))
        assert profile is None


class TestGetClimatologicalProfile:
    def test_climatology(self, patched_data_dir):
        profile = get_climatological_profile("86025200", 166)
        assert profile is not None
        assert len(profile) > 0

    def test_climatology_no_data(self, patched_data_dir):
        profile = get_climatological_profile("99999999", 166)
        assert profile is None


class TestGetThermoclineDepth:
    def test_thermocline_detected(self, patched_data_dir):
        thermo = get_thermocline_depth("86025200", date(2021, 6, 15))
        if thermo is not None:
            assert 1.0 <= thermo <= 20.0

    def test_thermocline_no_data(self, patched_data_dir):
        thermo = get_thermocline_depth("99999999", date(2021, 6, 15))
        assert thermo is None


class TestGetSpeciesDepthZones:
    def test_species_zones(self, patched_data_dir):
        prefs = [
            {"species": "Walleye", "optimum": "64-70", "lower_avoidance": 50, "upper_avoidance": 76},
            {"species": "Northern Pike", "optimum": "65", "lower_avoidance": 55, "upper_avoidance": 74},
        ]
        zones = get_species_depth_zones("86025200", date(2021, 6, 15), prefs)
        assert zones is not None
        assert len(zones) == 2
        assert zones[0]["species"] == "Walleye"
        assert "opt_depth_low_m" in zones[0]
        assert "thermo_depth_m" in zones[0]

    def test_species_zones_no_data(self, patched_data_dir):
        prefs = [{"species": "Walleye", "optimum": "64-70", "lower_avoidance": 50, "upper_avoidance": 76}]
        zones = get_species_depth_zones("99999999", date(2021, 6, 15), prefs)
        assert zones is None


class TestGenerateSampleData:
    def test_generates_parquet(self, tmp_data_dir, monkeypatch):
        import onkia.usgs_glm as mod
        monkeypatch.setattr(mod, "_DATA_DIR", tmp_data_dir)
        out = generate_sample_data("86025200", "Clearwater", 25.0, 45.3, output_dir=tmp_data_dir)
        assert out.exists()
        df = pd.read_parquet(out)
        assert {"date", "depth_m", "temp_c"}.issubset(df.columns)
        assert len(df) > 0
        assert df["depth_m"].max() <= 25.0

    def test_generate_all_wright_county(self, tmp_data_dir, monkeypatch):
        import onkia.usgs_glm as mod
        monkeypatch.setattr(mod, "_DATA_DIR", tmp_data_dir)
        from scripts.download_usgs_glm import WRIGHT_COUNTY_LAKES

        for name, (lat, lon, dow) in WRIGHT_COUNTY_LAKES.items():
            generate_sample_data(dow, name, 15.0, lat, output_dir=tmp_data_dir)

        lakes = available_lakes()
        assert len(lakes) == len(WRIGHT_COUNTY_LAKES)
