"""Tests for onkia.plan_generator — evening fishing plan generator."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pytest

from onkia.models import WaterTempPreference
from onkia.plan_generator import (
    EveningPlan,
    PlanBlock,
    _build_conditions_summary,
    _build_location_guidance,
    _build_technique_note,
    _compute_cpue_map,
    _compute_species_scores,
    _format_hour,
    _format_hour30,
    _select_block_species,
    _temp_condition,
    _wind_label,
    generate_evening_plan,
)
from onkia.weather import PressureInterpretation, PressureTrend, WeatherResult


def _make_pref(
    species: str,
    lower: Optional[float] = None,
    optimum: str = "65",
    upper: Optional[float] = None,
) -> WaterTempPreference:
    return WaterTempPreference(
        species=species,
        lower_avoidance=lower,
        optimum=optimum,
        upper_avoidance=upper,
    )


def _make_weather(
    air_temp_f: Optional[float] = 70.0,
    pressure_hpa: Optional[float] = 1013.0,
    wind_speed: Optional[float] = 8.0,
    wind_dir: Optional[float] = 180.0,
    cloud_cover: Optional[float] = 40.0,
    pressure_trend: Optional[PressureTrend] = None,
    fallback: bool = False,
) -> WeatherResult:
    pi = None
    if pressure_trend:
        pi = PressureInterpretation(
            trend=pressure_trend,
            label=pressure_trend.value,
            fishing_note="test",
        )
    return WeatherResult(
        air_temp_f=air_temp_f,
        pressure_hpa=pressure_hpa,
        pressure_inhg=30.0 if pressure_hpa else None,
        wind_speed_mph=wind_speed,
        wind_direction_deg=wind_dir,
        cloud_cover_pct=cloud_cover,
        pressure_trend=pressure_trend,
        pressure_interpretation=pi,
        fallback_used=fallback,
    )


_TARGET_PREFS = [
    _make_pref("Largemouth Bass", 50, "65-75", 85),
    _make_pref("Northern Pike", 55, "65", 74),
    _make_pref("Sunfish", 50, "58", 68),
    _make_pref("Black Crappie", 60, "71", 75),
    _make_pref("Walleye", 50, "64-70", 76),
]


class TestTempCondition:
    def test_optimal(self):
        pref = _make_pref("Walleye", 50, "64-70", 76)
        assert _temp_condition(67.0, pref) == "optimal"

    def test_cold(self):
        pref = _make_pref("Walleye", 50, "64-70", 76)
        assert _temp_condition(45.0, pref) == "cold"

    def test_warm(self):
        pref = _make_pref("Walleye", 50, "64-70", 76)
        assert _temp_condition(80.0, pref) == "warm"

    def test_below_lower_avoidance(self):
        pref = _make_pref("Walleye", 50, "64-70", 76)
        assert _temp_condition(40.0, pref) == "cold"

    def test_above_upper_avoidance(self):
        pref = _make_pref("Walleye", 50, "64-70", 76)
        assert _temp_condition(85.0, pref) == "warm"

    def test_unknown_optimum(self):
        pref = WaterTempPreference(species="Test", lower_avoidance=None, optimum="N/A", upper_avoidance=None)
        assert _temp_condition(65.0, pref) == "unknown"


class TestWindLabel:
    def test_north(self):
        assert _wind_label(0.0) == "N"

    def test_south(self):
        assert _wind_label(180.0) == "S"

    def test_east(self):
        assert _wind_label(90.0) == "E"

    def test_west(self):
        assert _wind_label(270.0) == "W"


class TestFormatHour:
    def test_morning(self):
        assert _format_hour(9) == "9:00 AM"

    def test_noon(self):
        assert _format_hour(12) == "12:00 PM"

    def test_evening(self):
        assert _format_hour(17) == "5:00 PM"

    def test_night(self):
        assert _format_hour(21) == "9:00 PM"


class TestFormatHour30:
    def test_530_pm(self):
        assert _format_hour30(17, half=True) == "5:30 PM"

    def test_900_pm(self):
        assert _format_hour30(21, half=False) == "9:00 PM"


class TestComputeSpeciesScores:
    def test_optimal_species_scores_higher(self):
        scores = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=PressureTrend.STABLE,
            wind_speed=5.0,
            block_key="early",
            cpue_by_species={"Sunfish": 10.0, "Black Crappie": 5.0, "Largemouth Bass": 8.0},
        )
        assert "Sunfish" in scores
        assert "Black Crappie" in scores
        assert "Largemouth Bass" in scores
        assert scores["Sunfish"] > 0

    def test_falling_pressure_boosts_scores(self):
        scores_stable = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=PressureTrend.STABLE,
            wind_speed=5.0,
            block_key="early",
            cpue_by_species={},
        )
        scores_falling = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=PressureTrend.FALLING,
            wind_speed=5.0,
            block_key="early",
            cpue_by_species={},
        )
        for sp in scores_stable:
            assert scores_falling.get(sp, 0) > scores_stable[sp]

    def test_dusk_bonus_for_walleye(self):
        scores_early = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=PressureTrend.STABLE,
            wind_speed=5.0,
            block_key="early",
            cpue_by_species={"Walleye": 5.0},
        )
        scores_dusk = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=PressureTrend.STABLE,
            wind_speed=5.0,
            block_key="dusk",
            cpue_by_species={"Walleye": 5.0},
        )
        assert scores_dusk.get("Walleye", 0) > scores_early.get("Walleye", 0)

    def test_windy_bonus_for_pike(self):
        scores_calm = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=PressureTrend.STABLE,
            wind_speed=3.0,
            block_key="late",
            cpue_by_species={"Northern Pike": 5.0},
        )
        scores_windy = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=PressureTrend.STABLE,
            wind_speed=12.0,
            block_key="late",
            cpue_by_species={"Northern Pike": 5.0},
        )
        assert scores_windy.get("Northern Pike", 0) > scores_calm.get("Northern Pike", 0)

    def test_no_pressure_trend(self):
        scores = _compute_species_scores(
            water_temp_f=67.0,
            prefs=_TARGET_PREFS,
            pressure_trend=None,
            wind_speed=5.0,
            block_key="early",
            cpue_by_species={},
        )
        assert len(scores) > 0


class TestSelectBlockSpecies:
    def test_highest_score_wins(self):
        scores = {"Walleye": 3.0, "Bass": 5.0, "Crappie": 1.0}
        assert _select_block_species("dusk", scores) == "Bass"

    def test_empty_scores(self):
        assert _select_block_species("dusk", {}) is None


class TestBuildLocationGuidance:
    def test_windward_shore_when_windy(self):
        loc, evidence = _build_location_guidance(
            "Walleye", "optimal", 180.0, 12.0,
        )
        assert "windward" in loc.lower() or "North shore" in loc
        assert len(evidence) > 0

    def test_calm_conditions(self):
        loc, evidence = _build_location_guidance(
            "Sunfish", "optimal", 90.0, 3.0,
        )
        assert "Weed" in loc or "shallow" in loc.lower()

    def test_no_wind_data(self):
        loc, evidence = _build_location_guidance(
            "Largemouth Bass", "optimal", None, None,
        )
        assert len(loc) > 0

    def test_walleye_dusk_location(self):
        loc, evidence = _build_location_guidance(
            "Walleye", "optimal", 180.0, 5.0,
        )
        assert "point" in loc.lower() or "Rock" in loc
        assert any("low-light" in e.lower() or "Walleye" in e for e in evidence)


class TestBuildTechniqueNote:
    def test_falling_pressure_aggressive(self):
        tech, evidence = _build_technique_note(
            "Walleye", "optimal", PressureTrend.FALLING,
        )
        assert "aggressive" in tech.lower()
        assert any("falling" in e.lower() for e in evidence)

    def test_rising_pressure_slow(self):
        tech, evidence = _build_technique_note(
            "Largemouth Bass", "optimal", PressureTrend.RISING,
        )
        assert "slow" in tech.lower()
        assert any("rising" in e.lower() for e in evidence)

    def test_stable_pressure_standard(self):
        tech, evidence = _build_technique_note(
            "Black Crappie", "optimal", PressureTrend.STABLE,
        )
        assert "optimal" in tech.lower()

    def test_high_steady_finesse(self):
        tech, evidence = _build_technique_note(
            "Northern Pike", "optimal", PressureTrend.HIGH_STEADY,
        )
        assert "finesse" in tech.lower() or "deep" in tech.lower()

    def test_no_pressure_trend(self):
        tech, evidence = _build_technique_note(
            "Sunfish", "optimal", None,
        )
        assert "optimal" in tech.lower()


class TestComputeCpueMap:
    def test_with_survey_records(self):
        records = [
            {"species_code": "WAE", "cpue": 12.5, "survey_date": "2022-06-15"},
            {"species_code": "LMB", "cpue": 8.3, "survey_date": "2022-06-15"},
            {"species_code": "BLC", "cpue": 15.2, "survey_date": "2022-06-15"},
        ]
        cpue_map = _compute_cpue_map(records)
        assert cpue_map["Walleye"] == 12.5
        assert cpue_map["Largemouth Bass"] == 8.3
        assert cpue_map["Black Crappie"] == 15.2

    def test_no_records(self):
        cpue_map = _compute_cpue_map(None)
        assert all(v is None for v in cpue_map.values())

    def test_sunfish_aggregates_codes(self):
        records = [
            {"species_code": "BLG", "cpue": 10.0, "survey_date": "2022-06-15"},
            {"species_code": "GSF", "cpue": 5.0, "survey_date": "2022-06-15"},
        ]
        cpue_map = _compute_cpue_map(records)
        assert cpue_map["Sunfish"] is not None

    def test_invalid_cpue_skipped(self):
        records = [
            {"species_code": "WAE", "cpue": "N/A", "survey_date": "2022-06-15"},
        ]
        cpue_map = _compute_cpue_map(records)
        assert cpue_map["Walleye"] is None


class TestBuildConditionsSummary:
    def test_full_conditions(self):
        summary = _build_conditions_summary(
            water_temp_f=68.0,
            pressure_trend=PressureTrend.FALLING,
            pressure_interp=PressureInterpretation(
                trend=PressureTrend.FALLING,
                label="Falling rapidly",
                fishing_note="Best bite!",
            ),
            wind_speed=12.0,
            wind_dir_deg=180.0,
            cloud_cover=45.0,
        )
        assert "68" in summary
        assert "Falling rapidly" in summary
        assert "12" in summary

    def test_minimal_conditions(self):
        summary = _build_conditions_summary(
            water_temp_f=55.0,
            pressure_trend=None,
            pressure_interp=None,
            wind_speed=None,
            wind_dir_deg=None,
            cloud_cover=None,
        )
        assert "55" in summary


class TestGenerateEveningPlan:
    def test_produces_3_blocks(self):
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=68.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
        )
        assert len(plan.blocks) >= 2
        assert len(plan.blocks) <= 3

    def test_each_block_has_required_fields(self):
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=68.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
        )
        for block in plan.blocks:
            assert block.species
            assert block.time_start
            assert block.time_end
            assert block.location
            assert block.technique
            assert block.lures
            assert block.evidence

    def test_plan_adapts_to_weather(self):
        plan_calm = generate_evening_plan(
            weather=_make_weather(
                air_temp_f=68.0,
                wind_speed=3.0,
                pressure_trend=PressureTrend.STABLE,
            ),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
        )
        plan_windy = generate_evening_plan(
            weather=_make_weather(
                air_temp_f=68.0,
                wind_speed=15.0,
                wind_dir=180.0,
                pressure_trend=PressureTrend.FALLING,
            ),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
        )
        assert plan_calm.conditions_summary != plan_windy.conditions_summary
        calm_locs = [b.location for b in plan_calm.blocks]
        windy_locs = [b.location for b in plan_windy.blocks]
        assert calm_locs != windy_locs

    def test_plan_with_survey_data(self):
        records = [
            {"species_code": "WAE", "cpue": 20.0, "survey_date": "2023-06-15"},
            {"species_code": "LMB", "cpue": 10.0, "survey_date": "2023-06-15"},
            {"species_code": "BLC", "cpue": 15.0, "survey_date": "2023-06-15"},
        ]
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=68.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
            survey_records=records,
        )
        for block in plan.blocks:
            has_cpue_evidence = any("CPUE" in e for e in block.evidence)
            has_cpue_data = block.species in ("Walleye", "Largemouth Bass", "Black Crappie")
            if has_cpue_data:
                assert has_cpue_evidence, f"Expected CPUE evidence for {block.species}"

    def test_conditions_summary(self):
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=68.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
        )
        assert "68" in plan.conditions_summary

    def test_data_sources(self):
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=68.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
            survey_records=[{"species_code": "WAE", "cpue": 5.0, "survey_date": "2022-06-15"}],
        )
        assert len(plan.data_sources) >= 2
        assert any("DNR" in ds for ds in plan.data_sources)

    def test_fallback_weather_data_source(self):
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=66.0, fallback=True),
            water_temp_f=66.0,
            target_species_prefs=_TARGET_PREFS,
        )
        assert any("Seasonal" in ds or "fallback" in ds.lower() for ds in plan.data_sources)

    def test_evidence_is_traceable(self):
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=68.0, pressure_trend=PressureTrend.FALLING),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
            survey_records=[{"species_code": "WAE", "cpue": 8.0, "survey_date": "2023-06-15"}],
        )
        for block in plan.blocks:
            assert len(block.evidence) >= 2, f"{block.species} block needs traceable evidence"

    def test_custom_time_window(self):
        plan = generate_evening_plan(
            weather=_make_weather(air_temp_f=68.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
            start_hour=18,
            end_hour=20,
        )
        assert len(plan.blocks) >= 1

    def test_cold_water_changes_plan(self):
        plan_warm = generate_evening_plan(
            weather=_make_weather(air_temp_f=80.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=70.0,
            target_species_prefs=_TARGET_PREFS,
        )
        plan_cold = generate_evening_plan(
            weather=_make_weather(air_temp_f=40.0, pressure_trend=PressureTrend.STABLE),
            water_temp_f=38.0,
            target_species_prefs=_TARGET_PREFS,
        )
        warm_species = [b.species for b in plan_warm.blocks]
        cold_species = [b.species for b in plan_cold.blocks]
        techniques_warm = [b.technique for b in plan_warm.blocks]
        techniques_cold = [b.technique for b in plan_cold.blocks]
        assert techniques_warm != techniques_cold or warm_species != cold_species


def _serialize_weather_like_fishing_page(weather: WeatherResult) -> str:
    """Reproduce the JSON serialization performed by fishing.py before calling the
    Streamlit-cached plan generator.  pressure_trend is stored as its .value string
    and pressure_interpretation is omitted entirely."""
    return json.dumps({
        "air_temp_f": weather.air_temp_f,
        "pressure_hpa": weather.pressure_hpa,
        "pressure_inhg": weather.pressure_inhg,
        "wind_speed_mph": weather.wind_speed_mph,
        "wind_direction_deg": weather.wind_direction_deg,
        "cloud_cover_pct": weather.cloud_cover_pct,
        "pressure_trend": weather.pressure_trend.value if weather.pressure_trend else None,
        "source": weather.source,
        "fallback_used": weather.fallback_used,
    })


def _deserialize_weather_like_cache_layer(weather_json: str) -> WeatherResult:
    """Reproduce the fixed deserialization in fishing_data._generate_plan_cached."""
    raw = json.loads(weather_json)
    if raw.get("pressure_trend"):
        trend = PressureTrend(raw["pressure_trend"])
        raw["pressure_trend"] = trend
        _TREND_DEFAULTS = {
            PressureTrend.FALLING: ("Falling rapidly", "Fish feed aggressively before a front — best bite window!"),
            PressureTrend.RISING: ("Rising", "Fish may be less active — try deeper water and slower presentations."),
            PressureTrend.RISING_AFTER_DROP: ("Rising after a drop", "Fish resume feeding after pressure stabilizes — good action."),
            PressureTrend.HIGH_STEADY: ("High and steady", "High pressure = slower fishing. Focus on deep structure and live bait."),
            PressureTrend.STABLE: ("Stable", "Normal fish activity expected."),
        }
        label, note = _TREND_DEFAULTS.get(trend, (trend.value, ""))
        raw["pressure_interpretation"] = PressureInterpretation(trend=trend, label=label, fishing_note=note)
    return WeatherResult(**raw)


class TestWeatherJsonRoundTrip:
    """Regression tests for the Streamlit cache JSON round-trip bug.

    Before the fix, fishing.py serialised pressure_trend as a plain string and
    _generate_plan_cached reconstructed WeatherResult without converting it back
    to the PressureTrend enum.  plan_generator._build_conditions_summary then
    called pressure_trend.value on the string, raising AttributeError.
    """

    @pytest.mark.parametrize("trend", list(PressureTrend))
    def test_round_trip_preserves_enum(self, trend):
        weather = _make_weather(pressure_trend=trend)
        recovered = _deserialize_weather_like_cache_layer(
            _serialize_weather_like_fishing_page(weather)
        )
        assert recovered.pressure_trend is trend

    @pytest.mark.parametrize("trend", list(PressureTrend))
    def test_round_trip_preserves_pressure_interpretation(self, trend):
        weather = _make_weather(pressure_trend=trend)
        recovered = _deserialize_weather_like_cache_layer(
            _serialize_weather_like_fishing_page(weather)
        )
        assert recovered.pressure_interpretation is not None
        assert recovered.pressure_interpretation.trend is trend
        assert recovered.pressure_interpretation.label

    @pytest.mark.parametrize("trend", list(PressureTrend))
    def test_generate_plan_does_not_raise_with_round_tripped_weather(self, trend):
        """The AttributeError must not be raised after the fix."""
        weather = _make_weather(air_temp_f=68.0, pressure_trend=trend)
        recovered = _deserialize_weather_like_cache_layer(
            _serialize_weather_like_fishing_page(weather)
        )
        plan = generate_evening_plan(
            weather=recovered,
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
        )
        assert plan.conditions_summary
        assert trend.value in plan.conditions_summary

    def test_none_pressure_trend_round_trips_safely(self):
        weather = _make_weather(pressure_trend=None)
        recovered = _deserialize_weather_like_cache_layer(
            _serialize_weather_like_fishing_page(weather)
        )
        assert recovered.pressure_trend is None
        assert recovered.pressure_interpretation is None
        plan = generate_evening_plan(
            weather=recovered,
            water_temp_f=68.0,
            target_species_prefs=_TARGET_PREFS,
        )
        assert plan.conditions_summary
