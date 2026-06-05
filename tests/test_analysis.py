"""Unit tests for onkia.analysis — CPUE trend analysis module."""
from __future__ import annotations

from typing import List

import pytest

from onkia.analysis import (
    MIN_SURVEYS_FOR_TREND,
    LakeAnalysis,
    SpeciesTrendSummary,
    StockingEvent,
    TrendStatus,
    WaterClarityTrend,
    analyze_lake,
    analyze_species_at_lake,
    analyze_water_clarity,
    compute_cpue_trend,
    parse_stocking_events,
)
from onkia.models import FishCatchSummary, Survey, SurveyOverview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_survey(survey_date: str, catches: List[dict]) -> Survey:
    """Build a Survey object from a date string and a list of catch dicts."""
    summaries = []
    for c in catches:
        summaries.append(
            FishCatchSummary(
                species=c["species"],
                total_catch=c.get("total_catch", 10),
                total_weight=c.get("total_weight", 15.0),
                cpue=str(c.get("cpue", 1.0)),
                average_weight=str(c.get("average_weight", 1.5)),
                gear=c.get("gear", "Standard gill nets"),
                gear_count=c.get("gear_count", 16),
                quartile_count="1.0-3.0",
                quartile_weight="0.5-2.0",
            )
        )
    return Survey(
        survey_sub_type="",
        survey_type="Standard Survey",
        survey_date=survey_date,
        narrative="",
        survey_id=1,
        fish_catch_summaries=summaries,
    )


def _make_overview(
    surveys: List[Survey],
    water_clarity: list = None,
) -> SurveyOverview:
    return SurveyOverview(
        littoral_acres=500.0,
        mean_depth_feet=15.0,
        max_depth_feet=30,
        area_acres=1000.0,
        shore_length_miles=5.0,
        water_clarity=water_clarity or [],
        surveys=surveys,
    )


# ---------------------------------------------------------------------------
# compute_cpue_trend
# ---------------------------------------------------------------------------


class TestComputeCpueTrend:
    def test_insufficient_data_below_min(self):
        assert compute_cpue_trend([1.0, 2.0], [2018, 2020]) == TrendStatus.INSUFFICIENT_DATA

    def test_insufficient_data_empty(self):
        assert compute_cpue_trend([], []) == TrendStatus.INSUFFICIENT_DATA

    def test_increasing_trend(self):
        # Clear upward slope: 1, 2, 3, 4, 5
        assert compute_cpue_trend([1.0, 2.0, 3.0, 4.0, 5.0], [2018, 2019, 2020, 2021, 2022]) == TrendStatus.INCREASING

    def test_decreasing_trend(self):
        # Clear downward slope: 5, 4, 3, 2, 1
        assert compute_cpue_trend([5.0, 4.0, 3.0, 2.0, 1.0], [2018, 2019, 2020, 2021, 2022]) == TrendStatus.DECREASING

    def test_stable_trend(self):
        # Nearly flat: tiny variation around 3.0
        assert compute_cpue_trend([3.0, 3.01, 2.99, 3.0, 3.01], [2018, 2019, 2020, 2021, 2022]) == TrendStatus.STABLE

    def test_exactly_min_surveys(self):
        result = compute_cpue_trend([1.0, 2.0, 3.0], [2018, 2020, 2022])
        assert result in (TrendStatus.INCREASING, TrendStatus.STABLE, TrendStatus.DECREASING)

    def test_all_zero_cpue(self):
        assert compute_cpue_trend([0.0, 0.0, 0.0], [2018, 2020, 2022]) == TrendStatus.STABLE

    def test_custom_stability_fraction(self):
        # With a very high threshold, a real slope should still be stable
        result = compute_cpue_trend(
            [1.0, 1.1, 1.2], [2018, 2020, 2022], stability_fraction=10.0
        )
        assert result == TrendStatus.STABLE


# ---------------------------------------------------------------------------
# analyze_species_at_lake
# ---------------------------------------------------------------------------


class TestAnalyzeSpeciesAtLake:
    def test_returns_none_when_no_matching_species(self):
        surveys = [
            _make_survey("2022-07-01", [{"species": "YEP", "cpue": 3.0}]),
            _make_survey("2020-07-01", [{"species": "YEP", "cpue": 4.0}]),
        ]
        result = analyze_species_at_lake("Walleye", ["WAE"], surveys)
        assert result is None

    def test_returns_summary_for_matching_species(self):
        surveys = [
            _make_survey("2022-07-01", [{"species": "WAE", "cpue": 2.0}]),
            _make_survey("2020-07-01", [{"species": "WAE", "cpue": 3.0}]),
        ]
        result = analyze_species_at_lake("Walleye", ["WAE"], surveys)
        assert isinstance(result, SpeciesTrendSummary)
        assert result.species == "Walleye"

    def test_insufficient_data_with_two_surveys(self):
        surveys = [
            _make_survey("2022-07-01", [{"species": "WAE", "cpue": 2.0}]),
            _make_survey("2020-07-01", [{"species": "WAE", "cpue": 3.0}]),
        ]
        result = analyze_species_at_lake("Walleye", ["WAE"], surveys)
        assert result.status == TrendStatus.INSUFFICIENT_DATA

    def test_trend_with_three_or_more_surveys(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "WAE", "cpue": 1.0}]),
            _make_survey("2022-07-01", [{"species": "WAE", "cpue": 2.0}]),
            _make_survey("2020-07-01", [{"species": "WAE", "cpue": 3.0}]),
        ]
        result = analyze_species_at_lake("Walleye", ["WAE"], surveys)
        assert result.status == TrendStatus.DECREASING

    def test_evidence_contains_cpue_values(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "WAE", "cpue": 1.0}]),
            _make_survey("2022-07-01", [{"species": "WAE", "cpue": 2.0}]),
            _make_survey("2020-07-01", [{"species": "WAE", "cpue": 3.0}]),
        ]
        result = analyze_species_at_lake("Walleye", ["WAE"], surveys)
        assert "3.00" in result.evidence
        assert "1.00" in result.evidence

    def test_survey_years_are_sorted(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "NOP", "cpue": 5.0}]),
            _make_survey("2018-07-01", [{"species": "NOP", "cpue": 3.0}]),
            _make_survey("2021-07-01", [{"species": "NOP", "cpue": 4.0}]),
        ]
        result = analyze_species_at_lake("Northern Pike", ["NOP"], surveys)
        assert result.survey_years == [2018, 2021, 2024]

    def test_sunfish_aggregates_multiple_codes(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "BLG", "cpue": 5.0}]),
            _make_survey("2022-07-01", [{"species": "GSF", "cpue": 4.0}]),
            _make_survey("2020-07-01", [{"species": "BLG", "cpue": 3.0}]),
        ]
        result = analyze_species_at_lake("Sunfish", ["BLG", "GSF", "HSF", "PMK"], surveys)
        assert result is not None
        assert len(result.survey_years) == 3

    def test_skips_zero_cpue(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "WAE", "cpue": 0.0}]),
            _make_survey("2022-07-01", [{"species": "WAE", "cpue": 2.0}]),
        ]
        result = analyze_species_at_lake("Walleye", ["WAE"], surveys)
        # Only one valid record (cpue > 0)
        assert result.latest_cpue == 2.0

    def test_best_gear_in_summary(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "LMB", "cpue": 3.0, "gear": "Standard trap nets"}]),
            _make_survey("2022-07-01", [{"species": "LMB", "cpue": 2.0, "gear": "Standard trap nets"}]),
            _make_survey("2020-07-01", [{"species": "LMB", "cpue": 1.0, "gear": "Standard gill nets"}]),
        ]
        result = analyze_species_at_lake("Largemouth Bass", ["LMB"], surveys)
        assert result.best_gear == "Standard trap nets"

    def test_avg_weight_computed(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "WAE", "cpue": 2.0, "average_weight": 2.0}]),
            _make_survey("2022-07-01", [{"species": "WAE", "cpue": 3.0, "average_weight": 1.8}]),
        ]
        result = analyze_species_at_lake("Walleye", ["WAE"], surveys)
        assert result.avg_weight_lbs is not None
        assert abs(result.avg_weight_lbs - 1.9) < 0.01


# ---------------------------------------------------------------------------
# analyze_water_clarity
# ---------------------------------------------------------------------------


class TestAnalyzeWaterClarity:
    def test_returns_none_for_empty_input(self):
        assert analyze_water_clarity([]) is None

    def test_returns_none_for_invalid_data(self):
        assert analyze_water_clarity([["bad", "data"]]) is None

    def test_returns_trend_object(self):
        data = [["07/21/2014", "4.5"], ["08/01/2022", "6.5"]]
        result = analyze_water_clarity(data)
        assert isinstance(result, WaterClarityTrend)

    def test_insufficient_data_for_two_readings(self):
        data = [["2018-01-01", "4.5"], ["2022-01-01", "5.5"]]
        result = analyze_water_clarity(data)
        assert result.status == TrendStatus.INSUFFICIENT_DATA

    def test_increasing_clarity(self):
        data = [
            ["2018-01-01", "3.0"],
            ["2020-01-01", "4.0"],
            ["2022-01-01", "5.0"],
            ["2024-01-01", "6.0"],
        ]
        result = analyze_water_clarity(data)
        assert result.status == TrendStatus.INCREASING

    def test_decreasing_clarity(self):
        data = [
            ["2018-01-01", "6.0"],
            ["2020-01-01", "5.0"],
            ["2022-01-01", "4.0"],
            ["2024-01-01", "3.0"],
        ]
        result = analyze_water_clarity(data)
        assert result.status == TrendStatus.DECREASING

    def test_evidence_contains_years_and_values(self):
        data = [
            ["2018-01-01", "4.0"],
            ["2020-01-01", "4.5"],
            ["2022-01-01", "5.0"],
        ]
        result = analyze_water_clarity(data)
        assert "2018" in result.evidence
        assert "2022" in result.evidence

    def test_recent_and_earliest_clarity_populated(self):
        data = [["2018-01-01", "4.0"], ["2022-01-01", "6.0"]]
        result = analyze_water_clarity(data)
        assert result.earliest_clarity_ft == 4.0
        assert result.recent_clarity_ft == 6.0

    def test_skips_zero_clarity(self):
        data = [["2018-01-01", "0.0"], ["2022-01-01", "5.0"]]
        result = analyze_water_clarity(data)
        assert result.earliest_clarity_ft == 5.0


# ---------------------------------------------------------------------------
# parse_stocking_events
# ---------------------------------------------------------------------------


class TestParseStockingEvents:
    def test_returns_empty_for_empty_input(self):
        assert parse_stocking_events([]) == []

    def test_parses_basic_record(self):
        records = [{"species": "Walleye", "year": "2022", "quantity": "50000"}]
        events = parse_stocking_events(records)
        assert len(events) == 1
        assert events[0].species == "Walleye"
        assert events[0].year == 2022
        assert events[0].quantity == 50000

    def test_quantity_handles_comma_formatting(self):
        records = [{"species": "Walleye", "year": "2022", "quantity": "50,000"}]
        events = parse_stocking_events(records)
        assert events[0].quantity == 50000

    def test_sorted_by_year_descending(self):
        records = [
            {"species": "Walleye", "year": "2018", "quantity": "10000"},
            {"species": "Walleye", "year": "2022", "quantity": "20000"},
            {"species": "Walleye", "year": "2015", "quantity": "5000"},
        ]
        events = parse_stocking_events(records)
        assert [e.year for e in events] == [2022, 2018, 2015]

    def test_skips_records_without_year(self):
        records = [{"species": "Walleye", "quantity": "50000"}]
        events = parse_stocking_events(records)
        assert events == []

    def test_skips_records_without_species(self):
        records = [{"year": "2022", "quantity": "50000"}]
        events = parse_stocking_events(records)
        assert events == []

    def test_handles_none_quantity(self):
        records = [{"species": "Walleye", "year": "2022"}]
        events = parse_stocking_events(records)
        assert len(events) == 1
        assert events[0].quantity is None


# ---------------------------------------------------------------------------
# analyze_lake (integration)
# ---------------------------------------------------------------------------


class TestAnalyzeLake:
    def _make_rich_overview(self) -> SurveyOverview:
        surveys = [
            _make_survey("2024-07-01", [
                {"species": "WAE", "cpue": 1.5, "average_weight": 1.8},
                {"species": "NOP", "cpue": 4.0, "average_weight": 3.0},
            ]),
            _make_survey("2022-07-01", [
                {"species": "WAE", "cpue": 2.0, "average_weight": 1.9},
                {"species": "NOP", "cpue": 5.0, "average_weight": 3.2},
            ]),
            _make_survey("2020-07-01", [
                {"species": "WAE", "cpue": 2.5, "average_weight": 2.0},
                {"species": "NOP", "cpue": 6.0, "average_weight": 3.5},
            ]),
        ]
        return _make_overview(
            surveys=surveys,
            water_clarity=[["2018-01-01", "4.0"], ["2022-01-01", "5.5"]],
        )

    def test_returns_lake_analysis(self):
        overview = self._make_rich_overview()
        result = analyze_lake("Test Lake", overview)
        assert isinstance(result, LakeAnalysis)
        assert result.lake_name == "Test Lake"

    def test_species_trends_only_include_present_species(self):
        overview = self._make_rich_overview()
        result = analyze_lake("Test Lake", overview)
        species_names = [t.species for t in result.species_trends]
        # Only WAE and NOP have data; LMB, BLC, Sunfish should be absent
        assert "Walleye" in species_names
        assert "Northern Pike" in species_names
        assert "Largemouth Bass" not in species_names

    def test_water_clarity_trend_included(self):
        overview = self._make_rich_overview()
        result = analyze_lake("Test Lake", overview)
        assert result.water_clarity is not None

    def test_stocking_events_parsed(self):
        overview = self._make_rich_overview()
        stocking = [{"species": "Walleye", "year": "2021", "quantity": "30000"}]
        result = analyze_lake("Test Lake", overview, stocking_records=stocking)
        assert len(result.stocking_events) == 1
        assert result.stocking_events[0].species == "Walleye"

    def test_no_stocking_records_defaults_to_empty(self):
        overview = self._make_rich_overview()
        result = analyze_lake("Test Lake", overview)
        assert result.stocking_events == []

    def test_empty_surveys_yields_no_species_trends(self):
        overview = _make_overview(surveys=[])
        result = analyze_lake("Empty Lake", overview)
        assert result.species_trends == []

    def test_walleye_decreasing_trend_detected(self):
        surveys = [
            _make_survey("2024-07-01", [{"species": "WAE", "cpue": 1.0}]),
            _make_survey("2022-07-01", [{"species": "WAE", "cpue": 2.0}]),
            _make_survey("2020-07-01", [{"species": "WAE", "cpue": 3.0}]),
        ]
        overview = _make_overview(surveys=surveys)
        result = analyze_lake("Clearwater", overview)
        wae = next(t for t in result.species_trends if t.species == "Walleye")
        assert wae.status == TrendStatus.DECREASING
