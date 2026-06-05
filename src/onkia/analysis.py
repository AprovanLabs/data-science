"""CPUE trend analysis for lake-specific fishing intelligence.

Computes population trend narratives with traceable evidence for
each target species at a selected lake, using linear regression over
historical survey data from the MN DNR.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from onkia.models import Survey, SurveyOverview


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SPECIES_CODES: Dict[str, List[str]] = {
    "Walleye": ["WAE"],
    "Northern Pike": ["NOP"],
    "Largemouth Bass": ["LMB"],
    "Black Crappie": ["BLC"],
    "Sunfish": ["BLG", "GSF", "HSF", "PMK"],
}

# Minimum number of data points required to compute a trend
MIN_SURVEYS_FOR_TREND = 3

# |slope| < this fraction of mean CPUE per year → classified as STABLE
_SLOPE_STABILITY_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TrendStatus(str, Enum):
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class SpeciesTrendSummary:
    """Trend narrative for a single species at a lake, with traceable evidence."""

    species: str
    status: TrendStatus
    # Human-readable evidence string citing survey years, CPUE values, and gear
    evidence: str
    latest_cpue: Optional[float]
    earliest_cpue: Optional[float]
    survey_years: List[int]
    avg_weight_lbs: Optional[float]
    avg_weight_trend: TrendStatus
    best_gear: Optional[str]


@dataclass
class WaterClarityTrend:
    """Secchi disk clarity trend with traceable evidence."""

    status: TrendStatus
    evidence: str
    recent_clarity_ft: Optional[float]
    earliest_clarity_ft: Optional[float]


@dataclass
class StockingEvent:
    """A single fish stocking event parsed from DNR stocking records."""

    species: str
    year: int
    quantity: Optional[int]
    life_stage: str = ""


@dataclass
class LakeAnalysis:
    """Complete trend analysis for all target species at a lake."""

    lake_name: str
    species_trends: List[SpeciesTrendSummary] = field(default_factory=list)
    water_clarity: Optional[WaterClarityTrend] = None
    stocking_events: List[StockingEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core trend computation
# ---------------------------------------------------------------------------


def _linear_slope(x: List[float], y: List[float]) -> float:
    """Return the slope of the least-squares regression line through (x, y)."""
    if len(x) < 2:
        return 0.0
    coeffs = np.polyfit(np.array(x, dtype=float), np.array(y, dtype=float), 1)
    return float(coeffs[0])


def compute_cpue_trend(
    cpue_values: List[float],
    years: List[int],
    *,
    stability_fraction: float = _SLOPE_STABILITY_FRACTION,
) -> TrendStatus:
    """Classify a CPUE time series as INCREASING, DECREASING, or STABLE.

    Returns INSUFFICIENT_DATA when fewer than MIN_SURVEYS_FOR_TREND points exist.
    The stability threshold is ``stability_fraction * mean(cpue)`` per year — changes
    smaller than this are classified as STABLE rather than noise-driven trends.
    """
    if len(cpue_values) < MIN_SURVEYS_FOR_TREND:
        return TrendStatus.INSUFFICIENT_DATA
    mean_cpue = float(np.mean(cpue_values))
    if mean_cpue == 0.0:
        return TrendStatus.STABLE
    slope = _linear_slope([float(y) for y in years], cpue_values)
    threshold = stability_fraction * mean_cpue
    if slope > threshold:
        return TrendStatus.INCREASING
    if slope < -threshold:
        return TrendStatus.DECREASING
    return TrendStatus.STABLE


# ---------------------------------------------------------------------------
# Survey data extraction helpers
# ---------------------------------------------------------------------------


def _collect_cpue_records(
    surveys: List[Survey],
    species_codes: List[str],
) -> List[Tuple[int, float, float, str]]:
    """Extract (year, cpue, avg_weight, gear) for the given species codes.

    Only records with a positive, parseable CPUE are included.
    Results are sorted by year ascending.
    """
    records: List[Tuple[int, float, float, str]] = []
    for survey in surveys:
        try:
            year = int(str(survey.survey_date)[:4])
        except (ValueError, TypeError, AttributeError):
            continue
        for catch in survey.fish_catch_summaries:
            if catch.species not in species_codes:
                continue
            try:
                cpue = float(catch.cpue)
            except (ValueError, TypeError):
                continue
            if cpue <= 0:
                continue
            try:
                avg_w = float(catch.average_weight)
            except (ValueError, TypeError):
                avg_w = float("nan")
            records.append((year, cpue, avg_w, catch.gear or ""))
    records.sort(key=lambda r: r[0])
    return records


def _best_gear(records: List[Tuple[int, float, float, str]]) -> Optional[str]:
    """Return the gear type with the highest mean CPUE across all records."""
    gear_totals: Dict[str, List[float]] = {}
    for _, cpue, _, gear in records:
        if gear:
            gear_totals.setdefault(gear, []).append(cpue)
    if not gear_totals:
        return None
    return max(gear_totals, key=lambda g: float(np.mean(gear_totals[g])))


def _build_species_evidence(
    species: str,
    status: TrendStatus,
    records: List[Tuple[int, float, float, str]],
) -> str:
    """Build a traceable evidence string citing survey years, CPUE values, and gear."""
    if not records:
        return f"No survey data found for {species}."
    if status == TrendStatus.INSUFFICIENT_DATA:
        years = sorted({r[0] for r in records})
        year_str = ", ".join(str(y) for y in years)
        return (
            f"Only {len(records)} survey record(s) available "
            f"(year(s): {year_str}) — insufficient for trend."
        )
    earliest_year, earliest_cpue = records[0][0], records[0][1]
    latest_year, latest_cpue = records[-1][0], records[-1][1]
    gear = _best_gear(records) or "surveys"
    arrow = {"INCREASING": "↑", "DECREASING": "↓", "STABLE": "→"}.get(status.value, "")
    return (
        f"CPUE {earliest_cpue:.2f} → {latest_cpue:.2f} {arrow} "
        f"over {earliest_year}–{latest_year} {gear} "
        f"({len(records)} survey point(s))"
    )


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------


def analyze_species_at_lake(
    species_name: str,
    species_codes: List[str],
    surveys: List[Survey],
) -> Optional[SpeciesTrendSummary]:
    """Compute a trend summary for a single species across all surveys at a lake.

    Returns None if the species has no data in any survey.
    """
    records = _collect_cpue_records(surveys, species_codes)
    if not records:
        return None

    years = [r[0] for r in records]
    cpue_values = [r[1] for r in records]
    weight_values = [r[2] for r in records if not math.isnan(r[2])]

    status = compute_cpue_trend(cpue_values, years)
    avg_weight = float(np.mean(weight_values)) if weight_values else None

    if len(weight_values) >= MIN_SURVEYS_FOR_TREND:
        weight_years = [records[i][0] for i in range(len(records)) if not math.isnan(records[i][2])]
        weight_trend = compute_cpue_trend(weight_values, weight_years)
    else:
        weight_trend = TrendStatus.INSUFFICIENT_DATA

    return SpeciesTrendSummary(
        species=species_name,
        status=status,
        evidence=_build_species_evidence(species_name, status, records),
        latest_cpue=cpue_values[-1] if cpue_values else None,
        earliest_cpue=cpue_values[0] if cpue_values else None,
        survey_years=sorted(set(years)),
        avg_weight_lbs=avg_weight,
        avg_weight_trend=weight_trend,
        best_gear=_best_gear(records),
    )


def analyze_water_clarity(
    water_clarity: List[List[Union[float, str]]],
) -> Optional[WaterClarityTrend]:
    """Compute a Secchi disk clarity trend from SurveyOverview.water_clarity data.

    Each item in water_clarity is expected to be [date_str_or_year, clarity_ft].
    Returns None if no valid readings are found.
    """
    records: List[Tuple[int, float]] = []
    for item in water_clarity:
        if len(item) < 2:
            continue
        try:
            date_str = str(item[0])
            # Handle both ISO format "YYYY-MM-DD" and US format "MM/DD/YYYY"
            if "/" in date_str:
                year = int(date_str.split("/")[-1])
            else:
                year = int(date_str[:4])
            clarity = float(item[1])
        except (ValueError, TypeError):
            continue
        if clarity > 0:
            records.append((year, clarity))
    records.sort(key=lambda r: r[0])
    if not records:
        return None

    years = [r[0] for r in records]
    clarity_values = [r[1] for r in records]
    status = compute_cpue_trend(clarity_values, years)

    earliest_year, earliest_val = records[0]
    latest_year, latest_val = records[-1]

    if status == TrendStatus.INSUFFICIENT_DATA:
        evidence = (
            f"Only {len(records)} Secchi reading(s) — insufficient for trend. "
            f"Most recent: {latest_val:.1f} ft ({latest_year})."
        )
    else:
        direction = {
            TrendStatus.INCREASING: "clearer",
            TrendStatus.DECREASING: "murkier",
            TrendStatus.STABLE: "stable",
        }.get(status, "unchanged")
        evidence = (
            f"Water clarity {earliest_val:.1f} ft ({earliest_year}) → "
            f"{latest_val:.1f} ft ({latest_year}): {direction} "
            f"({len(records)} Secchi readings)."
        )
    return WaterClarityTrend(
        status=status,
        evidence=evidence,
        recent_clarity_ft=latest_val,
        earliest_clarity_ft=earliest_val,
    )


def parse_stocking_events(stocking_records: List[Dict]) -> List[StockingEvent]:
    """Parse a list of stocking record dicts into StockingEvent objects.

    Handles the variable field names produced by ``app._parse_stocking``.
    """
    events: List[StockingEvent] = []
    for record in stocking_records:
        year_raw = (
            record.get("year")
            or record.get("Year")
            or record.get("YEAR")
            or str(record.get("stocking_date", ""))[:4]
            or str(record.get("date", ""))[:4]
        )
        try:
            year = int(str(year_raw))
        except (ValueError, TypeError):
            continue

        species = str(
            record.get("species")
            or record.get("Species")
            or record.get("SPECIES")
            or record.get("fish_species")
            or ""
        )
        if not species:
            continue

        quantity_raw = (
            record.get("quantity")
            or record.get("Quantity")
            or record.get("number")
            or record.get("Number")
        )
        try:
            quantity = int(str(quantity_raw).replace(",", ""))
        except (ValueError, TypeError):
            quantity = None

        life_stage = str(
            record.get("lifestage")
            or record.get("life_stage")
            or record.get("LifeStage")
            or record.get("stage")
            or ""
        )
        events.append(StockingEvent(species=species, year=year, quantity=quantity, life_stage=life_stage))

    return sorted(events, key=lambda e: e.year, reverse=True)


def analyze_lake(
    lake_name: str,
    survey_overview: SurveyOverview,
    stocking_records: Optional[List[Dict]] = None,
) -> LakeAnalysis:
    """Run full trend analysis for all target species at a lake.

    Analyzes Walleye, Northern Pike, Largemouth Bass, Black Crappie, and Sunfish
    (only species present in surveys are included in the result).

    Args:
        lake_name: Display name of the lake.
        survey_overview: Parsed SurveyOverview from MnDnrLakeTopographyService.
        stocking_records: Optional list of stocking record dicts from _parse_stocking.

    Returns:
        LakeAnalysis containing per-species trends, water clarity trend,
        and parsed stocking events.
    """
    surveys = survey_overview.surveys
    species_trends: List[SpeciesTrendSummary] = []
    for species_name, codes in TARGET_SPECIES_CODES.items():
        summary = analyze_species_at_lake(species_name, codes, surveys)
        if summary is not None:
            species_trends.append(summary)

    return LakeAnalysis(
        lake_name=lake_name,
        species_trends=species_trends,
        water_clarity=analyze_water_clarity(survey_overview.water_clarity),
        stocking_events=parse_stocking_events(stocking_records or []),
    )
