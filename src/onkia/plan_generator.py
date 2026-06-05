"""Evening fishing plan generator for Wright County lakes.

Produces a structured, time-sequenced plan for catching multiple species
during the evening league window (5:30-9 PM by default). All recommendations
are rule-based from data — no LLM generation.

Main entry point:
    generate_evening_plan(
        weather, water_temp_f, target_species_prefs, survey_records, wind_dir_deg,
        start_hour, end_hour,
    ) -> EveningPlan
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Optional, Tuple

from onkia.models import WaterTempPreference
from onkia.weather import PressureTrend, PressureInterpretation, WeatherResult


_DEFAULT_START = 17
_DEFAULT_END = 21


@dataclass
class PlanBlock:
    time_start: str
    time_end: str
    species: str
    location: str
    technique: str
    lures: List[str]
    depth: str
    evidence: List[str]


@dataclass
class EveningPlan:
    blocks: List[PlanBlock]
    conditions_summary: str
    data_sources: List[str]


_EVENING_SPECIES_ORDER = {
    "early": ["Sunfish", "Black Crappie", "Largemouth Bass"],
    "late": ["Largemouth Bass", "Northern Pike"],
    "dusk": ["Walleye", "Black Crappie"],
}

_EVENING_TIME_BLOCKS: Dict[str, Tuple[int, int]] = {
    "early": (17, 19),
    "late": (19, 20),
    "dusk": (20, 21),
}

_EVENING_BLOCK_LABELS: Dict[str, str] = {
    "early": "Early Evening",
    "late": "Late Evening",
    "dusk": "Dusk",
}

_TECHNIQUES: Dict[str, Dict[str, Dict]] = {
    "Largemouth Bass": {
        "cold": {
            "lures": ["Jig with paddle tail", "Drop shot with finesse worm"],
            "depth": "12-20 ft near structure",
        },
        "optimal": {
            "lures": ["Topwater frog/popper", "Spinnerbait", "Soft plastic Texas rig"],
            "depth": "Shallow flats, weed edges (4-10 ft)",
        },
        "warm": {
            "lures": ["Deep-diving crankbait", "Carolina rig", "Swimbait"],
            "depth": "Thermocline break (15-25 ft)",
        },
    },
    "Northern Pike": {
        "cold": {
            "lures": ["Jigging spoon", "Large blade bait"],
            "depth": "10-20 ft over weed flats",
        },
        "optimal": {
            "lures": ["Spinnerbait", "Inline spinner (Mepps)", "Sucker minnow on bobber"],
            "depth": "Weed edges 6-12 ft",
        },
        "warm": {
            "lures": ["Large glide bait", "Musky-style bucktail"],
            "depth": "Deep weed beds or cool tributaries",
        },
    },
    "Sunfish": {
        "cold": {
            "lures": ["Tiny jig (1/32 oz)", "Ice-fishing style teardrops"],
            "depth": "6-12 ft near structure",
        },
        "optimal": {
            "lures": ["Wax worm under bobber", "Beetle spin", "Small popper"],
            "depth": "Shallow weeds 2-6 ft",
        },
        "warm": {
            "lures": ["Night crawler pieces", "Small spinner"],
            "depth": "Slightly deeper 4-8 ft in shade",
        },
    },
    "Black Crappie": {
        "cold": {
            "lures": ["Tube jig 1/16 oz", "Marabou jig"],
            "depth": "15-25 ft suspended",
        },
        "optimal": {
            "lures": ["Small jig under bobber", "Minnow on hook", "Crappie tube"],
            "depth": "Brush piles / timber 8-15 ft",
        },
        "warm": {
            "lures": ["Tiny swimbaits", "Inline spinner 1/8 oz"],
            "depth": "Shaded docks / deep structure",
        },
    },
    "Walleye": {
        "cold": {
            "lures": ["Jigging rap / jigging spoon", "Live sucker minnow"],
            "depth": "20-35 ft on hard bottom",
        },
        "optimal": {
            "lures": ["Lindy rig with night crawler", "Jig + minnow", "Crankbait troll"],
            "depth": "Weed edges and rock bars 8-18 ft",
        },
        "warm": {
            "lures": ["Deep-troll crankbait", "Live minnow on slip sinker"],
            "depth": "Below thermocline 20-30 ft",
        },
    },
}

_WIND_POSITION: Dict[str, str] = {
    "N": "South shore (windward)",
    "NNE": "South-southwest shore (windward)",
    "NE": "Southwest shore (windward)",
    "ENE": "West-southwest shore (windward)",
    "E": "West shore (windward)",
    "ESE": "Northwest shore (windward)",
    "SE": "Northwest shore (windward)",
    "SSE": "North-northwest shore (windward)",
    "S": "North shore (windward)",
    "SSW": "North-northeast shore (windward)",
    "SW": "Northeast shore (windward)",
    "WSW": "East-northeast shore (windward)",
    "W": "East shore (windward)",
    "WNW": "Southeast shore (windward)",
    "NW": "Southeast shore (windward)",
    "NNW": "South-southeast shore (windward)",
}

_PRESSURE_ACTIVITY_SCORE: Dict[PressureTrend, float] = {
    PressureTrend.FALLING: 1.5,
    PressureTrend.RISING_AFTER_DROP: 1.3,
    PressureTrend.STABLE: 1.0,
    PressureTrend.RISING: 0.7,
    PressureTrend.HIGH_STEADY: 0.5,
}

_SPECIES_CODE_MAP: Dict[str, str] = {
    "WAE": "Walleye",
    "NOP": "Northern Pike",
    "LMB": "Largemouth Bass",
    "BLC": "Black Crappie",
    "BLG": "Bluegill",
    "GSF": "Green Sunfish",
    "YEP": "Yellow Perch",
    "RKB": "Rock Bass",
    "PMK": "Pumpkinseed",
    "HSF": "Hybrid Sunfish",
}

_SPECIES_TO_CODES: Dict[str, List[str]] = {
    "Walleye": ["WAE"],
    "Northern Pike": ["NOP"],
    "Largemouth Bass": ["LMB"],
    "Black Crappie": ["BLC"],
    "Sunfish": ["BLG", "GSF", "HSF", "PMK"],
}

_DUSK_BONUS_SPECIES = {"Walleye", "Black Crappie"}
_EARLY_BONUS_SPECIES = {"Sunfish", "Black Crappie", "Largemouth Bass"}
_LATE_BONUS_SPECIES = {"Largemouth Bass", "Northern Pike"}

_WINDY_THRESHOLD_MPH = 10.0
_CALM_THRESHOLD_MPH = 5.0


def _wind_label(deg: float) -> str:
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


def _temp_condition(temp_f: float, pref: WaterTempPreference) -> str:
    lower = pref.lower_avoidance
    upper = pref.upper_avoidance
    optimum_str = pref.optimum
    try:
        if "-" in optimum_str:
            opt_low, opt_high = (float(x) for x in optimum_str.split("-"))
        else:
            opt_low = opt_high = float(optimum_str)
    except (ValueError, AttributeError):
        return "unknown"
    if lower is not None and temp_f < lower:
        return "cold"
    if upper is not None and temp_f > upper:
        return "warm"
    if opt_low <= temp_f <= opt_high:
        return "optimal"
    if temp_f < opt_low:
        return "cold"
    return "warm"


def _format_hour(h: int) -> str:
    if h == 0 or h == 24:
        return "12:00 AM"
    if h < 12:
        return f"{h}:00 AM" if h > 0 else "12:00 AM"
    if h == 12:
        return "12:00 PM"
    return f"{h - 12}:00 PM"


def _format_hour30(h: int, half: bool = False) -> str:
    base = h if h <= 12 else h - 12
    if h == 0 or h == 24:
        base = 12
    suffix = "AM" if h < 12 else "PM"
    mins = ":30" if half else ":00"
    return f"{base}{mins} {suffix}"


def _compute_species_scores(
    water_temp_f: float,
    prefs: List[WaterTempPreference],
    pressure_trend: Optional[PressureTrend],
    wind_speed: Optional[float],
    block_key: str,
    cpue_by_species: Dict[str, Optional[float]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    pressure_mult = _PRESSURE_ACTIVITY_SCORE.get(pressure_trend, 1.0) if pressure_trend else 1.0

    for pref in prefs:
        if pref.species not in _EVENING_SPECIES_ORDER.get(block_key, []):
            continue
        condition = _temp_condition(water_temp_f, pref)
        temp_score = {"optimal": 3.0, "cold": 1.0, "warm": 1.5, "unknown": 0.5}.get(condition, 0.5)
        cpue = cpue_by_species.get(pref.species)
        cpue_score = 0.0
        if cpue is not None:
            if cpue > 15:
                cpue_score = 2.0
            elif cpue > 8:
                cpue_score = 1.5
            elif cpue > 3:
                cpue_score = 1.0
            elif cpue > 0:
                cpue_score = 0.5

        block_bonus = 0.0
        if block_key == "dusk" and pref.species in _DUSK_BONUS_SPECIES:
            block_bonus = 1.5
        elif block_key == "early" and pref.species in _EARLY_BONUS_SPECIES:
            block_bonus = 1.0
        elif block_key == "late" and pref.species in _LATE_BONUS_SPECIES:
            block_bonus = 1.0

        wind_bonus = 0.0
        if wind_speed is not None:
            if wind_speed >= _WINDY_THRESHOLD_MPH and pref.species in (
                "Northern Pike", "Largemouth Bass", "Walleye",
            ):
                wind_bonus = 0.5
            elif wind_speed < _CALM_THRESHOLD_MPH and pref.species in (
                "Sunfish", "Black Crappie",
            ):
                wind_bonus = 0.3

        scores[pref.species] = (temp_score + cpue_score + block_bonus + wind_bonus) * pressure_mult

    return scores


def _build_location_guidance(
    species: str,
    condition: str,
    wind_dir_deg: Optional[float],
    wind_speed: Optional[float],
) -> Tuple[str, List[str]]:
    base_loc = _TECHNIQUES.get(species, {}).get(condition, {}).get("depth", "")
    evidence = []
    wind_note = ""

    if wind_dir_deg is not None and wind_speed is not None and wind_speed >= _WINDY_THRESHOLD_MPH:
        wlabel = _wind_label(wind_dir_deg)
        windward = _WIND_POSITION.get(wlabel, "Windward shore")
        wind_note = f"Focus on {windward} — wind concentrates baitfish."
        evidence.append(f"Wind from {wlabel} at {wind_speed:.0f} mph pushes baitfish to windward shore")

    parts = [base_loc]
    if wind_note:
        parts.append(wind_note)

    if species == "Walleye" and condition in ("optimal", "cold"):
        parts.append("Rock bars and points as light fades.")
        evidence.append("Walleye are low-light specialists — move to points/structure at dusk")
    elif species == "Black Crappie" and condition == "optimal":
        parts.append("Brush piles and suspended timber in the basin.")
        evidence.append("Crappie suspend near structure during evening feed")
    elif species == "Largemouth Bass" and condition == "optimal":
        parts.append("Weed edges and shallow flats near cover.")
        evidence.append("Bass actively feed during low-light transitions")
    elif species == "Northern Pike" and condition in ("optimal", "cold"):
        parts.append("Weed edges and ambush points.")
        evidence.append("Pike station at weed edges waiting for baitfish")
    elif species == "Sunfish" and condition in ("optimal", "warm"):
        parts.append("Shallow weed beds and shoreline cover.")
        evidence.append("Sunfish hold near shallow structure in evening")

    return " | ".join(parts) if len(parts) > 1 else parts[0], evidence


def _select_block_species(
    block_key: str,
    scores: Dict[str, float],
) -> Optional[str]:
    if not scores:
        return None
    sorted_species = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_species[0][0]


def _build_technique_note(
    species: str,
    condition: str,
    pressure_trend: Optional[PressureTrend],
) -> Tuple[str, List[str]]:
    tech = _TECHNIQUES.get(species, {}).get(condition)
    if not tech:
        return "Standard presentations for species.", []

    lures = tech["lures"]
    technique = f"Use {', '.join(lures[:2])}"
    evidence = []

    if pressure_trend == PressureTrend.FALLING:
        technique += " — aggressive retrieves, fish are actively feeding before the front."
        evidence.append("Falling pressure triggers aggressive feeding (pressure trend: falling)")
    elif pressure_trend == PressureTrend.RISING:
        technique += " — slow down presentations, fish are less active."
        evidence.append("Rising pressure reduces activity — use slower presentations (pressure trend: rising)")
    elif pressure_trend == PressureTrend.RISING_AFTER_DROP:
        technique += " — moderate pace, fish resuming feeding after front."
        evidence.append("Fish resume feeding as pressure stabilizes after a drop (pressure trend: rising after drop)")
    elif pressure_trend == PressureTrend.HIGH_STEADY:
        technique += " — finesse approach, focus on deep structure."
        evidence.append("High steady pressure slows fishing — focus deep (pressure trend: high steady)")
    else:
        if condition == "optimal":
            technique += " — standard presentations in optimal temp range."
            evidence.append(f"Water temp in optimal range for {species}")
        elif condition == "cold":
            technique += " — slow presentation, fish are sluggish."
            evidence.append(f"Water temp below optimal for {species} — slow presentations")
        elif condition == "warm":
            technique += " — fish deeper, slower presentation."
            evidence.append(f"Water temp above optimal for {species} — fish deeper")

    return technique, evidence


def _compute_cpue_map(
    survey_records: Optional[List[Dict]],
) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    if not survey_records:
        for sp in _SPECIES_TO_CODES:
            result[sp] = None
        return result

    species_cpue: Dict[str, List[float]] = {}
    for rec in survey_records:
        code = rec.get("species_code", "")
        species_name = _SPECIES_CODE_MAP.get(code, code)
        cpue_val = rec.get("cpue")
        if cpue_val is not None:
            try:
                cpue_f = float(cpue_val)
            except (ValueError, TypeError):
                continue
            species_cpue.setdefault(species_name, []).append(cpue_f)

    for display_name, codes in _SPECIES_TO_CODES.items():
        all_cpues = []
        for code in codes:
            name = _SPECIES_CODE_MAP.get(code, code)
            if name in species_cpue:
                all_cpues.extend(species_cpue[name])
        if all_cpues:
            recent = sorted(species_cpue.get(display_name, all_cpues), reverse=True)[:3]
            result[display_name] = sum(recent) / len(recent) if recent else None
        else:
            result[display_name] = None

    return result


def _build_conditions_summary(
    water_temp_f: float,
    pressure_trend: Optional[PressureTrend],
    pressure_interp: Optional[PressureInterpretation],
    wind_speed: Optional[float],
    wind_dir_deg: Optional[float],
    cloud_cover: Optional[float],
) -> str:
    parts = [f"Water temp: ~{water_temp_f:.0f}°F"]
    if pressure_interp:
        parts.append(f"Pressure: {pressure_interp.label}")
    if pressure_trend:
        parts.append(f"Trend: {pressure_trend.value}")
    if wind_speed is not None:
        wdir = _wind_label(wind_dir_deg) if wind_dir_deg is not None else ""
        parts.append(f"Wind: {wind_speed:.0f} mph {wdir}".strip())
    if cloud_cover is not None:
        cloud_label = "Clear" if cloud_cover < 30 else "Partly cloudy" if cloud_cover < 60 else "Overcast"
        parts.append(f"Sky: {cloud_label} ({cloud_cover:.0f}%)")
    return " | ".join(parts)


def generate_evening_plan(
    weather: WeatherResult,
    water_temp_f: float,
    target_species_prefs: List[WaterTempPreference],
    survey_records: Optional[List[Dict]] = None,
    wind_dir_deg: Optional[float] = None,
    start_hour: int = _DEFAULT_START,
    end_hour: int = _DEFAULT_END,
) -> EveningPlan:
    """Generate a structured evening fishing plan.

    Args:
        weather: Weather result from Open-Meteo integration.
        water_temp_f: Estimated water temperature in Fahrenheit.
        target_species_prefs: Water temp preferences for target species.
        survey_records: DNR survey records for the lake (list of dicts with
            species_code, cpue, survey_date keys).
        wind_dir_deg: Wind direction in degrees (overrides weather if given).
        start_hour: Start of fishing window (24h, default 17).
        end_hour: End of fishing window (24h, default 21).

    Returns:
        EveningPlan with time-sequenced blocks, conditions summary, and data sources.
    """
    w_dir = wind_dir_deg if wind_dir_deg is not None else weather.wind_direction_deg
    w_speed = weather.wind_speed_mph
    p_trend = weather.pressure_trend

    cpue_map = _compute_cpue_map(survey_records)

    blocks: List[PlanBlock] = []
    all_data_sources: List[str] = []
    survey_year = None
    if survey_records:
        for rec in survey_records:
            sd = rec.get("survey_date", "")
            if sd:
                survey_year = sd[:4] if isinstance(sd, str) else str(sd)[:4]
                break

    if weather.fallback_used:
        all_data_sources.append("Seasonal averages (Open-Meteo unavailable)")
    else:
        all_data_sources.append(f"Open-Meteo {weather.source}")

    if survey_year:
        all_data_sources.append(f"DNR lake survey ({survey_year})")

    all_data_sources.append("MN DNR water temperature preferences")

    for block_key, (bh_start, bh_end) in _EVENING_TIME_BLOCKS.items():
        if bh_start < start_hour or bh_end > end_hour:
            continue

        scores = _compute_species_scores(
            water_temp_f,
            target_species_prefs,
            p_trend,
            w_speed,
            block_key,
            cpue_map,
        )

        top_species = _select_block_species(block_key, scores)
        if top_species is None:
            for sp in _EVENING_SPECIES_ORDER.get(block_key, []):
                if any(p.species == sp for p in target_species_prefs):
                    top_species = sp
                    break
        if top_species is None:
            continue

        pref = next((p for p in target_species_prefs if p.species == top_species), None)
        condition = _temp_condition(water_temp_f, pref) if pref else "unknown"

        location, loc_evidence = _build_location_guidance(
            top_species, condition, w_dir, w_speed,
        )

        technique, tech_evidence = _build_technique_note(
            top_species, condition, p_trend,
        )

        tech_data = _TECHNIQUES.get(top_species, {}).get(condition, {})
        lures = tech_data.get("lures", [])
        depth = tech_data.get("depth", "")

        evidence = []
        evidence.extend(loc_evidence)
        evidence.extend(tech_evidence)

        cpue_val = cpue_map.get(top_species)
        if cpue_val is not None:
            evidence.append(
                f"DNR survey CPUE for {top_species}: {cpue_val:.1f} (historical catch rate)"
            )

        if pref:
            evidence.append(
                f"Water temp {water_temp_f:.0f}°F vs {top_species} optimal {pref.optimum}°F — {condition} conditions"
            )

        block_label = _EVENING_BLOCK_LABELS.get(block_key, block_key.title())
        time_start_str = _format_hour30(bh_start, half=True)
        time_end_str = _format_hour30(bh_end, half=True)

        blocks.append(PlanBlock(
            time_start=f"{time_start_str}",
            time_end=f"{time_end_str}",
            species=top_species,
            location=location,
            technique=technique,
            lures=lures,
            depth=depth,
            evidence=evidence,
        ))

    conditions_summary = _build_conditions_summary(
        water_temp_f,
        p_trend,
        weather.pressure_interpretation,
        w_speed,
        w_dir,
        weather.cloud_cover_pct,
    )

    return EveningPlan(
        blocks=blocks,
        conditions_summary=conditions_summary,
        data_sources=all_data_sources,
    )
