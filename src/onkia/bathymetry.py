"""Bathymetry data loading and species depth zone computation.

Loads pre-processed GeoJSON contour files from ``data/bathymetry/`` and
provides helpers for computing species-specific preferred depth zones
based on water temperature, time of day, and wind direction.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from onkia.models import WaterTempPreference

logger = logging.getLogger(__name__)

_BATHYMETRY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bathymetry"

_DEPTH_CONTOUR_COLORS: Dict[int, str] = {
    0: "#b3d9ff",
    5: "#80caff",
    10: "#4db8ff",
    15: "#1aa3ff",
    20: "#008ae6",
    25: "#006bb3",
    30: "#004d80",
    35: "#003566",
    40: "#00264d",
    50: "#001a33",
    60: "#000f1f",
    70: "#000a14",
    80: "#00050a",
}

_SPECIES_DEPTH_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Largemouth Bass": {
        "cold": (12.0, 20.0),
        "optimal": (4.0, 10.0),
        "warm": (15.0, 25.0),
    },
    "Northern Pike": {
        "cold": (10.0, 20.0),
        "optimal": (6.0, 12.0),
        "warm": (12.0, 20.0),
    },
    "Walleye": {
        "cold": (20.0, 35.0),
        "optimal": (8.0, 18.0),
        "warm": (20.0, 30.0),
    },
    "Black Crappie": {
        "cold": (15.0, 25.0),
        "optimal": (8.0, 15.0),
        "warm": (4.0, 12.0),
    },
    "Sunfish": {
        "cold": (6.0, 12.0),
        "optimal": (2.0, 6.0),
        "warm": (4.0, 8.0),
    },
}

_TIME_DEPTH_SHIFT: Dict[str, float] = {
    "dawn": -2.0,
    "morning": -1.0,
    "midday": 2.0,
    "evening": -1.5,
    "night": 1.0,
}


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
        return "optimal"
    if lower is not None and temp_f < lower:
        return "cold"
    if upper is not None and temp_f > upper:
        return "warm"
    if opt_low <= temp_f <= opt_high:
        return "optimal"
    if temp_f < opt_low:
        return "cold"
    return "warm"


def contour_color(depth_ft: float) -> str:
    step = 5
    band = int(depth_ft // step) * step
    return _DEPTH_CONTOUR_COLORS.get(band, "#001a33")


def load_contours(lake_name: str, bathymetry_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    bdir = bathymetry_dir or _BATHYMETRY_DIR
    slug = lake_name.lower().replace(" ", "_")
    geojson_path = bdir / f"{slug}_contours.geojson"
    if not geojson_path.exists():
        logger.debug("No bathymetry data for %s at %s", lake_name, geojson_path)
        return None
    try:
        with open(geojson_path, "r") as f:
            data = json.load(f)
        logger.info("Loaded %d contour features for %s", len(data.get("features", [])), lake_name)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load bathymetry for %s: %s", lake_name, exc)
        return None


def load_depth_profile(lake_name: str, bathymetry_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    bdir = bathymetry_dir or _BATHYMETRY_DIR
    slug = lake_name.lower().replace(" ", "_")
    profile_path = bdir / f"{slug}_profile.json"
    if not profile_path.exists():
        return None
    try:
        with open(profile_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


_SQ_METERS_PER_ACRE = 4046.8564224
_METERS_PER_DEGREE = 111320.0


def _ring_area_acres(ring: List[List[float]]) -> float:
    """Planar shoelace area of a lon/lat ring, converted to acres.

    Uses an equirectangular approximation scaled by cos(mean latitude),
    which is plenty accurate for lake-sized polygons.
    """
    if len(ring) < 3:
        return 0.0
    mean_lat = sum(pt[1] for pt in ring) / len(ring)
    # Shoelace formula in degrees^2.
    area_deg2 = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[i + 1][0], ring[i + 1][1]
        area_deg2 += x1 * y2 - x2 * y1
    area_deg2 = abs(area_deg2) / 2.0
    area_m2 = area_deg2 * (_METERS_PER_DEGREE ** 2) * math.cos(math.radians(mean_lat))
    return area_m2 / _SQ_METERS_PER_ACRE


def _feature_rings(geometry: Dict[str, Any]) -> List[List[List[float]]]:
    """Extract closed exterior rings from a contour feature geometry."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon" and coords:
        return [coords[0]]
    if gtype == "MultiPolygon":
        return [poly[0] for poly in coords if poly]
    if gtype == "LineString" and len(coords) >= 4 and coords[0] == coords[-1]:
        return [coords]
    if gtype == "MultiLineString":
        return [line for line in coords if len(line) >= 4 and line[0] == line[-1]]
    return []


def depth_area_profile(
    lake_name: str, bathymetry_dir: Optional[Path] = None
) -> List[Tuple[float, float]]:
    """Compute the enclosed water surface area (acres) at each contour depth.

    Returns a list of ``(depth_ft, acres)`` tuples sorted by depth, suitable
    for plotting a hypsographic (area vs. depth) curve. Area at each depth is
    the sum of all closed contour rings at that depth.
    """
    contours = load_contours(lake_name, bathymetry_dir)
    if not contours:
        return []
    areas: Dict[float, float] = {}
    for feature in contours.get("features", []):
        props = feature.get("properties", {})
        depth = props.get("depth_ft")
        if depth is None:
            continue
        geometry = feature.get("geometry") or {}
        total = sum(_ring_area_acres(ring) for ring in _feature_rings(geometry))
        if total > 0:
            areas[float(depth)] = areas.get(float(depth), 0.0) + total
    profile = sorted(areas.items())
    # Enforce a physically valid hypsographic curve: the area enclosed at a
    # shallower depth can never be less than at a deeper depth. Some shoreline
    # contours are unclosed LineStrings and contribute no area, so backfill
    # from the deeper contours.
    running_max = 0.0
    for i in range(len(profile) - 1, -1, -1):
        depth, acres = profile[i]
        running_max = max(running_max, acres)
        profile[i] = (depth, running_max)
    return profile


def species_depth_zone(
    species: str,
    water_temp_f: float,
    time_of_day: str,
    wind_direction: Optional[str] = None,
    water_temp_pref: Optional[WaterTempPreference] = None,
) -> Optional[Tuple[float, float]]:
    ranges = _SPECIES_DEPTH_RANGES.get(species)
    if not ranges:
        return None
    if water_temp_pref is not None:
        condition = _temp_condition(water_temp_f, water_temp_pref)
    else:
        condition = "optimal"
    depth_range = ranges.get(condition, ranges.get("optimal"))
    if depth_range is None:
        return None
    shift = _TIME_DEPTH_SHIFT.get(time_of_day, 0.0)
    adjusted = (max(0.0, depth_range[0] + shift), depth_range[1] + shift)
    if wind_direction is not None:
        pass
    return adjusted


def available_lakes(bathymetry_dir: Optional[Path] = None) -> List[str]:
    bdir = bathymetry_dir or _BATHYMETRY_DIR
    if not bdir.is_dir():
        return []
    lakes = set()
    for p in bdir.glob("*_contours.geojson"):
        slug = p.stem.replace("_contours", "")
        name = slug.replace("_", " ").title()
        lakes.add(name)
    return sorted(lakes)
