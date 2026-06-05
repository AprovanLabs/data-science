"""Bathymetry overlay support — depth contours and species depth zones.

Provides:
    - Synthetic depth-contour GeoJSON generated from lake morphology data.
    - Per-species preferred depth zones adjusted for time of day and temperature.
    - Helpers for loading pre-generated GeoJSON and building Folium map layers.

GeoJSON files are stored in ``data/bathymetry/<lake_name>.geojson`` and generated
by ``scripts/generate_bathymetry.py``.  This module can also generate them
on-the-fly from morphology parameters when a file is not present.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Species depth preferences (feet)
# Each key maps to a dict of time-of-day windows → (min_depth, max_depth).
# "default" is used when the requested time-of-day key is absent.
# ---------------------------------------------------------------------------

SPECIES_DEPTH_PREFS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Largemouth Bass": {
        "default": (4.0, 15.0),
        "dawn":    (2.0,  8.0),
        "morning": (4.0, 12.0),
        "midday":  (10.0, 20.0),
        "evening": (3.0, 10.0),
        "night":   (2.0,  8.0),
    },
    "Northern Pike": {
        "default": (3.0, 12.0),
        "dawn":    (2.0,  8.0),
        "morning": (4.0, 12.0),
        "midday":  (8.0, 18.0),
        "evening": (3.0, 10.0),
        "night":   (4.0, 14.0),
    },
    "Sunfish": {
        "default": (2.0,  8.0),
        "dawn":    (1.0,  5.0),
        "morning": (2.0,  6.0),
        "midday":  (4.0, 10.0),
        "evening": (2.0,  6.0),
        "night":   (3.0,  8.0),
    },
    "Black Crappie": {
        "default": (6.0, 18.0),
        "dawn":    (4.0, 12.0),
        "morning": (6.0, 15.0),
        "midday":  (12.0, 22.0),
        "evening": (6.0, 15.0),
        "night":   (4.0, 14.0),
    },
    "Walleye": {
        "default": (8.0, 22.0),
        "dawn":    (4.0, 14.0),
        "morning": (8.0, 18.0),
        "midday":  (15.0, 30.0),
        "evening": (6.0, 18.0),
        "night":   (4.0, 15.0),
    },
}

# Temperature condition depth offsets (feet)
_WARM_DEPTH_OFFSET = 4.0
_COLD_DEPTH_OFFSET = -2.0

# ---------------------------------------------------------------------------
# Depth colour ramp: light blue (shallow) → dark blue (deep)
# ---------------------------------------------------------------------------

_DEPTH_RAMP: List[Tuple[float, str]] = [
    (0.00, "#c8e8f8"),
    (0.20, "#90cdf4"),
    (0.40, "#63b3ed"),
    (0.60, "#4299e1"),
    (0.80, "#2b6cb0"),
    (1.00, "#1a365d"),
]


def depth_contour_color(depth_ft: float, max_depth_ft: float) -> str:
    """Return a hex colour for *depth_ft* along the light-to-dark ramp."""
    if max_depth_ft <= 0:
        return _DEPTH_RAMP[0][1]
    ratio = min(max(depth_ft / max_depth_ft, 0.0), 1.0)
    for i in range(len(_DEPTH_RAMP) - 1):
        lo_r, lo_c = _DEPTH_RAMP[i]
        hi_r, hi_c = _DEPTH_RAMP[i + 1]
        if lo_r <= ratio <= hi_r:
            t = (ratio - lo_r) / (hi_r - lo_r) if hi_r > lo_r else 0.0
            return _blend_hex(lo_c, hi_c, t)
    return _DEPTH_RAMP[-1][1]


def _blend_hex(c1: str, c2: str, t: float) -> str:
    def parse(c: str) -> Tuple[int, int, int]:
        h = c.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r1, g1, b1 = parse(c1)
    r2, g2, b2 = parse(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )


# ---------------------------------------------------------------------------
# Synthetic contour generation (pure Python — no shapely required)
# ---------------------------------------------------------------------------

def _ellipse_coords(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    aspect: float = 1.3,
    rotation_deg: float = 25.0,
    n: int = 64,
) -> List[List[float]]:
    """Return closed GeoJSON [lon, lat] ring for an ellipse."""
    lat_m = 1.0 / 111_000.0
    lon_m = 1.0 / (111_000.0 * math.cos(math.radians(center_lat)))
    rot = math.radians(rotation_deg)
    pts: List[List[float]] = []
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        x = radius_m * math.cos(a)
        y = radius_m * aspect * math.sin(a)
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        pts.append([round(center_lon + xr * lon_m, 7), round(center_lat + yr * lat_m, 7)])
    return pts


def generate_depth_contours_geojson(
    lake_name: str,
    lat: float,
    lon: float,
    max_depth_ft: float,
    area_acres: float,
    num_contours: int = 8,
    aspect: float = 1.3,
    rotation_deg: float = 25.0,
) -> dict:
    """Generate a synthetic GeoJSON FeatureCollection of depth contour polygons.

    Contours are concentric ellipses using the same hypsometric curve used
    in the existing depth-profile chart::

        fraction_area(depth) = 1 - (depth / max_depth) ** 0.6

    Features are ordered shallowest-first so they render correctly in Folium
    (deepest on top, each ring visually narrower).

    Args:
        lake_name: Name stored in each feature's properties.
        lat: Lake centre latitude (WGS84).
        lon: Lake centre longitude (WGS84).
        max_depth_ft: Maximum depth in feet.
        area_acres: Surface area in acres.
        num_contours: Number of depth steps to generate.
        aspect: Ellipse y/x ratio for a natural look.
        rotation_deg: Rotation of the major axis in degrees from north.

    Returns:
        GeoJSON ``FeatureCollection`` dict.
    """
    if max_depth_ft <= 0 or area_acres <= 0:
        return {"type": "FeatureCollection", "features": []}

    area_m2 = area_acres * 4_046.86
    r_surface = math.sqrt(area_m2 / math.pi)

    step = max_depth_ft / num_contours
    depths = [step * (i + 1) for i in range(num_contours)]

    features: List[dict] = []
    for depth_ft in depths:
        frac = 1.0 - (depth_ft / max_depth_ft) ** 0.6
        r = r_surface * math.sqrt(max(frac, 0.0))
        if r < 2:
            continue
        color = depth_contour_color(depth_ft, max_depth_ft)
        features.append({
            "type": "Feature",
            "properties": {
                "lake": lake_name,
                "depth_ft": round(depth_ft, 1),
                "color": color,
                "weight": 1,
                "fill_opacity": 0.5,
                "line_opacity": 0.8,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_ellipse_coords(lat, lon, r, aspect, rotation_deg)],
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# GeoJSON persistence
# ---------------------------------------------------------------------------

_BATHYMETRY_DIR = Path(__file__).resolve().parents[2] / "data" / "bathymetry"


def _geojson_path(lake_name: str) -> Path:
    safe = lake_name.lower().replace(" ", "_")
    return _BATHYMETRY_DIR / f"{safe}.geojson"


def save_bathymetry_geojson(lake_name: str, geojson: dict) -> Path:
    """Write *geojson* to ``data/bathymetry/<lake_name>.geojson``."""
    _BATHYMETRY_DIR.mkdir(parents=True, exist_ok=True)
    path = _geojson_path(lake_name)
    path.write_text(json.dumps(geojson, indent=2))
    return path


def load_bathymetry_geojson(lake_name: str) -> Optional[dict]:
    """Load the stored GeoJSON for *lake_name*, or ``None`` if absent."""
    path = _geojson_path(lake_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Species zone depth helpers
# ---------------------------------------------------------------------------

def get_species_zone_depths(
    species: str,
    time_of_day: str,
    temp_condition: str,
    max_depth_ft: float,
) -> Optional[Tuple[float, float]]:
    """Return ``(min_depth_ft, max_depth_ft)`` for a species under current conditions.

    Applies a vertical offset for warm (fish go deeper) or cold (fish stay
    shallower) temperature conditions, then clamps to the lake's max depth.

    Args:
        species: Must match a key in :data:`SPECIES_DEPTH_PREFS`.
        time_of_day: ``dawn`` | ``morning`` | ``midday`` | ``evening`` | ``night``.
        temp_condition: ``"cold"`` | ``"optimal"`` | ``"warm"``.
        max_depth_ft: Lake max depth used for clamping.

    Returns:
        ``(min, max)`` tuple in feet, or ``None`` if species is not recognised.
    """
    prefs = SPECIES_DEPTH_PREFS.get(species)
    if prefs is None:
        return None
    min_d, max_d = prefs.get(time_of_day, prefs["default"])

    offset = 0.0
    if temp_condition == "warm":
        offset = _WARM_DEPTH_OFFSET
    elif temp_condition == "cold":
        offset = _COLD_DEPTH_OFFSET

    min_d = max(0.0, min_d + offset)
    max_d = min(max_depth_ft, max_d + offset)
    if min_d >= max_d:
        max_d = min(min_d + 3.0, max_depth_ft)
    return (round(min_d, 1), round(max_d, 1))


def get_wind_shore_note(wind_direction_deg: Optional[float]) -> Optional[str]:
    """Return a human-readable note about the windward shore concentration.

    Args:
        wind_direction_deg: Wind direction in degrees (0 = N, 90 = E, …)
            or ``None`` when unavailable.

    Returns:
        Short advisory string, or ``None``.
    """
    if wind_direction_deg is None:
        return None
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(wind_direction_deg / 45) % 8
    windward = dirs[idx]
    leeward = dirs[(idx + 4) % 8]
    return (
        f"Wind from {windward} — baitfish concentrate on the "
        f"{leeward} (leeward) shore. Focus structure there."
    )
