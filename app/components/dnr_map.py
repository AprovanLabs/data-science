"""Reusable Folium map component for Wright County lake visualization.

Supports depth contour overlays and species-specific preferred zone highlighting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium

logger = logging.getLogger(__name__)

WRIGHT_COUNTY_CENTER = (45.17, -94.05)
WRIGHT_COUNTY_ZOOM = 10

WRIGHT_COUNTY_LAKES: Dict[str, Tuple[float, float, str]] = {
    "Clearwater": (45.3052, -94.1184, "86025200"),
    "Lake Pulaski": (45.2703, -94.0398, "86099900"),
    "Lake Sylvia": (45.2853, -93.9960, "86099700"),
    "Pelican Lake": (45.3193, -93.9085, "86099600"),
    "Maple Lake": (45.2243, -94.0062, "86056300"),
    "Howard Lake": (45.0618, -94.0740, "86045400"),
    "Lake Montrose": (45.0698, -94.0169, "86063600"),
    "Lake Andrew": (45.1437, -94.2478, "86003800"),
    "Buffalo Lake": (45.1918, -94.2354, "86012000"),
    "Bass Lake": (45.2531, -94.1152, "86009200"),
    "South Center Lake": (45.3804, -93.9371, "86111700"),
    "Lake Francis": (45.1312, -93.9812, "86040700"),
    "Lake Charlotte": (45.1956, -93.9374, "86020400"),
    "Twin Lake": (45.3315, -94.1987, "86120900"),
    "Lake Ida": (45.0893, -94.2175, "86047300"),
}

_BATHYMETRY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bathymetry"

_CONTOUR_STYLE_BASE = {
    "weight": 1.5,
    "opacity": 0.7,
}

_SPECIES_ZONE_STYLE = {
    "weight": 2.5,
    "opacity": 0.85,
    "dashArray": "6 4",
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _depth_to_color(depth_ft: float) -> str:
    step = 5
    band = int(depth_ft // step) * step
    color_map = {
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
    return color_map.get(band, "#001a33")


def load_contours_geojson(lake_name: str, bathymetry_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    bdir = bathymetry_dir or _BATHYMETRY_DIR
    slug = _slug(lake_name)
    path = bdir / f"{slug}_contours.geojson"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load bathymetry for %s", lake_name)
        return None


def add_depth_contours(
    m: folium.Map,
    lake_name: str,
    bathymetry_dir: Optional[Path] = None,
    show: bool = True,
) -> Optional[folium.FeatureGroup]:
    geojson_data = load_contours_geojson(lake_name, bathymetry_dir)
    if geojson_data is None:
        return None

    fg = folium.FeatureGroup(name="Depth Contours", show=show)

    folium.GeoJson(
        geojson_data,
        style_function=lambda x: {
            **_CONTOUR_STYLE_BASE,
            "color": _depth_to_color(x["properties"].get("depth_ft", 0)),
        },
        tooltip=folium.GeoJsonTooltip(fields=["depth_ft"], aliases=["Depth (ft):"]),
    ).add_to(fg)

    fg.add_to(m)
    return fg


def add_species_zone_overlay(
    m: folium.Map,
    lake_name: str,
    species_name: str,
    depth_range: Tuple[float, float],
    zone_color: str,
    bathymetry_dir: Optional[Path] = None,
    show: bool = True,
) -> Optional[folium.FeatureGroup]:
    geojson_data = load_contours_geojson(lake_name, bathymetry_dir)
    if geojson_data is None:
        return None

    matching_features = []
    for feature in geojson_data.get("features", []):
        depth = feature.get("properties", {}).get("depth_ft", 0)
        if depth_range[0] <= depth <= depth_range[1]:
            matching_features.append(feature)

    if not matching_features:
        return None

    fg = folium.FeatureGroup(name=f"{species_name} Zone ({depth_range[0]:.0f}-{depth_range[1]:.0f} ft)", show=show)

    collection = {
        "type": "FeatureCollection",
        "features": matching_features,
    }

    style = {**_SPECIES_ZONE_STYLE, "color": zone_color, "fillColor": zone_color, "fillOpacity": 0.25}

    folium.GeoJson(
        collection,
        style_function=lambda x: style,
        tooltip=f"{species_name}: {depth_range[0]:.0f}-{depth_range[1]:.0f} ft preferred",
    ).add_to(fg)

    fg.add_to(m)
    return fg


def build_lake_map(
    selected_lake: Optional[str] = None,
    search_results: Optional[List[Dict]] = None,
    center: Tuple[float, float] = WRIGHT_COUNTY_CENTER,
    zoom: int = WRIGHT_COUNTY_ZOOM,
    show_bathymetry: bool = False,
    species_zones: Optional[List[Dict[str, Any]]] = None,
    bathymetry_dir: Optional[Path] = None,
) -> folium.Map:
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )

    for name, (lat, lon, dow) in WRIGHT_COUNTY_LAKES.items():
        is_selected = name == selected_lake
        folium.CircleMarker(
            location=(lat, lon),
            radius=10 if is_selected else 7,
            color="#e63946" if is_selected else "#1d3557",
            fill=True,
            fill_color="#e63946" if is_selected else "#457b9d",
            fill_opacity=0.9 if is_selected else 0.7,
            tooltip=f"{name} (DOW: {dow})",
            popup=folium.Popup(
                f"<b>{name}</b><br>DOW: {dow}<br>Wright County, MN",
                max_width=200,
            ),
        ).add_to(m)

    if search_results:
        for lake in search_results:
            lake_name = lake.get("name", "")
            point = lake.get("point", {})
            coords = point.get("epsg:4326", [])
            if len(coords) >= 2:
                lon_val, lat_val = float(coords[0]), float(coords[1])
                lake_id = lake.get("id", "")
                if lake_name not in WRIGHT_COUNTY_LAKES:
                    folium.Marker(
                        location=(lat_val, lon_val),
                        tooltip=f"{lake_name} (DOW: {lake_id})",
                        popup=folium.Popup(
                            f"<b>{lake_name}</b><br>DOW: {lake_id}<br>Wright County, MN",
                            max_width=200,
                        ),
                        icon=folium.Icon(color="green", icon="info-sign"),
                    ).add_to(m)

    if selected_lake and show_bathymetry:
        add_depth_contours(m, selected_lake, bathymetry_dir, show=True)

    if selected_lake and species_zones:
        for zone in species_zones:
            add_species_zone_overlay(
                m,
                selected_lake,
                zone["species"],
                zone["depth_range"],
                zone["color"],
                bathymetry_dir,
                show=True,
            )

    if show_bathymetry or species_zones:
        folium.LayerControl(collapsed=False).add_to(m)

    return m
