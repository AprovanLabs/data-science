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

# Verified against the MN DNR LakeFinder API (search.cgi, county=86).
# Coordinates are the DNR lake centroid points; DOW numbers are the real
# Division of Waters lake IDs used by survey/stocking/bathymetry services.
WRIGHT_COUNTY_LAKES: Dict[str, Tuple[float, float, str]] = {
    "Clearwater": (45.305151, -94.118377, "86025200"),
    "Lake Pulaski": (45.201091, -93.853719, "86005300"),
    "West Lake Sylvia": (45.248600, -94.213272, "86027900"),
    "East Lake Sylvia": (45.252140, -94.195182, "86028900"),
    "Pelican Lake": (45.232986, -93.761074, "86003100"),
    "Maple Lake": (45.233273, -93.965263, "86013401"),
    "Howard Lake": (45.072609, -94.069390, "86019900"),
    "Lake Charlotte": (45.151031, -93.747086, "86001100"),
    "Buffalo Lake": (45.162705, -93.893336, "86009000"),
    "Bass Lake": (45.321655, -94.102800, "86023400"),
    "Lake Ida": (45.303705, -93.904368, "86014600"),
    "Ramsey Lake": (45.210702, -93.996352, "86012000"),
    "Sugar Lake": (45.317764, -94.038057, "86023300"),
    "Waverly Lake": (45.075360, -93.972443, "86011400"),
    "Cedar Lake": (45.265732, -94.067116, "86022700"),
    "Granite Lake": (45.184752, -94.110140, "86021700"),
    "French Lake": (45.208502, -94.163034, "86027300"),
}

_BATHYMETRY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bathymetry"

# Contour geometries are closed polygons; render them as outline-only so
# overlapping depth rings stay visible instead of filling the lake solid.
_CONTOUR_STYLE_BASE = {
    "weight": 1.5,
    "opacity": 0.7,
    "fill": False,
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
    lake_coord_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
) -> folium.Map:
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )

    for name, (lat, lon, dow) in WRIGHT_COUNTY_LAKES.items():
        if lake_coord_overrides and name in lake_coord_overrides:
            lat, lon = lake_coord_overrides[name]
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
