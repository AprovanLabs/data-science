"""Reusable Folium map component for Wright County lake visualization."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import folium

# Wright County, MN center coordinates
WRIGHT_COUNTY_CENTER = (45.17, -94.05)
WRIGHT_COUNTY_ZOOM = 10

# Curated list of major Wright County lakes: name -> (lat, lon, dow_number)
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


def build_lake_map(
    selected_lake: Optional[str] = None,
    search_results: Optional[List[Dict]] = None,
    center: Tuple[float, float] = WRIGHT_COUNTY_CENTER,
    zoom: int = WRIGHT_COUNTY_ZOOM,
) -> folium.Map:
    """Build a Folium map showing Wright County lakes.

    Args:
        selected_lake: Name of currently selected lake (highlighted in red).
        search_results: Additional lake results from DNR API search to overlay.
        center: Map center as (lat, lon).
        zoom: Initial zoom level.

    Returns:
        Configured folium.Map instance.
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )

    # Add curated lakes
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

    # Overlay any additional search results from DNR API
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

    return m
