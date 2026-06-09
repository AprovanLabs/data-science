#!/usr/bin/env python3
"""Prepare MN DNR lake bathymetry GeoJSON for Folium.

Default mode fetches real contour geometry from the MN Geospatial Commons
ArcGIS REST service (Lake Bathymetric Contours, layer 0 of
us_mn_state_dnr/water_lake_bathymetry) and writes one
``<slug>_contours.geojson`` + ``<slug>_profile.json`` pair per lake into
``data/bathymetry/``:

    python scripts/prepare_bathymetry.py --output-dir data/bathymetry

A local geodatabase export of the same dataset can be used instead:

    pip install geopandas shapely fiona
    python scripts/prepare_bathymetry.py \\
        --gdb path/to/LakeBathymetry.gdb \\
        --lakes "Clearwater" "Lake Pulaski" \\
        --output-dir data/bathymetry
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Verified DOW numbers from the MN DNR LakeFinder API (county=86).
WRIGHT_COUNTY_DOW_MAP: Dict[str, str] = {
    "Clearwater": "86025200",
    "Lake Pulaski": "86005300",
    "West Lake Sylvia": "86027900",
    "East Lake Sylvia": "86028900",
    "Pelican Lake": "86003100",
    "Maple Lake": "86013401",
    "Howard Lake": "86019900",
    "Lake Charlotte": "86001100",
    "Buffalo Lake": "86009000",
    "Bass Lake": "86023400",
    "Lake Ida": "86014600",
    "Ramsey Lake": "86012000",
    "Sugar Lake": "86023300",
    "Waverly Lake": "86011400",
    "Cedar Lake": "86022700",
    "Granite Lake": "86021700",
    "French Lake": "86027300",
}

CONTOUR_SERVICE_URL = (
    "https://enterprise.gisdata.mn.gov/aghost/rest/services/"
    "us_mn_state_dnr/water_lake_bathymetry/MapServer/0/query"
)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _round_coords(coords: Any, ndigits: int = 6) -> Any:
    if isinstance(coords, (int, float)):
        return round(coords, ndigits)
    return [_round_coords(c, ndigits) for c in coords]


def _line_to_feature_geometry(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert closed LineString rings to Polygons so overlays can be filled."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "LineString":
        if len(coords) >= 4 and coords[0] == coords[-1]:
            return {"type": "Polygon", "coordinates": [coords]}
        return geometry
    if gtype == "MultiLineString":
        if all(len(part) >= 4 and part[0] == part[-1] for part in coords):
            return {"type": "Polygon", "coordinates": coords}
        return geometry
    return geometry


def _keep_basin(dowlknum: str, lake_dow: str) -> bool:
    """Keep a contour feature for the requested lake.

    Contours are stored per sub-basin (e.g. Clearwater 86025200 is split
    into 86025201/86025202). A parent DOW (ending in "00") matches all of
    its basins; a specific basin DOW matches only itself.
    """
    if dowlknum == lake_dow:
        return True
    return lake_dow.endswith("00") and dowlknum[:6] == lake_dow[:6]


def fetch_lake_contours(lake_name: str, dow: str) -> List[Dict[str, Any]]:
    """Fetch real bathymetric contours for one lake from the DNR REST service."""
    import requests

    features: List[Dict[str, Any]] = []
    offset = 0
    while True:
        resp = requests.get(
            CONTOUR_SERVICE_URL,
            params={
                "where": f"dowlknum LIKE '{dow[:6]}%'",
                "outFields": "depth,abs_depth,lake_name,dowlknum",
                "outSR": "4326",
                "f": "geojson",
                "resultOffset": str(offset),
            },
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
        page = payload.get("features", [])
        features.extend(page)
        if not payload.get("properties", {}).get("exceededTransferLimit") and not payload.get(
            "exceededTransferLimit"
        ):
            break
        offset += len(page)

    out: List[Dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {})
        dowlknum = str(props.get("dowlknum", ""))
        if not _keep_basin(dowlknum, dow):
            continue
        geometry = _line_to_feature_geometry(feature.get("geometry", {}))
        geometry["coordinates"] = _round_coords(geometry["coordinates"])
        out.append(
            {
                "type": "Feature",
                "properties": {
                    "depth_ft": float(props.get("abs_depth", 0)),
                    "lake": lake_name,
                    "dow": dowlknum,
                },
                "geometry": geometry,
            }
        )
    return out


def _write_lake_files(lake_name: str, dow: str, features: List[Dict[str, Any]], output_dir: Path) -> None:
    slug = _slug(lake_name)
    geojson = {"type": "FeatureCollection", "features": features}
    out_path = output_dir / f"{slug}_contours.geojson"
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    logger.info("Wrote %d contours to %s", len(features), out_path)

    depth_values = [f["properties"]["depth_ft"] for f in features]
    if depth_values:
        profile = {
            "lake": lake_name,
            "dow": dow,
            "max_depth": max(depth_values),
            "min_depth": min(depth_values),
            "contour_count": len(depth_values),
            "depths": sorted(set(int(d) for d in depth_values)),
        }
        profile_path = output_dir / f"{slug}_profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)


def fetch_from_dnr(lake_names: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for lake_name in lake_names:
        dow = WRIGHT_COUNTY_DOW_MAP.get(lake_name)
        if dow is None:
            logger.warning("No DOW number for %s, skipping", lake_name)
            continue
        try:
            features = fetch_lake_contours(lake_name, dow)
        except Exception as exc:
            logger.error("Failed to fetch contours for %s: %s", lake_name, exc)
            continue
        if not features:
            logger.warning("No contours found for %s (DOW %s)", lake_name, dow)
            continue
        _write_lake_files(lake_name, dow, features, output_dir)


def process_geodatabase(
    gdb_path: str,
    lake_names: List[str],
    output_dir: Path,
    tolerance: float = 0.00005,
) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        logger.error("geopandas is required: pip install geopandas shapely fiona")
        sys.exit(1)

    logger.info("Opening geodatabase: %s", gdb_path)
    try:
        layers = gpd.io.file.fiona.listlayers(gdb_path)
    except Exception:
        layers = []
    logger.info("Available layers: %s", layers)

    contour_layer = None
    for candidate in ["contours", "LakeBathymetryContours", "lake_bathymetry_contours", "Contour_Lines"]:
        if candidate in layers:
            contour_layer = candidate
            break
    if contour_layer is None and layers:
        contour_layer = layers[0]

    if contour_layer is None:
        logger.error("No contour layer found in geodatabase")
        sys.exit(1)

    gdf = gpd.read_file(gdb_path, layer=contour_layer)
    logger.info("Loaded %d contour features", len(gdf))

    if gdf.crs is None:
        logger.warning("No CRS defined, assuming EPSG:26915 (UTM zone 15N)")
        gdf = gdf.set_crs("EPSG:26915")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    output_dir.mkdir(parents=True, exist_ok=True)

    dow_col = next((c for c in ["DOW_NUMBER", "dow_num", "dowlknum", "DOWLKNUM"] if c in gdf.columns), None)
    depth_col = next(
        (c for c in ["abs_depth", "DEPTH_FT", "depth_ft", "DEPTH", "Depth", "CONTOUR", "ELEV"] if c in gdf.columns),
        None,
    )

    for lake_name in lake_names:
        dow = WRIGHT_COUNTY_DOW_MAP.get(lake_name)
        if dow is None:
            logger.warning("No DOW number for %s, skipping", lake_name)
            continue

        if dow_col is None:
            logger.warning("No DOW column found in geodatabase; skipping %s", lake_name)
            continue
        mask = gdf[dow_col].astype(str).apply(lambda v: _keep_basin(v, dow))
        lake_gdf = gdf[mask]

        if lake_gdf.empty:
            logger.warning("No contours found for %s", lake_name)
            continue

        lake_gdf = lake_gdf.copy()
        lake_gdf["geometry"] = lake_gdf["geometry"].simplify(tolerance=tolerance)

        features = []
        for _, row in lake_gdf.iterrows():
            depth_val = abs(float(row[depth_col])) if depth_col else 0.0
            geom = _line_to_feature_geometry(row["geometry"].__geo_interface__)
            geom["coordinates"] = _round_coords(geom["coordinates"])
            features.append(
                {
                    "type": "Feature",
                    "properties": {"depth_ft": depth_val, "lake": lake_name, "dow": dow},
                    "geometry": geom,
                }
            )

        _write_lake_files(lake_name, dow, features, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MN DNR lake bathymetry GeoJSON")
    parser.add_argument("--gdb", help="Optional path to LakeBathymetry geodatabase (.gdb); default fetches from the DNR REST service")
    parser.add_argument("--lakes", nargs="+", default=list(WRIGHT_COUNTY_DOW_MAP.keys()), help="Lake names to process")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bathymetry"), help="Output directory")
    parser.add_argument("--tolerance", type=float, default=0.00005, help="Geometry simplification tolerance (gdb mode)")

    args = parser.parse_args()

    if args.gdb:
        process_geodatabase(args.gdb, args.lakes, args.output_dir, args.tolerance)
    else:
        fetch_from_dnr(args.lakes, args.output_dir)


if __name__ == "__main__":
    main()
