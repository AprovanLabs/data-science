#!/usr/bin/env python3
"""Pre-process MN DNR bathymetry GIS data into GeoJSON for Folium.

Usage:
    pip install geopandas shapely fiona

    python scripts/prepare_bathymetry.py \\
        --gdb path/to/LakeBathymetry.gdb \\
        --lakes "Clearwater" "Pulaski" "Sylvia" \\
        --output-dir data/bathymetry

Or generate sample data for testing:

    python scripts/prepare_bathymetry.py --sample --output-dir data/bathymetry
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WRIGHT_COUNTY_DOW_MAP: Dict[str, str] = {
    "Clearwater": "86025200",
    "Lake Pulaski": "86099900",
    "Lake Sylvia": "86099700",
    "Pelican Lake": "86099600",
    "Maple Lake": "86056300",
    "Howard Lake": "86045400",
    "Lake Montrose": "86063600",
    "Lake Andrew": "86003800",
    "Buffalo Lake": "86012000",
    "Bass Lake": "86009200",
    "South Center Lake": "86111700",
    "Lake Francis": "86040700",
    "Lake Charlotte": "86020400",
    "Twin Lake": "86120900",
    "Lake Ida": "86047300",
}

CONTOUR_INTERVALS_FT = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


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

    for lake_name in lake_names:
        dow = WRIGHT_COUNTY_DOW_MAP.get(lake_name)
        if dow is None:
            logger.warning("No DOW number for %s, skipping", lake_name)
            continue

        lake_gdf = gdf[gdf["DOW_NUMBER"] == int(dow)] if "DOW_NUMBER" in gdf.columns else gpd.GeoDataFrame()
        if lake_gdf.empty and "dow_num" in gdf.columns:
            lake_gdf = gdf[gdf["dow_num"] == int(dow)]

        if lake_gdf.empty:
            logger.warning("No contours found for %s", lake_name)
            continue

        lake_gdf = lake_gdf.copy()
        lake_gdf["geometry"] = lake_gdf["geometry"].simplify(tolerance=tolerance)

        depth_col = None
        for col in ["DEPTH_FT", "depth_ft", "DEPTH", "Depth", "CONTOUR", "ELEV"]:
            if col in lake_gdf.columns:
                depth_col = col
                break

        features = []
        depth_values = []
        for _, row in lake_gdf.iterrows():
            depth_val = float(row[depth_col]) if depth_col else 0.0
            depth_values.append(depth_val)
            geom = row["geometry"]
            feature = {
                "type": "Feature",
                "properties": {"depth_ft": depth_val, "lake": lake_name, "dow": dow},
                "geometry": geom.__geo_interface__,
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}
        slug = _slug(lake_name)
        out_path = output_dir / f"{slug}_contours.geojson"
        with open(out_path, "w") as f:
            json.dump(geojson, f)
        logger.info("Wrote %d contours to %s", len(features), out_path)

        if depth_values:
            profile = {
                "lake": lake_name, "dow": dow,
                "max_depth": max(depth_values), "min_depth": min(depth_values),
                "contour_count": len(depth_values),
                "depths": sorted(set(int(d) for d in depth_values)),
            }
            profile_path = output_dir / f"{slug}_profile.json"
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2)


def generate_sample_data(output_dir: Path) -> None:
    try:
        from shapely.geometry import Point
        from shapely.affinity import scale
    except ImportError:
        logger.error("shapely is required for sample data generation")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    lake_centers = {
        "Clearwater": (45.3052, -94.1184),
        "Lake Pulaski": (45.2703, -94.0398),
        "Lake Sylvia": (45.2853, -93.9960),
        "Pelican Lake": (45.3193, -93.9085),
        "Maple Lake": (45.2243, -94.0062),
    }

    logger.info("Generating sample bathymetry data for %d lakes", len(lake_centers))

    for lake_name, (lat, lon) in lake_centers.items():
        features = []
        for depth in CONTOUR_INTERVALS_FT:
            radius_deg = 0.015 * (1.0 - depth / 100.0)
            if radius_deg < 0.002:
                break
            center = Point(lon, lat)
            contour = scale(center.buffer(radius_deg), xfact=1.3, yfact=0.9)
            features.append({
                "type": "Feature",
                "properties": {"depth_ft": float(depth), "lake": lake_name, "dow": WRIGHT_COUNTY_DOW_MAP.get(lake_name, "")},
                "geometry": contour.__geo_interface__,
            })

        geojson = {"type": "FeatureCollection", "features": features}
        slug = _slug(lake_name)
        out_path = output_dir / f"{slug}_contours.geojson"
        with open(out_path, "w") as f:
            json.dump(geojson, f)
        logger.info("Wrote %d sample contours to %s", len(features), out_path)

        profile = {
            "lake": lake_name, "dow": WRIGHT_COUNTY_DOW_MAP.get(lake_name, ""),
            "max_depth": max(f["properties"]["depth_ft"] for f in features),
            "min_depth": min(f["properties"]["depth_ft"] for f in features),
            "contour_count": len(features),
            "depths": sorted(set(int(f["properties"]["depth_ft"]) for f in features)),
        }
        profile_path = output_dir / f"{slug}_profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-process MN DNR bathymetry GIS data")
    parser.add_argument("--gdb", help="Path to LakeBathymetry geodatabase (.gdb)")
    parser.add_argument("--lakes", nargs="+", default=list(WRIGHT_COUNTY_DOW_MAP.keys()), help="Lake names to process")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bathymetry"), help="Output directory")
    parser.add_argument("--tolerance", type=float, default=0.00005, help="Geometry simplification tolerance")
    parser.add_argument("--sample", action="store_true", help="Generate sample data without real GIS input")

    args = parser.parse_args()

    if args.sample:
        generate_sample_data(args.output_dir)
    elif args.gdb:
        process_geodatabase(args.gdb, args.lakes, args.output_dir, args.tolerance)
    else:
        logger.error("Provide --gdb <path> or --sample")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
