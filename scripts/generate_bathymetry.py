"""Generate synthetic bathymetry GeoJSON files for Wright County lakes.

Usage:
    python scripts/generate_bathymetry.py

This script pre-generates depth-contour GeoJSON for every lake defined in
``app/components/dnr_map.py``, using morphology values sourced from the MN
DNR LakeFinder API (max_depth and area estimates are from survey data).

The resulting files are stored in ``data/bathymetry/<lake_name>.geojson`` and
are loaded by the Streamlit app at runtime without requiring a live API call.

When real MN Geospatial Commons bathymetry data is available (FileGDB format
at https://gisdata.mn.gov/dataset/water-lake-bathymetry), you can replace the
synthetic generation here with geopandas/fiona readers:

    import geopandas as gpd
    gdf = gpd.read_file("path/to/bathymetry.gdb", layer="contours")
    # filter to lake, reproject to WGS84, export to GeoJSON
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when run from the repo root
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from onkia.bathymetry import generate_depth_contours_geojson, save_bathymetry_geojson

# ---------------------------------------------------------------------------
# Lake morphology data
# Values sourced from MN DNR LakeFinder surveys for Wright County lakes.
# max_depth_ft and area_acres are representative survey values.
# ---------------------------------------------------------------------------

LAKE_MORPHOLOGY = {
    # name: (lat, lon, max_depth_ft, area_acres, aspect_ratio, rotation_deg)
    "Clearwater":       (45.3052, -94.1184, 73.0,  3186.9, 1.4, 30.0),
    "Lake Pulaski":     (45.2703, -94.0398, 42.0,   620.0, 1.2, 15.0),
    "Lake Sylvia":      (45.2853, -93.9960, 52.0,   420.0, 1.3, 45.0),
    "Pelican Lake":     (45.3193, -93.9085, 38.0,   640.0, 1.5, 20.0),
    "Maple Lake":       (45.2243, -94.0062, 28.0,   390.0, 1.2, 60.0),
    "Howard Lake":      (45.0618, -94.0740, 36.0,   344.0, 1.3, 10.0),
    "Lake Montrose":    (45.0698, -94.0169, 22.0,   228.0, 1.1, 80.0),
    "Lake Andrew":      (45.1437, -94.2478, 30.0,   510.0, 1.4, 35.0),
    "Buffalo Lake":     (45.1918, -94.2354, 18.0,   680.0, 1.6, 50.0),
    "Bass Lake":        (45.2531, -94.1152, 24.0,   185.0, 1.2, 20.0),
    "South Center Lake":(45.3804, -93.9371, 48.0,   330.0, 1.3, 15.0),
    "Lake Francis":     (45.1312, -93.9812, 30.0,   560.0, 1.5, 40.0),
    "Lake Charlotte":   (45.1956, -93.9374, 35.0,   280.0, 1.2, 25.0),
    "Twin Lake":        (45.3315, -94.1987, 26.0,   310.0, 1.3, 70.0),
    "Lake Ida":         (45.0893, -94.2175, 32.0,   450.0, 1.4, 55.0),
}


def main() -> None:
    generated = []
    errors = []
    for name, (lat, lon, max_d, area, aspect, rot) in LAKE_MORPHOLOGY.items():
        try:
            geojson = generate_depth_contours_geojson(
                lake_name=name,
                lat=lat,
                lon=lon,
                max_depth_ft=max_d,
                area_acres=area,
                num_contours=8,
                aspect=aspect,
                rotation_deg=rot,
            )
            path = save_bathymetry_geojson(name, geojson)
            n = len(geojson["features"])
            print(f"  {name:25s} → {path.name}  ({n} contours, max {max_d:.0f} ft)")
            generated.append(name)
        except Exception as exc:
            print(f"  ERROR {name}: {exc}")
            errors.append(name)

    print(f"\nDone: {len(generated)} lakes generated, {len(errors)} errors.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    print("Generating bathymetry GeoJSON for Wright County lakes…\n")
    main()
