#!/usr/bin/env python3
"""Spike experiments: validate USGS STAC API as GEE replacement.

Run each spike individually:
  python3 scripts/spike_stac_validation.py spike1   # Landsat thermal
  python3 scripts/spike_stac_validation.py spike2   # Heatmap
  python3 scripts/spike_stac_validation.py spike3   # Sentinel-2 NDCI
  python3 scripts/spike_stac_validation.py spike4   # USGS pre-computed dataset
  python3 scripts/spike_stac_validation.py all       # Run all spikes
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta, timezone

CLEARWATER_LAT = 45.3052
CLEARWATER_LON = -94.1184
CLEARWATER_NAME = "Clearwater"


def spike1_landsat_thermal() -> dict:
    """Spike 1: Fetch Landsat thermal tile via USGS STAC, compute mean temp."""
    print("=" * 60)
    print("SPIKE 1: Landsat thermal via USGS STAC")
    print("=" * 60)

    from onkia.satellite_lst_stac import get_latest_lst

    start = time.time()
    result = get_latest_lst(CLEARWATER_NAME, CLEARWATER_LAT, CLEARWATER_LON, days_back=30)
    elapsed = time.time() - start

    print(f"Lake: {result.lake_name}")
    print(f"Temp (C): {result.temp_celsius}")
    print(f"Temp (F): {result.temp_fahrenheit}")
    print(f"Obs date: {result.observation_date}")
    print(f"Scene count: {result.scene_count}")
    print(f"Pixel count: {result.pixel_count}")
    print(f"Fallback used: {result.fallback_used}")
    if result.error_msg:
        print(f"Error: {result.error_msg}")
    print(f"Elapsed: {elapsed:.1f}s")

    return {
        "spike": 1,
        "success": not result.fallback_used,
        "temp_celsius": result.temp_celsius,
        "temp_fahrenheit": result.temp_fahrenheit,
        "observation_date": str(result.observation_date),
        "scene_count": result.scene_count,
        "pixel_count": result.pixel_count,
        "elapsed_s": round(elapsed, 1),
        "error_msg": result.error_msg,
    }


def spike2_heatmap() -> dict:
    """Spike 2: Generate heatmap thumbnail from STAC raster."""
    print("=" * 60)
    print("SPIKE 2: LST heatmap via USGS STAC + matplotlib")
    print("=" * 60)

    from onkia.satellite_lst_stac import get_lst_heatmap

    start = time.time()
    png_bytes = get_lst_heatmap(
        CLEARWATER_LAT, CLEARWATER_LON, 3000.0, days_back=30, image_size=400
    )
    elapsed = time.time() - start

    if png_bytes is not None:
        out_path = "/tmp/clearwater_lst_heatmap.png"
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"Heatmap saved to: {out_path}")
        print(f"Image size: {len(png_bytes)} bytes")
        print(f"Elapsed: {elapsed:.1f}s")
    else:
        print("No heatmap generated (no data available)")

    return {
        "spike": 2,
        "success": png_bytes is not None,
        "image_bytes": len(png_bytes) if png_bytes else 0,
        "output_path": "/tmp/clearwater_lst_heatmap.png" if png_bytes else None,
        "elapsed_s": round(elapsed, 1),
    }


def spike3_ndci() -> dict:
    """Spike 3: Fetch Sentinel-2 NDCI via STAC (Planetary Computer or Copernicus)."""
    print("=" * 60)
    print("SPIKE 3: Sentinel-2 NDCI via STAC")
    print("=" * 60)

    from onkia.satellite_lst_stac import get_ndci, _HAS_PLANETARY_COMPUTER

    if not _HAS_PLANETARY_COMPUTER:
        import os
        token = os.getenv("CDSE_ACCESS_TOKEN", "").strip()
        if not token:
            print("SKIP: planetary_computer not installed and CDSE_ACCESS_TOKEN not set")
            return {"spike": 3, "success": False, "error_msg": "No auth available"}
    else:
        print("Using Planetary Computer (free, no auth required)")

    start = time.time()
    result = get_ndci(CLEARWATER_NAME, CLEARWATER_LAT, CLEARWATER_LON, days_back=30)
    elapsed = time.time() - start

    print(f"Lake: {result.lake_name}")
    print(f"NDCI: {result.ndci_value}")
    print(f"Category: {result.chlorophyll_category}")
    print(f"Obs date: {result.observation_date}")
    print(f"Fallback used: {result.fallback_used}")
    if result.error_msg:
        print(f"Error: {result.error_msg}")
    print(f"Elapsed: {elapsed:.1f}s")

    return {
        "spike": 3,
        "success": not result.fallback_used,
        "ndci_value": result.ndci_value,
        "chlorophyll_category": result.chlorophyll_category,
        "observation_date": str(result.observation_date),
        "elapsed_s": round(elapsed, 1),
        "error_msg": result.error_msg,
    }


def spike4_sciencebase() -> dict:
    """Spike 4: Evaluate USGS pre-computed lake temperature dataset."""
    print("=" * 60)
    print("SPIKE 4: USGS pre-computed lake temperature (ScienceBase)")
    print("=" * 60)

    import requests

    sb_url = "https://www.sciencebase.gov/catalog/items"
    params = {
        "parentId": "5f5b51f282ce3f6c8a1e2b3c",
        "q": "lake temperature",
        "max": 10,
        "format": "json",
    }

    try:
        start = time.time()
        resp = requests.get(sb_url, params=params, timeout=15)
        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"ScienceBase API returned status {resp.status_code}")
            return {
                "spike": 4,
                "success": False,
                "error_msg": f"HTTP {resp.status_code}",
                "elapsed_s": round(elapsed, 1),
            }

        data = resp.json()
        items = data.get("items", [])
        print(f"Found {len(items)} items on ScienceBase")
        for item in items[:5]:
            print(f"  - {item.get('title', 'untitled')} (id: {item.get('id', '?')})")
            if item.get("summary"):
                print(f"    {item['summary'][:120]}...")

        print(f"Elapsed: {elapsed:.1f}s")

        detailed_items = []
        for item in items[:5]:
            detailed_items.append({
                "id": item.get("id"),
                "title": item.get("title"),
                "summary": (item.get("summary") or "")[:200],
            })

        return {
            "spike": 4,
            "success": len(items) > 0,
            "item_count": len(items),
            "items": detailed_items,
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as exc:
        print(f"ScienceBase query failed: {exc}")
        return {"spike": 4, "success": False, "error_msg": str(exc)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python spike_stac_validation.py [spike1|spike2|spike3|spike4|all]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    results = {}

    if cmd in ("spike1", "all"):
        results["spike1"] = spike1_landsat_thermal()
        print()

    if cmd in ("spike2", "all"):
        results["spike2"] = spike2_heatmap()
        print()

    if cmd in ("spike3", "all"):
        results["spike3"] = spike3_ndci()
        print()

    if cmd in ("spike4", "all"):
        results["spike4"] = spike4_sciencebase()
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, val in results.items():
        status = "PASS" if val.get("success") else "FAIL"
        print(f"  {key}: {status}")

    out_path = "/tmp/stac_spike_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to: {out_path}")


if __name__ == "__main__":
    main()
