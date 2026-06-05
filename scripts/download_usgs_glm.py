#!/usr/bin/env python3
"""Download and pre-process USGS GLM temperature data for Wright County lakes.

Usage:
    python scripts/download_usgs_glm.py [--sample] [--output-dir DIR]

Modes:
  --sample    Generate synthetic test data instead of downloading from USGS.
              Useful for development when network access to ScienceBase is
              unavailable.

  default     Download crosswalk + metadata from ScienceBase, identify
              Wright County lakes, extract their temperature profiles from
              the GLM NetCDF files, and save as per-lake parquet files.

Data source:
  ScienceBase item 6206d3c2d34ec05caca53071
  "Daily water column temperature predictions for thousands of Midwest
   U.S. lakes between 1979-2022"

Output:
  data/usgs_glm/<dow_id>.parquet   per-lake depth-resolved daily temps
  data/usgs_glm/crosswalk.csv       lake ID crosswalk (cached)
  data/usgs_glm/lake_metadata.csv   lake metadata (cached)
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "usgs_glm"

SCIENCEBASE_ITEM = "6206d3c2d34ec05caca53071"

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

DOW_IDS = {v[2] for v in WRIGHT_COUNTY_LAKES.values()}


def _try_import_requests():
    try:
        import requests
        return requests
    except ImportError:
        logger.error("requests is required for downloading. Install it with: pip install requests")
        sys.exit(1)


def _try_import_xarray():
    try:
        import xarray as xr
        return xr
    except ImportError:
        logger.error("xarray is required for reading NetCDF. Install it with: pip install xarray")
        sys.exit(1)


def _sciencebase_url(item_id: str) -> str:
    return f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json"


def download_crosswalk(requests_module) -> Optional[Path]:
    """Download lake_id_crosswalk.csv from ScienceBase."""
    out_path = DATA_DIR / "crosswalk.csv"
    if out_path.exists():
        logger.info("Crosswalk already cached at %s", out_path)
        return out_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests_module.get(_sciencebase_url(SCIENCEBASE_ITEM), timeout=60)
        item = resp.json()
    except Exception:
        logger.error("Failed to fetch ScienceBase item metadata")
        return None

    for f in item.get("files", []):
        name = f.get("name", "")
        if "crosswalk" in name.lower() and name.endswith(".csv"):
            url = f.get("downloadUrl", f.get("uri", ""))
            if not url:
                continue
            logger.info("Downloading crosswalk from %s", url)
            try:
                r = requests_module.get(url, timeout=120)
                out_path.write_bytes(r.content)
                logger.info("Saved crosswalk → %s", out_path)
                return out_path
            except Exception:
                logger.error("Failed to download crosswalk file")
                return None

    logger.error("crosswalk CSV not found in ScienceBase item files")
    return None


def download_metadata(requests_module) -> Optional[Path]:
    """Download lake_metadata.csv from ScienceBase."""
    out_path = DATA_DIR / "lake_metadata.csv"
    if out_path.exists():
        logger.info("Metadata already cached at %s", out_path)
        return out_path

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests_module.get(_sciencebase_url(SCIENCEBASE_ITEM), timeout=60)
        item = resp.json()
    except Exception:
        logger.error("Failed to fetch ScienceBase item metadata")
        return None

    for f in item.get("files", []):
        name = f.get("name", "")
        if "metadata" in name.lower() and name.endswith(".csv"):
            url = f.get("downloadUrl", f.get("uri", ""))
            if not url:
                continue
            logger.info("Downloading metadata from %s", url)
            try:
                r = requests_module.get(url, timeout=120)
                out_path.write_bytes(r.content)
                logger.info("Saved metadata → %s", out_path)
                return out_path
            except Exception:
                logger.error("Failed to download metadata file")
                return None

    logger.error("metadata CSV not found in ScienceBase item files")
    return None


def find_matching_site_ids(crosswalk_path: Path) -> Dict[str, str]:
    """Map DOW IDs to USGS site_ids via the crosswalk file.

    Returns {dow_id: site_id} for Wright County lakes found in the data.
    """
    import pandas as pd

    df = pd.read_csv(crosswalk_path)
    mndow_col = None
    for col in df.columns:
        if "mndow" in col.lower() or "dow" in col.lower():
            mndow_col = col
            break

    if mndow_col is None:
        logger.error("No MNDOW column found in crosswalk. Columns: %s", list(df.columns))
        return {}

    site_col = None
    for col in df.columns:
        if "site_id" in col.lower() or "siteid" in col.lower():
            site_col = col
            break
    if site_col is None:
        logger.error("No site_id column found in crosswalk. Columns: %s", list(df.columns))
        return {}

    df[mndow_col] = df[mndow_col].astype(str).str.strip()

    mapping: Dict[str, str] = {}
    for dow_id in DOW_IDS:
        matches = df[df[mndow_col].str.contains(dow_id, na=False, regex=False)]
        if not matches.empty:
            site_id = str(matches.iloc[0][site_col])
            mapping[dow_id] = site_id
            lake_name = [n for n, v in WRIGHT_COUNTY_LAKES.items() if v[2] == dow_id]
            logger.info("  %s (DOW %s) → site_id %s", lake_name[0] if lake_name else "?", dow_id, site_id)
        else:
            logger.warning("  DOW %s not found in crosswalk", dow_id)

    return mapping


def download_netcdf_data(
    requests_module,
    site_ids: Dict[str, str],
    model: str = "GLM",
) -> bool:
    """Download and extract temperature NetCDF data for matched lakes.

    Returns True if any lake data was extracted successfully.
    """
    import pandas as pd
    import tempfile

    try:
        resp = requests_module.get(_sciencebase_url(SCIENCEBASE_ITEM), timeout=60)
        item = resp.json()
    except Exception:
        logger.error("Failed to fetch ScienceBase item metadata for NetCDF download")
        return False

    zip_name_pattern = f"lake_temp_preds_{model}_NLDAS"
    zip_url = None
    for f in item.get("files", []):
        name = f.get("name", "")
        if zip_name_pattern in name and name.endswith(".zip"):
            zip_url = f.get("downloadUrl", f.get("uri", ""))
            break

    if not zip_url:
        logger.error("%s NetCDF zip not found in ScienceBase files", model)
        return False

    logger.info("Downloading %s zip from %s (this may take a while)...", model, zip_url)

    target_sites = set(site_ids.values())
    if not target_sites:
        logger.warning("No matched site IDs — nothing to extract")
        return False

    try:
        r = requests_module.get(zip_url, timeout=600, stream=True)
        tmp_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        for chunk in r.iter_content(chunk_size=8192):
            tmp_zip.write(chunk)
        tmp_zip.close()
        zip_path = tmp_zip.name
    except Exception:
        logger.error("Failed to download NetCDF zip")
        return False

    extracted = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_files = [n for n in zf.namelist() if n.endswith(".nc")]
            logger.info("NetCDF zip contains %d .nc files", len(nc_files))

            xr = _try_import_xarray()
            for nc_name in nc_files:
                with zf.open(nc_name) as f:
                    try:
                        ds = xr.open_dataset(io.BytesIO(f.read()))
                    except Exception:
                        logger.warning("Failed to open %s", nc_name)
                        continue

                    if "site_id" not in ds.coords and "site_id" not in ds.dims:
                        ds.close()
                        continue

                    site_ids_in_file = ds.coords.get("site_id", ds.dims.get("site_id"))
                    if site_ids_in_file is None:
                        ds.close()
                        continue

                    ids_values = set(str(v) for v in site_ids_in_file.values)
                    matched = ids_values & target_sites
                    if not matched:
                        ds.close()
                        continue

                    for sid in matched:
                        dow_id = next(k for k, v in site_ids.items() if v == sid)
                        try:
                            sub = ds.sel(site_id=sid)
                            df = sub.to_dataframe().reset_index()
                            df = df.rename(columns={
                                "depth": "depth_m",
                                "temp": "temp_c",
                            })
                            if "time" in df.columns:
                                df = df.rename(columns={"time": "date"})
                            df = df[["date", "depth_m", "temp_c"]].dropna(subset=["temp_c"])
                            out_path = DATA_DIR / f"{dow_id}.parquet"
                            df.to_parquet(out_path, index=False)
                            lake_name = [n for n, v in WRIGHT_COUNTY_LAKES.items() if v[2] == dow_id]
                            logger.info(
                                "  Extracted %s (%s): %d rows → %s",
                                lake_name[0] if lake_name else "?", dow_id, len(df), out_path,
                            )
                            extracted += 1
                        except Exception:
                            logger.warning("  Failed to extract site_id=%s", sid, exc_info=True)

                    ds.close()
    finally:
        os.unlink(zip_path)

    logger.info("Extracted data for %d lakes", extracted)
    return extracted > 0


def generate_sample_data(output_dir: Optional[Path] = None) -> None:
    """Generate synthetic test data for all Wright County lakes."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from onkia.usgs_glm import generate_sample_data as _gen

    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    for name, (lat, lon, dow) in WRIGHT_COUNTY_LAKES.items():
        max_depth_map = {
            "Clearwater": 25, "Lake Pulaski": 18, "Lake Sylvia": 19,
            "Pelican Lake": 14, "Maple Lake": 12, "Howard Lake": 9,
            "Lake Montrose": 8, "Lake Andrew": 22, "Buffalo Lake": 10,
            "Bass Lake": 11, "South Center Lake": 20, "Lake Francis": 15,
            "Lake Charlotte": 8, "Twin Lake": 10, "Lake Ida": 16,
        }
        max_depth = max_depth_map.get(name, 15.0)
        _gen(dow, name, max_depth, lat, output_dir=out)
        logger.info("  Generated sample data for %s (DOW %s)", name, dow)

    logger.info("Sample data generation complete — %d lakes", len(WRIGHT_COUNTY_LAKES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download USGS GLM temperature data for Wright County lakes")
    parser.add_argument("--sample", action="store_true", help="Generate synthetic test data instead of downloading")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()

    out_dir = args.output_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        logger.info("=== Generating synthetic sample data ===")
        generate_sample_data(output_dir=out_dir)
        return

    logger.info("=== Downloading USGS GLM data from ScienceBase ===")
    requests = _try_import_requests()

    cw_path = download_crosswalk(requests)
    if cw_path is None:
        logger.error("Cannot proceed without crosswalk file. Use --sample for test data.")
        sys.exit(1)

    mapping = find_matching_site_ids(cw_path)
    if not mapping:
        logger.warning("No Wright County lakes matched in crosswalk. Trying EA-LSTM model.")
        success = download_netcdf_data(requests, {}, model="EALSTM")
        if not success:
            logger.error("No data extracted. Use --sample for test data.")
            sys.exit(1)
        return

    logger.info("Matched %d Wright County lakes in crosswalk", len(mapping))

    meta_path = download_metadata(requests)
    if meta_path:
        logger.info("Lake metadata cached at %s", meta_path)

    success = download_netcdf_data(requests, mapping, model="GLM")
    if not success:
        logger.warning("GLM extraction failed — trying EA-LSTM model (broader coverage)")
        success = download_netcdf_data(requests, mapping, model="EALSTM")

    if success:
        logger.info("Data download and extraction complete.")
    else:
        logger.error("Failed to extract any lake data. Use --sample for test data.")


if __name__ == "__main__":
    main()
