"""Satellite-derived lake surface temperature via Google Earth Engine.

Fetches Landsat 8/9 thermal band (ST_B10) data for Wright County lakes.
Provides:
  - Most recent cloud-free LST observation per lake
  - Historical LST trend (last 3-6 months)
  - Sentinel-2 NDCI (chlorophyll-a proxy)
  - Spatial temperature heatmap URL for large lakes (>1000 acres)

GEE authentication is required for live data. Configure via:
  EE_SERVICE_ACCOUNT_EMAIL    — service account email
  EE_SERVICE_ACCOUNT_KEY_PATH — path to private key JSON file
  EE_SERVICE_ACCOUNT_KEY_JSON — private key JSON contents (alternative to key path)

If credentials are absent, all public functions return results with
``fallback_used=True`` and ``None`` temperature values.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lake metadata
# ---------------------------------------------------------------------------

#: Approximate surface area in acres for Wright County lakes.
LAKE_ACRES: dict[str, float] = {
    "Clearwater": 3158.0,
    "Buffalo Lake": 1552.0,
    "Lake Sylvia": 904.0,
    "Twin Lake": 872.0,
    "Lake Pulaski": 813.0,
    "Maple Lake": 777.0,
    "Howard Lake": 711.0,
    "Pelican Lake": 3800.0,
    "Lake Charlotte": 253.0,
    "Lake Ida": 226.0,
    "Bass Lake": 218.0,
    "Lake Francis": 460.0,
    "Lake Andrew": 200.0,
    "Lake Montrose": 150.0,
    "South Center Lake": 100.0,
}

#: Lakes below this threshold get a single mean LST value only.
LST_USEFUL_THRESHOLD_ACRES: float = 700.0

#: Lakes at or above this threshold get a spatial heatmap overlay.
HEATMAP_THRESHOLD_ACRES: float = 1000.0

# Landsat scale factor: ST_B10 DN → Kelvin
_LANDSAT_SCALE = 0.00341802
_LANDSAT_OFFSET = 149.0  # Kelvin offset

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LSTObservation:
    """Surface temperature result for a single lake snapshot."""

    lake_name: str
    temp_celsius: Optional[float] = None
    temp_fahrenheit: Optional[float] = None
    observation_date: Optional[date] = None
    #: Number of cloud-free Landsat scenes composited.
    scene_count: int = 0
    #: Pixel count used in the mean reduction.
    pixel_count: Optional[int] = None
    satellite: str = "Landsat 8/9"
    fallback_used: bool = False
    error_msg: Optional[str] = None


@dataclass
class LSTHistoryPoint:
    """One LST observation in a historical time series."""

    observation_date: date
    temp_celsius: float
    temp_fahrenheit: float


@dataclass
class NDCIObservation:
    """Sentinel-2 Normalised Difference Chlorophyll Index result."""

    lake_name: str
    ndci_value: Optional[float] = None
    #: Qualitative category derived from NDCI.
    chlorophyll_category: Optional[str] = None
    observation_date: Optional[date] = None
    fallback_used: bool = False
    error_msg: Optional[str] = None


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0


def kelvin_to_celsius(k: float) -> float:
    """Convert Kelvin to Celsius."""
    return k - 273.15


def landsat_dn_to_celsius(dn: float) -> float:
    """Apply Landsat Collection-2 ST scale factor and convert to Celsius."""
    kelvin = dn * _LANDSAT_SCALE + _LANDSAT_OFFSET
    return kelvin_to_celsius(kelvin)


def ndci_to_category(ndci: float) -> str:
    """Map NDCI value to a qualitative chlorophyll category.

    Thresholds from Mishra & Mishra (2012):
      ndci < 0.0   → low
      0.0–0.1     → moderate
      > 0.1       → high
    """
    if ndci < 0.0:
        return "low"
    if ndci <= 0.1:
        return "moderate"
    return "high"


def lake_radius_m(lake_name: str) -> float:
    """Return a search-radius (metres) appropriate for the lake's area.

    Larger lakes need a bigger buffer to capture enough pixels.
    """
    acres = LAKE_ACRES.get(lake_name, 300.0)
    if acres >= 2000:
        return 3000.0
    if acres >= 1000:
        return 2000.0
    if acres >= 500:
        return 1200.0
    return 700.0


# ---------------------------------------------------------------------------
# GEE initialisation
# ---------------------------------------------------------------------------

_gee_initialized: bool = False
_gee_available: bool = False
_temp_key_path: Optional[str] = None  # path of any temp key file written


def _init_gee() -> bool:
    """Initialise the Earth Engine API. Returns True if ready.

    Reads credentials from environment variables:
      EE_SERVICE_ACCOUNT_EMAIL    — service account email
      EE_SERVICE_ACCOUNT_KEY_PATH — path to JSON private key file
      EE_SERVICE_ACCOUNT_KEY_JSON — JSON key contents (alternative to key path)

    Falls back to application default credentials when env vars are absent.
    """
    global _gee_initialized, _gee_available, _temp_key_path
    if _gee_initialized:
        return _gee_available
    _gee_initialized = True

    try:
        import ee  # type: ignore[import]
    except ImportError:
        _log.warning("earthengine-api not installed; satellite LST unavailable")
        return False

    service_account = os.getenv("EE_SERVICE_ACCOUNT_EMAIL", "").strip()
    key_path = os.getenv("EE_SERVICE_ACCOUNT_KEY_PATH", "").strip()
    key_json = os.getenv("EE_SERVICE_ACCOUNT_KEY_JSON", "").strip()

    try:
        if service_account and (key_path or key_json):
            if key_json and not key_path:
                # Write the JSON contents to a temporary file
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                )
                tmp.write(key_json)
                tmp.close()
                _temp_key_path = tmp.name
                key_path = _temp_key_path
            credentials = ee.ServiceAccountCredentials(service_account, key_path)
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        _gee_available = True
        _log.info("Google Earth Engine initialised successfully")
    except Exception as exc:
        _log.warning("GEE initialisation failed: %s", exc)
        _gee_available = False

    return _gee_available


def _reset_gee_state() -> None:
    """Reset GEE init state — used in tests only."""
    global _gee_initialized, _gee_available
    _gee_initialized = False
    _gee_available = False


# ---------------------------------------------------------------------------
# Internal GEE helpers
# ---------------------------------------------------------------------------


def _build_lake_region(lat: float, lon: float, radius_m: float):  # type: ignore[return]
    """Return an ee.Geometry circle for the lake."""
    import ee  # type: ignore[import]
    return ee.Geometry.Point([lon, lat]).buffer(radius_m)


def _apply_lst_mask(image):  # type: ignore[return]
    """Apply Landsat C2 cloud/shadow mask and scale ST_B10 to Celsius."""
    import ee  # type: ignore[import]
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    clear_mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
        qa.bitwiseAnd(shadow_bit).eq(0)
    )
    lst_celsius = (
        image.select("ST_B10")
        .multiply(_LANDSAT_SCALE)
        .add(_LANDSAT_OFFSET)
        .subtract(273.15)
        .rename("LST_C")
    )
    return (
        lst_celsius
        .updateMask(clear_mask)
        .copyProperties(image, ["system:time_start"])
    )


def _landsat_collection(region, start_iso: str, end_iso: str):  # type: ignore[return]
    """Return a merged Landsat 8+9 C2 L2 collection, filtered and mapped."""
    import ee  # type: ignore[import]
    col8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    col9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    return (
        col8.merge(col9)
        .filterBounds(region)
        .filterDate(start_iso, end_iso)
        .map(_apply_lst_mask)
    )


def _mean_lst_over_region(composite, region) -> Optional[float]:
    """Reduce composite to a mean LST_C over region; returns None on error."""
    import ee  # type: ignore[import]
    try:
        result = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,
            maxPixels=int(1e6),
        )
        val = result.get("LST_C").getInfo()
        return float(val) if val is not None else None
    except Exception as exc:
        _log.debug("reduceRegion failed: %s", exc)
        return None


def _pixel_count_over_region(composite, region) -> Optional[int]:
    """Return the count of valid pixels in composite over region."""
    import ee  # type: ignore[import]
    try:
        result = composite.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=100,
            maxPixels=int(1e6),
        )
        val = result.get("LST_C").getInfo()
        return int(val) if val is not None else None
    except Exception:
        return None


def _most_recent_scene_date(collection) -> Optional[date]:
    """Return the acquisition date of the most recent image in collection."""
    import ee  # type: ignore[import]
    try:
        size = collection.size().getInfo()
        if size == 0:
            return None
        latest = collection.sort("system:time_start", False).first()
        ts_ms = latest.get("system:time_start").getInfo()
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_latest_lst(
    lake_name: str,
    lat: float,
    lon: float,
    days_back: int = 30,
) -> LSTObservation:
    """Fetch the most recent cloud-free Landsat LST for a lake.

    Uses a 30-day composite (median) when no single scene is cloud-free enough.

    Args:
        lake_name: Display name for labelling results.
        lat:       Lake centroid latitude (WGS-84).
        lon:       Lake centroid longitude (WGS-84).
        days_back: Look-back window in days (default 30).

    Returns:
        LSTObservation with temp in Celsius and Fahrenheit, or
        ``fallback_used=True`` if GEE is unavailable.
    """
    if not _init_gee():
        return LSTObservation(
            lake_name=lake_name,
            fallback_used=True,
            error_msg="GEE not initialised",
        )

    try:
        import ee  # type: ignore[import]

        radius_m = lake_radius_m(lake_name)
        region = _build_lake_region(lat, lon, radius_m)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        collection = _landsat_collection(
            region, start_date.isoformat(), end_date.isoformat()
        )
        scene_count = collection.size().getInfo()
        composite = collection.median()
        lst_c = _mean_lst_over_region(composite, region)
        pixel_count = _pixel_count_over_region(composite, region)
        obs_date = _most_recent_scene_date(collection)

        if lst_c is None:
            return LSTObservation(
                lake_name=lake_name,
                observation_date=obs_date,
                scene_count=scene_count,
                fallback_used=True,
                error_msg="No valid pixels in composite",
            )

        return LSTObservation(
            lake_name=lake_name,
            temp_celsius=round(lst_c, 1),
            temp_fahrenheit=round(celsius_to_fahrenheit(lst_c), 1),
            observation_date=obs_date,
            scene_count=scene_count,
            pixel_count=pixel_count,
            fallback_used=False,
        )
    except Exception as exc:
        _log.warning("get_latest_lst failed for %s: %s", lake_name, exc)
        return LSTObservation(
            lake_name=lake_name,
            fallback_used=True,
            error_msg=str(exc),
        )


def get_lst_history(
    lake_name: str,
    lat: float,
    lon: float,
    days_back: int = 180,
    interval_days: int = 16,
) -> List[LSTHistoryPoint]:
    """Fetch a historical LST time series for a lake.

    Samples the Landsat archive in ``interval_days`` windows to build a trend.

    Args:
        lake_name:     Display name for labelling.
        lat:           Lake centroid latitude.
        lon:           Lake centroid longitude.
        days_back:     Total look-back window (default 180 days / ~6 months).
        interval_days: Size of each compositing window (default 16, ~Landsat repeat).

    Returns:
        List of LSTHistoryPoint sorted oldest-first.
        Empty list if GEE is unavailable or no data found.
    """
    if not _init_gee():
        return []

    try:
        import ee  # type: ignore[import]

        radius_m = lake_radius_m(lake_name)
        region = _build_lake_region(lat, lon, radius_m)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        points: List[LSTHistoryPoint] = []
        cursor = start_date
        while cursor < end_date:
            window_end = min(cursor + timedelta(days=interval_days), end_date)
            collection = _landsat_collection(
                region, cursor.isoformat(), window_end.isoformat()
            )
            size = collection.size().getInfo()
            if size > 0:
                composite = collection.median()
                lst_c = _mean_lst_over_region(composite, region)
                window_mid = cursor + (window_end - cursor) / 2
                if lst_c is not None:
                    points.append(
                        LSTHistoryPoint(
                            observation_date=window_mid,
                            temp_celsius=round(lst_c, 1),
                            temp_fahrenheit=round(celsius_to_fahrenheit(lst_c), 1),
                        )
                    )
            cursor = window_end

        return sorted(points, key=lambda p: p.observation_date)
    except Exception as exc:
        _log.warning("get_lst_history failed for %s: %s", lake_name, exc)
        return []


def get_ndci(
    lake_name: str,
    lat: float,
    lon: float,
    days_back: int = 30,
) -> NDCIObservation:
    """Fetch the most recent Sentinel-2 NDCI (chlorophyll-a proxy).

    NDCI = (B05 - B04) / (B05 + B04)
    where B05 = red-edge (705 nm) and B04 = red (665 nm).

    Args:
        lake_name: Display name for labelling.
        lat:       Lake centroid latitude.
        lon:       Lake centroid longitude.
        days_back: Look-back window in days (default 30).

    Returns:
        NDCIObservation with ndci_value and chlorophyll_category, or
        ``fallback_used=True`` if GEE is unavailable.
    """
    if not _init_gee():
        return NDCIObservation(
            lake_name=lake_name,
            fallback_used=True,
            error_msg="GEE not initialised",
        )

    try:
        import ee  # type: ignore[import]

        radius_m = lake_radius_m(lake_name)
        region = _build_lake_region(lat, lon, radius_m)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        def _apply_ndci(image):
            red_edge = image.select("B5")  # 705 nm
            red = image.select("B4")       # 665 nm
            # Mask clouds/water using Scene Classification Layer
            scl = image.select("SCL")
            valid_mask = scl.neq(9).And(scl.neq(8)).And(scl.neq(3))
            ndci = red_edge.subtract(red).divide(red_edge.add(red)).rename("NDCI")
            return ndci.updateMask(valid_mask).copyProperties(
                image, ["system:time_start"]
            )

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(_apply_ndci)
        )

        size = collection.size().getInfo()
        if size == 0:
            return NDCIObservation(
                lake_name=lake_name,
                fallback_used=True,
                error_msg="No Sentinel-2 scenes in window",
            )

        composite = collection.median()
        try:
            result = composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=20,
                maxPixels=int(1e6),
            )
            ndci_val = result.get("NDCI").getInfo()
        except Exception:
            ndci_val = None

        obs_date = _most_recent_scene_date(collection)

        if ndci_val is None:
            return NDCIObservation(
                lake_name=lake_name,
                observation_date=obs_date,
                fallback_used=True,
                error_msg="No valid NDCI pixels",
            )

        ndci_f = round(float(ndci_val), 3)
        return NDCIObservation(
            lake_name=lake_name,
            ndci_value=ndci_f,
            chlorophyll_category=ndci_to_category(ndci_f),
            observation_date=obs_date,
            fallback_used=False,
        )
    except Exception as exc:
        _log.warning("get_ndci failed for %s: %s", lake_name, exc)
        return NDCIObservation(
            lake_name=lake_name,
            fallback_used=True,
            error_msg=str(exc),
        )


def get_lst_heatmap_url(
    lat: float,
    lon: float,
    radius_m: float,
    days_back: int = 30,
    min_temp_c: float = 5.0,
    max_temp_c: float = 30.0,
    image_size: int = 400,
) -> Optional[str]:
    """Return a signed GEE thumbnail URL for the LST spatial heatmap.

    Uses a thermal colour palette (blue → red). Suitable for display via
    ``st.image()`` in Streamlit or an <img> tag.  URL is valid for ~1 hour.

    Args:
        lat:         Lake centroid latitude.
        lon:         Lake centroid longitude.
        radius_m:    Buffer radius in metres.
        days_back:   Composite window in days (default 30).
        min_temp_c:  Minimum temperature for colour scale.
        max_temp_c:  Maximum temperature for colour scale.
        image_size:  Output image pixel dimension (square).

    Returns:
        HTTPS URL string, or None if GEE is unavailable.
    """
    if not _init_gee():
        return None

    try:
        import ee  # type: ignore[import]

        region = _build_lake_region(lat, lon, radius_m)
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        collection = _landsat_collection(
            region, start_date.isoformat(), end_date.isoformat()
        )
        if collection.size().getInfo() == 0:
            return None

        composite = collection.median()
        url: str = composite.getThumbURL(
            {
                "min": min_temp_c,
                "max": max_temp_c,
                "palette": [
                    "000080",  # navy (cold)
                    "0000ff",  # blue
                    "00ffff",  # cyan
                    "00ff00",  # green
                    "ffff00",  # yellow
                    "ff8000",  # orange
                    "ff0000",  # red (warm)
                ],
                "region": region,
                "dimensions": image_size,
                "format": "png",
            }
        )
        return url
    except Exception as exc:
        _log.warning("get_lst_heatmap_url failed: %s", exc)
        return None
