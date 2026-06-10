"""Satellite-derived lake surface temperature via STAC APIs.

Replaces the former GEE-based satellite_lst module.
Uses the Microsoft Planetary Computer STAC API (pystac-client + planetary_computer)
to fetch Landsat 8/9 Collection 2 Level-2 thermal data (ST_B10 band).
No credentials required — the Planetary Computer provides free SAS-signed
URLs to Azure Blob Storage.

For Sentinel-2 NDCI, also uses the Planetary Computer (free). Falls back to
the Copernicus Data Space Ecosystem STAC API (requires CDSE_ACCESS_TOKEN).

Public API:
  - get_latest_lst()  — most recent cloud-free LST per lake
  - get_lst_history()  — historical LST trend
  - get_ndci()         — Sentinel-2 NDCI (chlorophyll-a proxy)
  - get_lst_heatmap()  — spatial temperature heatmap as PNG bytes
"""
from __future__ import annotations

import io
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import numpy as np

try:
    import pystac_client
    import rasterio
    import rioxarray
    import xarray as xr
    from rasterio.windows import from_bounds as rio_window_from_bounds
except ImportError as _exc:
    raise ImportError(
        "satellite_lst_stac requires pystac-client, rasterio, and rioxarray. "
        "Install with: pip install pystac-client rioxarray"
    ) from _exc

try:
    import planetary_computer
    _HAS_PLANETARY_COMPUTER = True
except ImportError:
    _HAS_PLANETARY_COMPUTER = False

# Tune GDAL for remote COG access: skip directory listings on open and
# bound every HTTP request so a stalled read fails fast instead of
# hanging the Streamlit worker indefinitely.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

#: Connect/read timeout (seconds) for STAC API searches.
_STAC_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Lake metadata
# ---------------------------------------------------------------------------

#: Surface area in acres for Wright County lakes (MN DNR LakeFinder morphology).
LAKE_ACRES: dict[str, float] = {
    "Clearwater": 3187.0,
    "Pelican Lake": 3800.0,
    "Buffalo Lake": 1552.0,
    "Sugar Lake": 1014.0,
    "West Lake Sylvia": 904.0,
    "Lake Pulaski": 813.0,
    "Cedar Lake": 790.0,
    "Howard Lake": 745.0,
    "East Lake Sylvia": 669.0,
    "Maple Lake": 633.0,
    "Waverly Lake": 494.0,
    "Granite Lake": 362.0,
    "French Lake": 346.0,
    "Ramsey Lake": 317.0,
    "Lake Charlotte": 253.0,
    "Lake Ida": 226.0,
    "Bass Lake": 222.0,
}

#: Lakes below this threshold get a single mean LST value only.
LST_USEFUL_THRESHOLD_ACRES: float = 700.0

#: Lakes at or above this threshold get a spatial heatmap overlay.
HEATMAP_THRESHOLD_ACRES: float = 1000.0

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


@dataclass
class SatelliteOverlayGrids:
    """Georeferenced satellite grids for map overlays (EPSG:4326).

    ``lst_grid`` holds surface temperature in °C; ``ndci_grid`` holds NDCI
    (chlorophyll/vegetation proxy). Both are 2-D xarray DataArrays with
    ``y``/``x`` lat/lon coordinates, so map bounds can be derived via
    :func:`grid_bounds`. Either grid may be None if no cloud-free scene
    was found.
    """

    lake_name: str
    lst_grid: Optional["xr.DataArray"] = None
    lst_date: Optional[date] = None
    ndci_grid: Optional["xr.DataArray"] = None
    ndci_date: Optional[date] = None
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


def lake_radius_m(lake_name: str, acres: Optional[float] = None) -> float:
    """Return a search-radius (metres) appropriate for the lake's area.

    Larger lakes need a bigger buffer to capture enough pixels. ``acres``
    overrides the built-in Wright County table for lakes added at runtime
    (e.g. from a DNR LakeFinder search in another county).
    """
    if acres is None or acres <= 0:
        acres = LAKE_ACRES.get(lake_name, 300.0)
    if acres >= 2000:
        return 3000.0
    if acres >= 1000:
        return 2000.0
    if acres >= 500:
        return 1200.0
    return 700.0

_log = logging.getLogger(__name__)

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_C2L2_COLLECTION = "landsat-c2-l2"

USGS_STAC_URL = "https://landsatlook.usgs.gov/stac-server"
LANDSAT_ST_COLLECTION = "landsat-c2l2-st"

COPERNICUS_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
SENTINEL2_L2A_COLLECTION = "sentinel-2-l2a"

_LANDSAT_SCALE = 0.00341802
_LANDSAT_OFFSET = 149.0

_LANDSAT_THERMAL_ASSET = "lwir11"
_LANDSAT_QA_ASSET = "qa_pixel"


def _point_to_bbox(lat: float, lon: float, radius_m: float) -> list:
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    dlat = radius_m / m_per_deg_lat
    dlon = radius_m / m_per_deg_lon
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def _landsat_stac_client() -> pystac_client.Client:
    if _HAS_PLANETARY_COMPUTER:
        return pystac_client.Client.open(
            MPC_STAC_URL,
            modifier=planetary_computer.sign_inplace,
            timeout=_STAC_TIMEOUT,
        )
    _log.warning(
        "planetary_computer not installed — falling back to USGS STAC. "
        "Landsat data URLs require ERS authentication. "
        "Install with: pip install planetary-computer"
    )
    return pystac_client.Client.open(USGS_STAC_URL, timeout=_STAC_TIMEOUT)


def _copernicus_stac_client() -> pystac_client.Client:
    headers: dict[str, str] = {}
    token = os.getenv("CDSE_ACCESS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return pystac_client.Client.open(COPERNICUS_STAC_URL, headers=headers, timeout=_STAC_TIMEOUT)


def _cloud_mask_landsat(qa_pixel) -> np.ndarray:
    vals = np.array(qa_pixel, dtype=np.uint16)
    cloud_bit = np.uint16(1 << 3)
    shadow_bit = np.uint16(1 << 4)
    return ((vals & cloud_bit) == 0) & ((vals & shadow_bit) == 0)


def _fetch_landsat_items(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    radius_m: float,
    max_items: int = 20,
) -> list:
    client = _landsat_stac_client()
    collection = LANDSAT_C2L2_COLLECTION if _HAS_PLANETARY_COMPUTER else LANDSAT_ST_COLLECTION
    bbox = _point_to_bbox(lat, lon, radius_m)
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        max_items=max_items,
    )
    items = list(search.items())
    items.sort(
        key=lambda i: i.datetime or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return items


def _fetch_sentinel2_items(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    radius_m: float,
    max_items: int = 10,
) -> list:
    if _HAS_PLANETARY_COMPUTER:
        client = pystac_client.Client.open(
            MPC_STAC_URL,
            modifier=planetary_computer.sign_inplace,
            timeout=_STAC_TIMEOUT,
        )
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=_point_to_bbox(lat, lon, radius_m),
            datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
            max_items=max_items,
            query={"eo:cloud_cover": {"lt": 30}},
        )
    else:
        token = os.getenv("CDSE_ACCESS_TOKEN", "").strip()
        if not token:
            return []
        client = _copernicus_stac_client()
        search = client.search(
            collections=[SENTINEL2_L2A_COLLECTION],
            bbox=_point_to_bbox(lat, lon, radius_m),
            datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
            max_items=max_items,
            query={"eo:cloud_cover": {"lt": 30}},
        )
    items = list(search.items())
    items.sort(
        key=lambda i: i.datetime or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return items


def _s2_band_keys(item: pystac_client.Item):
    if "B04" in item.assets and "B05" in item.assets:
        return "B04", "B05", "SCL"
    if "B04_20m" in item.assets and "B05_20m" in item.assets:
        return "B04_20m", "B05_20m", "SCL_20m"
    return None, None, None


def _read_windowed(
    asset_href: str,
    lat: float,
    lon: float,
    radius_m: float,
    target_shape: Optional[tuple] = None,
) -> Optional[np.ndarray]:
    """Read a windowed subset of a raster in its native CRS.

    Args:
        asset_href: URL to the raster asset.
        lat: Lake centroid latitude.
        lon: Lake centroid longitude.
        radius_m: Buffer radius in metres.
        target_shape: If given, resample to this (rows, cols) shape.
    """
    try:
        from pyproj import Transformer
        from scipy.ndimage import zoom as ndimage_zoom

        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        dlat = radius_m / m_per_deg_lat
        dlon = radius_m / m_per_deg_lon
        min_lon, max_lon = lon - dlon, lon + dlon
        min_lat, max_lat = lat - dlat, lat + dlat

        with rasterio.open(asset_href) as ds:
            src_crs = ds.crs
            if src_crs is not None:
                transformer = Transformer.from_crs(
                    "EPSG:4326", src_crs, always_xy=True
                )
                min_x, min_y = transformer.transform(min_lon, min_lat)
                max_x, max_y = transformer.transform(max_lon, max_lat)
            else:
                min_x, min_y, max_x, max_y = min_lon, min_lat, max_lon, max_lat

            window = rio_window_from_bounds(min_x, min_y, max_x, max_y, ds.transform)
            window = window.round_offsets().round_lengths()
            data = ds.read(1, window=window, masked=True)

        if target_shape is not None and data.shape != target_shape:
            if data.count() > 0:
                row_factor = target_shape[0] / data.shape[0]
                col_factor = target_shape[1] / data.shape[1]
                filled = data.filled(0)
                resampled = ndimage_zoom(filled, (row_factor, col_factor), order=0)
                resampled_mask = ndimage_zoom(
                    data.mask.astype(np.uint8), (row_factor, col_factor), order=0
                )
                data = np.ma.masked_where(resampled_mask > 0.5, resampled)
            else:
                data = np.ma.masked_all(target_shape)

        return data
    except Exception as exc:
        _log.warning("Windowed read failed for %s: %s", asset_href[:80], exc)
        return None


def _read_windowed_xr(
    asset_href: str,
    lat: float,
    lon: float,
    radius_m: float,
) -> Optional[xr.DataArray]:
    """Read a windowed subset and return as an EPSG:4326 xarray DataArray.

    Clips to the lake window in the raster's *native* CRS first so only a
    few hundred kilobytes are fetched from the remote COG, then reprojects
    just that clip. (Reprojecting before clipping would download and warp
    the entire scene — hundreds of megabytes per band.)
    """
    try:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        dlat = radius_m / m_per_deg_lat
        dlon = radius_m / m_per_deg_lon
        min_lon, max_lon = lon - dlon, lon + dlon
        min_lat, max_lat = lat - dlat, lat + dlat

        da = rioxarray.open_rasterio(
            asset_href, masked=True, default_name="band"
        ).squeeze("band", drop=True)

        # clip_box transforms the lat/lon box into the raster's CRS, so the
        # clip happens before any pixels are pulled over the network.
        da = da.rio.clip_box(
            minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat,
            crs="EPSG:4326",
        ).load()

        if da.rio.crs is not None and da.rio.crs.to_epsg() != 4326:
            da = da.rio.reproject("EPSG:4326")
        return da
    except Exception as exc:
        _log.warning("Windowed xr read failed for %s: %s", asset_href[:80], exc)
        return None


def get_latest_lst(
    lake_name: str,
    lat: float,
    lon: float,
    days_back: int = 30,
) -> LSTObservation:
    """Fetch the most recent cloud-free Landsat LST for a lake via STAC.

    Uses the Microsoft Planetary Computer for free access to Landsat
    Collection 2 Level-2 Surface Temperature data. No credentials needed
    when the ``planetary_computer`` package is installed.

    Args:
        lake_name: Display name for labelling results.
        lat:       Lake centroid latitude (WGS-84).
        lon:       Lake centroid longitude (WGS-84).
        days_back: Look-back window in days (default 30).

    Returns:
        LSTObservation with temp in Celsius and Fahrenheit, or
        ``fallback_used=True`` if no data found.
    """
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)
        radius_m = lake_radius_m(lake_name)

        items = _fetch_landsat_items(lat, lon, start_date, end_date, radius_m)
        if not items:
            return LSTObservation(
                lake_name=lake_name,
                fallback_used=True,
                error_msg="No Landsat scenes found in window",
            )

        best_lst = None
        best_date = None
        best_pixel_count = 0
        scene_count = len(items)

        for item in items[:5]:
            thermal_asset = item.assets.get(_LANDSAT_THERMAL_ASSET)
            qa_asset = item.assets.get(_LANDSAT_QA_ASSET)
            if thermal_asset is None:
                continue

            thermal_data = _read_windowed(thermal_asset.href, lat, lon, radius_m)
            if thermal_data is None:
                continue

            qa_data = None
            if qa_asset is not None:
                qa_data = _read_windowed(qa_asset.href, lat, lon, radius_m)

            lst_c = thermal_data * _LANDSAT_SCALE + _LANDSAT_OFFSET - 273.15

            if qa_data is not None:
                mask = _cloud_mask_landsat(qa_data)
                lst_c = np.ma.masked_where(~mask, lst_c)

            pixel_count = int(lst_c.count())
            if pixel_count < 5:
                continue

            mean_val = float(lst_c.mean())

            if best_lst is None or (
                item.datetime and (best_date is None or item.datetime.date() > best_date)
            ):
                best_lst = round(mean_val, 1)
                best_date = item.datetime.date() if item.datetime else None
                best_pixel_count = pixel_count

        if best_lst is None:
            return LSTObservation(
                lake_name=lake_name,
                fallback_used=True,
                error_msg="No valid thermal pixels in any scene",
            )

        return LSTObservation(
            lake_name=lake_name,
            temp_celsius=best_lst,
            temp_fahrenheit=round(celsius_to_fahrenheit(best_lst), 1),
            observation_date=best_date,
            scene_count=scene_count,
            pixel_count=best_pixel_count,
            fallback_used=False,
        )
    except Exception as exc:
        _log.warning("get_latest_lst (STAC) failed for %s: %s", lake_name, exc)
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
    """Fetch a historical LST time series for a lake via STAC.

    Each interval window finds the most recent scene and computes
    mean LST over the lake area.

    Args:
        lake_name:     Display name for labelling.
        lat:           Lake centroid latitude.
        lon:           Lake centroid longitude.
        days_back:     Total look-back window (default 180 days).
        interval_days: Size of each compositing window (default 16).

    Returns:
        List of LSTHistoryPoint sorted oldest-first.
    """
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)
        radius_m = lake_radius_m(lake_name)

        items = _fetch_landsat_items(lat, lon, start_date, end_date, radius_m, max_items=50)
        if not items:
            return []

        scene_dates: dict[date, pystac_client.Item] = {}
        for item in items:
            if item.datetime:
                scene_dates[item.datetime.date()] = item

        points: List[LSTHistoryPoint] = []
        cursor = start_date
        while cursor < end_date:
            window_end = min(cursor + timedelta(days=interval_days), end_date)
            best_item = None
            best_d = None
            for d, item in scene_dates.items():
                if cursor <= d < window_end:
                    if best_d is None or d > best_d:
                        best_d = d
                        best_item = item

            if best_item is not None:
                thermal_asset = best_item.assets.get(_LANDSAT_THERMAL_ASSET)
                qa_asset = best_item.assets.get(_LANDSAT_QA_ASSET)
                if thermal_asset is not None:
                    thermal_data = _read_windowed(thermal_asset.href, lat, lon, radius_m)
                    if thermal_data is not None:
                        lst_c = thermal_data * _LANDSAT_SCALE + _LANDSAT_OFFSET - 273.15

                        if qa_asset is not None:
                            qa_data = _read_windowed(qa_asset.href, lat, lon, radius_m)
                            if qa_data is not None:
                                mask = _cloud_mask_landsat(qa_data)
                                lst_c = np.ma.masked_where(~mask, lst_c)

                        pixel_count = int(lst_c.count())
                        if pixel_count >= 5:
                            mean_val = float(lst_c.mean())
                            window_mid = cursor + (window_end - cursor) / 2
                            points.append(
                                LSTHistoryPoint(
                                    observation_date=window_mid,
                                    temp_celsius=round(mean_val, 1),
                                    temp_fahrenheit=round(celsius_to_fahrenheit(mean_val), 1),
                                )
                            )

            cursor = window_end

        return sorted(points, key=lambda p: p.observation_date)
    except Exception as exc:
        _log.warning("get_lst_history (STAC) failed for %s: %s", lake_name, exc)
        return []


def get_lst_heatmap(
    lat: float,
    lon: float,
    radius_m: float,
    days_back: int = 30,
    min_temp_c: float = 5.0,
    max_temp_c: float = 30.0,
    image_size: int = 400,
) -> Optional[bytes]:
    """Generate a PNG heatmap of LST from the most recent Landsat scene.

    Unlike the GEE version which returns a URL, this returns the raw PNG
    bytes suitable for writing to a file or serving via HTTP.

    Args:
        lat:         Lake centroid latitude.
        lon:         Lake centroid longitude.
        radius_m:    Buffer radius in metres.
        days_back:   Look-back window in days.
        min_temp_c:  Minimum temperature for colour scale.
        max_temp_c:  Maximum temperature for colour scale.
        image_size:  Output image pixel width (square).

    Returns:
        PNG image bytes, or None if no data available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        items = _fetch_landsat_items(lat, lon, start_date, end_date, radius_m, max_items=5)
        if not items:
            return None

        for item in items:
            thermal_asset = item.assets.get(_LANDSAT_THERMAL_ASSET)
            qa_asset = item.assets.get(_LANDSAT_QA_ASSET)
            if thermal_asset is None:
                continue

            da = _read_windowed_xr(thermal_asset.href, lat, lon, radius_m)
            if da is None:
                continue

            lst_c = da * _LANDSAT_SCALE + _LANDSAT_OFFSET - 273.15

            if qa_asset is not None:
                qa_da = _read_windowed_xr(qa_asset.href, lat, lon, radius_m)
                if qa_da is not None:
                    mask = _cloud_mask_landsat(qa_da.values)
                    lst_c = lst_c.where(mask)

            data = lst_c.compute()
            if data.isnull().all():
                continue

            colors = [
                (0.0, "#000080"),
                (0.166, "#0000ff"),
                (0.333, "#00ffff"),
                (0.5, "#00ff00"),
                (0.666, "#ffff00"),
                (0.833, "#ff8000"),
                (1.0, "#ff0000"),
            ]
            cmap = LinearSegmentedColormap.from_list("lst", colors)

            fig, ax = plt.subplots(1, 1, figsize=(image_size / 100, image_size / 100), dpi=100)
            data.plot(ax=ax, cmap=cmap, vmin=min_temp_c, vmax=max_temp_c, add_colorbar=False)
            ax.set_axis_off()
            fig.patch.set_alpha(0)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)
            buf.seek(0)
            return buf.read()

        return None
    except Exception as exc:
        _log.warning("get_lst_heatmap (STAC) failed: %s", exc)
        return None


def get_ndci(
    lake_name: str,
    lat: float,
    lon: float,
    days_back: int = 30,
) -> NDCIObservation:
    """Fetch the most recent Sentinel-2 NDCI (chlorophyll-a proxy) via STAC.

    NDCI = (B05 - B04) / (B05 + B04)
    where B05 = red-edge (705 nm) and B04 = red (665 nm).

    Uses the Microsoft Planetary Computer (free, no auth required when
    ``planetary_computer`` package is installed). Falls back to Copernicus
    Data Space Ecosystem (requires CDSE_ACCESS_TOKEN env var).

    Args:
        lake_name: Display name for labelling.
        lat:       Lake centroid latitude.
        lon:       Lake centroid longitude.
        days_back: Look-back window in days (default 30).

    Returns:
        NDCIObservation with ndci_value and chlorophyll_category, or
        ``fallback_used=True`` if no data available.
    """
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)
        radius_m = lake_radius_m(lake_name)

        items = _fetch_sentinel2_items(lat, lon, start_date, end_date, radius_m)
        if not items:
            if not _HAS_PLANETARY_COMPUTER:
                return NDCIObservation(
                    lake_name=lake_name,
                    fallback_used=True,
                    error_msg="CDSE_ACCESS_TOKEN not set and planetary_computer not installed",
                )
            return NDCIObservation(
                lake_name=lake_name,
                fallback_used=True,
                error_msg="No Sentinel-2 scenes in window",
            )

        for item in items[:3]:
            b04_key, b05_key, scl_key = _s2_band_keys(item)
            if b04_key is None:
                continue

            b04_asset = item.assets.get(b04_key)
            b05_asset = item.assets.get(b05_key)
            if b04_asset is None or b05_asset is None:
                continue

            b04_data = _read_windowed(b04_asset.href, lat, lon, radius_m)
            if b04_data is None:
                continue

            b05_data = _read_windowed(
                b05_asset.href, lat, lon, radius_m,
                target_shape=b04_data.shape,
            )
            if b05_data is None:
                continue

            scl_data = None
            if scl_key and scl_key in item.assets:
                scl_data = _read_windowed(
                    item.assets[scl_key].href, lat, lon, radius_m,
                    target_shape=b04_data.shape,
                )

            if scl_data is not None:
                valid_mask = (
                    (scl_data != 3)
                    & (scl_data != 8)
                    & (scl_data != 9)
                    & (scl_data != 10)
                )
                b04_data = np.ma.masked_where(~valid_mask, b04_data)
                b05_data = np.ma.masked_where(~valid_mask, b05_data)

            denom = b05_data.astype(np.float64) + b04_data.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                ndci = np.ma.where(
                    denom != 0,
                    (b05_data.astype(np.float64) - b04_data.astype(np.float64)) / denom,
                    np.nan,
                )

            mean_ndci = float(np.nanmean(ndci.compressed())) if ndci.count() > 0 else np.nan
            if np.isnan(mean_ndci):
                continue

            ndci_f = round(mean_ndci, 3)
            obs_date = item.datetime.date() if item.datetime else None
            return NDCIObservation(
                lake_name=lake_name,
                ndci_value=ndci_f,
                chlorophyll_category=ndci_to_category(ndci_f),
                observation_date=obs_date,
                fallback_used=False,
            )

        return NDCIObservation(
            lake_name=lake_name,
            fallback_used=True,
            error_msg="No valid NDCI pixels in any scene",
        )
    except Exception as exc:
        _log.warning("get_ndci (STAC) failed for %s: %s", lake_name, exc)
        return NDCIObservation(
            lake_name=lake_name,
            fallback_used=True,
            error_msg=str(exc),
        )


# ---------------------------------------------------------------------------
# Map overlay grids (for folium ImageOverlay)
# ---------------------------------------------------------------------------


def grid_bounds(da: "xr.DataArray") -> list:
    """Return ``[[south, west], [north, east]]`` bounds of an EPSG:4326 grid."""
    lats = da["y"].values
    lons = da["x"].values
    return [
        [float(lats.min()), float(lons.min())],
        [float(lats.max()), float(lons.max())],
    ]


def get_lst_grid(
    lat: float,
    lon: float,
    radius_m: float,
    days_back: int = 45,
) -> tuple:
    """Cloud-masked surface temperature grid (°C) from the latest Landsat scene.

    Lake bounding boxes can straddle two Landsat path/rows, so the most
    recent scene may only clip a corner of the window and miss the lake
    entirely. Scenes are therefore screened: the first one whose grid
    covers the lake centroid with enough valid pixels wins; otherwise the
    best-covered candidate seen is returned.

    Returns ``(DataArray, observation_date)`` or ``(None, None)``.
    """
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        items = _fetch_landsat_items(lat, lon, start_date, end_date, radius_m, max_items=10)
        best_score = 0.0
        best_grid = None
        best_date = None
        for item in items:
            thermal_asset = item.assets.get(_LANDSAT_THERMAL_ASSET)
            qa_asset = item.assets.get(_LANDSAT_QA_ASSET)
            if thermal_asset is None:
                continue

            da = _read_windowed_xr(thermal_asset.href, lat, lon, radius_m)
            if da is None:
                continue

            lst_c = da * _LANDSAT_SCALE + _LANDSAT_OFFSET - 273.15

            if qa_asset is not None:
                qa_da = _read_windowed_xr(qa_asset.href, lat, lon, radius_m)
                if qa_da is not None and qa_da.shape == lst_c.shape:
                    mask = _cloud_mask_landsat(np.nan_to_num(qa_da.values, nan=0.0))
                    lst_c = lst_c.where(mask)

            lst_c = lst_c.compute()
            valid = int(lst_c.notnull().sum())
            if valid < 5:
                continue
            obs_date = item.datetime.date() if item.datetime else None

            lats = lst_c["y"].values
            lons = lst_c["x"].values
            covers_lake = (
                float(lats.min()) <= lat <= float(lats.max())
                and float(lons.min()) <= lon <= float(lons.max())
            )
            valid_frac = valid / max(lst_c.size, 1)

            # Newest scene that clearly covers the lake wins outright.
            if covers_lake and valid_frac >= 0.3:
                return lst_c, obs_date

            score = valid_frac if covers_lake else valid_frac * 0.1
            if score > best_score:
                best_score = score
                best_grid = lst_c
                best_date = obs_date

        return best_grid, best_date
    except Exception as exc:
        _log.warning("get_lst_grid (STAC) failed: %s", exc)
        return None, None


def get_ndci_grid(
    lat: float,
    lon: float,
    radius_m: float,
    days_back: int = 45,
) -> tuple:
    """NDCI grid from the latest low-cloud Sentinel-2 scene.

    NDCI = (B05 - B04) / (B05 + B04), masked via the SCL scene
    classification (clouds, shadows, cirrus). Returns
    ``(DataArray, observation_date)`` or ``(None, None)``.
    """
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)

        items = _fetch_sentinel2_items(lat, lon, start_date, end_date, radius_m)
        for item in items[:3]:
            b04_key, b05_key, scl_key = _s2_band_keys(item)
            if b04_key is None:
                continue
            b04_asset = item.assets.get(b04_key)
            b05_asset = item.assets.get(b05_key)
            if b04_asset is None or b05_asset is None:
                continue

            # B05 (red edge) is the coarser 20 m band — use its grid as the base.
            b05 = _read_windowed_xr(b05_asset.href, lat, lon, radius_m)
            b04 = _read_windowed_xr(b04_asset.href, lat, lon, radius_m)
            if b04 is None or b05 is None:
                continue
            if b04.shape != b05.shape:
                b04 = b04.interp_like(b05, method="nearest")

            ndci = (b05 - b04) / (b05 + b04)

            if scl_key and scl_key in item.assets:
                scl = _read_windowed_xr(item.assets[scl_key].href, lat, lon, radius_m)
                if scl is not None:
                    if scl.shape != b05.shape:
                        scl = scl.interp_like(b05, method="nearest")
                    valid = (scl != 3) & (scl != 8) & (scl != 9) & (scl != 10)
                    ndci = ndci.where(valid)

            ndci = ndci.where(np.isfinite(ndci)).compute()
            if int(ndci.notnull().sum()) < 5:
                continue
            obs_date = item.datetime.date() if item.datetime else None
            return ndci, obs_date
        return None, None
    except Exception as exc:
        _log.warning("get_ndci_grid (STAC) failed: %s", exc)
        return None, None


def get_overlay_grids(
    lake_name: str,
    lat: float,
    lon: float,
    radius_m: Optional[float] = None,
    days_back: int = 45,
) -> SatelliteOverlayGrids:
    """Fetch LST and NDCI grids for map overlays in one call."""
    if radius_m is None:
        radius_m = lake_radius_m(lake_name)
    lst_grid, lst_date = get_lst_grid(lat, lon, radius_m, days_back=days_back)
    ndci_grid, ndci_date = get_ndci_grid(lat, lon, radius_m, days_back=days_back)
    error = None
    if lst_grid is None and ndci_grid is None:
        error = "No cloud-free Landsat or Sentinel-2 scenes found in window"
    return SatelliteOverlayGrids(
        lake_name=lake_name,
        lst_grid=lst_grid,
        lst_date=lst_date,
        ndci_grid=ndci_grid,
        ndci_date=ndci_date,
        error_msg=error,
    )
