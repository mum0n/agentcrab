"""Area of Interest (AOI) functions for AgentCrab.

Provides functions to read assimilated environmental data (temperature, bathymetry)
from local Zarr stores, and to compute habitat suitability indices.

Data sources:
- World Ocean Atlas 2023 (WOA23): climatological temperature/salinity
- NWA bathymetry: high-resolution bathymetry for the Northwest Atlantic
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_ENVIRONMENT_ROOT = Path(__file__).resolve().parents[2] / "data" / "environmental"

WOA23_TEMP_PATH = "woa23/woa23_temperature_00_0.25deg.zarr"
NWA_BATHY_PATH = "nwa/bathymetry.zarr"


def _resolve_environment_root() -> Path:
    """Resolve environmental data from this project only."""
    return _ENVIRONMENT_ROOT


@lru_cache(maxsize=2)
def _open_zarr(path: Path):
    """Open a Zarr store, returning None if not available."""
    try:
        import zarr
        return zarr.open(str(path), mode="r")
    except Exception:
        return None


def read_woa23_temperature(lat: float, lon: float, depth: float) -> Optional[float]:
    """Read temperature from WOA23 zarr store at given lat/lon/depth.

    Returns None if data is not available.
    """
    root = _resolve_environment_root()
    store_path = root / WOA23_TEMP_PATH
    if not store_path.exists():
        return None
    store = _open_zarr(store_path)
    if store is None:
        return None
    try:
        t_an = store["t_an"]
        lats = store["lat"][:]
        lons = store["lon"][:]
        depths = store["depth"][:]
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))
        depth_idx = int(np.argmin(np.abs(depths - depth)))
        temp = float(t_an[0, depth_idx, lat_idx, lon_idx])
        if temp != temp:  # NaN check
            return None
        return temp
    except Exception:
        return None


def read_bathymetry(lat: float, lon: float) -> Optional[float]:
    """Read bathymetry from NWA store at given lat/lon.

    Returns depth in meters, or None if not available.
    """
    root = _resolve_environment_root()
    store_path = root / NWA_BATHY_PATH
    if not store_path.exists():
        return None
    store = _open_zarr(store_path)
    if store is None:
        return None
    try:
        bathy = store["depth"] if "depth" in store else store["elevation"]
        lats = store["lat"][:] if "lat" in store else store["y"][:]
        lons = store["lon"][:] if "lon" in store else store["x"][:]
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))
        val = float(bathy[lat_idx, lon_idx])
        if val != val:  # NaN
            return None
        return abs(val)  # depth as positive
    except Exception:
        return None


def aoi_bottom_temperature(depth: float, lat: float) -> float:
    """Evaluate climatological ambient bottom temperature.

    First tries WOA23 data (using default Scotian Shelf lon=-64), then falls back
    to empirical formula.

    Formula: T_bottom = 2.0 + 2.5 * exp(-((z - 150.0) / 60.0)^2) + 0.3 * (45.0 - phi)
    """
    return aoi_bottom_temperature_at(-64.0, lat, depth)


def aoi_bottom_temperature_at(lon: float, lat: float, depth: float) -> float:
    """Evaluate bottom temperature at a specific lon/lat/depth."""
    temp = read_woa23_temperature(lat, lon, depth)
    if temp is not None:
        return temp
    # Fallback to empirical formula
    z = float(depth)
    phi = float(lat)
    t = 2.0 + 2.5 * math.exp(-((z - 150.0) / 60.0) ** 2) + 0.3 * (45.0 - phi)
    return t


def aoi_habitat_suitability(temp: float, depth: float, mult: float = 1.0, offset: float = 0.0) -> float:
    """Evaluate Habitat Suitability Index (HSI).

    Optimum thermal window: 0-4 C
    Optimum depth window: 60-280 m
    """
    t_opt = math.exp(-0.5 * ((float(temp) - 2.0) / 2.2) ** 2)
    z_opt = math.exp(-0.5 * ((float(depth) - 150.0) / 75.0) ** 2)
    raw_hsi = t_opt * z_opt
    realized = raw_hsi * float(mult) + float(offset)
    return max(0.0, min(1.0, realized))


def aoi_domain_bounds(lon: float, lat: float, span_meters: float = 1000.0):
    """Compute geographic bounding boxes centered at (lon, lat)."""
    half_m = float(span_meters) / 2.0
    lat_deg_len = 111139.0
    lon_deg_len = max(1000.0, 111139.0 * math.cos(math.radians(float(lat))))
    dlat = half_m / lat_deg_len
    dlon = half_m / lon_deg_len
    return {
        "lon": [float(lon) - dlon, float(lon) + dlon],
        "lat": [float(lat) - dlat, float(lat) + dlat],
    }


def environmental_input_status() -> dict:
    """Check availability of environmental data sources."""
    root = _resolve_environment_root()
    status = {
        "environmental_data_dir": str(root),
        "woa23_temperature": (root / WOA23_TEMP_PATH).exists(),
        "nwa_bathymetry": (root / NWA_BATHY_PATH).exists(),
    }
    return status
