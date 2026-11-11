"""Network access (only used by `build`; the demo and tests run offline from the committed derived tables).

Sources (cite when you use the output):
  DEM: swissALTIRegio 10 m, Federal Office of Topography swisstopo, cloud-optimised GeoTIFF on data.geo.admin.ch
  Rail network: 'Railway network' basic data set, Federal Office of Transport FOT, data.geo.admin.ch
  Weather: MeteoSwiss SwissMetNet open data (CC BY 4.0), data.geo.admin.ch/ch.meteoschweiz.ogd-smn
  Snow/wind: WSL Institute for Snow and Avalanche Research SLF, IMIS public API (CC BY 4.0), last 7 days only
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .climate import parse_smn_hourly
from .terrain import Grid

DEM_URL = "https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/swissaltiregio_2056_5728.tif"
RAIL_URL = "https://data.geo.admin.ch/ch.bav.schienennetz/schienennetz/schienennetz_2056_de.gpkg"
SMN_BASE = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
SMN_PERIOD = "historical_2020-2029"
IMIS_BASE = "https://measurement-api.slf.ch/public/api/imis"
UA = {"User-Agent": "alpine-line-exposure-map/0.1 (open-source research tool)"}


def get(url: str, timeout: int = 180, retries: int = 3) -> bytes:
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # network hiccup: back off and retry
            last = exc
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"download failed: {url}: {last}")


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(get(url))
        tmp.replace(dest)
    return dest


def smn_hourly(abbr: str, cache: Path) -> pd.DataFrame:
    """Hourly 2020-2025 extract for one station; the raw CSV is parsed in memory and only a compact copy is cached."""
    low = abbr.lower()
    out = cache / "smn" / f"{low}.csv.gz"
    if out.exists():
        df = pd.read_csv(out)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df
    raw = get(f"{SMN_BASE}/{low}/ogd-smn_{low}_h_{SMN_PERIOD}.csv").decode("utf-8", errors="replace")
    df = parse_smn_hourly(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4g", compression="gzip")
    return df


def dem_window(e: float, n: float, half: float, cache: Path, key: str, res: float = 10.0) -> Grid:
    """Read a square DEM window (centre e,n; half-size `half` m) from the national COG via HTTP range requests."""
    return dem_bounds(e - half, n - half, e + half, n + half, cache, key, res)


def dem_bounds(west: float, south: float, east: float, north: float, cache: Path, key: str, res: float = 10.0) -> Grid:
    """Read a DEM window by LV95 bounds (snapped outward to the 10 m grid); cached as int16 decimetres."""
    out = cache / "dem" / f"{key}.npz"
    if out.exists():
        g = Grid.load(out)
        if not np.nanmin(g.z) < -500:  # files written by an early int16 version overflowed above 3276 m: refetch
            return g
    import rasterio  # optional dependency, only needed to rebuild the derived data
    from rasterio.windows import from_bounds

    west, south = np.floor(west / res) * res, np.floor(south / res) * res
    east, north = np.ceil(east / res) * res, np.ceil(north / res) * res
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif", "GDAL_NUM_THREADS": "1"}
    with rasterio.Env(**env), rasterio.open("/vsicurl/" + DEM_URL) as ds:
        win = from_bounds(west, south, east, north, ds.transform)
        z = ds.read(1, window=win, boundless=True, fill_value=np.nan, masked=False).astype(float)
        nod = ds.nodata
    if nod is not None:
        z[np.isclose(z, nod) | (z > 1e30)] = np.nan
    g = Grid(z, float(west), float(north), res)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.save(out)
    return Grid.load(out)  # same decimetre rounding as later runs


def imis_stations(cache: Path) -> list[dict]:
    dest = download(f"{IMIS_BASE}/stations", cache / "imis" / "stations.json")
    return json.loads(dest.read_text(encoding="utf-8"))


def imis_week(code: str, cache: Path) -> list[dict]:
    dest = download(f"{IMIS_BASE}/station/{code}/measurements?period_in_days=7", cache / "imis" / f"{code}.json")
    return json.loads(dest.read_text(encoding="utf-8"))
