"""Station climatology from open data: wind speed, gusts, storm directions and snow-on-ground.

Inputs are the MeteoSwiss SwissMetNet open-data CSVs (semicolon separated, hourly 'h' files) and the SLF IMIS
public API (30-minute records, last 7 days only). All speeds are m/s; directions are degrees the wind comes FROM.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

WINTER_MONTHS = (11, 12, 1, 2, 3, 4)
HOURLY_COLS = {
    "fkl010h0": "wind",  # scalar mean wind, m/s
    "fkl010h1": "gust",  # 1 s gust peak, hourly max, m/s
    "dkl010h0": "dir",  # mean direction, degrees
    "htoauths": "snow_cm",  # automatic snow depth, cm
    "tre200h0": "temp",  # 2 m air temperature, C
}


def read_smn_meta(path_or_text) -> pd.DataFrame:
    """SwissMetNet station list -> abbr, name, elev, e, n, lat, lon, exposition."""
    if isinstance(path_or_text, (str, Path)) and Path(str(path_or_text)).exists():
        raw = Path(path_or_text).read_bytes()
    else:
        raw = path_or_text.encode("utf-8") if isinstance(path_or_text, str) else bytes(path_or_text)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep=";")
    out = pd.DataFrame(
        {
            "abbr": df["station_abbr"].astype(str),
            "name": df["station_name"].astype(str),
            "elev": pd.to_numeric(df["station_height_masl"], errors="coerce"),
            "e": pd.to_numeric(df["station_coordinates_lv95_east"], errors="coerce"),
            "n": pd.to_numeric(df["station_coordinates_lv95_north"], errors="coerce"),
            "lat": pd.to_numeric(df["station_coordinates_wgs84_lat"], errors="coerce"),
            "lon": pd.to_numeric(df["station_coordinates_wgs84_lon"], errors="coerce"),
            "exposition": df.get("station_exposition_en", pd.Series([""] * len(df))).fillna("").astype(str),
        }
    )
    return out.dropna(subset=["elev", "e", "n"]).reset_index(drop=True)


def parse_smn_hourly(path_or_text) -> pd.DataFrame:
    """Parse a SwissMetNet hourly CSV into time_utc + the columns in HOURLY_COLS (renamed). Missing columns -> NaN."""
    if isinstance(path_or_text, (str, Path)) and Path(str(path_or_text)).exists():
        src = path_or_text
    else:
        src = io.StringIO(path_or_text)
    head = pd.read_csv(src, sep=";", nrows=0)
    if "reference_timestamp" not in head.columns:
        raise ValueError("not a SwissMetNet CSV: no reference_timestamp column")
    if isinstance(src, io.StringIO):
        src.seek(0)
    use = ["reference_timestamp"] + [c for c in HOURLY_COLS if c in head.columns]
    df = pd.read_csv(src, sep=";", usecols=use, dtype={c: "float64" for c in use[1:]})
    out = pd.DataFrame({"time_utc": pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M", utc=True)})
    for src_col, name in HOURLY_COLS.items():
        out[name] = df[src_col].to_numpy(float) if src_col in df.columns else np.nan
    return out


def dir_bins(direction_deg, n_dirs: int = 16) -> np.ndarray:
    """Sector index for directions (sector 0 is centred on north). NaN -> -1."""
    d = np.asarray(direction_deg, dtype=float)
    width = 360.0 / n_dirs
    idx = np.floor(((np.nan_to_num(d) + width / 2.0) % 360.0) / width).astype(int)
    return np.where(np.isfinite(d), idx % n_dirs, -1)


def rose(direction_deg, n_dirs: int = 16) -> np.ndarray:
    """Normalised direction histogram (sums to 1, or all zeros if no valid direction)."""
    b = dir_bins(direction_deg, n_dirs)
    counts = np.bincount(b[b >= 0], minlength=n_dirs).astype(float)
    return counts / counts.sum() if counts.sum() > 0 else counts


def is_winter(t: pd.Series) -> np.ndarray:
    return t.dt.month.isin(WINTER_MONTHS).to_numpy()


def station_stats(h: pd.DataFrame, n_dirs: int = 16, storm_quantile: float = 0.9, snow_cm: float = 20.0,
                  drift_wind: float = 8.0) -> dict:
    """Per-station climatology from an hourly frame (output of parse_smn_hourly)."""
    w = is_winter(h["time_utc"])
    wind = h["wind"].to_numpy(float)
    gust = h["gust"].to_numpy(float)
    ok = np.isfinite(wind)
    wok = ok & w
    out = {
        "hours": int(len(h)),
        "wind_coverage": float(ok.mean()) if len(h) else 0.0,
        "wind_mean_all": float(np.nanmean(wind)) if ok.any() else np.nan,
        "wind_mean_winter": float(wind[wok].mean()) if wok.any() else np.nan,
        "gust_p95_winter": float(np.nanpercentile(gust[w], 95)) if np.isfinite(gust[w]).any() else np.nan,
    }
    rose_storm = np.zeros(n_dirs)
    if wok.sum() > 100:
        thr = np.quantile(wind[wok], storm_quantile)
        sel = wok & (wind >= thr)
        rose_storm = rose(h["dir"].to_numpy(float)[sel], n_dirs)
    for k in range(n_dirs):
        out[f"storm_rose_{k:02d}"] = float(rose_storm[k])
    # snow on ground: daily 06 UTC value of the hourly automatic snow depth
    snow = h[["time_utc", "snow_cm"]].dropna()
    snow = snow[(snow["time_utc"].dt.hour == 6) & is_winter(snow["time_utc"])]
    n_winter_days = int(w.sum() / 24)
    if n_winter_days and len(snow) >= 0.5 * n_winter_days:
        out["snow_day_frac"] = float((snow["snow_cm"] >= snow_cm).mean())
    else:
        out["snow_day_frac"] = np.nan
    # heuristic drifting-snow hours: snow on ground, freezing, mean wind above a rule-of-thumb threshold
    sd = h["snow_cm"].to_numpy(float)
    tt = h["temp"].to_numpy(float)
    have = np.isfinite(sd) & np.isfinite(tt) & ok
    if have.sum() > 1000:
        drift = have & (sd >= 10) & (tt < 0) & (wind >= drift_wind)
        years = have.sum() / (24 * 365.25)
        out["drift_hours_per_year"] = float(drift.sum() / years)
        out["drift_hours_total"] = int(drift.sum())
        dr = rose(h["dir"].to_numpy(float)[drift], n_dirs)
        for k in range(n_dirs):
            out[f"drift_rose_{k:02d}"] = float(dr[k])
    return out


def rose_columns(prefix: str, n_dirs: int = 16) -> list[str]:
    return [f"{prefix}_{k:02d}" for k in range(n_dirs)]


def pooled_rose(df: pd.DataFrame, prefix: str = "storm_rose", n_dirs: int = 16, mask=None) -> np.ndarray:
    """Average of the per-station roses (rows with an all-zero rose are skipped); normalised to 1."""
    r = df[rose_columns(prefix, n_dirs)].to_numpy(float)
    keep = np.isfinite(r).all(axis=1) & (r.sum(axis=1) > 0)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    if not keep.any():
        raise ValueError("no station with a valid rose")
    m = r[keep].mean(axis=0)
    return m / m.sum()


def snow_logistic(elev, snow_frac, iters: int = 200):
    """Fit snow_frac ~ 1/(1+exp(-(elev-e50)/s)) by least squares on a grid + refinement. Returns (e50, s)."""
    elev = np.asarray(elev, dtype=float)
    y = np.asarray(snow_frac, dtype=float)
    ok = np.isfinite(elev) & np.isfinite(y)
    elev, y = elev[ok], y[ok]
    if len(y) < 3:
        raise ValueError("need >= 3 stations with snow data")
    best = (np.inf, 1500.0, 300.0)
    for e50 in np.arange(200.0, 3500.0, 25.0):
        for s in np.arange(50.0, 1000.0, 25.0):
            p = 1.0 / (1.0 + np.exp(-(elev - e50) / s))
            sse = float(((p - y) ** 2).sum())
            if sse < best[0]:
                best = (sse, e50, s)
    return best[1], best[2]


def snow_factor(elev, e50: float, s: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(elev, dtype=float) - e50) / s))


def parse_imis(payload: list[dict]) -> pd.DataFrame:
    """SLF IMIS 30-min records -> time_utc, wind (VW_30MIN_MEAN), gust (VW_30MIN_MAX), dir (DW_30MIN_MEAN), snow_cm."""
    if not isinstance(payload, list):
        raise ValueError("IMIS payload must be a list of records")
    rows = []
    for r in payload:
        if not isinstance(r, dict) or "measure_date" not in r:
            continue
        hs = r.get("HS")
        rows.append(
            {
                "time_utc": r["measure_date"],
                "wind": r.get("VW_30MIN_MEAN"),
                "gust": r.get("VW_30MIN_MAX"),
                "dir": r.get("DW_30MIN_MEAN"),
                "snow_cm": hs,
            }
        )
    df = pd.DataFrame(rows, columns=["time_utc", "wind", "gust", "dir", "snow_cm"])
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    for c in ("wind", "gust", "dir", "snow_cm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[(df["wind"] < 0) | (df["wind"] > 80), "wind"] = np.nan
    return df.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
