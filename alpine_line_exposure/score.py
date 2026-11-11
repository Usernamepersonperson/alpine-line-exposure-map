"""Per-section scores along the line and the priority ranking.

wind_pred     : winter mean wind (m/s) predicted from elevation + Sx exposure, with the regression fitted on the
                SwissMetNet stations (the part that is validated).
crosswind     : share of storm-hour wind directions that blow across the line (|sin| of the angle to the track).
wind_score    : wind_pred * crosswind (m/s of storm wind across the track; crosswind weighting not validated).
drift_index   : 0-100 heuristic for snow deposition on the track: shelter angle Sx where drifting-snow winds cross
                the line, boosted when the terrain 200 m upwind is exposed (a snow source), times the share of winter
                days with >= 20 cm of snow on the ground at that elevation. Not validated against drift observations.
priority      : mean of the percentile ranks of wind_score and drift_index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .climate import rose_columns, snow_factor
from .terrain import direction_centers

N_DIRS = 16


def crosswind_matrix(bearing_deg, n_dirs: int = N_DIRS) -> np.ndarray:
    """|sin(angle between wind direction and track bearing)| -> (n_sections, n_dirs); 1 = square across the track."""
    d = direction_centers(n_dirs)
    return np.abs(np.sin(np.radians(d[None, :] - np.asarray(bearing_deg, dtype=float)[:, None])))


def pct_rank(x) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rank(pct=True, method="average").to_numpy()


def score_sections(sec: pd.DataFrame, coef: np.ndarray, w_storm: np.ndarray, w_drift: np.ndarray, snow_e50: float,
                   snow_s: float, dmax: int = 300) -> pd.DataFrame:
    """Add score columns to the section terrain table. coef = [b0, b_elev_per_km, b_exposure]."""
    w_storm = np.asarray(w_storm, dtype=float) / np.sum(w_storm)
    w_drift = np.asarray(w_drift, dtype=float) / np.sum(w_drift)
    sx = sec[rose_columns(f"sx{dmax}", N_DIRS)].to_numpy(float)
    sxup = sec[rose_columns("sxup300", N_DIRS)].to_numpy(float)
    cross = crosswind_matrix(sec["bearing_deg"])
    elev = sec["dem_elev"].to_numpy(float)
    out = sec[["section", "km_from", "km_to", "e", "n", "bearing_deg", "dem_elev", "slope_deg"]].copy()
    out["exposure"] = -(sx @ w_storm)
    out["wind_pred"] = coef[0] + coef[1] * elev / 1000.0 + coef[2] * out["exposure"].to_numpy()
    out["crosswind"] = cross @ w_storm
    out["wind_score"] = out["wind_pred"] * out["crosswind"]
    shelter = np.clip(np.nan_to_num(sx), 0, None)
    source = 1.0 + np.clip(np.nan_to_num(-sxup), 0, None) / 5.0
    raw = (shelter * source * cross) @ w_drift * snow_factor(elev, snow_e50, snow_s)
    out["snow_days_frac"] = snow_factor(elev, snow_e50, snow_s)
    out["drift_index"] = 100.0 * raw / raw.max() if np.nanmax(raw) > 0 else 0.0
    out["priority"] = 100.0 * (pct_rank(out["wind_score"]) + pct_rank(out["drift_index"])) / 2.0
    # DEM grade between neighbours: a rail line cannot climb much more than ~12 %, so larger values mark sections
    # where the line is probably in a tunnel, gallery or deep cutting (or the DEM and line disagree)
    z = pd.Series(elev).interpolate(limit_direction="both").to_numpy()
    km = ((out["km_from"] + out["km_to"]) / 2).to_numpy()
    grade = np.gradient(z, km * 1000.0) if len(z) > 1 else np.zeros_like(z)
    out["dem_grade_pct"] = 100.0 * grade
    out["grade_flag"] = np.abs(out["dem_grade_pct"]) > 15.0
    return out


def top_n(scored: pd.DataFrame, col: str, n: int = 10, min_gap_km: float = 0.3) -> pd.DataFrame:
    """Top-n sections by `col`, skipping sections within `min_gap_km` of an already picked one (spread the list)."""
    order = scored.sort_values(col, ascending=False)
    picked: list[int] = []
    centers = ((scored["km_from"] + scored["km_to"]) / 2).to_numpy()
    for i in order.index:
        if not np.isfinite(scored.at[i, col]):
            continue
        if all(abs(centers[i] - centers[j]) >= min_gap_km for j in picked):
            picked.append(i)
        if len(picked) == n:
            break
    res = scored.loc[picked].copy()
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    return res


def hotspots(scored: pd.DataFrame, col: str = "priority", quantile: float = 0.9) -> pd.DataFrame:
    """Merge consecutive sections above the `quantile` of `col` into stretches."""
    thr = np.nanquantile(scored[col], quantile)
    hot = (scored[col] >= thr).to_numpy()
    rows = []
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and hot[j + 1]:
                j += 1
            part = scored.iloc[i : j + 1]
            rows.append(
                {
                    "km_from": float(part["km_from"].iloc[0]),
                    "km_to": float(part["km_to"].iloc[-1]),
                    "n_sections": int(len(part)),
                    "max_" + col: float(part[col].max()),
                    "mean_wind_score": float(part["wind_score"].mean()),
                    "max_drift_index": float(part["drift_index"].max()),
                    "elev_m": float(part["dem_elev"].mean()),
                }
            )
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows).sort_values("max_" + col, ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
