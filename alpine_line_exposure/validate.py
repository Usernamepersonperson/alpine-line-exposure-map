"""Does the terrain exposure index explain observed wind at stations it was not fitted on?

Leave-one-station-out: for every held-out station the storm-direction weights (pooled from high stations) and the
regression coefficients are re-estimated without it, then its wind is predicted. Compared models:
  elevation only (baseline)  vs  elevation + Sx exposure  (and a few variants, reported together).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .climate import pooled_rose, rose_columns
from .terrain import weighted_exposure

N_DIRS = 16


def rankdata(a) -> np.ndarray:
    return pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy()


def spearman(a, b) -> float:
    """Spearman rank correlation on the pairs where both values are finite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with intercept; returns [b0, b1, ...]."""
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + np.asarray(X, dtype=float) @ coef[1:]


def exposure(df: pd.DataFrame, dmax: int, weights: np.ndarray) -> np.ndarray:
    return weighted_exposure(df[rose_columns(f"sx{dmax}", N_DIRS)].to_numpy(float), weights)


class _Arrays:
    """Columns pulled out once, so the leave-one-out loop is plain numpy."""

    def __init__(self, df: pd.DataFrame):
        self.elev = df["elev"].to_numpy(float)[:, None] / 1000.0
        self.tpi = df["tpi500"].to_numpy(float)
        self.sx = {d: df[rose_columns(f"sx{d}", N_DIRS)].to_numpy(float) for d in (100, 300, 1000)}


def _features(a: _Arrays, model: str, weights: np.ndarray) -> np.ndarray:
    if model == "elevation":
        return a.elev
    parts = [a.elev]
    for term in model.split("+")[1:]:
        if term.startswith("sx"):
            d = int(term.removeprefix("sx").split("_")[0])
            w = np.ones(N_DIRS) if term.endswith("_uniform") else weights
            parts.append(weighted_exposure(a.sx[d], w))
        elif term == "tpi500":
            parts.append(a.tpi)
        else:
            raise ValueError(f"unknown model {model}")
    if len(parts) == 1:
        raise ValueError(f"unknown model {model}")
    return np.column_stack(parts)


# pre-registered models; the last one was added after seeing that TPI did well (reported as post-hoc)
MODELS = ("elevation", "elevation+sx300", "elevation+sx100", "elevation+sx1000", "elevation+sx300_uniform", "elevation+tpi500",
          "elevation+sx300+tpi500")


def loo(df: pd.DataFrame, target: str, rose_prefix: str, rose_mask, models=MODELS) -> pd.DataFrame:
    """Leave-one-out predictions for each model. `rose_mask` selects the stations that define storm directions."""
    y = df[target].to_numpy(float)
    mask = np.asarray(rose_mask, dtype=bool)
    arr = _Arrays(df)
    roses = df[rose_columns(rose_prefix, N_DIRS)].to_numpy(float)
    valid = np.isfinite(roses).all(axis=1) & (roses.sum(axis=1) > 0) & mask
    out = {"abbr": df["abbr"].to_numpy(), "obs": y}
    for m in models:
        pred = np.full(len(df), np.nan)
        for i in range(len(df)):
            train = np.ones(len(df), dtype=bool)
            train[i] = False
            keep = valid & train
            if not keep.any():
                raise ValueError("no station with a valid rose")
            w = roses[keep].mean(axis=0)
            X = _features(arr, m, w / w.sum())
            coef = ols(X[train], y[train])
            pred[i] = predict(coef, X[i : i + 1])[0]
        out[m] = pred
    return pd.DataFrame(out)


def metrics(pred: pd.DataFrame, models=MODELS, n_boot: int = 2000, seed: int = 7) -> pd.DataFrame:
    y = pred["obs"].to_numpy(float)
    base_err = np.abs(pred["elevation"].to_numpy(float) - y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    rows = []
    for m in models:
        p = pred[m].to_numpy(float)
        err = p - y
        rmse = float(np.sqrt(np.mean(err**2)))
        base_rmse = float(np.sqrt(np.mean(base_err**2)))
        # paired bootstrap of the RMSE difference vs the elevation-only baseline
        d = np.sqrt((err[idx] ** 2).mean(axis=1)) - np.sqrt((base_err[idx] ** 2).mean(axis=1))
        rows.append(
            {
                "model": m,
                "loo_rmse": rmse,
                "loo_mae": float(np.mean(np.abs(err))),
                "loo_r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
                "spearman_pred_obs": spearman(p, y),
                "rmse_change_vs_elev_pct": 100.0 * (rmse - base_rmse) / base_rmse,
                "rmse_diff_ci95_lo": float(np.percentile(d, 2.5)),
                "rmse_diff_ci95_hi": float(np.percentile(d, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, target: str, rose_prefix: str, rose_mask) -> dict:
    """Full validation summary for one station set."""
    df = df.reset_index(drop=True)
    pred = loo(df, target, rose_prefix, rose_mask)
    met = metrics(pred)
    w_all = pooled_rose(df, rose_prefix, N_DIRS, rose_mask)
    e300 = exposure(df, 300, w_all)
    y = df[target].to_numpy(float)
    resid_elev = y - pred["elevation"].to_numpy(float)
    return {
        "n_stations": int(len(df)),
        "target": target,
        "spearman_elev_obs": spearman(df["elev"], y),
        "spearman_sx300_exposure_obs": spearman(e300, y),
        "spearman_sx300_exposure_vs_elev_residual": spearman(e300, resid_elev),
        "storm_weights": [round(float(v), 4) for v in w_all],
        "models": met,
        "predictions": pred.assign(exposure_sx300=e300, elev=df["elev"].to_numpy(float)),
    }
