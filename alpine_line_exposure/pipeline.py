"""Offline demo: validation at stations + section ranking + figures, from the committed tables in data/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .climate import pooled_rose, rose_columns, snow_logistic
from .score import hotspots, score_sections, top_n
from .terrain import Grid
from .validate import exposure, ols, spearman, validate

N_DIRS = 16
REF_STATION = "GUE"  # nearest crest weather station to the line, used for its storm and drift directions
MIN_COVERAGE = 0.7


def usable_smn(smn: pd.DataFrame) -> pd.DataFrame:
    sxcols = [c for d in (100, 300, 1000) for c in rose_columns(f"sx{d}", N_DIRS)]
    ok = (smn["wind_coverage"] >= MIN_COVERAGE) & smn["wind_mean_winter"].notna() & smn[sxcols].notna().all(axis=1)
    ok &= smn["tpi500"].notna() & (smn[rose_columns("storm_rose", N_DIRS)].sum(axis=1) > 0)
    return smn[ok].reset_index(drop=True)


def usable_imis(imis: pd.DataFrame) -> pd.DataFrame:
    sxcols = [c for d in (100, 300, 1000) for c in rose_columns(f"sx{d}", N_DIRS)]
    ok = imis["wind_mean_week"].notna() & imis[sxcols].notna().all(axis=1) & imis["tpi500"].notna()
    return imis[ok].reset_index(drop=True)


def _models_json(v: dict) -> dict:
    out = {k: v[k] for k in v if k not in ("models", "predictions")}
    out["models"] = v["models"].round(4).to_dict(orient="records")
    return out


def demo(data: Path, out: Path, docs: Path, top_count: int = 10, figures: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    smn = usable_smn(pd.read_csv(data / "stations_smn.csv"))
    sec = pd.read_csv(data / "sections_terrain.csv")
    crest = (smn["elev"] >= 2000).to_numpy()

    v_mean = validate(smn, "wind_mean_winter", "storm_rose", crest)
    v_gust = validate(smn.dropna(subset=["gust_p95_winter"]), "gust_p95_winter", "storm_rose",
                      smn.dropna(subset=["gust_p95_winter"])["elev"].to_numpy() >= 2000)
    alp = smn[smn["elev"] >= 1000].reset_index(drop=True)
    v_alp = validate(alp, "wind_mean_winter", "storm_rose", (alp["elev"] >= 2000).to_numpy())
    results = {"smn_wind_mean_winter": v_mean, "smn_gust_p95_winter": v_gust, "smn_alpine_ge1000m_wind_mean_winter": v_alp}
    imis_path = data / "stations_imis.csv"
    if imis_path.exists():
        imis = usable_imis(pd.read_csv(imis_path))
        if len(imis) >= 10:
            imis = imis.assign(elev=imis["elev"].astype(float))
            results["imis_week_wind_mean"] = validate(imis, "wind_mean_week", "week_rose", (imis["type"] == "WIND").to_numpy())

    rows = []
    for name, v in results.items():
        m = v["models"].copy()
        m.insert(0, "dataset", name)
        m.insert(1, "n_stations", v["n_stations"])
        rows.append(m)
    pd.concat(rows).round(4).to_csv(out / "validation_models.csv", index=False)
    v_mean["predictions"].round(4).to_csv(out / "smn_loo_predictions.csv", index=False)

    # wind model for the line: fitted on all usable stations with regional storm weights
    w_regional = pooled_rose(smn, "storm_rose", N_DIRS, crest)
    X = np.column_stack([smn["elev"].to_numpy(float) / 1000.0, exposure(smn, 300, w_regional)])
    coef = ols(X, smn["wind_mean_winter"].to_numpy(float))
    ref = pd.read_csv(data / "stations_smn.csv").set_index("abbr").loc[REF_STATION]
    w_storm = ref[rose_columns("storm_rose", N_DIRS)].to_numpy(float)
    drift_cols = rose_columns("drift_rose", N_DIRS)
    use_drift = all(c in ref.index for c in drift_cols) and float(ref.get("drift_hours_total", 0) or 0) >= 50
    w_drift = ref[drift_cols].to_numpy(float) if use_drift else w_storm
    snow = smn.dropna(subset=["snow_day_frac"])
    e50, s = snow_logistic(snow["elev"], snow["snow_day_frac"])

    scored = score_sections(sec, coef, w_storm, w_drift, e50, s)
    scored.round(4).to_csv(out / "sections_ranked.csv", index=False)
    cols = ["rank", "section", "km_from", "km_to", "dem_elev", "bearing_deg", "wind_pred", "crosswind", "wind_score",
            "drift_index", "priority", "grade_flag"]
    top = top_n(scored, "priority", top_count)
    top[cols].round(3).to_csv(out / "top_priority.csv", index=False)
    top_n(scored, "wind_score", top_count)[cols].round(3).to_csv(out / "top_wind.csv", index=False)
    top_n(scored, "drift_index", top_count)[cols].round(3).to_csv(out / "top_drift.csv", index=False)
    hs = hotspots(scored)
    hs.round(3).to_csv(out / "hotspots.csv", index=False)

    # robustness: would the simpler TPI model (which validated better) rank the line differently?
    Xt = np.column_stack([smn["elev"].to_numpy(float) / 1000.0, smn["tpi500"].to_numpy(float)])
    coef_t = ols(Xt, smn["wind_mean_winter"].to_numpy(float))
    wind_t = coef_t[0] + coef_t[1] * sec["dem_elev"].to_numpy(float) / 1000.0 + coef_t[2] * sec["tpi500"].to_numpy(float)
    alt = scored.assign(wind_score=wind_t * scored["crosswind"])
    km_w = top_n(scored, "wind_score", top_count)["km_from"].to_numpy()
    km_t = top_n(alt, "wind_score", top_count)["km_from"].to_numpy()
    robust = {
        "spearman_section_wind_pred_sx_vs_tpi": spearman(scored["wind_pred"], wind_t),
        "top_wind_within_300m_sx_vs_tpi": int(sum(np.min(np.abs(km_t - k)) <= 0.3 for k in km_w)),
        "top_n": top_count,
    }

    met = v_mean["models"].set_index("model")
    summary = {
        "line_km": round(float(sec["km_to"].max()), 2),
        "n_sections": int(len(sec)),
        "section_m": round(float((sec["km_to"] - sec["km_from"]).median() * 1000), 1),
        "reference_station": REF_STATION,
        "ref_storm_rose": [round(float(x), 4) for x in w_storm],
        "ref_drift_rose_used": bool(use_drift),
        "ref_drift_hours_per_year": float(ref.get("drift_hours_per_year", np.nan)),
        "wind_model_coef": {"intercept": float(coef[0]), "per_km_elev": float(coef[1]), "per_deg_exposure": float(coef[2])},
        "snow_logistic": {"elev_50pct_m": float(e50), "scale_m": float(s), "n_stations": int(len(snow))},
        "validation": {k: _models_json(v) for k, v in results.items()},
        "headline": {
            "n_stations": v_mean["n_stations"],
            "rmse_elev": float(met.loc["elevation", "loo_rmse"]),
            "rmse_elev_sx300": float(met.loc["elevation+sx300", "loo_rmse"]),
            "rmse_change_pct": float(met.loc["elevation+sx300", "rmse_change_vs_elev_pct"]),
            "spearman_elev_sx300": float(met.loc["elevation+sx300", "spearman_pred_obs"]),
            "spearman_elev": float(met.loc["elevation", "spearman_pred_obs"]),
            "spearman_exposure_vs_residual": v_mean["spearman_sx300_exposure_vs_elev_residual"],
        },
        "robustness": robust,
        "grade_flag_sections": int(scored["grade_flag"].sum()),
        "top_priority_km": [[round(float(a), 1), round(float(b), 1)] for a, b in zip(top["km_from"], top["km_to"], strict=True)],
        "n_hotspots": int(len(hs)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if figures:
        from .plots import figure, post_figure

        docs.mkdir(parents=True, exist_ok=True)
        dem = Grid.load(data / "corridor_dem_50m.npz")
        figure(scored, top, dem, v_mean, docs / "figure.png", REF_STATION)
        post_figure(scored, top, dem, summary, docs / "post-figure.png")
    return summary
