import numpy as np
import pandas as pd
import pytest

from alpine_line_exposure.score import crosswind_matrix, hotspots, pct_rank, score_sections, top_n
from alpine_line_exposure.validate import loo, metrics, ols, spearman, validate

N = 16


def test_spearman_basic():
    a = np.arange(10.0)
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(spearman(a, np.ones(10)))
    assert np.isnan(spearman([1.0, np.nan], [2.0, 3.0]))


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = 3 + 2 * X[:, 0] - 0.5 * X[:, 1]
    np.testing.assert_allclose(ols(X, y), [3, 2, -0.5], atol=1e-10)


def stations(n=60, sx_effect=0.25, seed=3):
    """Synthetic stations: wind = 1 + 1.5 * elev_km + sx_effect * exposure + noise; storms from the west."""
    rng = np.random.default_rng(seed)
    elev = rng.uniform(300, 3500, n)
    sx = rng.normal(0, 4, size=(n, N))
    w = np.zeros(N)
    w[12] = 1.0
    expo = -sx[:, 12]
    wind = 1 + 1.5 * elev / 1000 + sx_effect * expo + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"abbr": [f"S{i}" for i in range(n)], "elev": elev, "wind": wind, "tpi500": rng.normal(0, 20, n)})
    for d in (100, 300, 1000):
        for k in range(N):
            df[f"sx{d}_{k:02d}"] = sx[:, k] if d == 300 else sx[:, k] + rng.normal(0, 2, n)
    for k in range(N):
        df[f"storm_rose_{k:02d}"] = w[k]
    return df


def test_loo_detects_real_terrain_signal():
    df = stations()
    v = validate(df, "wind", "storm_rose", np.ones(len(df), bool))
    m = v["models"].set_index("model")
    assert m.loc["elevation+sx300", "loo_rmse"] < 0.5 * m.loc["elevation", "loo_rmse"]
    assert m.loc["elevation+sx300", "rmse_diff_ci95_hi"] < 0
    assert v["spearman_sx300_exposure_vs_elev_residual"] > 0.8


def test_loo_reports_no_gain_without_signal():
    df = stations(sx_effect=0.0)
    m = validate(df, "wind", "storm_rose", np.ones(len(df), bool))["models"].set_index("model")
    assert m.loc["elevation+sx300", "loo_rmse"] >= 0.97 * m.loc["elevation", "loo_rmse"]
    assert m.loc["elevation+sx300", "rmse_diff_ci95_lo"] < 0 < m.loc["elevation+sx300", "rmse_diff_ci95_hi"] + 1e-9


def test_loo_never_uses_the_held_out_target():
    df = stations(n=25)
    p1 = loo(df, "wind", "storm_rose", np.ones(len(df), bool), models=("elevation",))
    df2 = df.copy()
    df2.loc[0, "wind"] = 1e6  # a wild value at station 0 must not change its own prediction
    p2 = loo(df2, "wind", "storm_rose", np.ones(len(df), bool), models=("elevation",))
    assert p1.loc[0, "elevation"] == pytest.approx(p2.loc[0, "elevation"])
    assert metrics(p1, models=("elevation",)).loc[0, "rmse_change_vs_elev_pct"] == 0


def test_crosswind_matrix():
    c = crosswind_matrix([0.0, 90.0], 4)  # directions N, E, S, W
    np.testing.assert_allclose(c, [[0, 1, 0, 1], [1, 0, 1, 0]], atol=1e-12)


def sections(n=30):
    df = pd.DataFrame({"section": np.arange(n), "km_from": np.arange(n) * 0.1, "km_to": np.arange(1, n + 1) * 0.1,
                       "e": 2_690_000 + np.arange(n) * 100.0, "n": 1_170_000.0, "bearing_deg": 90.0,
                       "dem_elev": 2000.0, "slope_deg": 5.0})
    for k in range(N):
        df[f"sx300_{k:02d}"] = 0.0
        df[f"sxup300_{k:02d}"] = 0.0
    return df


def test_score_sections_ranks_exposed_and_sheltered_spots():
    df = sections()
    w = np.zeros(N)
    w[0] = 1.0  # storms from the north, i.e. square across an east-west track
    df.loc[5, "sx300_00"] = -8.0  # exposed crest
    df.loc[20, "sx300_00"] = 6.0  # sheltered lee: drift deposition
    df.loc[20, "sxup300_00"] = -5.0
    sc = score_sections(df, np.array([1.0, 1.5, 0.2]), w, w, 1500.0, 300.0)
    assert sc["wind_score"].idxmax() == 5
    assert sc["drift_index"].idxmax() == 20
    assert sc["drift_index"].max() == pytest.approx(100.0)
    assert sc["crosswind"].to_numpy() == pytest.approx(1.0)
    assert not sc["grade_flag"].any()


def test_score_sections_parallel_wind_gives_no_crosswind():
    df = sections()
    w = np.zeros(N)
    w[4] = 1.0  # from the east, along the track
    sc = score_sections(df, np.array([1.0, 1.5, 0.2]), w, w, 1500.0, 300.0)
    assert sc["crosswind"].max() == pytest.approx(0.0, abs=1e-12)


def test_grade_flag_marks_impossible_rail_grades():
    df = sections()
    df.loc[10:12, "dem_elev"] = 2100.0  # +100 m within 100 m: tunnel under a spur
    sc = score_sections(df, np.array([1.0, 1.5, 0.2]), np.ones(N), np.ones(N), 1500.0, 300.0)
    assert sc.loc[9:13, "grade_flag"].any()
    assert not sc.loc[0:7, "grade_flag"].any()


def test_top_n_spreads_picks_and_hotspots_merge():
    df = sections()
    df["priority"] = 0.0
    df.loc[[10, 11, 12], "priority"] = [90, 95, 92]
    df.loc[25, "priority"] = 80
    df["wind_score"] = 1.0
    df["drift_index"] = 0.0
    t = top_n(df, "priority", 2, min_gap_km=0.3)
    assert t["section"].tolist() == [11, 25]
    hs = hotspots(df, "priority", 0.9)
    assert hs.loc[0, "n_sections"] == 3
    assert hs.loc[0, "km_from"] == pytest.approx(1.0)


def test_pct_rank_bounds():
    r = pct_rank([3, 1, 2])
    assert r.tolist() == pytest.approx([1.0, 1 / 3, 2 / 3])
