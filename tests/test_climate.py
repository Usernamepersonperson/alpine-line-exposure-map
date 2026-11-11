import numpy as np
import pandas as pd
import pytest

from alpine_line_exposure.climate import (
    dir_bins,
    parse_imis,
    parse_smn_hourly,
    pooled_rose,
    read_smn_meta,
    rose,
    snow_factor,
    snow_logistic,
    station_stats,
)

SMN_TEXT = """station_abbr;reference_timestamp;tre200h0;fkl010h1;dkl010h0;fkl010h0;htoauths;extra
XYZ;01.01.2024 00:00;-6.2;5.5;25;1.8;23;9
XYZ;01.01.2024 01:00;-6.1;2.8;346;1.1;;9
XYZ;01.01.2024 02:00;-5.7;;203;;23;9
"""


def test_parse_smn_hourly():
    df = parse_smn_hourly(SMN_TEXT)
    assert list(df.columns) == ["time_utc", "wind", "gust", "dir", "snow_cm", "temp"]
    assert df["time_utc"].iloc[1] == pd.Timestamp("2024-01-01 01:00", tz="UTC")
    assert df["wind"].iloc[0] == pytest.approx(1.8)
    assert np.isnan(df["wind"].iloc[2])
    assert np.isnan(df["snow_cm"].iloc[1])


def test_parse_smn_hourly_missing_columns_and_bad_file():
    df = parse_smn_hourly("station_abbr;reference_timestamp;fkl010h0\nX;01.02.2024 00:00;3\n")
    assert df["gust"].isna().all() and df["wind"].iloc[0] == 3
    with pytest.raises(ValueError):
        parse_smn_hourly("a;b\n1;2\n")


def test_read_smn_meta():
    text = (
        "station_abbr;station_name;station_height_masl;station_coordinates_lv95_east;station_coordinates_lv95_north;"
        "station_coordinates_wgs84_lat;station_coordinates_wgs84_lon;station_exposition_en\n"
        "AAA;Alpha;2000;2690000;1170000;46.6;8.6;crest\nBBB;Beta;;2600000;1200000;46.9;7.4;plain\n"
    )
    m = read_smn_meta(text)
    assert list(m["abbr"]) == ["AAA"]  # station without elevation dropped
    assert m["exposition"].iloc[0] == "crest"


def test_dir_bins_edges():
    b = dir_bins([0, 11.2, 11.3, 180, 340, 349, np.nan], 16)
    assert b.tolist() == [0, 0, 1, 8, 15, 0, -1]


def test_rose_normalised_and_empty():
    r = rose([90, 90, 270, np.nan], 4)
    np.testing.assert_allclose(r, [0, 2 / 3, 0, 1 / 3])
    assert rose([np.nan], 4).sum() == 0


def synthetic_hours(n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2021-01-01", periods=n_days * 24, freq="h", tz="UTC")
    calm = rng.uniform(0, 5, len(t))
    storm = rng.random(len(t)) < 0.1
    wind = np.where(storm, rng.uniform(15, 25, len(t)), calm)
    direction = np.where(storm, 270.0, rng.uniform(0, 360, len(t)))
    snow = np.where(t.month.isin([12, 1, 2, 3]), 50.0, 0.0)
    return pd.DataFrame({"time_utc": t, "wind": wind, "gust": wind * 1.6, "dir": direction, "snow_cm": snow, "temp": -5.0})


def test_station_stats_storm_rose_points_to_storm_direction():
    s = station_stats(synthetic_hours())
    rose16 = np.array([s[f"storm_rose_{k:02d}"] for k in range(16)])
    assert np.argmax(rose16) == 12  # 270 deg = sector 12 of 16
    assert rose16[12] > 0.9
    assert 0 < s["snow_day_frac"] < 1
    assert s["wind_coverage"] == 1.0
    assert s["drift_hours_per_year"] > 0
    assert s["drift_rose_12"] == pytest.approx(1.0)


def test_station_stats_without_snow_sensor():
    h = synthetic_hours(200)
    h["snow_cm"] = np.nan
    s = station_stats(h)
    assert np.isnan(s["snow_day_frac"])
    assert "drift_hours_per_year" not in s


def test_pooled_rose_mask_and_error():
    df = pd.DataFrame({f"storm_rose_{k:02d}": [1.0 if k == 0 else 0.0, 1.0 if k == 4 else 0.0] for k in range(16)})
    r = pooled_rose(df, mask=[True, False])
    assert r[0] == 1.0 and r[4] == 0.0
    with pytest.raises(ValueError):
        pooled_rose(df, mask=[False, False])


def test_snow_logistic_recovers_parameters():
    elev = np.linspace(300, 3000, 40)
    frac = snow_factor(elev, 1400.0, 250.0)
    e50, s = snow_logistic(elev, frac)
    assert e50 == pytest.approx(1400.0, abs=25)
    assert s == pytest.approx(250.0, abs=25)
    with pytest.raises(ValueError):
        snow_logistic([1000, 2000], [0.1, 0.5])


def test_parse_imis_cleans_records():
    payload = [
        {"measure_date": "2026-11-10T00:30:00Z", "VW_30MIN_MEAN": 3.0, "VW_30MIN_MAX": 7.0, "DW_30MIN_MEAN": 120.0, "HS": 0.0},
        {"measure_date": "2026-11-10T00:00:00Z", "VW_30MIN_MEAN": -9999, "DW_30MIN_MEAN": 100.0},
        {"no_date": 1},
        "junk",
    ]
    df = parse_imis(payload)
    assert len(df) == 2
    assert df["time_utc"].is_monotonic_increasing
    assert np.isnan(df["wind"].iloc[0]) and df["wind"].iloc[1] == 3.0
    with pytest.raises(ValueError):
        parse_imis({"detail": "error"})
