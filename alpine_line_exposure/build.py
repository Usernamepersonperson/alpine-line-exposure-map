"""Rebuild the committed derived tables in data/ from the open sources (network + optional rasterio needed)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import fetch
from .climate import parse_imis, read_smn_meta, rose, station_stats
from .gpkg import read_table
from .line import chain, cut_sections, shortest_route, simplify
from .terrain import Grid, direction_centers, slope_deg, sx, sx_rose, tpi

DMAXES = (100, 300, 1000)
N_DIRS = 16
UPWIND_M = 200.0
STATION_HALF = 1000 + UPWIND_M + 100


def terrain_features(grid: Grid, x, y, upwind: bool = False) -> pd.DataFrame:
    """Elevation, slope, TPI and Sx roses at points (same definition for line sections and stations)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cols = {
        "dem_elev": grid.sample(x, y),
        "slope_deg": slope_deg(grid, x, y),
        "tpi250": tpi(grid, x, y, 250.0),
        "tpi500": tpi(grid, x, y, 500.0),
    }
    for d in DMAXES:
        r = sx_rose(grid, x, y, d, N_DIRS)
        for k in range(N_DIRS):
            cols[f"sx{d}_{k:02d}"] = r[:, k]
    if upwind:  # Sx of the terrain UPWIND of the point (snow source area), per direction
        for k, a in enumerate(direction_centers(N_DIRS)):
            ux = x + UPWIND_M * np.sin(np.radians(a))
            uy = y + UPWIND_M * np.cos(np.radians(a))
            cols[f"sxup300_{k:02d}"] = sx(grid, ux, uy, a, 300.0)
    return pd.DataFrame(cols)


def build_line(gpkg: Path, start: str, end: str) -> np.ndarray:
    nodes = {r["Betriebspunkt_Name"]: r for r in read_table(gpkg, "Netzknoten", ["interlis_oid", "Betriebspunkt_Name"])}
    segs = read_table(gpkg, "Netzsegment", ["rAnfangsknoten", "rEndknoten"])
    edges = [{"a": s["rAnfangsknoten"], "b": s["rEndknoten"], "xy": chain(s["geom"])} for s in segs]
    a = nodes[start]["interlis_oid"]
    b = nodes[end]["interlis_oid"]
    return simplify(chain(shortest_route(edges, a, b)), 0.5)


def run(data_dir: Path, cache: Path, start: str = "Andermatt", end: str = "Disentis/Mustér", section_m: float = 100.0,
        log=print) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    info = {"built_utc": datetime.now(UTC).isoformat(timespec="seconds"), "sources": {}}

    # 1. line geometry from the federal railway network
    gpkg = fetch.download(fetch.RAIL_URL, cache / "raw" / "schienennetz_2056_de.gpkg")
    xy = build_line(gpkg, start, end)
    pd.DataFrame({"e": xy[:, 0].round(1), "n": xy[:, 1].round(1)}).to_csv(data_dir / "line.csv", index=False)
    log(f"line: {len(xy)} vertices")
    info["sources"]["line"] = fetch.RAIL_URL

    # 2. corridor DEM and section terrain
    pad = 1000 + UPWIND_M + 200
    w, s = xy.min(axis=0) - pad
    e, n = xy.max(axis=0) + pad
    grid = fetch.dem_bounds(w, s, e, n, cache, "corridor")
    sec = cut_sections(xy, section_m)
    feats = terrain_features(grid, sec["e"], sec["n"], upwind=True)
    pd.concat([sec, feats], axis=1).to_csv(data_dir / "sections_terrain.csv", index=False, float_format="%.3f")
    grid.coarsen(5).save(data_dir / "corridor_dem_50m.npz", scale=1.0)
    log(f"sections: {len(sec)}")
    info["sources"]["dem"] = fetch.DEM_URL

    # 3. SwissMetNet stations: climatology 2020-2025 + terrain at the station
    meta_path = fetch.download(f"{fetch.SMN_BASE}/ogd-smn_meta_stations.csv", cache / "raw" / "ogd-smn_meta_stations.csv")
    inv_path = fetch.download(f"{fetch.SMN_BASE}/ogd-smn_meta_datainventory.csv", cache / "raw" / "ogd-smn_meta_datainventory.csv")
    meta = read_smn_meta(meta_path)
    inv = pd.read_csv(inv_path, sep=";", encoding="latin-1")
    with_wind = set(inv.loc[inv["parameter_shortname"] == "fkl010h0", "station_abbr"])
    rows = []
    for i, st in meta[meta["abbr"].isin(with_wind)].iterrows():
        try:
            h = fetch.smn_hourly(st["abbr"], cache)
            h = h[(h["time_utc"] >= "2020-01-01") & (h["time_utc"] < "2026-01-01")]
            stats = station_stats(h, N_DIRS)
            g = fetch.dem_window(st["e"], st["n"], STATION_HALF, cache, f"smn_{st['abbr'].lower()}")
            tf = terrain_features(g, [st["e"]], [st["n"]]).iloc[0].to_dict()
            rows.append({**st.to_dict(), **stats, **tf, "t_first": str(h["time_utc"].min()), "t_last": str(h["time_utc"].max())})
            log(f"smn {st['abbr']} ok ({i + 1}/{len(meta)})")
        except Exception as exc:  # keep going; record what failed
            log(f"smn {st['abbr']} skipped: {exc}")
    pd.DataFrame(rows).to_csv(data_dir / "stations_smn.csv", index=False, float_format="%.4f")
    info["sources"]["smn"] = f"{fetch.SMN_BASE} (hourly, {fetch.SMN_PERIOD}, 2020-01-01..2025-12-31 UTC)"

    # 4. IMIS stations: 7-day snapshot (the public API serves no longer history)
    from .line import wgs84_to_lv95

    rows = []
    stations = fetch.imis_stations(cache)
    for st in stations:
        try:
            df = parse_imis(fetch.imis_week(st["code"], cache))
            ok = df["wind"].notna()
            if ok.sum() < 200:  # need most of the 336 half-hours
                continue
            ee, nn = wgs84_to_lv95(st["lat"], st["lon"])
            g = fetch.dem_window(float(ee), float(nn), STATION_HALF, cache, f"imis_{st['code'].lower()}")
            tf = terrain_features(g, [float(ee)], [float(nn)]).iloc[0].to_dict()
            r = rose(df.loc[ok, "dir"], N_DIRS)
            rec = {"abbr": st["code"], "name": st.get("label", ""), "type": st.get("type", ""), "elev": st.get("elevation"),
                   "e": float(ee), "n": float(nn), "wind_mean_week": float(df.loc[ok, "wind"].mean()),
                   "gust_p95_week": float(df["gust"].quantile(0.95)), "n_records": int(ok.sum()),
                   "t_first": str(df["time_utc"].min()), "t_last": str(df["time_utc"].max())}
            rec.update({f"week_rose_{k:02d}": float(r[k]) for k in range(N_DIRS)})
            rows.append({**rec, **tf})
            log(f"imis {st['code']} ok")
        except Exception as exc:
            log(f"imis {st['code']} skipped: {exc}")
    pd.DataFrame(rows).to_csv(data_dir / "stations_imis.csv", index=False, float_format="%.4f")
    info["sources"]["imis"] = f"{fetch.IMIS_BASE} (30-min records, last 7 days at build time)"
    (data_dir / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
