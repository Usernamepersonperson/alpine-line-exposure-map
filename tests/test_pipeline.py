"""End-to-end checks on the committed derived data (offline)."""

import json
import struct
from pathlib import Path

import pandas as pd
import pytest

from alpine_line_exposure.pipeline import demo
from alpine_line_exposure.terrain import Grid

DATA = Path(__file__).resolve().parent.parent / "data"


def png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    docs = tmp_path_factory.mktemp("docs")
    summary = demo(DATA, out, docs, top_count=10, figures=True)
    return summary, out, docs


def test_committed_tables_are_consistent():
    line = pd.read_csv(DATA / "line.csv")
    sec = pd.read_csv(DATA / "sections_terrain.csv")
    smn = pd.read_csv(DATA / "stations_smn.csv")
    assert len(line) > 100 and len(sec) > 200
    assert sec["dem_elev"].between(1000, 3000).all()
    assert (smn["abbr"] == "GUE").any()
    # DEM elevation at the station agrees with the published station height for most stations
    diff = (smn["dem_elev"] - smn["elev"]).abs()
    assert (diff < 30).mean() > 0.8
    dem = Grid.load(DATA / "corridor_dem_50m.npz")
    w, s, e, n = dem.bounds
    assert w < sec["e"].min() and e > sec["e"].max() and s < sec["n"].min() and n > sec["n"].max()


def test_demo_outputs(run):
    summary, out, _ = run
    for name in ("sections_ranked.csv", "top_priority.csv", "top_wind.csv", "top_drift.csv", "hotspots.csv",
                 "validation_models.csv", "summary.json"):
        assert (out / name).exists(), name
    top = pd.read_csv(out / "top_priority.csv")
    assert len(top) == 10 and top["rank"].tolist() == list(range(1, 11))
    assert summary["headline"]["n_stations"] >= 100
    assert json.loads((out / "summary.json").read_text())["n_sections"] == summary["n_sections"]


def test_figure_sizes(run):
    _, _, docs = run
    assert png_size(docs / "figure.png") == (1600, 900)
    assert png_size(docs / "post-figure.png") == (960, 700)
