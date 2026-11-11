import json

import numpy as np
import pytest

from alpine_line_exposure.line import (
    chain,
    cut_sections,
    interpolate,
    lv95_to_wgs84,
    polyline_length,
    read_polyline,
    shortest_route,
    simplify,
    wgs84_to_lv95,
)


def test_wgs84_to_lv95_reference_point():
    # swisstopo's worked example: 46 deg 2' 38.87" N, 8 deg 43' 49.79" E -> E 2 699 999.76, N 1 099 999.97
    lat = 46 + 2 / 60 + 38.87 / 3600
    lon = 8 + 43 / 60 + 49.79 / 3600
    e, n = wgs84_to_lv95(lat, lon)
    assert e == pytest.approx(2699999.76, abs=0.5)
    assert n == pytest.approx(1099999.97, abs=0.5)


def test_lv95_roundtrip():
    lat, lon = lv95_to_wgs84(*wgs84_to_lv95(46.65, 8.65))
    assert lat == pytest.approx(46.65, abs=2e-5)
    assert lon == pytest.approx(8.65, abs=2e-5)


def test_shortest_route_orients_reversed_edges():
    edges = [
        {"a": "A", "b": "B", "xy": np.array([[0, 0], [10, 0]])},
        {"a": "C", "b": "B", "xy": np.array([[20, 0], [10, 0]])},  # stored backwards
        {"a": "A", "b": "C", "xy": np.array([[0, 0], [0, 50], [20, 50], [20, 0]])},  # longer detour
    ]
    xy = chain(shortest_route(edges, "A", "C"))
    np.testing.assert_allclose(xy, [[0, 0], [10, 0], [20, 0]])


def test_shortest_route_errors():
    edges = [{"a": "A", "b": "B", "xy": np.array([[0, 0], [1, 0]])}, {"a": "C", "b": "D", "xy": np.array([[5, 0], [6, 0]])}]
    with pytest.raises(ValueError):
        shortest_route(edges, "A", "D")
    with pytest.raises(KeyError):
        shortest_route(edges, "A", "Z")


def test_simplify_keeps_corner_drops_collinear():
    xy = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], dtype=float)
    np.testing.assert_allclose(simplify(xy, 0.01), [[0, 0], [2, 0], [2, 2]])


def test_cut_sections_lengths_and_bearing():
    xy = np.array([[2_690_000.0, 1_170_000.0], [2_691_000.0, 1_170_000.0]])
    sec = cut_sections(xy, 100.0)
    assert len(sec) == 10
    assert sec["bearing_deg"].to_numpy() == pytest.approx(90.0)
    assert sec["e"].iloc[0] == pytest.approx(2_690_050.0)
    assert (sec["km_to"] - sec["km_from"]).sum() == pytest.approx(1.0)


def test_cut_sections_merges_short_tail():
    xy = np.array([[0.0, 0.0], [0.0, 1030.0]])
    sec = cut_sections(xy, 100.0)
    assert len(sec) == 10
    assert sec["km_to"].iloc[-1] == pytest.approx(1.03)
    assert sec["bearing_deg"].iloc[0] == pytest.approx(0.0)


def test_cut_sections_rejects_bad_input():
    with pytest.raises(ValueError):
        cut_sections(np.array([[0.0, 0.0], [1.0, 0.0]]), 0)
    with pytest.raises(ValueError):
        cut_sections(np.array([[0.0, 0.0], [0.0, 0.0]]), 100)


def test_interpolate_and_length():
    xy = np.array([[0, 0], [3, 4], [3, 10]], dtype=float)
    assert polyline_length(xy) == pytest.approx(11.0)
    np.testing.assert_allclose(interpolate(xy, [0, 5, 8, 99]), [[0, 0], [3, 4], [3, 7], [3, 10]])


def test_read_polyline_formats(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("e,n\n2690000,1170000\n2690100,1170000\n")
    assert read_polyline(p).shape == (2, 2)
    q = tmp_path / "b.csv"
    q.write_text("lat,lon\n46.65,8.65\n46.66,8.66\n")
    xy = read_polyline(q)
    assert 2_680_000 < xy[0, 0] < 2_700_000
    g = tmp_path / "c.geojson"
    g.write_text(json.dumps({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[8.65, 46.65], [8.66, 46.66]]}}))
    np.testing.assert_allclose(read_polyline(g), xy, atol=1e-6)


def test_read_polyline_rejects_bad_files(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("x,y\n1,2\n3,4\n")
    with pytest.raises(ValueError):
        read_polyline(p)
    q = tmp_path / "far.csv"
    q.write_text("e,n\n100,200\n300,400\n")
    with pytest.raises(ValueError):
        read_polyline(q)
    g = tmp_path / "pt.geojson"
    g.write_text(json.dumps({"type": "Point", "coordinates": [8.6, 46.6]}))
    with pytest.raises(ValueError):
        read_polyline(g)
