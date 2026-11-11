import math
import sqlite3
import struct

import numpy as np
import pytest

from alpine_line_exposure.gpkg import WKBError, arc_points, read_table, strip_gpkg_header, wkb_to_lines


def pts_wkb(code, pts, bo="<", ndim=2):
    head = (b"\x01" if bo == "<" else b"\x00") + struct.pack(bo + "I", code)
    flat = [v for p in pts for v in (list(p) + [0.0] * (ndim - len(p)))]
    return head + struct.pack(bo + "I", len(pts)) + struct.pack(bo + "d" * len(flat), *flat)


def multi_wkb(code, parts, bo="<"):
    return (b"\x01" if bo == "<" else b"\x00") + struct.pack(bo + "I", code) + struct.pack(bo + "I", len(parts)) + b"".join(parts)


def gpkg_blob(wkb, envelope=1):
    flags = 1 | (envelope << 1)
    env = struct.pack("<4d", 0, 1, 0, 1) if envelope == 1 else b""
    return b"GP" + bytes([0, flags]) + struct.pack("<i", 2056) + env + wkb


def test_linestring_roundtrip():
    lines = wkb_to_lines(pts_wkb(2, [(0, 0), (3, 4)]))
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0], [[0, 0], [3, 4]])


def test_big_endian_linestring():
    lines = wkb_to_lines(pts_wkb(2, [(1, 2), (5, 6)], bo=">"))
    np.testing.assert_allclose(lines[0], [[1, 2], [5, 6]])


def test_quarter_arc_points_lie_on_circle():
    r = 100.0
    p0, p1, p2 = (r, 0.0), (r * math.cos(math.pi / 4), r * math.sin(math.pi / 4)), (0.0, r)
    pts = arc_points(p0, p1, p2, max_step_m=2.0)
    np.testing.assert_allclose(np.hypot(pts[:, 0], pts[:, 1]), r, rtol=1e-9)
    length = np.hypot(*np.diff(pts, axis=0).T).sum()
    assert length == pytest.approx(math.pi * r / 2, rel=1e-3)
    np.testing.assert_allclose(pts[0], p0)
    np.testing.assert_allclose(pts[-1], p2)


def test_clockwise_arc_goes_through_middle_point():
    # same end points, but the middle point is on the long (clockwise) side
    r = 10.0
    pts = arc_points((r, 0.0), (-r * math.cos(math.pi / 4), -r * math.sin(math.pi / 4)), (0.0, r), max_step_m=0.5)
    length = np.hypot(*np.diff(pts, axis=0).T).sum()
    assert length == pytest.approx(1.5 * math.pi * r, rel=1e-3)


def test_collinear_arc_is_straight():
    pts = arc_points((0, 0), (1, 1), (2, 2))
    np.testing.assert_allclose(pts, [[0, 0], [2, 2]])


def test_full_circle():
    pts = arc_points((10, 0), (-10, 0), (10, 0), max_step_m=1.0)
    length = np.hypot(*np.diff(pts, axis=0).T).sum()
    assert length == pytest.approx(2 * math.pi * 10, rel=1e-3)


def test_multicurve_with_m_values_and_compound_curve():
    # XYM (ISO 2000 offset) members: a LineStringM and a CompoundCurveM(LineStringM, CircularStringM)
    ls = pts_wkb(2002, [(0, 0, 5), (10, 0, 6)], ndim=3)
    arc = pts_wkb(2008, [(10, 0, 0), (10 + 10 * math.sin(math.pi / 4), 10 - 10 * math.cos(math.pi / 4), 0), (20, 10, 0)], ndim=3)
    cc = multi_wkb(2009, [pts_wkb(2002, [(0, 0, 0), (10, 0, 0)], ndim=3), arc])
    lines = wkb_to_lines(multi_wkb(2011, [ls, cc]))
    assert len(lines) == 2
    assert lines[1].shape[1] == 2
    np.testing.assert_allclose(lines[1][0], [0, 0])
    np.testing.assert_allclose(lines[1][-1], [20, 10])
    # joint between the straight part and the arc is not duplicated
    assert not (np.hypot(*np.diff(lines[1], axis=0).T) < 1e-9).any()


def test_ewkb_z_flag():
    code = 2 | 0x80000000
    lines = wkb_to_lines(pts_wkb(code, [(0, 0, 100), (1, 1, 200)], ndim=3))
    np.testing.assert_allclose(lines[0], [[0, 0], [1, 1]])


def test_points_and_multipoints():
    pt = b"\x01" + struct.pack("<I", 1) + struct.pack("<2d", 7, 8)
    assert wkb_to_lines(pt)[0].tolist() == [[7, 8]]
    mp = multi_wkb(4, [pt, pt])
    assert len(wkb_to_lines(mp)) == 2


def test_even_circular_string_rejected():
    with pytest.raises(WKBError):
        wkb_to_lines(pts_wkb(8, [(0, 0), (1, 1), (2, 0), (3, 1)]))


def test_truncated_and_unsupported():
    with pytest.raises(WKBError):
        wkb_to_lines(pts_wkb(2, [(0, 0), (1, 1)])[:-4])
    poly = b"\x01" + struct.pack("<I", 3) + struct.pack("<I", 0)
    with pytest.raises(WKBError):
        wkb_to_lines(poly)
    with pytest.raises(WKBError):
        wkb_to_lines(b"\x07" + struct.pack("<I", 2))


def test_gpkg_header():
    wkb = pts_wkb(2, [(0, 0), (1, 0)])
    out, srs = strip_gpkg_header(gpkg_blob(wkb))
    assert out == wkb and srs == 2056
    with pytest.raises(WKBError):
        strip_gpkg_header(b"XX" + gpkg_blob(wkb)[2:])


def test_read_table(tmp_path):
    db = tmp_path / "t.gpkg"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE seg (id INTEGER, geom BLOB, "a" TEXT)')
    con.execute("INSERT INTO seg VALUES (1, ?, 'x')", (gpkg_blob(pts_wkb(2, [(0, 0), (5, 0)]), envelope=0),))
    con.commit()
    con.close()
    rows = read_table(db, "seg", ["a"])
    assert rows[0]["a"] == "x"
    np.testing.assert_allclose(rows[0]["geom"][0], [[0, 0], [5, 0]])
