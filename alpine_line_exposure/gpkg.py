"""Minimal GeoPackage geometry reader with curve support (no GDAL needed).

GeoPackage stores geometries as a small header followed by ISO WKB. Railway networks published from INTERLIS
models often use curved geometries (CircularString / CompoundCurve / MultiCurve, sometimes with M values), which
common WKB readers reject. This module parses them and linearises circular arcs into short chords.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path

import numpy as np

_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

POINT, LINESTRING, POLYGON, MULTIPOINT, MULTILINESTRING = 1, 2, 3, 4, 5
CIRCULARSTRING, COMPOUNDCURVE, MULTICURVE = 8, 9, 11
_CURVE_TYPES = {LINESTRING, CIRCULARSTRING, COMPOUNDCURVE}
_MULTI_TYPES = {MULTILINESTRING, MULTICURVE}


class WKBError(ValueError):
    """Raised for malformed or unsupported WKB / GeoPackage blobs."""


def strip_gpkg_header(blob: bytes) -> tuple[bytes, int]:
    """Return (wkb, srs_id) from a GeoPackage geometry blob."""
    if len(blob) < 8 or blob[:2] != b"GP":
        raise WKBError("not a GeoPackage geometry blob (missing 'GP' magic)")
    flags = blob[3]
    env = (flags >> 1) & 0b111
    if env not in _ENVELOPE_BYTES:
        raise WKBError(f"invalid envelope indicator {env}")
    bo = "<" if flags & 1 else ">"
    srs_id = struct.unpack(bo + "i", blob[4:8])[0]
    start = 8 + _ENVELOPE_BYTES[env]
    if len(blob) <= start:
        raise WKBError("empty geometry")
    return bytes(blob[start:]), srs_id


def _type_and_dims(raw: int) -> tuple[int, int]:
    """Decode an ISO or EWKB geometry type code into (base type, coordinates per point)."""
    has_z = bool(raw & 0x80000000)
    has_m = bool(raw & 0x40000000)
    raw &= 0x0FFFFFFF
    base = raw % 1000
    family = raw // 1000
    if family == 1:
        has_z = True
    elif family == 2:
        has_m = True
    elif family == 3:
        has_z = has_m = True
    elif family != 0:
        raise WKBError(f"unknown geometry type code {raw}")
    return base, 2 + int(has_z) + int(has_m)


def arc_points(p0, p1, p2, max_step_m: float = 5.0) -> np.ndarray:
    """Linearise the circular arc through p0 -> p1 -> p2 (p1 lies on the arc). Returns (n, 2) incl. both ends."""
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    closed = math.isclose(x0, x2, abs_tol=1e-9) and math.isclose(y0, y2, abs_tol=1e-9)
    if closed:  # full circle: p1 is the diametrically opposite point (ISO convention)
        if math.isclose(x0, x1, abs_tol=1e-9) and math.isclose(y0, y1, abs_tol=1e-9):
            return np.array([p0, p2], dtype=float)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sweep = 2 * math.pi
        r = math.hypot(x0 - cx, y0 - cy)
        a0 = math.atan2(y0 - cy, x0 - cx)
    else:
        d = 2.0 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        if abs(d) < 1e-9:  # collinear: a straight segment
            return np.array([p0, p2], dtype=float)
        s0, s1, s2 = x0 * x0 + y0 * y0, x1 * x1 + y1 * y1, x2 * x2 + y2 * y2
        cx = (s0 * (y1 - y2) + s1 * (y2 - y0) + s2 * (y0 - y1)) / d
        cy = (s0 * (x2 - x1) + s1 * (x0 - x2) + s2 * (x1 - x0)) / d
        r = math.hypot(x0 - cx, y0 - cy)
        a0 = math.atan2(y0 - cy, x0 - cx)
        a1 = math.atan2(y1 - cy, x1 - cx)
        a2 = math.atan2(y2 - cy, x2 - cx)
        # counter-clockwise sweep from a0 to a2; if the middle point is not on it, go clockwise
        ccw = (a2 - a0) % (2 * math.pi)
        mid = (a1 - a0) % (2 * math.pi)
        sweep = ccw if mid <= ccw else ccw - 2 * math.pi
    n = max(2, int(math.ceil(abs(sweep) * r / max_step_m)) + 1)
    t = np.linspace(0.0, 1.0, n)
    ang = a0 + sweep * t
    pts = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
    pts[0], pts[-1] = p0, p2
    return pts


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise WKBError("truncated WKB")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def header(self) -> tuple[str, int, int]:
        order = self.take(1)[0]
        if order not in (0, 1):
            raise WKBError(f"bad byte-order flag {order}")
        bo = "<" if order == 1 else ">"
        base, ndim = _type_and_dims(struct.unpack(bo + "I", self.take(4))[0])
        return bo, base, ndim

    def points(self, bo: str, ndim: int) -> np.ndarray:
        n = struct.unpack(bo + "I", self.take(4))[0]
        if n > 10_000_000:
            raise WKBError("implausible point count")
        vals = struct.unpack(bo + "d" * (n * ndim), self.take(8 * n * ndim))
        return np.asarray(vals, dtype=float).reshape(n, ndim)[:, :2]


def _read_curve(r: _Reader, max_step_m: float) -> np.ndarray:
    bo, base, ndim = r.header()
    if base == LINESTRING:
        return r.points(bo, ndim)
    if base == CIRCULARSTRING:
        pts = r.points(bo, ndim)
        if len(pts) < 3 or len(pts) % 2 == 0:
            raise WKBError("circular string needs an odd number (>= 3) of points")
        parts = [arc_points(pts[i], pts[i + 1], pts[i + 2], max_step_m) for i in range(0, len(pts) - 2, 2)]
        return _join(parts)
    if base == COMPOUNDCURVE:
        n = struct.unpack(bo + "I", r.take(4))[0]
        return _join([_read_curve(r, max_step_m) for _ in range(n)])
    raise WKBError(f"geometry type {base} is not a curve")


def _join(parts: list[np.ndarray]) -> np.ndarray:
    out = [parts[0]]
    for p in parts[1:]:
        if len(p) and len(out[-1]) and np.allclose(p[0], out[-1][-1]):
            p = p[1:]
        out.append(p)
    return np.vstack(out)


def wkb_to_lines(wkb: bytes, max_step_m: float = 5.0) -> list[np.ndarray]:
    """Parse a (multi) linear or curved WKB geometry into a list of (n, 2) XY arrays."""
    r = _Reader(wkb)
    bo, base, ndim = r.header()
    if base in _MULTI_TYPES:
        n = struct.unpack(bo + "I", r.take(4))[0]
        return [_read_curve(r, max_step_m) for _ in range(n)]
    if base == POINT:
        return [np.asarray(struct.unpack(bo + "d" * ndim, r.take(8 * ndim))[:2], dtype=float).reshape(1, 2)]
    if base == MULTIPOINT:
        n = struct.unpack(bo + "I", r.take(4))[0]
        out = []
        for _ in range(n):
            pbo, pbase, pndim = r.header()
            if pbase != POINT:
                raise WKBError("multipoint member is not a point")
            out.append(np.asarray(struct.unpack(pbo + "d" * pndim, r.take(8 * pndim))[:2], dtype=float).reshape(1, 2))
        return out
    r.pos = 0
    if base in _CURVE_TYPES:
        return [_read_curve(r, max_step_m)]
    raise WKBError(f"unsupported geometry type {base}")


def read_table(path: str | Path, table: str, columns: list[str]) -> list[dict]:
    """Read rows of a GeoPackage feature table; the 'geom' column is returned as a list of XY arrays."""
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        cols = ", ".join(f'"{c}"' for c in ["geom", *columns])
        out = []
        for row in con.execute(f'SELECT {cols} FROM "{table}"'):
            wkb, _ = strip_gpkg_header(row[0])
            rec = dict(zip(columns, row[1:], strict=True))
            rec["geom"] = wkb_to_lines(wkb)
            out.append(rec)
        return out
    finally:
        con.close()
