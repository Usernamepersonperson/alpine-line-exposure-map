"""Line geometry: route a corridor through a network, read/write polylines, cut it into fixed-length sections.

All coordinates are Swiss LV95 (EPSG:2056) metres: E (east), N (north).
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def wgs84_to_lv95(lat_deg, lon_deg):
    """Approximate WGS84 -> LV95 (swisstopo's published formulas, ~1 m accuracy). Vectorised."""
    lat = np.asarray(lat_deg, dtype=float) * 3600.0
    lon = np.asarray(lon_deg, dtype=float) * 3600.0
    p = (lat - 169028.66) / 10000.0
    lam = (lon - 26782.5) / 10000.0
    e = 2600072.37 + 211455.93 * lam - 10938.51 * lam * p - 0.36 * lam * p**2 - 44.54 * lam**3
    n = 1200147.07 + 308807.95 * p + 3745.25 * lam**2 + 76.63 * p**2 - 194.56 * lam**2 * p + 119.79 * p**3
    return e, n


def lv95_to_wgs84(e, n):
    """Approximate LV95 -> WGS84 (swisstopo's published formulas). Returns (lat, lon) in degrees."""
    y = (np.asarray(e, dtype=float) - 2600000.0) / 1e6
    x = (np.asarray(n, dtype=float) - 1200000.0) / 1e6
    lon = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x**2 - 0.0436 * y**3
    lat = 16.9023892 + 3.238272 * x - 0.270978 * y**2 - 0.002528 * x**2 - 0.0447 * y**2 * x - 0.0140 * x**3
    return lat * 100.0 / 36.0, lon * 100.0 / 36.0


def polyline_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return 0.0
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def shortest_route(edges: list[dict], start: str, end: str) -> list[np.ndarray]:
    """Dijkstra over edges {'a': node, 'b': node, 'xy': (n,2)}. Returns the edge geometries oriented start->end."""
    adj: dict[str, list[tuple[float, str, int]]] = {}
    for i, ed in enumerate(edges):
        w = polyline_length(ed["xy"])
        adj.setdefault(ed["a"], []).append((w, ed["b"], i))
        adj.setdefault(ed["b"], []).append((w, ed["a"], i))
    if start not in adj or end not in adj:
        raise KeyError(f"node not in network: {start if start not in adj else end}")
    dist = {start: 0.0}
    prev: dict[str, tuple[str, int]] = {}
    heap = [(0.0, start)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        if u == end:
            break
        for w, v, i in adj[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = (u, i)
                heapq.heappush(heap, (nd, v))
    if end not in done:
        raise ValueError(f"no route from {start} to {end}")
    path = []
    node = end
    while node != start:
        u, i = prev[node]
        xy = np.asarray(edges[i]["xy"], dtype=float)
        if edges[i]["a"] != u:  # traversed b -> a
            xy = xy[::-1]
        path.append(xy)
        node = u
    return path[::-1]


def chain(parts: list[np.ndarray], tol: float = 1.0) -> np.ndarray:
    """Concatenate oriented pieces into one polyline, dropping duplicated joints."""
    out = [np.asarray(parts[0], dtype=float)]
    for p in parts[1:]:
        p = np.asarray(p, dtype=float)
        if np.hypot(*(p[0] - out[-1][-1])) <= tol:
            p = p[1:]
        out.append(p)
    xy = np.vstack(out)
    keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-6])
    return xy[keep]


def simplify(xy: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification (iterative)."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return xy
    keep = np.zeros(len(xy), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        seg = b - a
        L = math.hypot(*seg)
        pts = xy[i + 1 : j]
        if L == 0:
            d = np.hypot(*(pts - a).T)
        else:
            d = np.abs(seg[0] * (pts[:, 1] - a[1]) - seg[1] * (pts[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return xy[keep]


def interpolate(xy: np.ndarray, s) -> np.ndarray:
    """Points at chainage s (metres from the first vertex) along the polyline."""
    xy = np.asarray(xy, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    s = np.clip(np.asarray(s, dtype=float), 0.0, cum[-1])
    return np.column_stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, 90 = east) of a vector."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def cut_sections(xy: np.ndarray, length_m: float = 100.0) -> pd.DataFrame:
    """Cut a polyline into consecutive sections of `length_m` (the last one may be shorter, min 1/2 length kept)."""
    if length_m <= 0:
        raise ValueError("section length must be positive")
    total = polyline_length(xy)
    if total == 0:
        raise ValueError("polyline has zero length")
    starts = np.arange(0.0, total, length_m)
    ends = np.minimum(starts + length_m, total)
    if len(starts) > 1 and ends[-1] - starts[-1] < length_m / 2:  # merge a short tail into the previous section
        ends[-2] = ends[-1]
        starts, ends = starts[:-1], ends[:-1]
    mids = (starts + ends) / 2.0
    p0, p1, pm = interpolate(xy, starts), interpolate(xy, ends), interpolate(xy, mids)
    return pd.DataFrame(
        {
            "section": np.arange(len(starts)),
            "km_from": starts / 1000.0,
            "km_to": ends / 1000.0,
            "e": pm[:, 0],
            "n": pm[:, 1],
            "bearing_deg": bearing_deg(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]),
        }
    )


def read_polyline(path: str | Path) -> np.ndarray:
    """Read a line from CSV (columns e,n in LV95 or lat,lon in WGS84) or GeoJSON (LineString, WGS84)."""
    path = Path(path)
    if path.suffix.lower() in (".geojson", ".json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        geom = obj
        if obj.get("type") == "FeatureCollection":
            geom = obj["features"][0]["geometry"]
        elif obj.get("type") == "Feature":
            geom = obj["geometry"]
        if geom.get("type") != "LineString":
            raise ValueError("GeoJSON must contain a LineString")
        lonlat = np.asarray(geom["coordinates"], dtype=float)[:, :2]
        e, n = wgs84_to_lv95(lonlat[:, 1], lonlat[:, 0])
        return np.column_stack([e, n])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "e" in cols and "n" in cols:
        xy = df[[cols["e"], cols["n"]]].to_numpy(float)
    elif "lat" in cols and "lon" in cols:
        e, n = wgs84_to_lv95(df[cols["lat"]].to_numpy(float), df[cols["lon"]].to_numpy(float))
        xy = np.column_stack([e, n])
    else:
        raise ValueError("CSV needs columns e,n (LV95) or lat,lon (WGS84)")
    if len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("polyline needs >= 2 finite vertices")
    if xy[:, 0].min() < 2_400_000 or xy[:, 0].max() > 2_900_000 or xy[:, 1].min() < 1_050_000 or xy[:, 1].max() > 1_320_000:
        raise ValueError("coordinates outside the Swiss LV95 range")
    return xy
