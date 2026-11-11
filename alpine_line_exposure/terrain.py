"""Terrain exposure from a gridded DEM, evaluated at points.

Main index: the upwind maximum-slope parameter Sx (Winstral, Marks & Gurney 2002). For a point and a wind
direction A (the direction the wind comes FROM), Sx is the steepest elevation angle, in degrees, to any terrain
within `dmax` metres upwind. Sx < 0 means the ground falls away upwind (exposed: ridge, crest, convex slope);
Sx > 0 means terrain upwind rises above the point (sheltered: lee slope, hollow, cutting).
Like the original, the value is averaged over a 30 degree window of sub-azimuths around A.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NODATA = -(2**31)


@dataclass
class Grid:
    """Regular north-up grid. z[0, 0] is the north-west cell; (x0, y0) is the north-west corner, res in metres."""

    z: np.ndarray
    x0: float
    y0: float
    res: float

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        if self.z.ndim != 2 or min(self.z.shape) < 2:
            raise ValueError("DEM must be a 2-D array of at least 2x2 cells")
        if not self.res > 0:
            raise ValueError("resolution must be positive")

    @property
    def shape(self):
        return self.z.shape

    @property
    def bounds(self):
        """(west, south, east, north) of the grid edges."""
        ny, nx = self.z.shape
        return self.x0, self.y0 - ny * self.res, self.x0 + nx * self.res, self.y0

    def centers(self):
        ny, nx = self.z.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.res
        ys = self.y0 - (np.arange(ny) + 0.5) * self.res
        return xs, ys

    def sample(self, x, y) -> np.ndarray:
        """Bilinear interpolation between cell centres; NaN outside the grid or next to NaN cells."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x - self.x0) / self.res - 0.5
        row = (self.y0 - y) / self.res - 0.5
        ny, nx = self.z.shape
        eps = 1e-9
        ok = (col >= -eps) & (col <= nx - 1 + eps) & (row >= -eps) & (row <= ny - 1 + eps)
        c0 = np.clip(np.floor(np.nan_to_num(col)), 0, nx - 2).astype(np.int64)
        r0 = np.clip(np.floor(np.nan_to_num(row)), 0, ny - 2).astype(np.int64)
        out = np.full(np.broadcast(x, y).shape, np.nan)
        if not ok.any():
            return out
        cc, rr = c0[ok], r0[ok]
        fc, fr = np.clip(col[ok] - cc, 0, 1), np.clip(row[ok] - rr, 0, 1)
        z = self.z
        v = (
            z[rr, cc] * (1 - fc) * (1 - fr)
            + z[rr, cc + 1] * fc * (1 - fr)
            + z[rr + 1, cc] * (1 - fc) * fr
            + z[rr + 1, cc + 1] * fc * fr
        )
        out[ok] = v
        return out

    def coarsen(self, factor: int) -> Grid:
        """Block-average by an integer factor (for figures)."""
        if factor <= 1:
            return self
        ny, nx = (s // factor * factor for s in self.z.shape)
        z = self.z[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
        return Grid(z, self.x0, self.y0, self.res * factor)

    def save(self, path: str | Path, scale: float = 10.0) -> None:
        """Store as int32 (decimetres by default) in a compressed .npz (int16 would overflow above 3276.7 m)."""
        zi = np.where(np.isfinite(self.z), np.round(self.z * scale), NODATA).astype(np.int32)
        np.savez_compressed(path, z=zi, x0=self.x0, y0=self.y0, res=self.res, scale=scale)

    @classmethod
    def load(cls, path: str | Path) -> Grid:
        with np.load(path) as f:
            z = f["z"].astype(float)
            z[f["z"] == (NODATA if f["z"].dtype == np.int32 else -32768)] = np.nan
            return cls(z / float(f["scale"]), float(f["x0"]), float(f["y0"]), float(f["res"]))


def sx(grid: Grid, x, y, azimuth_deg: float, dmax: float, *, window_deg: float = 30.0, n_sub: int = 7,
       step: float | None = None, strict: bool = True) -> np.ndarray:
    """Winstral Sx (degrees) at points for wind FROM `azimuth_deg`, searching up to `dmax` metres upwind.

    strict=True returns NaN if any part of the upwind search leaves the grid (avoids silently short fetches).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if dmax <= 0:
        raise ValueError("dmax must be positive")
    step = grid.res if step is None else float(step)
    dist = np.arange(step, dmax + 1e-9, step)
    if len(dist) == 0:
        raise ValueError("dmax must be at least one step")
    z0 = grid.sample(x, y)
    subs = np.linspace(azimuth_deg - window_deg / 2, azimuth_deg + window_deg / 2, n_sub) if n_sub > 1 else [azimuth_deg]
    res = []
    for a in np.radians(subs):
        xs = x[:, None] + dist[None, :] * np.sin(a)
        ys = y[:, None] + dist[None, :] * np.cos(a)
        zs = grid.sample(xs, ys)
        ang = np.degrees(np.arctan((zs - z0[:, None]) / dist[None, :]))
        with np.errstate(all="ignore"):
            m = np.nanmax(np.where(np.isfinite(ang), ang, -np.inf), axis=1)
        m[~np.isfinite(m)] = np.nan
        if strict:
            m[~np.isfinite(zs).all(axis=1)] = np.nan
        res.append(m)
    out = np.mean(res, axis=0)
    out[~np.isfinite(z0)] = np.nan
    return out


def direction_centers(n_dirs: int = 16) -> np.ndarray:
    return np.arange(n_dirs) * 360.0 / n_dirs


def sx_rose(grid: Grid, x, y, dmax: float, n_dirs: int = 16, **kw) -> np.ndarray:
    """Sx for each of `n_dirs` equally spaced wind directions -> array (n_points, n_dirs)."""
    return np.column_stack([sx(grid, x, y, a, dmax, **kw) for a in direction_centers(n_dirs)])


def tpi(grid: Grid, x, y, radius: float, n_az: int = 16) -> np.ndarray:
    """Topographic position index: elevation minus the mean elevation on a ring of `radius` metres."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    az = np.radians(np.arange(n_az) * 360.0 / n_az)
    ring = grid.sample(x[:, None] + radius * np.sin(az)[None, :], y[:, None] + radius * np.cos(az)[None, :])
    return grid.sample(x, y) - ring.mean(axis=1)


def slope_deg(grid: Grid, x, y) -> np.ndarray:
    """Terrain slope (degrees) from central differences at +-1 cell."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    h = grid.res
    dzdx = (grid.sample(x + h, y) - grid.sample(x - h, y)) / (2 * h)
    dzdy = (grid.sample(x, y + h) - grid.sample(x, y - h)) / (2 * h)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def hillshade(z: np.ndarray, res: float, azimuth: float = 315.0, altitude: float = 40.0) -> np.ndarray:
    """Classic hillshade in [0, 1] for figures."""
    gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def weighted_exposure(sx_rose_vals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exposure = -(sum_d w_d * Sx_d) with weights normalised to 1; higher = more exposed to the weighted winds."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != sx_rose_vals.shape[-1] or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per direction, and not all zero")
    return -(sx_rose_vals @ (w / w.sum()))
