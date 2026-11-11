import math

import numpy as np
import pytest

from alpine_line_exposure.terrain import Grid, hillshade, slope_deg, sx, sx_rose, tpi, weighted_exposure

RES = 10.0


def plane(gx=0.0, gy=0.0, n=201):
    """z = gx * x + gy * y on an n x n grid centred on (0, 0); y grows to the north."""
    half = n * RES / 2
    xs = -half + (np.arange(n) + 0.5) * RES
    ys = half - (np.arange(n) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    return Grid(1000 + gx * X + gy * Y, -half, half, RES)


def ridge(n=201, height=100.0, width=300.0):
    """North-south ridge along x = 0."""
    g = plane(n=n)
    xs, ys = g.centers()
    X, _ = np.meshgrid(xs, ys)
    g.z = 1000 + height * np.exp(-((X / width) ** 2))
    return g


def test_sample_exact_on_plane():
    g = plane(0.1, -0.2)
    pts = np.array([[3.3, -7.1], [0.0, 0.0], [123.4, 456.7]])
    np.testing.assert_allclose(g.sample(pts[:, 0], pts[:, 1]), 1000 + 0.1 * pts[:, 0] - 0.2 * pts[:, 1])


def test_sample_outside_is_nan():
    g = plane()
    assert np.isnan(g.sample(5000.0, 0.0))
    assert np.isnan(g.sample(0.0, -5000.0))


def test_grid_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Grid(np.zeros(5), 0, 0, 10)
    with pytest.raises(ValueError):
        Grid(np.zeros((3, 3)), 0, 0, 0)


def test_sx_flat_is_zero():
    assert sx(plane(), 0.0, 0.0, 270.0, 300.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_sx_on_inclined_plane_matches_slope_angle():
    g = plane(gy=0.2)  # rises to the north
    ang = math.degrees(math.atan(0.2))
    assert sx(g, 0.0, 0.0, 0.0, 300.0, n_sub=1)[0] == pytest.approx(ang, abs=1e-6)  # wind from north: sheltered
    assert sx(g, 0.0, 0.0, 180.0, 300.0, n_sub=1)[0] == pytest.approx(-ang, abs=1e-6)  # from south: exposed
    assert sx(g, 0.0, 0.0, 90.0, 300.0, n_sub=1)[0] == pytest.approx(0.0, abs=1e-6)


def test_sx_window_average_is_smaller_than_centre():
    g = plane(gy=0.2)
    centre = sx(g, 0.0, 0.0, 0.0, 300.0, n_sub=1)[0]
    window = sx(g, 0.0, 0.0, 0.0, 300.0)[0]
    expected = np.mean([math.degrees(math.atan(0.2 * math.cos(math.radians(a)))) for a in np.linspace(-15, 15, 7)])
    assert window == pytest.approx(expected, abs=1e-6)
    assert window < centre


def test_sx_ridge_top_exposed_valley_sheltered():
    g = ridge()
    top = sx(g, 0.0, 0.0, 270.0, 300.0)[0]
    lee = sx(g, 250.0, 0.0, 270.0, 300.0)[0]  # east of the crest, westerly wind
    assert top < 0
    assert lee > 0


def test_sx_strict_nan_when_search_leaves_grid():
    g = plane(n=21)  # 210 m wide
    assert np.isnan(sx(g, 0.0, 0.0, 0.0, 300.0)[0])
    assert np.isfinite(sx(g, 0.0, 0.0, 0.0, 300.0, strict=False)[0])


def test_sx_bad_dmax():
    with pytest.raises(ValueError):
        sx(plane(), 0.0, 0.0, 0.0, 0.0)


def test_sx_rose_shape_and_symmetry():
    g = plane(gx=0.1)
    r = sx_rose(g, [0.0, 10.0], [0.0, 0.0], 200.0, 8)
    assert r.shape == (2, 8)
    assert r[0, 2] == pytest.approx(-r[0, 6], abs=1e-9)  # east vs west


def test_tpi_and_slope():
    g = ridge()
    assert tpi(g, [0.0], [0.0], 250.0)[0] > 0
    assert tpi(plane(0.3, 0.1), [0.0], [0.0], 250.0)[0] == pytest.approx(0.0, abs=1e-6)
    assert slope_deg(plane(gy=1.0), [0.0], [0.0])[0] == pytest.approx(45.0, abs=1e-6)


def test_weighted_exposure():
    vals = np.array([[1.0, -3.0], [2.0, 2.0]])
    np.testing.assert_allclose(weighted_exposure(vals, np.array([1.0, 1.0])), [1.0, -2.0])
    np.testing.assert_allclose(weighted_exposure(vals, np.array([0.0, 2.0])), [3.0, -2.0])
    with pytest.raises(ValueError):
        weighted_exposure(vals, np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        weighted_exposure(vals, np.array([1.0, -1.0]))


def test_save_load_and_coarsen(tmp_path):
    g = plane(0.05, 0.02, n=40)
    g.z[0, 0] = np.nan
    p = tmp_path / "g.npz"
    g.save(p)
    h = Grid.load(p)
    assert np.isnan(h.z[0, 0])
    np.testing.assert_allclose(h.z[1:, 1:], g.z[1:, 1:], atol=0.05)
    c = h.coarsen(4)
    assert c.shape == (10, 10) and c.res == 40.0
    assert hillshade(g.z, g.res).min() >= 0 and hillshade(g.z, g.res).max() <= 1


def test_save_load_high_summits(tmp_path):
    # regression: int16 decimetres overflowed above 3276.7 m (two summit stations came back negative)
    g = Grid(np.full((4, 4), 3571.3), 0.0, 40.0, 10.0)
    p = tmp_path / "high.npz"
    g.save(p)
    np.testing.assert_allclose(Grid.load(p).z, 3571.3, atol=0.05)
