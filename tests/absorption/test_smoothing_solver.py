import numpy as np
from numpy.polynomial.polynomial import polyvander
import numpy.linalg as nla

from quasar_utils.absorption import smoothing


def make_random_slides(n_slides=10, w=7, seed=0):
    rng = np.random.default_rng(seed)
    x_slides = rng.normal(scale=0.1, size=(n_slides, w))
    # ensure increasing x per slide
    x_slides = np.sort(x_slides, axis=1)
    y_slides = np.sin(x_slides) + 0.1 * rng.normal(size=(n_slides, w))
    dy_slides = 0.02 + 0.01 * rng.random(size=(n_slides, w))
    # introduce some NaNs randomly
    mask = rng.random((n_slides, w)) < 0.05
    y_slides[mask] = np.nan
    dy_slides[mask] = np.nan
    return x_slides, y_slides, dy_slides


def reference_lstsq(x_s, y_s, dy_s, p):
    # select finite rows
    valid = np.isfinite(x_s) & np.isfinite(y_s) & np.isfinite(dy_s) & (dy_s > 0)
    if valid.sum() == 0:
        return np.full(p + 1, np.nan)
    X = polyvander(x_s[valid], deg=p)
    sqrt_w = 1.0 / dy_s[valid]
    sol, *_ = nla.lstsq(X * sqrt_w[:, None], y_s[valid] * sqrt_w, rcond=None)
    return sol


def test_solve_weighted_poly_matches_lstsq_for_random_slides():
    x_slides, y_slides, dy_slides = make_random_slides(n_slides=20, w=9)
    p = 2
    # call the implementation
    coeffs = smoothing.solve_weighted_poly.__wrapped__(x_slides, y_slides, dy_slides, p, full=True)

    for i in range(x_slides.shape[0]):
        ref = reference_lstsq(x_slides[i], y_slides[i], dy_slides[i], p)
        # coeffs shape (n_slides, p+1)
        if np.all(np.isnan(ref)):
            assert np.all(np.isnan(coeffs[i]))
        else:
            assert np.allclose(coeffs[i], ref, rtol=1e-6, atol=1e-8)


def test_solve_weighted_poly_handles_underdetermined():
    # create slides with fewer valid rows than p+1
    x_slides = np.array([[0.0, 1.0, np.nan]])
    y_slides = np.array([[1.0, 2.0, np.nan]])
    dy_slides = np.array([[0.1, 0.1, np.nan]])
    p = 3
    coeffs = smoothing.solve_weighted_poly.__wrapped__(x_slides, y_slides, dy_slides, p, full=True)
    # behavior: lstsq returns least-norm solution; ensure it returns finite values or nan
    assert coeffs.shape == (1, p + 1)
