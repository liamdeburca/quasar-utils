import os
import sys
import numpy as np

# Ensure the package dir is importable during tests
sys.path.insert(0, os.path.abspath("src"))

from quasar_utils.absorption import smoothing


def solve_weighted_poly_lstsq(x_slides, y_slides, dy_slides, p, full=False):
    """Reference per-slide weighted polynomial solver using lstsq.

    Returns the constant term per slide when full=False, matching the
    behaviour expected by the existing code.
    """
    from numpy.polynomial.polynomial import polyvander
    from numpy import isfinite, nan, zeros

    n_slides = y_slides.shape[0]
    coeffs = zeros((n_slides, p + 1), dtype=float)

    for i in range(n_slides):
        x_s = x_slides[i]
        y_s = y_slides[i]
        dy_s = dy_slides[i]

        valid = isfinite(x_s) & isfinite(y_s) & isfinite(dy_s) & (dy_s > 0)
        if valid.sum() == 0:
            coeffs[i, :] = nan
            continue

        X = polyvander(x_s[valid], deg=p)
        sqrt_w = 1.0 / dy_s[valid]
        Xw = X * sqrt_w[:, None]
        yw = y_s[valid] * sqrt_w

        sol, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        coeffs[i, :] = sol

    return coeffs if full else coeffs[:, 0]


def test_weighted_savgol_equivalence_random():
    # Synthetic spectrum
    rng = np.random.default_rng(1)
    N = 300
    x = np.linspace(4000.0, 5000.0, N)
    # Smooth underlying function plus small features
    y_true = np.sin(x / 200.0) * 1.5 + 0.001 * (x - 4500.0)
    y = y_true + rng.normal(scale=0.02, size=N)
    # Heteroscedastic uncertainties
    dy = 0.02 + 0.005 * rng.random(N)

    w = 31
    p = 2

    mask = np.ones_like(x, dtype=bool)

    # Old implementation result
    y_old = smoothing.weighted_savgol_filter.__wrapped__(x, y, dy, w=w, p=p, mask=mask, interpolate=False)

    # Recompute valid indices, slides and solve with lstsq-based solver
    valid_indices = smoothing.get_valid_indices.__wrapped__(mask, w, p, mode="standard") & np.isfinite(x)
    slides = smoothing.create_slides.__wrapped__(np.where(np.isfinite(x), x, np.nan), np.where(mask, y, np.nan), np.where(mask, dy, np.nan), w, mask=valid_indices)
    # slides are (x_slides, y_slides, dy_slides)
    x_slides, y_slides, dy_slides = slides

    solved = solve_weighted_poly_lstsq(x_slides, y_slides, dy_slides, p=p, full=False)

    y_new = y.copy()
    y_new[valid_indices] = solved

    # Since both approaches should be numerically very close, assert closeness
    assert np.allclose(y_old, y_new, rtol=1e-6, atol=1e-8)
