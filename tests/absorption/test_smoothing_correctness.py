import numpy as np
from numpy.testing import assert_allclose
import scipy.signal as sps

from quasar_utils.absorption import smoothing


def synth_spectrum(N=300, seed=1):
    x = np.linspace(4000.0, 5000.0, N)
    y_true = np.sin(x / 200.0) * 1.5 + 0.001 * (x - 4500.0)
    rng = np.random.default_rng(seed)
    y = y_true + rng.normal(scale=0.02, size=N)
    dy = np.full_like(x, 0.02)
    return x, y, dy


def test_uniform_uncertainties_matches_scipy_center():
    x, y, dy = synth_spectrum()
    w = 31
    p = 2

    # Our smoothing (no interpolation)
    y_custom = smoothing.weighted_savgol_filter.__wrapped__(x, y, dy, w=w, p=p, mask=None, interpolate=False)

    # SciPy reference
    y_scipy = sps.savgol_filter(y, window_length=w, polyorder=p, mode='interp')

    # Compare only central region to avoid edge padding differences
    left = w // 2
    right = len(x) - w // 2

    assert_allclose(y_custom[left:right], y_scipy[left:right], rtol=1e-6, atol=1e-8)


def test_uniform_uncertainties_full_mask():
    x, y, dy = synth_spectrum()
    w = 31
    p = 2

    mask = np.ones_like(x, dtype=bool)
    y_custom = smoothing.weighted_savgol_filter.__wrapped__(x, y, dy, w=w, p=p, mask=mask, interpolate=False)

    y_scipy = sps.savgol_filter(y, window_length=w, polyorder=p, mode='interp')

    left = w // 2
    right = len(x) - w // 2
    assert_allclose(y_custom[left:right], y_scipy[left:right], rtol=1e-6, atol=1e-8)
