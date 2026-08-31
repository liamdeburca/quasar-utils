import numpy as np
import pytest
import scipy.signal as sps

from numpy.testing import assert_allclose
from quasar_utils.absorption import smoothing


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_randomized_equivalence_with_uniform_dy(seed):
    rng = np.random.default_rng(seed)
    N = 500
    x = np.linspace(4000.0, 5000.0, N)
    y_true = np.sin(x / 200.0) * 1.5 + 0.001 * (x - 4500.0)
    y = y_true + rng.normal(scale=0.02, size=N)
    dy = np.full_like(x, 0.02)

    w = 31
    p = 2

    y_custom = smoothing.weighted_savgol_filter.__wrapped__(x, y, dy, w=w, p=p, mask=None, interpolate=False)
    y_scipy = sps.savgol_filter(y, window_length=w, polyorder=p, mode='interp')

    left = w // 2
    right = len(x) - w // 2
    assert_allclose(y_custom[left:right], y_scipy[left:right], rtol=1e-6, atol=1e-8)


def test_stability_under_small_perturbations():
    rng = np.random.default_rng(0)
    N = 400
    x = np.linspace(4000.0, 5000.0, N)
    y_true = np.sin(x / 200.0) * 1.5
    base_noise = rng.normal(scale=0.02, size=N)
    y0 = y_true + base_noise
    y1 = y_true + base_noise + rng.normal(scale=1e-4, size=N)

    dy = np.full_like(x, 0.02)
    w = 31
    p = 2

    out0 = smoothing.weighted_savgol_filter.__wrapped__(x, y0, dy, w=w, p=p, mask=None, interpolate=False)
    out1 = smoothing.weighted_savgol_filter.__wrapped__(x, y1, dy, w=w, p=p, mask=None, interpolate=False)

    # Small perturbations should only cause small changes in smoothed output
    diff = np.max(np.abs(out0 - out1))
    assert diff < 1e-3
