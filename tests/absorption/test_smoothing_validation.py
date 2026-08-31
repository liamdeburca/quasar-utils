import numpy as np
import pytest

from quasar_utils.absorption import smoothing
from quasar_typing.errors import SmoothingError


def synth_spectrum(N=100):
    x = np.linspace(4000.0, 5000.0, N)
    y_true = np.sin(x / 200.0) * 1.5 + 0.001 * (x - 4500.0)
    rng = np.random.default_rng(0)
    y = y_true + rng.normal(scale=0.02, size=N)
    dy = np.full_like(x, 0.02)
    return x, y, dy


def test_even_window_raises():
    x, y, dy = synth_spectrum(50)
    with pytest.raises(SmoothingError, match="odd"):
        smoothing.weighted_savgol_filter(x, y, dy, w=20, p=2)


def test_w_le_p_raises():
    x, y, dy = synth_spectrum(50)
    with pytest.raises(SmoothingError, match="greater than polynomial order"):
        smoothing.weighted_savgol_filter(x, y, dy, w=3, p=3)


def test_short_input_raises():
    x, y, dy = synth_spectrum(10)
    # choose w larger than length
    with pytest.raises(SmoothingError, match="at least window size"):
        smoothing.weighted_savgol_filter(x, y, dy, w=21, p=2)


def test_no_valid_indices_returns_original():
    x, y, dy = synth_spectrum(50)
    # Make dy invalid (<=0) so mask yields no valid indices
    bad_dy = np.zeros_like(dy)
    # interpolate=False should return original array (copy)
    out = smoothing.weighted_savgol_filter(x, y, bad_dy, w=11, p=2, mask=None, interpolate=False)
    assert np.allclose(out, y)
