import numpy as np
import pytest

from quasar_utils.absorption import smoothing


def test_interpolate_missing_no_points():
    x = np.linspace(0, 1, 5)
    y = np.arange(5).astype(float)
    mask = np.array([False, False, False, False, False])
    # Current behaviour: numpy.interp would fail; we expect function to raise or handle.
    with pytest.raises(ValueError):
        smoothing.interpolate_missing.__wrapped__(x, y, mask)


def test_interpolate_missing_single_point():
    x = np.linspace(0, 1, 5)
    y = np.arange(5).astype(float)
    mask = np.array([False, False, True, False, False])
    out = smoothing.interpolate_missing.__wrapped__(x, y, mask)
    assert np.allclose(out, np.full_like(y, y[2]))


def test_create_slides_shapes_and_masking():
    x = np.linspace(0, 1, 11)
    y = np.sin(x)
    dy = np.full_like(x, 0.1)
    w = 5
    x_slides, y_slides, dy_slides = smoothing.create_slides.__wrapped__(x, y, dy, w)
    assert x_slides.shape[1] == w
    assert y_slides.shape[1] == w
    assert dy_slides.shape[1] == w

    # mask filtering
    mask = np.zeros_like(x, dtype=bool)
    mask[3:8] = True
    xs2, ys2, dys2 = smoothing.create_slides.__wrapped__(x, y, dy, w, mask=mask)
    assert xs2.shape[0] == mask.sum()
