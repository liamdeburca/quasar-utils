import numpy as np

from quasar_utils.absorption import absorption


def test_remove_single_behaviour():
    mask = np.array([False, True, False], dtype=bool)
    out = absorption.remove_single.__wrapped__(mask)
    assert not out.any()

    mask = np.array([True, True, True], dtype=bool)
    out = absorption.remove_single.__wrapped__(mask)
    assert np.array_equal(out, mask)


def test_join_regions_merges_close_regions():
    N = 30
    mask = np.zeros(N, dtype=bool)
    mask[10:12] = True
    mask[14:16] = True
    joined = absorption.join_regions.__wrapped__(mask, iterations=1)
    # with iterations=1, separation of 2 F pixels should be joined
    assert joined[10:16].all()


def test_refine_regions_shapes_and_types():
    N = 200
    x = np.linspace(4000, 5000, N)
    rng = np.random.default_rng(0)
    y = np.sin(x / 200.0) + rng.normal(scale=0.1, size=N)
    dy = np.full_like(x, 0.1)
    y_smooth = y.copy()
    y_bg = np.zeros_like(y)
    mask = np.zeros_like(y, dtype=bool)
    # inject an anomaly region
    mask[50:55] = True

    new_mask, new_y_smooth = absorption.refine_regions.__wrapped__(mask, x, y, dy, y_smooth, y_bg, valid_pixels=None)
    assert new_mask.dtype == bool
    assert new_mask.shape == mask.shape
    assert new_y_smooth.shape == y_smooth.shape
