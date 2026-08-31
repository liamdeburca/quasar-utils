import numpy as np

from quasar_utils.absorption import absorption


def test_fft_approach_detects_localized_negative_excursion():
    N = 1024
    rng = np.random.default_rng(0)
    z = rng.normal(size=N)
    center = 500
    z[center - 3 : center + 3] += -5.0
    ps, mask = absorption.fft_approach.__wrapped__(z, p_crit=1e-2, z_crit=-2, w=25)
    assert mask[center]
    assert ps[center] < 1e-2


def test_fft_approach_does_not_mark_edges():
    N = 200
    rng = np.random.default_rng(0)
    z = rng.normal(size=N)
    # inject anomaly near edge
    z[2:6] += -5.0
    ps, mask = absorption.fft_approach.__wrapped__(z, p_crit=1e-2, z_crit=-2, w=25)
    # not_edge masks out the first/last w pixels; ensure no True in first w pixels
    assert not mask[:25].any()
