import numpy as np
import pytest
from quasar_typing.numpy import FloatVector

from quasar_utils import resolving


def test_mask_kernels(
    subtests: pytest.Subtests,
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    w: int,
    n_sigma: float,
    rng: np.random.Generator,
):
    kernels = resolving.create_kernels(
        R, x, dx, 
        w=w, 
        n_sigma=n_sigma, 
        normalize=False,
    )

    for i in range(100):
        mask = rng.choice([True, False], size=x.size)
        y = rng.normal(size=mask.sum())

        _R = R[mask]
        _x = x[mask]
        _dx = dx[mask]
        with subtests.test(i):
            test = resolving.mask_kernels(kernels, mask)
            control = resolving.create_kernels(
                _R, _x, _dx,
                w=w,
                n_sigma=n_sigma,
                normalize=False,
            )
            assert test.shape == control.shape
            # assert np.allclose(test, control)

            for kt, kc in zip(test, control):
                assert np.allclose(kt, kc)

            test = resolving.apply_resolution(y, test)
            control = resolving.apply_resolution(y, control)
            assert np.isclose(np.dot(test, _dx), np.dot(control, _dx))