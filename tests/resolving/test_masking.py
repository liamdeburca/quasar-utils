import timeit

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

    for i in range(10):
        mask = rng.choice([True, False], size=x.size)
        y = rng.normal(size=mask.sum())

        _R = R[mask]
        _x = x[mask]
        _dx = dx[mask]

        test = resolving._mask_kernels(kernels, mask)
        control = resolving._create_kernels(
            _R, _x, _dx,
            w=w,
            n_sigma=n_sigma,
            normalize=False,
        )
        with subtests.test(i):
            assert test.shape == control.shape
            for kt, kc in zip(test, control):
                assert np.allclose(kt, kc)

            test = resolving.apply_resolution(y, test)
            control = resolving.apply_resolution(y, control)
            assert np.allclose(test, control)

    mask = rng.choice([True, False], size=x.size)
    t_masking = timeit.repeat(
        "f()",
        globals={"f": lambda: resolving._mask_kernels(kernels, mask)},
        repeat=1000,
        number=1,
    )
    t_new = timeit.repeat(
        "f()",
        globals={"f": lambda: resolving._create_kernels(
            R[mask], x[mask], dx[mask],
            w=w,
            n_sigma=n_sigma,
            normalize=False,
        )},
        repeat=1000,
        number=1,
    )

    with subtests.test(t_masking=min(t_masking), t_new=min(t_new)):
        pass


def test_delta_function(
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
    signal = np.zeros_like(x)
    idx = rng.choice(x.size)
    signal[idx] = 4.0
    result = resolving.apply_resolution(signal, kernels)
    assert result.shape == signal.shape
    assert np.isclose(np.dot(signal, dx), np.dot(result, dx), rtol=1e-2)
    
    