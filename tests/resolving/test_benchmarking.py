import timeit

import numpy as np
import pytest
from quasar_typing.numpy import FloatVector

from quasar_utils.resolving import (
    _apply_resolution,
    _create_kernels,
    _mask_kernels,
)


class TestMaskKernels:
    def test_benchmark(
        self,
        subtests: pytest.Subtests,
        R: FloatVector,
        x: FloatVector,
        dx: FloatVector,
        w: int,
        n_sigma: float,
        rng: np.random.Generator,
    ):
        kernels = _create_kernels(
            R, x, dx,
            w=w,
            n_sigma=n_sigma,
            normalize=False,
        )
        mask = rng.choice([True, False], size=x.size)
        f = lambda: _mask_kernels(kernels, mask)

        t = min(timeit.repeat(f, number=1000, repeat=1))
        with subtests.test(f"`mask_kernels`: {t:.1f}"):
            pass


class TestCreateKernels:
    def test_benchmark(
        self,
        subtests: pytest.Subtests,
        R: FloatVector,
        x: FloatVector,
        dx: FloatVector,
        w: int,
        n_sigma: float,
    ):
        f = lambda: _create_kernels(
            R, x, dx,
            w=w,
            n_sigma=n_sigma,
            normalize=False,
        )

        t = min(timeit.repeat(f, number=1000, repeat=1))
        with subtests.test(f"`create_kernels`: {t:.1f}"):
            pass


class TestApplyResolution:
    def test_benchmark(
        self,
        subtests: pytest.Subtests,
        R: FloatVector,
        x: FloatVector,
        dx: FloatVector,
        w: int,
        n_sigma: float,
        rng: np.random.Generator,
    ):
        kernels = _create_kernels(
            R, x, dx,
            w=w,
            n_sigma=n_sigma,
            normalize=False,
        )
        y = rng.random(x.size)
        f = lambda: _apply_resolution.__wrapped__(y, kernels)

        t = min(timeit.repeat(f, number=1000, repeat=1))
        with subtests.test(f"`apply_resolution`: {t:.1f}"):
            pass