import numpy as np
import pytest
from quasar_typing.numpy import FloatArray, FloatMatrix, FloatVector

from quasar_utils import resolving


def _explicit_method(
    y: FloatArray, 
    kernels: FloatMatrix, 
    out: FloatArray | None = None,
) -> FloatArray:
    out = np.empty_like(y) if out is None else out
    w = kernels.shape[1]
    l = w // 2

    padded_y = np.pad(y, [(0,0)]*(y.ndim-1) + [(l, l)])

    for i in range(y.shape[-1]):
        window = padded_y[..., i:i+w]
        out[..., i] = (window * kernels[i]).sum(axis=-1)

    return out


def test_create_kernels(
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    w: int,
    n_sigma: float,
):
    kernels = resolving.create_kernels(R, x, dx, w=w, n_sigma=n_sigma)
    assert kernels.shape[0] == x.size
    assert kernels.shape[1] == w
    assert (kernels.sum(axis=-1) <= 1).all()

def test_application(
    subtests: pytest.Subtests,
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    w: int,
    n_sigma: float,
    rng: np.random.Generator,
):
    kernels = resolving.create_kernels(R, x, dx, w=w, n_sigma=n_sigma)

    for i in range(100):
        y = rng.normal(size=x.size)
        with subtests.test(i):
            test = resolving.apply_resolution(y, kernels)
            control = _explicit_method(y, kernels)
            np.testing.assert_allclose(test, control)
            