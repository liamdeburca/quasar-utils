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
    kernels = resolving.create_kernels(
        R, x, dx, 
        w=w, 
        n_sigma=n_sigma, 
        normalize=True,
        sparse=False,
    )
    assert kernels.shape[0] == x.size
    assert kernels.shape[1] == w
    assert np.allclose(kernels.sum(axis=1), 1)

def test_create_kernels_sparse(
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    w: int,
    n_sigma: float,
):
    kernels = resolving.create_kernels(
        R, x, dx, 
        w=w, 
        n_sigma=n_sigma, 
        normalize=True,
        sparse=True,
    )
    assert kernels.shape[0] == x.size
    assert kernels.shape[1] == w
    assert np.allclose(kernels.sum(axis=1), 1)

def test_application(
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
        normalize=True,
        sparse=False,
    )

    for i in range(100):
        y = rng.normal(size=x.size)
        with subtests.test(i):
            test = resolving.apply_resolution(y, kernels)
            control = _explicit_method(y, kernels)
            assert np.allclose(test, control)
            assert np.isclose(np.dot(test, dx), np.dot(control, dx))

def test_application_sparse(
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
        normalize=True,
        sparse=True,
    )

    for i in range(100):
        y = rng.normal(size=x.size)
        with subtests.test(i):
            test = resolving.apply_resolution(y, kernels)
            control = _explicit_method(y, kernels.toarray())
            assert np.allclose(test, control)
            assert np.isclose(np.dot(test, dx), np.dot(control, dx))
            