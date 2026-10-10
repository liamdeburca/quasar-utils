from functools import wraps
from math import floor, log

from numpy import (
    arange,
    empty,
    exp,
    float64,
    gradient,
    int32,
    stack,
)
from pydantic import validate_call
from quasar_typing.numpy import FloatVector, IntVector, SortedFloatVector
from quasar_typing.scipy import csr_matrix_
from scipy.sparse import csr_matrix, diags_array

from .alpha_matrix_elements import (
    _alpha_matrix_elements as cy_alpha_matrix_elements,
)
from .alpha_matrix_elements import (
    _alpha_matrix_elements_conserved as cy_alpha_matrix_elements_conserved,
)

### Public functions

@validate_call
def log_edges(
    x: SortedFloatVector,
    v_res: float,
    *,
    dx: FloatVector | None = None,
    x_edges: SortedFloatVector | None = None,
) -> tuple[SortedFloatVector, SortedFloatVector, SortedFloatVector]:
    """Construct edge arrays for logarithmic resampling."""
    return _log_edges(x, v_res, dx=dx, x_edges=x_edges)


@validate_call
def alpha_matrix_elements(
    x: SortedFloatVector,
    xr: SortedFloatVector,
    dx: FloatVector,
) -> tuple[IntVector, IntVector, FloatVector]:
    return _alpha_matrix_elements(x, xr, dx)

@validate_call
def alpha_matrix_elements_conserved(
    x: SortedFloatVector,
    xr: SortedFloatVector,
    dxr: FloatVector,
) -> tuple[IntVector, IntVector, FloatVector]:
    return _alpha_matrix_elements_conserved(x, xr, dxr)

@validate_call
def alpha_matrix_sparse(
    x_edges: SortedFloatVector,
    xr_edges: SortedFloatVector,
    *,
    dx: FloatVector | None = None,
    dxr: FloatVector | None = None,
    conserve: bool = False,
) -> csr_matrix_:
    """Construct a sparse alpha resampling matrix between two edge grids."""
    return _alpha_matrix_sparse(x_edges, xr_edges, dx=dx, dxr=dxr, conserve=conserve)


@validate_call
def log_resample(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    v_res: float,
    *,
    dx: FloatVector | None = None,
    conserve: bool = True,
    covariance: bool = False,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """Resample the input data (x, y, dy) onto a logarithmic grid."""
    return _log_resample(x, y, dy, v_res, dx=dx, conserve=conserve, covariance=covariance)

### Private functions

@wraps(log_edges)
def _log_edges(
    x: SortedFloatVector,
    v_res: float,
    *,
    dx: FloatVector | None = None,
    x_edges: SortedFloatVector | None = None,
) -> tuple[SortedFloatVector, SortedFloatVector, SortedFloatVector]:
    if dx is None:
        dx = gradient(x)
    assert dx.size == x.size

    if x_edges is None:
        x_edges = empty(x.size + 1, dtype=float64)
        x_edges[:-1] = x - dx / 2
        x_edges[-1] = x[-1] + dx[-1] / 2
    assert x_edges.size == x.size + 1

    n_xr: int = floor(log(x_edges[-1] / x_edges[0]) / log(1 + v_res))
    log_xr_edges = log(x_edges[0]) + log(1 + v_res) * arange(n_xr + 1)
    xr_edges = exp(log_xr_edges)
    xr = exp(0.5 * (log_xr_edges[:-1] + log_xr_edges[1:]))

    return x_edges, xr_edges, xr


@wraps(alpha_matrix_elements)
def _alpha_matrix_elements(
    x: SortedFloatVector,
    xr: SortedFloatVector,
    dx: FloatVector,
) -> tuple[IntVector, IntVector, FloatVector]:
    nx = x.size - 1
    nxr = xr.size - 1

    i_indices = empty(nx + nxr, dtype=int32)
    j_indices = empty(nx + nxr, dtype=int32)
    vals = empty(nx + nxr, dtype=float64)

    count = cy_alpha_matrix_elements(x, xr, dx, i_indices, j_indices, vals)

    return i_indices[:count], j_indices[:count], vals[:count]

@wraps(alpha_matrix_elements_conserved)
def _alpha_matrix_elements_conserved(
    x: SortedFloatVector,
    xr: SortedFloatVector,
    dxr: FloatVector,
) -> tuple[IntVector, IntVector, FloatVector]:
    nx = x.size - 1
    nxr = xr.size - 1

    i_indices = empty(nx + nxr, dtype=int32)
    j_indices = empty(nx + nxr, dtype=int32)
    vals = empty(nx + nxr, dtype=float64)

    count = cy_alpha_matrix_elements_conserved(
        x, xr, dxr, i_indices, j_indices, vals
    )

    return i_indices[:count], j_indices[:count], vals[:count]


@wraps(alpha_matrix_sparse)
def _alpha_matrix_sparse(
    x_edges: SortedFloatVector,
    xr_edges: SortedFloatVector,
    *,
    dx: FloatVector | None = None,
    dxr: FloatVector | None = None,
    conserve: bool = False,
) -> csr_matrix_:
    if conserve:
        assert dxr is not None
        i, j, data = _alpha_matrix_elements_conserved(x_edges, xr_edges, dxr)
    else:
        assert dx is not None
        i, j, data = _alpha_matrix_elements(x_edges, xr_edges, dx)

    ij = stack([i, j], axis=0, dtype=int32)
    return csr_matrix((data, ij), shape=(xr_edges.size - 1, x_edges.size - 1))


@wraps(log_resample)
def _log_resample(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    v_res: float,
    *,
    dx: FloatVector | None = None,
    conserve: bool = True,
    covariance: bool = False,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    if dx is None:
        dx = gradient(x)

    x_edges, xr_edges, xr = log_edges(x, v_res, dx=dx)
    alpha_matrix = _alpha_matrix_sparse(
        x_edges,
        xr_edges,
        dx=dx,
        dxr=xr * v_res,
        conserve=conserve,
    )

    cov = diags_array(dy**2) if (dy.ndim == 1) else csr_matrix(dy)

    yr = alpha_matrix.dot(y)
    covr = alpha_matrix.dot(cov.dot(alpha_matrix.T))
    dyr = covr if covariance else covr.diagonal() ** 0.5

    return xr, yr, dyr