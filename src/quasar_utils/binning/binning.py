from math import floor

from functools import wraps

from numpy import arange, empty, exp, float64, int32, log, roll, stack
from numpy.typing import NDArray
from quasar_typing.numpy import FloatVector, SortedFloatVector
from quasar_typing.scipy import csr_matrix_
from scipy.sparse import csr_matrix, diags_array

from ..decorators import validate_call
from .alpha_matrix_elements import (
    _alpha_matrix_elements as cy_alpha_matrix_elements,
    _alpha_matrix_elements_conserved as cy_alpha_matrix_elements_conserved,
)


# Public functions (validated)
@validate_call
def alpha_matrix_sparse(
    x_edges: SortedFloatVector,
    xr_edges: SortedFloatVector,
    *,
    dx: FloatVector | None = None,
    dxr: FloatVector | None = None,
    conserve: bool = False,
) -> csr_matrix_:
    """Construct a sparse alpha resampling matrix between two edge grids.

    This is the validated public wrapper. The implementation is available as
    `_alpha_matrix_sparse` which performs the actual computation without
    Pydantic validation (used internally to avoid double-validation).
    """
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
    """Resample the input data (x, y, dy) onto a logarithmic grid.

    The validated public wrapper. See `_log_resample` for the unvalidated
    implementation used internally.
    """
    return _log_resample(x, y, dy, v_res, dx=dx, conserve=conserve, covariance=covariance)


@validate_call
def lin_dx(x: NDArray[float64]) -> NDArray[float64]:
    """Compute pixel widths for a 1D coordinate array.

    Public validated wrapper. See `_lin_dx` for the unvalidated
    implementation.
    """
    return _lin_dx(x)


@validate_call
def log_edges(
    x: NDArray[float64],
    v_res: float,
    *,
    dx: NDArray[float64] | None = None,
    x_edges: NDArray[float64] | None = None,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Construct logarithmically-spaced resampling edges.

    Public validated wrapper. See `_log_edges` for the unvalidated
    implementation.
    """
    return _log_edges(x, v_res, dx=dx, x_edges=x_edges)


# Private implementations (no pydantic validation)
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
        dx = lin_dx(x)

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


def alpha_matrix_elements(
    x: NDArray[float64],
    xr: NDArray[float64],
    dx: NDArray[float64],
) -> tuple[NDArray[int32], NDArray[int32], NDArray[float64]]:
    """
    Compute the non-zero elements of the alpha resampling matrix for
    logarithmic binning.

    Returns row_indices, col_indices, values for sparse matrix
    construction.
    """
    nx = x.size - 1
    nxr = xr.size - 1

    i_indices = empty(nx + nxr, dtype=int32)
    j_indices = empty(nx + nxr, dtype=int32)
    vals = empty(nx + nxr, dtype=float64)

    count = cy_alpha_matrix_elements(x, xr, dx, i_indices, j_indices, vals)

    return i_indices[:count], j_indices[:count], vals[:count]


def alpha_matrix_elements_conserved(
    x: NDArray[float64],
    xr: NDArray[float64],
    dxr: NDArray[float64],
) -> tuple[NDArray[int32], NDArray[int32], NDArray[float64]]:
    """
    Compute the non-zero elements of the alpha resampling matrix for
    logarithmic binning using conserved weights (dxr).

    Returns row_indices, col_indices, values for sparse matrix
    construction.
    """
    nx = x.size - 1
    nxr = xr.size - 1

    i_indices = empty(nx + nxr, dtype=int32)
    j_indices = empty(nx + nxr, dtype=int32)
    vals = empty(nx + nxr, dtype=float64)

    count = cy_alpha_matrix_elements_conserved(
        x, xr, dxr, i_indices, j_indices, vals
    )

    return i_indices[:count], j_indices[:count], vals[:count]


def _alpha_matrix_elements(
    x: NDArray[float64],
    xr: NDArray[float64],
    dx: NDArray[float64],
) -> tuple[NDArray[int32], NDArray[int32], NDArray[float64]]:
    """Backward-compatible wrapper name used internally by the module.

    This small shim calls the public `alpha_matrix_elements` helper. It
    exists to provide a stable internal name while avoiding exporting
    the helper at package level.
    """
    return alpha_matrix_elements(x, xr, dx)


def _alpha_matrix_elements_conserved(
    x: NDArray[float64],
    xr: NDArray[float64],
    dxr: NDArray[float64],
) -> tuple[NDArray[int32], NDArray[int32], NDArray[float64]]:
    """Backward-compatible wrapper name used internally by the module.

    See `_alpha_matrix_elements` for details.
    """
    return alpha_matrix_elements_conserved(x, xr, dxr)


### Private helpers (no validation) - low-level helpers kept last
@wraps(lin_dx)
def _lin_dx(x: NDArray[float64]) -> NDArray[float64]:
    dx = 0.5 * (roll(x, -1) - roll(x, 1))
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    return dx


@wraps(log_edges)
def _log_edges(
    x: NDArray[float64],
    v_res: float,
    *,
    dx: NDArray[float64] | None = None,
    x_edges: NDArray[float64] | None = None,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:

    if dx is None:
        dx = _lin_dx(x)
    else:
        assert dx.size == x.size
    if x_edges is None:
        x_edges = empty(x.size + 1, dtype=float64)
        x_edges[:-1] = x - dx / 2
        x_edges[-1] = x[-1] + dx[-1] / 2
    else:
        assert x_edges.size == x.size + 1

    n_xr: int = floor(log(x_edges[-1] / x_edges[0]) / log(1 + v_res))
    log_xr_edges = log(x_edges[0]) + log(1 + v_res) * arange(n_xr + 1)
    xr_edges = exp(log_xr_edges)
    xr = exp(0.5 * (log_xr_edges[:-1] + log_xr_edges[1:]))

    return x_edges, xr_edges, xr
