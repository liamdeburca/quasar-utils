from numpy import roll, float64, empty, log, arange, exp, stack, int32
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags_array

from quasar_typing.numpy import FloatVector, SortedFloatVector
from quasar_typing.scipy import csr_matrix_

from ..decorators import validate_call
from .alpha_matrix_elements import _alpha_matrix_elements

def lin_dx(x: NDArray[float64]) -> NDArray[float64]:
    dx = 0.5 * (roll(x, -1) - roll(x, 1))
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    return dx

def log_edges(
    x: NDArray[float64], 
    v_res: float,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    dx = lin_dx(x)
    x_edges = empty(x.size + 1, dtype=float64)
    x_edges[:-1] = x - dx / 2
    x_edges[-1] = x[-1] + dx[-1] / 2

    n_xr = log(x_edges[-1] / x_edges[0]) // log(1 + v_res) + 1
    log_xr_edges = log(x_edges[0]) + log(1 + v_res) * arange(n_xr + 1)
    xr_edges = exp(log_xr_edges)
    xr = exp(0.5 * (log_xr_edges[:-1] + log_xr_edges[1:]))

    return x_edges, xr_edges, xr

def alpha_matrix_elements(
    x: NDArray[float64], 
    xr: NDArray[float64],
) -> tuple[NDArray[int32], NDArray[int32], NDArray[float64]]:
    """
    Numba-optimised function to compute the non-zero elements of the alpha 
    resampling matrix for logarithmic binning.

    Returns row_indices, col_indices, values and bias for sparse matrix 
    construction.
    """
    nx  = x.size - 1
    nxr = xr.size - 1

    i_indices = empty(nx + nxr, dtype=int32)
    j_indices = empty(nx + nxr, dtype=int32)
    vals = empty(nx + nxr, dtype=float64)

    count = _alpha_matrix_elements(x, xr, i_indices, j_indices, vals)

    return i_indices[:count], j_indices[:count], vals[:count]

@validate_call
def alpha_matrix_sparse(
    x_edges: SortedFloatVector,
    xr_edges: SortedFloatVector,
) -> csr_matrix_:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    i, j, data = alpha_matrix_elements(x_edges, xr_edges)
    ij = stack([i, j], axis=0, dtype=int32)

    return csr_matrix(
        (data, ij), 
        shape = (xr_edges.size - 1, x_edges.size - 1),
    )

@validate_call
def log_resample(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    v_res: float,
    *,
    conserve: bool = False,
    covariance: bool = False,
) -> tuple[FloatVector, FloatVector, FloatVector]:
    """
    ** PYDANTIC VALIDATED METHOD **

    Resample the input data (x, y, dy) onto a logarithmic grid defined by the 
    velocity resolution v_res. The function returns the resampled x, y, and dy 
    (or covariance)
    """
    x_edges, xr_edges, xr = log_edges(x, v_res)
    alpha_matrix = alpha_matrix_sparse(x_edges, xr_edges)

    if conserve: 
        alpha_matrix /= alpha_matrix.sum(axis=1)[:,None]

    cov = diags_array(dy**2) if (dy.ndim == 1) else csr_matrix(dy)

    yr = alpha_matrix.dot(y)
    covr = alpha_matrix.dot(cov.dot(alpha_matrix.T))
    dyr = covr if covariance else covr.diagonal()**0.5

    return xr, yr, dyr