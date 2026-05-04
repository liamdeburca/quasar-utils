"""
File for performing flux-density or flux conserving logarithmic binning. 
"""

__all__ = ["log_resample"]

from numpy import empty, arange, log, exp, roll, stack, float64, int_
from numpy.typing import NDArray
from numba import njit
from scipy.sparse import csr_matrix, diags_array

from pydantic import validate_call
from quasar_typing.numpy import FloatVector, SortedFloatVector
from quasar_typing.scipy import csr_matrix_

###

@njit(fastmath=True)
def lin_dx(x: NDArray[float64]) -> NDArray[float64]:
    """
    ** NUMBA OPTIMISED FUNCTION **
    """
    dx = 0.5 * (roll(x, -1) - roll(x, 1))
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    return dx
    
@njit(fastmath=True)
def log_edges(
    x: NDArray[float64], 
    v_res: float,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """
    ** NUMBA OPTIMISED FUNCTION **
    """
    dx = lin_dx(x)
    x_edges     = empty(x.size + 1, dtype=float64)
    x_edges[:-1] = x - dx / 2
    x_edges[-1] = x[-1] + dx[-1] / 2

    n_xr = log(x_edges[-1] / x_edges[0]) // log(1 + v_res) + 1
    log_xr_edges = log(x_edges[0]) + log(1 + v_res) * arange(n_xr + 1)
    xr_edges = exp(log_xr_edges)
    xr = exp(0.5 * (log_xr_edges[:-1] + log_xr_edges[1:]))

    return x_edges, xr_edges, xr

@njit(fastmath=True)
def alpha_matrix_elements(
    x: NDArray[float64], 
    xr: NDArray[float64],
) -> tuple[NDArray[int_], NDArray[int_], NDArray[float64]]:
    """
    ** NUMBA OPTIMISED FUNCTION **

    Numba-optimised function to compute the non-zero elements of the alpha 
    resampling matrix for logarithmic binning.

    Returns row_indices, col_indices, values and bias for sparse matrix 
    construction.
    """
    nx  = x.size - 1
    nxr = xr.size - 1

    i_indices = empty(nx + nxr, dtype=int_)
    j_indices = empty(nx + nxr, dtype=int_)
    data      = empty(nx + nxr, dtype=float64)

    count: int = 0
    j_start: int = 0
    for i in range(nxr):
        for j in range(j_start, nx):
            if x[j+1] <= xr[i]: 
                continue
            if x[j] >= xr[i+1]:
                break

            i_indices[count] = i
            j_indices[count] = j
            data[count] = (min(x[j+1], xr[i+1]) - max(x[j], xr[i])) \
                / (x[j+1] - x[j])
            count += 1

            if x[j+1] == xr[i+1]:
                j_start = j + 1
                break
            if x[j+1] > xr[i+1]:
                j_start = j
                break

    return i_indices[:count], j_indices[:count], data[:count]

@validate_call
def alpha_matrix_sparse(
    x_edges: SortedFloatVector,
    xr_edges: SortedFloatVector,
) -> csr_matrix_:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    i, j, data = alpha_matrix_elements(x_edges, xr_edges)
    ij = stack([i, j], axis=0, dtype=int_)

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