"""
Python file containing utilities for performing efficient linear interpolation
through sparse matrix multiplication.
"""

__all__ = ["create_interp_matrix"]

from numpy import float64, empty, zeros, int_
from numpy.typing import NDArray
from numba import njit
from scipy.sparse import csr_matrix

from pydantic import validate_call
from quasar_typing.numpy import SortedFloatVector, FittableFloatVector
from quasar_typing.scipy import csr_matrix_

@njit(fastmath=True)
def _build_interp_data(
    x_in: NDArray[float64],
    x_out: NDArray[float64], 
    left: float,
    right: float,
) -> tuple[NDArray[int_], NDArray[int_], NDArray[float64], NDArray[float64]]:
    """
    Numba-optimized function to build interpolation matrix data.
    
    Returns row_indices, col_indices, values and bias for sparse matrix 
    construction with extrapolation.
    """
    n_out: int = x_out.size
    n_in:  int = x_in.size
    
    row_indices = empty(2 * n_out, dtype=int_)
    col_indices = empty(2 * n_out, dtype=int_)
    values      = empty(2 * n_out, dtype=float64)
    bias        = zeros(n_out,     dtype=float64)
    
    count: int = 0
    j: int = 0
    indices = empty(n_out, dtype=int_)
    for i, xi in enumerate(x_out):
        while j < n_in and x_in[j] < xi:
            j += 1

        indices[i] = j

    for i, xi in enumerate(x_out):
        idx = indices[i]
        
        if idx == 0:
            if xi == x_in[0]:
                # Exact match at the left boundary
                row_indices[count] = i
                col_indices[count] = 0
                values[count] = 1.0
                count += 1
            else:
                # Left extrapolation
                bias[i] = left
                continue
            
        elif idx >= n_in:
            # Right extrapolation
            bias[i] = right
            continue
            
        else:
            # Interpolate between x_in[idx-1] and x_in[idx]
            x0, x1 = x_in[idx-1:idx+1]

            w0: float = (x1 - xi) / (x1 - x0)
            w1: float = 1.0 - w0
                            
            if w0 != 0.0:
                row_indices[count] = i
                col_indices[count] = idx - 1
                values[count] = w0
                count += 1
            
            if w1 != 0.0:
                row_indices[count] = i
                col_indices[count] = idx
                values[count] = w1
                count += 1
    
    return row_indices[:count], col_indices[:count], values[:count], bias

@validate_call(validate_return=False)
def create_interp_matrix(
    x_in: SortedFloatVector, 
    x_out: FittableFloatVector, 
    left: float = 0, 
    right: float = 0,
) -> tuple[csr_matrix_, FittableFloatVector]:
    """
    Create a sparse linear interpolation matrix M such that:
    M @ _a ≈ np.interp(x_out, x_in, _a, left=left, right=right)
    
    Uses Numba JIT compilation for optimised performance.
    
    Parameters
    ----------
    x_in : array_like
        Source x-coordinates (input space), must be monotonically increasing
    x_out : array_like
        Target x-coordinates (output space)
    left : float, optional
        Value to return for x < _x[0], default is 0
    right : float, optional
        Value to return for x > _x[-1], default is 0
    
    Returns
    -------
    M : scipy.sparse.csr_matrix
        Sparse interpolation matrix of shape (len(x_out), len(x_in))

    bias : NDArray[float64]
        Array of bias values for extrapolated points.
    """        
    row_indices, col_indices, values, bias = _build_interp_data(
        x_in, x_out, left, right,
    )
    return csr_matrix(
        (values, (row_indices, col_indices)), 
        shape = (x_out.size, x_in.size),
    ), bias