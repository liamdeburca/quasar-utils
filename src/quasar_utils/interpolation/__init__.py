"""
Python file containing utilities for performing efficient linear interpolation
through sparse matrix multiplication.
"""

__all__ = ["create_interp_matrix"]

from numpy import float64, empty, zeros, int32
from scipy.sparse import csr_matrix

from quasar_typing.numpy import SortedFloatVector, FittableFloatVector, IntVector, FloatVector
from quasar_typing.scipy import csr_matrix_

from ..decorators import validate_call

from .interp_matrix_elements import (
    _interp_matrix_elements_no_bias, 
    _interp_matrix_elements,
)

@validate_call
def build_interp_data(
    x: SortedFloatVector,
    xb: SortedFloatVector,
    left: float = 0.0,
    right: float = 0.0,
) -> tuple[IntVector, IntVector, FloatVector, FloatVector]:    
    indices = empty(xb.size, dtype=int32)
    i_indices = empty(2 * xb.size, dtype=int32)
    j_indices = empty(2 * xb.size, dtype=int32)
    vals = empty(2 * xb.size, dtype=float64)

    bias = zeros(xb.size, dtype=float64)
    if left == 0.0 and right == 0.0:
        count = _interp_matrix_elements_no_bias(
            x, xb, indices, i_indices, j_indices, vals,
        )
    else:
        count = _interp_matrix_elements(
            x, xb, indices, i_indices, j_indices, vals, bias, left, right,
        )
    
    return i_indices[:count], j_indices[:count], vals[:count], bias

@validate_call
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
    row_indices, col_indices, values, bias = build_interp_data.__wrapped__(
        x_in, x_out, left, right,
    )
    return csr_matrix(
        (values, (row_indices, col_indices)), 
        shape = (x_out.size, x_in.size),
    ), bias