"""
Python file containing utilities for performing efficient linear interpolation
through sparse matrix multiplication.
"""

__all__ = ["build_interp_data", "create_interp_matrix"]

from functools import wraps

import numpy as np
from pydantic import validate_call
from quasar_typing.numpy import (
    FittableFloatVector,
    FloatVector,
    IntVector,
    SortedFloatVector,
)
from quasar_typing.scipy import CSRMatrix
from scipy.sparse import csr_matrix

### Public functions (validated)


@validate_call
def build_interp_data(
    x: SortedFloatVector,
    xb: SortedFloatVector,
    left: float = 0.0,
    right: float = 0.0,
) -> tuple[IntVector, IntVector, FloatVector, FloatVector]:
    """Validated public wrapper that prepares indices and values for
    constructing a sparse interpolation matrix.

    See `_build_interp_data` for the unvalidated implementation.
    """
    return _build_interp_data(x, xb, left=left, right=right)


@validate_call
def create_interp_matrix(
    x_in: SortedFloatVector,
    x_out: FittableFloatVector,
    left: float = 0,
    right: float = 0,
) -> tuple[CSRMatrix, FittableFloatVector]:
    """Validated public wrapper that returns a CSR interpolation matrix and
    bias vector.

    The implementation is in `_create_interp_matrix`.
    """
    return _create_interp_matrix(x_in, x_out, left=left, right=right)


### Private implementations (unvalidated)


@wraps(build_interp_data)
def _build_interp_data(
    x: SortedFloatVector,
    xb: SortedFloatVector,
    left: float = 0.0,
    right: float = 0.0,
) -> tuple[IntVector, IntVector, FloatVector, FloatVector]:
    indices = np.empty(xb.size, dtype=np.int32)
    i_indices = np.empty(2 * xb.size, dtype=np.int32)
    j_indices = np.empty(2 * xb.size, dtype=np.int32)
    vals = np.empty(2 * xb.size, dtype=np.float64)

    bias = np.zeros(xb.size, dtype=np.float64)
    if left == 0.0 and right == 0.0:
        from .interp_matrix_elements import (
            _interp_matrix_elements_no_bias as func_no_bias,
        )
        count = func_no_bias(
            x,
            xb,
            indices,
            i_indices,
            j_indices,
            vals,
        )
    else:
        from .interp_matrix_elements import _interp_matrix_elements as func
        count = func(
            x,
            xb,
            indices,
            i_indices,
            j_indices,
            vals,
            bias,
            left,
            right,
        )

    return i_indices[:count], j_indices[:count], vals[:count], bias


@wraps(create_interp_matrix)
def _create_interp_matrix(
    x_in: SortedFloatVector,
    x_out: FittableFloatVector,
    left: float = 0,
    right: float = 0,
) -> tuple[CSRMatrix, FittableFloatVector]:
    row_indices, col_indices, values, bias = _build_interp_data(
        x_in,
        x_out,
        left,
        right,
    )
    return csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(x_out.size, x_in.size),
    ), bias
