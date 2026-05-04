"""
Test to validate that Cython implementation matches Numba implementation
for interpolation matrix creation.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

# Import old (Numba) implementation
from quasar_utils.interpolation.interpolation import create_interp_matrix as create_interp_matrix_numba

# Import new (Cython) implementation
from quasar_utils.interpolation import create_interp_matrix as create_interp_matrix_cython


def test_cython_vs_numba_no_bias():
    """
    Test that Cython and Numba implementations produce identical results
    for the no-bias case (left=right=0).
    """
    # Create random sorted arrays
    np.random.seed(42)
    x = np.sort(np.random.uniform(0, 100, 50)).astype(np.float64)
    xb = np.sort(np.random.uniform(0, 100, 75)).astype(np.float64)
    
    # Create matrices using both implementations
    M_numba, bias_numba = create_interp_matrix_numba(x, xb, left=0.0, right=0.0)
    M_cython, bias_cython = create_interp_matrix_cython(x, xb, left=0.0, right=0.0)
    
    # Verify sparse matrix data is identical
    assert M_numba.shape == M_cython.shape, "Shape mismatch"
    assert np.array_equal(M_numba.data, M_cython.data), "Sparse matrix data mismatch"
    assert np.array_equal(M_numba.indices, M_cython.indices), "Column indices mismatch"
    assert np.array_equal(M_numba.indptr, M_cython.indptr), "Row pointers mismatch"
    
    # Verify bias arrays are identical
    assert np.array_equal(bias_numba, bias_cython), "Bias arrays mismatch"
    
    # Verify the matrices produce identical dense representations
    assert np.allclose(M_numba.toarray(), M_cython.toarray()), "Dense matrix mismatch"


def test_cython_vs_numba_no_bias_with_extrapolation_points():
    """
    Test with xb values that include points outside the range of x
    to ensure the no-bias case correctly handles them.
    """
    np.random.seed(123)
    x = np.sort(np.random.uniform(10, 90, 40)).astype(np.float64)
    # xb includes some points outside [x.min(), x.max()]
    xb = np.sort(np.concatenate([
        np.array([0, 5]),  # left extrapolation
        np.random.uniform(x.min(), x.max(), 50),
        np.array([95, 100])  # right extrapolation
    ])).astype(np.float64)
    
    # Create matrices using both implementations
    M_numba, bias_numba = create_interp_matrix_numba(x, xb, left=0.0, right=0.0)
    M_cython, bias_cython = create_interp_matrix_cython(x, xb, left=0.0, right=0.0)
    
    # Verify sparse matrix data is identical
    assert M_numba.shape == M_cython.shape, "Shape mismatch"
    assert np.array_equal(M_numba.data, M_cython.data), "Sparse matrix data mismatch"
    assert np.array_equal(M_numba.indices, M_cython.indices), "Column indices mismatch"
    assert np.array_equal(M_numba.indptr, M_cython.indptr), "Row pointers mismatch"
    
    # Verify bias arrays are identical
    assert np.array_equal(bias_numba, bias_cython), "Bias arrays mismatch"
    
    # Verify the matrices produce identical dense representations
    assert np.allclose(M_numba.toarray(), M_cython.toarray()), "Dense matrix mismatch"


def test_cython_vs_numba_small_arrays():
    """
    Test with small arrays to ensure correctness in edge cases.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    xb = np.array([0.0, 0.5, 1.5, 2.5, 3.0])
    
    M_numba, bias_numba = create_interp_matrix_numba(x, xb, left=0.0, right=0.0)
    M_cython, bias_cython = create_interp_matrix_cython(x, xb, left=0.0, right=0.0)
    
    assert M_numba.shape == M_cython.shape, "Shape mismatch"
    assert np.array_equal(M_numba.data, M_cython.data), "Sparse matrix data mismatch"
    assert np.array_equal(M_numba.indices, M_cython.indices), "Column indices mismatch"
    assert np.array_equal(M_numba.indptr, M_cython.indptr), "Row pointers mismatch"
    assert np.array_equal(bias_numba, bias_cython), "Bias arrays mismatch"


def test_cython_vs_numpy_interp():
    """
    Test that Cython matrix interpolation produces identical results to NumPy.interp.
    """
    # Create test data
    np.random.seed(42)
    x = np.sort(np.random.uniform(0, 100, 50)).astype(np.float64)
    xb = np.sort(np.random.uniform(0, 100, 75)).astype(np.float64)
    values = np.random.uniform(-10, 10, len(x)).astype(np.float64)
    
    # Get NumPy.interp result
    numpy_result = np.interp(xb, x, values, left=0.0, right=0.0)
    
    # Get Cython matrix result
    M, bias = create_interp_matrix_cython(x, xb, left=0.0, right=0.0)
    cython_result = M @ values + bias
    
    # Results should be identical (within floating point precision)
    assert np.allclose(numpy_result, cython_result, rtol=1e-14, atol=1e-14), \
        f"Results differ: numpy={numpy_result}, cython={cython_result}"


def test_cython_vs_numpy_interp_with_extrapolation():
    """
    Test Cython vs NumPy.interp with points outside the interpolation range.
    """
    np.random.seed(123)
    x = np.sort(np.random.uniform(10, 90, 40)).astype(np.float64)
    xb = np.sort(np.concatenate([
        np.array([0, 5]),  # left extrapolation
        np.random.uniform(x.min(), x.max(), 50),
        np.array([95, 100])  # right extrapolation
    ])).astype(np.float64)
    values = np.random.uniform(-5, 5, len(x)).astype(np.float64)
    
    # Define extrapolation values
    left_val = -1.0
    right_val = 2.0
    
    # Get NumPy.interp result
    numpy_result = np.interp(xb, x, values, left=left_val, right=right_val)
    
    # Get Cython matrix result
    M, bias = create_interp_matrix_cython(x, xb, left=left_val, right=right_val)
    cython_result = M @ values + bias
    
    # Results should be identical
    assert np.allclose(numpy_result, cython_result, rtol=1e-14, atol=1e-14), \
        f"Results differ: numpy={numpy_result}, cython={cython_result}"


def test_cython_vs_numpy_interp_small_arrays():
    """
    Test Cython vs NumPy.interp with small known arrays for easy debugging.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    xb = np.array([0.0, 0.5, 1.5, 2.5, 3.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Get NumPy.interp result
    numpy_result = np.interp(xb, x, values)
    
    # Get Cython matrix result
    M, bias = create_interp_matrix_cython(x, xb, left=0.0, right=0.0)
    cython_result = M @ values + bias
    
    # Results should be identical
    assert np.allclose(numpy_result, cython_result, rtol=1e-14, atol=1e-14), \
        f"Expected {numpy_result}, got {cython_result}"


if __name__ == "__main__":
    test_cython_vs_numba_no_bias()
    test_cython_vs_numba_no_bias_with_extrapolation_points()
    test_cython_vs_numba_small_arrays()
    test_cython_vs_numpy_interp()
    test_cython_vs_numpy_interp_with_extrapolation()
    test_cython_vs_numpy_interp_small_arrays()
    print("All tests passed!")
