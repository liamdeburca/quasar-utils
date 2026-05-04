"""
Test to validate that the Cython implementation of alpha_matrix_elements
produces identical results to the original Numba implementation.
"""

import numpy as np
from numpy.typing import NDArray
import pytest

# Import both implementations
from quasar_utils.binning.binning import alpha_matrix_elements as numba_alpha_matrix_elements
from quasar_utils.binning.alpha_matrix_elements import _alpha_matrix_elements as cython_alpha_matrix_elements


def generate_test_arrays(n_test: int = 100) -> list[tuple[NDArray, NDArray]]:
    """
    Generate 100 different sorted x and xr arrays for testing.
    
    Args:
        n_test: Number of test cases to generate.
    
    Returns:
        List of tuples (x_edges, xr_edges) where both are sorted float arrays.
    """
    test_cases = []
    
    for seed in range(n_test):
        np.random.seed(seed)
        
        # Generate random sorted x_edges (bin edges)
        n_x = np.random.randint(5, 50)
        x_edges = np.sort(np.random.uniform(1, 1000, n_x)).astype(np.float64)
        
        # Generate random sorted xr_edges (resampling bin edges)
        n_xr = np.random.randint(5, 50)
        xr_edges = np.sort(np.random.uniform(x_edges[0] * 0.9, x_edges[-1] * 1.1, n_xr)).astype(np.float64)
        
        test_cases.append((x_edges, xr_edges))
    
    return test_cases


def test_cython_vs_numba_alpha_matrix_elements():
    """
    Test that Cython and Numba implementations produce identical results.
    
    Compares i_indices, j_indices, and vals arrays for 100 different
    x and xr input arrays using numpy.array_equal.
    """
    test_cases = generate_test_arrays(n_test=100)
    
    for idx, (x_edges, xr_edges) in enumerate(test_cases):
        # Get results from Numba implementation
        numba_i, numba_j, numba_vals = numba_alpha_matrix_elements(x_edges, xr_edges)
        
        # Get results from Cython implementation
        # Cython version requires pre-allocated arrays and returns count
        nx = x_edges.size - 1
        nxr = xr_edges.size - 1
        max_elements = nx + nxr
        
        cython_i_arr = np.empty(max_elements, dtype=np.int32)
        cython_j_arr = np.empty(max_elements, dtype=np.int32)
        cython_vals_arr = np.empty(max_elements, dtype=np.float64)
        
        count = cython_alpha_matrix_elements(x_edges, xr_edges, cython_i_arr, cython_j_arr, cython_vals_arr)
        
        cython_i = cython_i_arr[:count].astype(np.int_)
        cython_j = cython_j_arr[:count].astype(np.int_)
        cython_vals = cython_vals_arr[:count]
        
        # Compare i_indices
        assert np.array_equal(numba_i, cython_i), (
            f"Test case {idx}: i_indices differ\n"
            f"Numba:  {numba_i}\n"
            f"Cython: {cython_i}"
        )
        
        # Compare j_indices
        assert np.array_equal(numba_j, cython_j), (
            f"Test case {idx}: j_indices differ\n"
            f"Numba:  {numba_j}\n"
            f"Cython: {cython_j}"
        )
        
        # Compare vals (allowing for small floating-point differences)
        assert np.allclose(numba_vals, cython_vals), (
            f"Test case {idx}: vals differ\n"
            f"Numba:  {numba_vals}\n"
            f"Cython: {cython_vals}\n"
            f"Max diff: {np.max(np.abs(numba_vals - cython_vals))}"
        )


if __name__ == "__main__":
    test_cython_vs_numba_alpha_matrix_elements()
    print("All tests passed!")
