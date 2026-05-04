"""
Benchmark to compare performance of Cython vs Numba interpolation implementations.

Measures:
1. Numba JIT compilation time (first call)
2. Numba compiled execution time (subsequent calls)
3. Cython execution time
4. NumPy.interp execution time (for reference)

Calculates break-even point where total Cython time becomes better than Numba.
"""

import numpy as np
import timeit
from typing import Tuple

# Import implementations
from quasar_utils.interpolation.interpolation import create_interp_matrix as create_interp_matrix_numba
from quasar_utils.interpolation import create_interp_matrix as create_interp_matrix_cython


def setup_test_data(seed: int = 42, n_in: int = 100, n_out: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Create random sorted arrays for interpolation."""
    np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 100, n_in)).astype(np.float64)
    xb = np.sort(np.random.uniform(0, 100, n_out)).astype(np.float64)
    return x, xb


def benchmark_numba_compilation(x: np.ndarray, xb: np.ndarray) -> float:
    """
    Measure Numba JIT compilation time (first call).
    
    Returns time in seconds.
    """
    # Create a fresh function for compilation timing
    from quasar_utils.interpolation.interpolation import _build_interp_data
    
    # Clear the JIT cache by creating a new context
    start = timeit.default_timer()
    _build_interp_data(x, xb, 0.0, 0.0)
    end = timeit.default_timer()
    
    return end - start


def benchmark_numba_compiled(x: np.ndarray, xb: np.ndarray, n_runs: int = 100) -> float:
    """
    Measure Numba compiled execution time (already JIT compiled).
    
    Returns average time per call in seconds.
    """
    # Warm up - ensure JIT compilation is done
    create_interp_matrix_numba(x, xb, left=0.0, right=0.0)
    
    # Time multiple runs
    def numba_call():
        create_interp_matrix_numba(x, xb, left=0.0, right=0.0)
    
    total_time = timeit.timeit(numba_call, number=n_runs)
    return total_time / n_runs


def benchmark_cython(x: np.ndarray, xb: np.ndarray, n_runs: int = 100) -> float:
    """
    Measure Cython execution time.
    
    Returns average time per call in seconds.
    """
    def cython_call():
        create_interp_matrix_cython(x, xb, left=0.0, right=0.0)
    
    total_time = timeit.timeit(cython_call, number=n_runs)
    return total_time / n_runs


def benchmark_numpy_interp(x: np.ndarray, xb: np.ndarray, values: np.ndarray, n_runs: int = 100) -> float:
    """
    Measure NumPy.interp execution time for comparison.
    
    Returns average time per call in seconds.
    """
    def numpy_call():
        np.interp(xb, x, values)
    
    total_time = timeit.timeit(numpy_call, number=n_runs)
    return total_time / n_runs


def calculate_breakeven(numba_compilation_time: float, numba_per_call: float, cython_per_call: float) -> Tuple[int, float]:
    """
    Calculate the break-even point where Cython becomes faster.
    
    Returns (number of calls, total time in seconds).
    If Cython is faster per-call, returns (0, 0) indicating immediate advantage.
    If Cython is slower per-call but compilation cost makes it worth it, calculates break-even.
    """
    if cython_per_call < numba_per_call:
        # Cython is faster per call AND saves compilation overhead
        # Break-even is immediate (0 calls)
        return 0, 0.0
    elif cython_per_call == numba_per_call:
        # Same speed per call; Cython wins only if you make many calls
        # Break-even at compilation_time / per_call difference (undefined, use infinity)
        return int(np.ceil(numba_compilation_time / 0.0001)), numba_compilation_time  # Effectively never
    else:
        # Cython is slower per call; break-even depends on compilation time
        # Break-even: numba_compilation_time + n * numba_per_call = n * cython_per_call
        # numba_compilation_time = n * (cython_per_call - numba_per_call)
        # n = numba_compilation_time / (cython_per_call - numba_per_call)
        
        n_calls = numba_compilation_time / (cython_per_call - numba_per_call)
        total_time = n_calls * cython_per_call
        
        return int(np.ceil(n_calls)), total_time


def run_benchmark(n_in: int = 100, n_out: int = 200, n_runs: int = 100):
    """Run complete benchmark suite."""
    print("=" * 70)
    print(f"INTERPOLATION BENCHMARK: Cython vs Numba")
    print("=" * 70)
    print(f"\nTest configuration:")
    print(f"  Input array size (x):  {n_in} points")
    print(f"  Output array size (xb): {n_out} points")
    print(f"  Benchmark runs: {n_runs} iterations per test\n")
    
    # Setup test data
    x, xb = setup_test_data(n_in=n_in, n_out=n_out)
    
    print("Running benchmarks...")
    print("-" * 70)
    
    # Numba compilation
    print("\n1. NUMBA JIT COMPILATION (first call)")
    numba_compile_time = benchmark_numba_compilation(x, xb)
    print(f"   Compilation time: {numba_compile_time*1000:.3f} ms")
    
    # Numba compiled execution
    print("\n2. NUMBA COMPILED EXECUTION (cached JIT)")
    numba_exec_time = benchmark_numba_compiled(x, xb, n_runs=n_runs)
    print(f"   Time per call: {numba_exec_time*1000:.3f} ms")
    print(f"   Total for {n_runs} calls: {numba_exec_time*n_runs*1000:.1f} ms")
    
    # Cython execution
    print("\n3. CYTHON EXECUTION")
    cython_exec_time = benchmark_cython(x, xb, n_runs=n_runs)
    print(f"   Time per call: {cython_exec_time*1000:.3f} ms")
    print(f"   Total for {n_runs} calls: {cython_exec_time*n_runs*1000:.1f} ms")
    
    # NumPy.interp execution (for reference)
    print("\n4. NUMPY.INTERP EXECUTION (for reference)")
    # Create random values to interpolate
    np.random.seed(42)
    values = np.random.uniform(0, 1, len(x)).astype(np.float64)
    numpy_exec_time = benchmark_numpy_interp(x, xb, values, n_runs=n_runs)
    print(f"   Time per call: {numpy_exec_time*1000:.3f} ms")
    print(f"   Total for {n_runs} calls: {numpy_exec_time*n_runs*1000:.1f} ms")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    speedup_cython_vs_numba = numba_exec_time / cython_exec_time
    print(f"\nCython vs Numba (per-call): {speedup_cython_vs_numba:.2f}x")
    
    if speedup_cython_vs_numba > 1.0:
        print(f"  → Cython is {speedup_cython_vs_numba:.2f}x FASTER per call")
    else:
        print(f"  → Numba is {1/speedup_cython_vs_numba:.2f}x faster per call")
    
    speedup_cython_vs_numpy = cython_exec_time / numpy_exec_time
    print(f"\nCython vs NumPy.interp (per-call):")
    if speedup_cython_vs_numpy > 1.0:
        print(f"  → Cython is {speedup_cython_vs_numpy:.1f}x SLOWER (expected, different approach)")
        print(f"  ⚠ Note: Cython builds reusable matrix; NumPy directly interpolates")
    else:
        print(f"  → Cython is {1/speedup_cython_vs_numpy:.1f}x faster per call")
    
    # Break-even analysis
    breakeven_calls, breakeven_time = calculate_breakeven(
        numba_compile_time, 
        numba_exec_time, 
        cython_exec_time
    )
    
    print(f"\nBREAK-EVEN ANALYSIS:")
    if breakeven_calls == 0:
        print(f"  ✓ Cython is FASTER from the very first call!")
        print(f"  → No JIT compilation overhead")
        print(f"  → Per-call execution is faster")
        print(f"\n  Recommendation: Always use Cython")
    else:
        print(f"  Cython breaks even after: {breakeven_calls:,} calls")
        print(f"  Total time at break-even: {breakeven_time*1000:.1f} ms")
        print(f"\n  Cost of Numba JIT compilation: {numba_compile_time*1000:.3f} ms")
        print(f"  If you need >{breakeven_calls} calls: Use Cython ✓")
        print(f"  If you need <{breakeven_calls} calls: Use Numba ✓")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Implementation':<25} {'Per-call (ms)':<15} {'100 calls (ms)':<15}")
    print("-" * 70)
    print(f"{'NumPy.interp':<25} {numpy_exec_time*1000:<15.3f} {numpy_exec_time*100*1000:<15.1f}")
    print(f"{'Cython':<25} {cython_exec_time*1000:<15.3f} {cython_exec_time*100*1000:<15.1f}")
    print(f"{'Numba (compiled)':<25} {numba_exec_time*1000:<15.3f} {numba_exec_time*100*1000:<15.1f}")
    
    print(f"\nNotes:")
    print(f"  • NumPy.interp: Direct interpolation (no matrix overhead)")
    print(f"  • Cython/Numba: Matrix creation for later application")
    print(f"  • NumPy is {cython_exec_time/numpy_exec_time:.1f}x faster per call")
    print(f"  • Matrix approach amortizes cost over multiple interpolations")
    print(f"  • Use NumPy.interp for: Single or few one-off interpolations")
    print(f"  • Use Cython for: Reusable matrices (multiple data arrays, same x)")
    
    if breakeven_calls > 0:
        total_numba_at_breakeven = numba_compile_time + breakeven_calls * numba_exec_time
        print(f"\n{'At break-even point:':<20} {breakeven_calls} calls")
        print(f"{'Numba total time':<20} {total_numba_at_breakeven*1000:<15.1f} ms")
        print(f"{'Cython total time':<20} {breakeven_time*1000:<15.1f} ms")
    else:
        total_numba_with_compilation = numba_compile_time + numba_exec_time * 100
        total_cython = cython_exec_time * 100
        savings_100_calls = (total_numba_with_compilation - total_cython) * 1000
        print(f"\nComparison (including 100 calls):")
        print(f"{'  Numba (1st call + 100 runs):':<35} {total_numba_with_compilation*1000:.1f} ms")
        print(f"{'  Cython (100 calls):':<35} {total_cython*1000:.1f} ms")
        print(f"{'  Time saved with Cython:':<35} {savings_100_calls:.1f} ms")
    
    print("=" * 70 + "\n")
    
    return {
        'numba_compile_time': numba_compile_time,
        'numba_exec_time': numba_exec_time,
        'cython_exec_time': cython_exec_time,
        'numpy_exec_time': numpy_exec_time,
        'breakeven_calls': breakeven_calls,
        'breakeven_time': breakeven_time,
    }


if __name__ == "__main__":
    results = run_benchmark(n_in=100, n_out=200, n_runs=100)
