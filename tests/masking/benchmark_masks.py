import time

import numpy as np

from quasar_utils.masking import get_mask

N: int = 10_000
N_TESTS: int = 10_000


def create_irregular() -> np.ndarray:
    x0, x1 = sorted(np.random.uniform(1000, 10_000, 2))
    return np.sort(np.random.uniform(x0, x1, N))


def create_linear() -> np.ndarray:
    x0 = np.random.uniform(1000, 2000)
    dx = round(np.random.uniform(1, 2), 6)
    return x0 + dx * np.arange(N)


def create_logarithmic() -> np.ndarray:
    x0 = np.random.uniform(1000, 2000)
    v_res = round(10 ** np.random.uniform(-4, -2), 6)
    return x0 * (1 + v_res) ** np.arange(N)


def get_bounds(arr: np.ndarray) -> tuple[float, float]:
    indices = sorted(np.random.choice(len(arr), 2, replace=False))
    return tuple(arr[indices])


def benchmark_irregular():
    print("\n" + "=" * 60)
    print("IRREGULAR ARRAY BENCHMARK")
    print("=" * 60)

    # Pre-generate arrays and bounds
    arrays_and_bounds = [
        (create_irregular(), get_bounds(create_irregular()))
        for _ in range(N_TESTS)
    ]

    # Time the specialized indexing
    start = time.perf_counter()
    for arr, bounds in arrays_and_bounds:
        _ = get_mask.__wrapped__(arr, bounds, array_type="irregular")
    specialized_time = time.perf_counter() - start
    specialized_time *= 1e6 / N_TESTS

    # Time the numpy method
    start = time.perf_counter()
    for arr, bounds in arrays_and_bounds:
        _ = get_mask.__wrapped__(arr, bounds)
    numpy_time = time.perf_counter() - start
    numpy_time *= 1e6 / N_TESTS

    print(f"Specialized (array_type='irregular'): {specialized_time:.3f}μs")
    print(f"NumPy method:                         {numpy_time:.3f}μs")
    print(f"Speedup: {numpy_time / specialized_time:.2f}x")


def benchmark_linear():
    print("\n" + "=" * 60)
    print("LINEAR ARRAY BENCHMARK")
    print("=" * 60)

    # Pre-generate arrays and bounds
    arrays_and_bounds = [
        (create_linear(), get_bounds(create_linear())) for _ in range(N_TESTS)
    ]

    # Time the specialized indexing
    start = time.perf_counter()
    for arr, bounds in arrays_and_bounds:
        _ = get_mask.__wrapped__(arr, bounds, array_type="linear")
    specialized_time = time.perf_counter() - start
    specialized_time *= 1e6 / N_TESTS

    # Time the numpy method
    start = time.perf_counter()
    for arr, bounds in arrays_and_bounds:
        _ = get_mask.__wrapped__(arr, bounds)
    numpy_time = time.perf_counter() - start
    numpy_time *= 1e6 / N_TESTS

    print(f"Specialized (array_type='linear'):   {specialized_time:.3f}μs")
    print(f"NumPy method:                        {numpy_time:.3f}μs")
    print(f"Speedup: {numpy_time / specialized_time:.2f}x")


def benchmark_logarithmic():
    print("\n" + "=" * 60)
    print("LOGARITHMIC ARRAY BENCHMARK")
    print("=" * 60)

    # Pre-generate arrays and bounds
    arrays_and_bounds = [
        (create_logarithmic(), get_bounds(create_logarithmic()))
        for _ in range(N_TESTS)
    ]

    # Time the specialized indexing
    start = time.perf_counter()
    for arr, bounds in arrays_and_bounds:
        _ = get_mask.__wrapped__(arr, bounds, array_type="logarithmic")
    specialized_time = time.perf_counter() - start
    specialized_time *= 1e6 / N_TESTS

    # Time the numpy method
    start = time.perf_counter()
    for arr, bounds in arrays_and_bounds:
        _ = get_mask.__wrapped__(arr, bounds)
    numpy_time = time.perf_counter() - start
    numpy_time *= 1e6 / N_TESTS

    print(f"Specialized (array_type='logarithmic'): {specialized_time:.3f}μs")
    print(f"NumPy method:                           {numpy_time:.3f}μs")
    print(f"Speedup: {numpy_time / specialized_time:.2f}x")


if __name__ == "__main__":
    print(f"Running {N_TESTS} iterations with array size {N}")
    benchmark_irregular()
    benchmark_linear()
    benchmark_logarithmic()
    print("\n" + "=" * 60)
