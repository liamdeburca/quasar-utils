"""Benchmark matrix-based interpolation vs numpy.interp.

This script measures the time to build the interpolation matrix and to apply
it, and compares that with the time for numpy.interp. Times are reported in
microseconds (1e-6 seconds) with a simple table for a range of sizes.
"""
from time import perf_counter

import numpy as np

from quasar_utils.interpolation import create_interp_matrix


def benchmark_case(n_in: int, n_out: int, repeats: int = 5) -> dict:
    rng = np.random.default_rng(42)

    x0 = rng.uniform(0.1, 10.0)
    x1 = x0 + rng.uniform(1.0, 100.0)
    x_in = np.sort(rng.random(n_in) * (x1 - x0) + x0)

    low = x0 - rng.uniform(0.0, 2.0)
    high = x1 + rng.uniform(0.0, 2.0)
    x_out = np.sort(rng.random(n_out) * (high - low) + low)

    a = rng.normal(size=n_in)

    # Warm up once
    M, bias = create_interp_matrix(x_in, x_out)
    _ = M.dot(a) + bias
    _ = np.interp(x_out, x_in, a)

    # Measure numpy.interp
    t_interp = 0.0
    for _ in range(repeats):
        t0 = perf_counter()
        _ = np.interp(x_out, x_in, a)
        t_interp += perf_counter() - t0

    # Measure matrix build
    t_build = 0.0
    for _ in range(repeats):
        t0 = perf_counter()
        M, bias = create_interp_matrix(x_in, x_out)
        t_build += perf_counter() - t0

    # Measure matrix apply
    t_apply = 0.0
    for _ in range(repeats):
        t0 = perf_counter()
        _ = M.dot(a) + bias
        t_apply += perf_counter() - t0

    return {
        "n_in": n_in,
        "n_out": n_out,
        "interp_us": (t_interp / repeats) * 1e6,
        "build_us": (t_build / repeats) * 1e6,
        "apply_us": (t_apply / repeats) * 1e6,
    }


def main():
    sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

    print(
        "Benchmark: numpy.interp vs matrix-based interpolation\n"
        "Times reported in microseconds (1e-6 seconds).\n"
    )

    header = f"{'n_in':>6} {'n_out':>6} {'interp(us)':>12} {'build(us)':>12} {'apply(us)':>12}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        n_in = n
        n_out = max(1, int(n * 1.2))
        repeats = 20 if n <= 200 else 5
        res = benchmark_case(n_in, n_out, repeats=repeats)
        print(
            f"{res['n_in']:6d} {res['n_out']:6d} {res['interp_us']:12.2f} {res['build_us']:12.2f} {res['apply_us']:12.2f}"
        )


if __name__ == '__main__':
    main()
