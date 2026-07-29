import time

import numpy as np
from tqdm import tqdm

from quasar_utils.masking import linear, logarithmic

RNG = np.random.default_rng(seed=42)

V_RES: float = 1e-4
X0: float = 1000.0
DX: float = 1.0
N_PIX: int = 10_000

N_TESTS: int = 100
N_BENCHMARKS: int = 1_000


def reference_get_mask(
    array: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    return (bounds[0] <= array) & (array <= bounds[1])


def test_irregular(n_tests: int = N_TESTS):
    for _ in range(n_tests):
        x = X0 + np.cumsum(RNG.uniform(0.5, 1.5, size=N_PIX))
        bounds = tuple(sorted(RNG.uniform(x[0], x[-1], size=2)))

        mask_ref = get_mask(
            x,
            bounds,
        )
        mask_test = get_mask.__wrapped__(
            x,
            bounds,
            array_type="irregular",
        )
        assert np.array_equal(mask_ref, mask_test)


def test_linear(n_tests: int = N_TESTS):
    x = X0 + np.arange(N_PIX) * DX

    for _ in range(n_tests):
        bounds = tuple(sorted(RNG.uniform(x[0], x[-1], size=2)))

        mask_ref = get_mask.__wrapped__(
            x,
            bounds,
        )
        mask_test = get_mask.__wrapped__(
            x,
            bounds,
            array_type="linear",
            dx=DX,
        )
        assert np.array_equal(mask_ref, mask_test)


def test_logarithmic(n_tests: int = N_TESTS):
    x = X0 * (1 + V_RES) ** np.arange(N_PIX)

    for _ in range(n_tests):
        bounds = tuple(sorted(RNG.uniform(x[0], x[-1], size=2)))

        mask_ref = get_mask.__wrapped__(
            x,
            bounds,
        )
        mask_test = get_mask.__wrapped__(
            x,
            bounds,
            array_type="logarithmic",
            v_res=V_RES,
        )
        assert np.array_equal(mask_ref, mask_test)


###


def time_irregular(n: int = N_BENCHMARKS) -> dict[str, list[float]]:

    reference_times: list[float] = []
    test_times: list[float] = []

    for _ in tqdm(
        range(n + 1),
        desc="irregular",
        leave=False,
    ):
        x = X0 + np.cumsum(RNG.uniform(0.5, 1.5, size=N_PIX))
        bounds = tuple(sorted(RNG.uniform(x[0], x[-1], size=2)))

        t0 = time.perf_counter()
        _ = get_mask.__wrapped__(
            x,
            bounds,
        )
        reference_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = get_mask.__wrapped__(
            x,
            bounds,
            array_type="irregular",
        )
        test_times.append(time.perf_counter() - t0)

    return dict(
        reference=reference_times[1:],
        test=test_times[1:],
    )


def time_linear(n: int = N_BENCHMARKS) -> dict[str, list[float]]:

    reference_times: list[float] = []
    test_times: list[float] = []

    x = X0 + np.arange(N_PIX) * DX

    for _ in tqdm(
        range(n + 1),
        desc="linear",
        leave=False,
    ):
        bounds = tuple(sorted(RNG.uniform(x[0], x[-1], size=2)))

        t0 = time.perf_counter()
        _ = get_mask.__wrapped__(
            x,
            bounds,
        )
        reference_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = linear.get_mask(
            x,
            bounds,
            DX,
            where=None,
        )
        test_times.append(time.perf_counter() - t0)

    return dict(
        reference=reference_times[1:],
        test=test_times[1:],
    )


def time_logarithmic(n: int = N_BENCHMARKS) -> dict[str, list[float]]:

    reference_times: list[float] = []
    test_times: list[float] = []

    x = X0 * (1 + V_RES) ** np.arange(N_PIX)

    for _ in tqdm(
        range(n + 1),
        desc="logarithmic",
        leave=False,
    ):
        bounds = tuple(sorted(RNG.uniform(x[0], x[-1], size=2)))

        t0 = time.perf_counter()
        _ = get_mask.__wrapped__(
            x,
            bounds,
        )
        reference_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = logarithmic.get_mask(
            x,
            bounds,
            V_RES,
            None,
        )
        test_times.append(time.perf_counter() - t0)

    return dict(
        reference=reference_times[1:],
        test=test_times[1:],
    )


def plot(
    irregular_times: dict[str, list[float]],
    linear_times: dict[str, list[float]],
    logarithmic_times: dict[str, list[float]],
) -> None:
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, times, title in zip(
        axes,
        [irregular_times, linear_times, logarithmic_times],
        ["irregular", "linear", "logarithmic"],
    ):
        ax.violinplot(
            [np.log10(times["reference"]), np.log10(times["test"])],
            positions=[1, 2],
            showmeans=True,
            showextrema=False,
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["reference", "test"])
        ax.set_title(title)
        ax.set_ylabel("log10(time) [s]")

    axes[0].set_ylim(-6, -3)

    plt.tight_layout()
    plt.show()


def main():
    plot(
        time_irregular(),
        time_linear(),
        time_logarithmic(),
    )


if __name__ == "__main__":
    main()
