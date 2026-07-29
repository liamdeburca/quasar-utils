import numpy as np

from quasar_utils.masking import get_mask

N: int = 10_000
N_TESTS: int = 1_000


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


def _irregular():
    arr = create_irregular()
    bounds = get_bounds(arr)

    mask = get_mask.__wrapped__(arr, bounds, array_type="irregular")
    mask_test = get_mask.__wrapped__(arr, bounds)

    assert np.array_equal(mask, mask_test)


def test_irregular():
    for _ in range(N_TESTS):
        _irregular()


def _linear():
    arr = create_linear()
    bounds = get_bounds(arr)

    mask = get_mask.__wrapped__(arr, bounds, array_type="linear")
    mask_test = get_mask.__wrapped__(arr, bounds)

    assert np.array_equal(mask, mask_test)


def test_linear():
    for _ in range(N_TESTS):
        _linear()


def _logarithmic():
    arr = create_logarithmic()
    bounds = get_bounds(arr)

    mask = get_mask.__wrapped__(arr, bounds, array_type="logarithmic")
    mask_test = get_mask.__wrapped__(arr, bounds)

    assert np.array_equal(mask, mask_test)


def test_logarithmic():
    for _ in range(N_TESTS):
        _logarithmic()
