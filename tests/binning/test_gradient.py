import numpy as np


def lin_dx_old(x: np.ndarray) -> np.ndarray:
    """Legacy lin_dx implementation using roll-based differences.

    This reproduces the previous behaviour we replaced with
    `numpy.gradient`.
    """
    dx = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    return dx   


def test_gradient_equivalence_random():
    """Randomised tests comparing legacy lin_dx and numpy.gradient.

    We test a range of array sizes and many random seeds to ensure the
    replacement with numpy.gradient is behaviourally equivalent to the
    original roll-based implementation used historically.
    """
    rng = np.random.default_rng(12345)

    sizes = [2, 3, 4, 5, 10, 50, 100]
    for n in sizes:
        for _ in range(50):
            # generate a sorted, positive array
            x = np.sort(rng.random(n) * 1e3 + 1.0)

            old = lin_dx_old(x)
            new = np.gradient(x)

            # Use allclose with a tight tolerance — these should be equal
            # up to floating point rounding.
            np.testing.assert_allclose(old, new, rtol=1e-12, atol=0)


def test_gradient_equivalence_structured():
    """Structured inputs where behaviour should be exact/evident."""
    # Uniform spacing
    x = np.linspace(1.0, 10.0, 11)
    np.testing.assert_allclose(lin_dx_old(x), np.gradient(x))

    # Exponential spacing
    x = np.exp(np.linspace(0.0, 5.0, 20))
    np.testing.assert_allclose(lin_dx_old(x), np.gradient(x))
