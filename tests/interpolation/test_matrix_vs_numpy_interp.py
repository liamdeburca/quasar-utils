import numpy as np

from quasar_utils.interpolation import create_interp_matrix


def test_matrix_interp_equivalence(subtests):
    """Verify matrix-based interpolation equals numpy.interp for many cases.

    Uses pytest subtests to run 100 randomized scenarios.
    """
    rng = np.random.default_rng(123456)

    for case in range(100):
        # random sizes
        n_in = rng.integers(5, 60)
        n_out = rng.integers(1, 120)

        # construct sorted, positive x_in
        x0 = rng.uniform(0.1, 10.0)
        x1 = x0 + rng.uniform(1.0, 100.0)
        x_in = np.sort(rng.random(n_in) * (x1 - x0) + x0)

        # allow x_out to include points outside the input range to test bias
        low = x0 - rng.uniform(0.0, 2.0)
        high = x1 + rng.uniform(0.0, 2.0)
        x_out = np.sort(rng.random(n_out) * (high - low) + low)

        a = rng.normal(size=n_in)

        # random left/right extrapolation values (occasionally non-zero)
        left = 0.0 if rng.random() < 0.5 else rng.normal()
        right = 0.0 if rng.random() < 0.5 else rng.normal()

        with subtests.test(case=case):
            M, bias = create_interp_matrix(x_in, x_out, left=left, right=right)

            # numpy.interp result
            expected = np.interp(x_out, x_in, a, left=left, right=right)

            # matrix multiplication + bias
            got = M.dot(a) + bias

            # allow small floating point differences
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
