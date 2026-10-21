import numpy as np
import pytest
from scipy.stats import kstest

from quasar_utils.absorption import absorption


def test_fft_pvalues_uniform(subtests: pytest.Subtests):
    """Verify that p-values from fft_test on Gaussian noise are uniform.

    For each sample length, generate many independent Gaussian residual arrays,
    compute p-values using absorption.fft_test (vectorized over the first axis),
    and compare the empirical distribution of p-values to a Uniform(0,1)
    distribution using the Kolmogorov-Smirnov test. The subtest will fail if
    the KS-test p-value is below alpha.
    """

    sizes = [25, 50, 100, 500, 1000, 5000, 10_000]
    n_trials = 1_000
    alpha = 0.01

    for N in sizes:
        with subtests.test(msg=f"N={N}"):
            rng = np.random.default_rng(42 + N)
            # Generate shape (n_trials, N) of independent N(0,1) residuals
            zs = rng.normal(size=(n_trials, N))

            # absorption.fft_test accepts a matrix and returns vectorized p-values
            pvals = absorption.fft_test(zs, k=None)[1]

            # Validate shapes
            assert pvals.shape[0] == n_trials, pvals.shape

            ks_res = kstest(pvals, "uniform")

            # Include the KS p-value in the subtest message
            with subtests.test(msg=f"N={N}, n_trials={n_trials}, ks_p={ks_res.pvalue:.3g}"):
                assert ks_res.pvalue > alpha, (
                    f"KS p-value {ks_res.pvalue:.6g} below alpha={alpha} for N={N}"
                )
