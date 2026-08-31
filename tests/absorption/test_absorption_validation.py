import numpy as np
import pytest
from pydantic import ValidationError

from quasar_utils.absorption import absorption


def test_fft_small_N_raises():
    # aggregate mode requires N >= 6 per implementation
    z = np.zeros(5)
    with pytest.raises(ValueError):
        absorption.fft_test(z)


def test_fft_k_out_of_range_raises():
    N = 128
    z = np.random.default_rng(0).normal(size=N)
    with pytest.raises(ValueError):
        absorption.fft_test(z, k=N)  # out of range


def test_fft_with_nan_raises_validation_error():
    N = 64
    z = np.random.default_rng(0).normal(size=N)
    z[10] = np.nan
    with pytest.raises(ValidationError):
        absorption.fft_test(z, k=5)
