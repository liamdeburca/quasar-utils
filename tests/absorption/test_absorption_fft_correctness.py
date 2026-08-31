import numpy as np

from quasar_utils.absorption import absorption


def synth_strong_sinusoid(N=256, k0=5, amp=10.0, noise_sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(N)
    z = amp * np.sin(2 * np.pi * k0 * i / N) + rng.normal(scale=noise_sigma, size=N)
    return z


def test_fft_single_bin_detection_low_p():
    N = 256
    k0 = 5
    z = synth_strong_sinusoid(N=N, k0=k0, amp=10.0, noise_sigma=0.1)
    stat, p = absorption.fft_test(z, k=k0)
    assert p < 1e-6


def test_fft_aggregate_detection_low_p_for_multifrequency():
    N = 512
    rng = np.random.default_rng(0)
    i = np.arange(N)
    z = (
        5.0 * np.sin(2 * np.pi * 3 * i / N)
        + 3.0 * np.sin(2 * np.pi * 7 * i / N)
        + rng.normal(scale=0.2, size=N)
    )
    stat, p = absorption.fft_test(z, k=None)
    assert p < 1e-6


def test_fft_random_noise_high_p():
    rng = np.random.default_rng(1)
    N = 512
    z = rng.normal(size=N)
    stat, p = absorption.fft_test(z, k=None)
    assert p > 1e-6
