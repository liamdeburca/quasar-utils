"""Tests for the dust_extinction submodule."""
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestGetDustCurve:
    """Tests for the get_dust_curve factory function."""

    def test_returns_ccm89(self):
        from quasar_utils.dereddening.dust_extinction import get_dust_curve
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        curve = get_dust_curve('ccm89')
        assert isinstance(curve, CCM89)

    def test_returns_o94(self):
        from quasar_utils.dereddening.dust_extinction import get_dust_curve
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        curve = get_dust_curve('o94')
        assert isinstance(curve, O94)

    def test_case_insensitive(self):
        from quasar_utils.dereddening.dust_extinction import get_dust_curve
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        assert isinstance(get_dust_curve('CCM89'), CCM89)
        assert isinstance(get_dust_curve('O94'), O94)

    def test_unsupported_curve_raises(self):
        from quasar_utils.dereddening.dust_extinction import get_dust_curve
        with pytest.raises(NotImplementedError):
            get_dust_curve('fitzpatrick99')


class TestCCM89:
    """Tests for the CCM89 extinction curve."""

    K_OPTICAL = np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64)

    def test_evaluate_returns_array(self):
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        result = CCM89.evaluate(self.K_OPTICAL, Rv=3.1)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.K_OPTICAL.shape

    def test_evaluate_at_v_band(self):
        """At x = 1/0.55 µm ≈ 1.818 µm⁻¹, A(λ)/A(V) should be ~1.0."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k_v = np.array([1.0 / 0.55], dtype=np.float64)
        result = CCM89.evaluate(k_v, Rv=3.1)
        assert_allclose(result, 1.0, atol=0.02)

    def test_evaluate_monotonic_in_optical(self):
        """Extinction should increase with wavenumber in the optical."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        result = CCM89.evaluate(self.K_OPTICAL, Rv=3.1)
        assert np.all(np.diff(result) > 0)

    def test_get_ab_arrays_shape(self):
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        a, b = CCM89.get_ab_arrays(self.K_OPTICAL)
        assert a.shape == self.K_OPTICAL.shape
        assert b.shape == self.K_OPTICAL.shape

    def test_evaluate_equals_a_plus_b_over_rv(self):
        """evaluate(k, Rv) should equal a(k) + b(k)/Rv."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        Rv = 3.1
        a, b = CCM89.get_ab_arrays(self.K_OPTICAL)
        expected = a + b / Rv
        result = CCM89.evaluate(self.K_OPTICAL, Rv=Rv)
        assert_allclose(result, expected)

    def test_ir_range(self):
        """Test evaluation in the infrared (0.3 <= k < 1.1)."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k_ir = np.array([0.5, 0.8, 1.0], dtype=np.float64)
        result = CCM89.evaluate(k_ir, Rv=3.1)
        assert result.shape == k_ir.shape
        assert np.all(np.isfinite(result))

    def test_nuv_range(self):
        """Test evaluation in the NUV (3.3 <= k <= 8.0)."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k_nuv = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float64)
        result = CCM89.evaluate(k_nuv, Rv=3.1)
        assert result.shape == k_nuv.shape
        assert np.all(np.isfinite(result))

    def test_fuv_range(self):
        """Test evaluation in the FUV (8 < k <= 10)."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k_fuv = np.array([8.5, 9.0, 9.5], dtype=np.float64)
        result = CCM89.evaluate(k_fuv, Rv=3.1)
        assert result.shape == k_fuv.shape
        assert np.all(np.isfinite(result))

    def test_full_range(self):
        """Evaluate across the full valid range without errors."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k_full = np.linspace(0.3, 10.0, 200, dtype=np.float64)
        result = CCM89.evaluate(k_full, Rv=3.1)
        assert result.shape == k_full.shape
        assert np.all(np.isfinite(result))


class TestO94:
    """Tests for the O94 extinction curve."""

    K_OPTICAL = np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64)

    def test_evaluate_returns_array(self):
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        result = O94.evaluate(self.K_OPTICAL, Rv=3.1)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.K_OPTICAL.shape

    def test_evaluate_at_v_band(self):
        """At x = 1/0.55 µm ≈ 1.818 µm⁻¹, A(λ)/A(V) should be ~1.0."""
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        k_v = np.array([1.0 / 0.55], dtype=np.float64)
        result = O94.evaluate(k_v, Rv=3.1)
        assert_allclose(result, 1.0, atol=0.02)

    def test_evaluate_monotonic_in_optical(self):
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        result = O94.evaluate(self.K_OPTICAL, Rv=3.1)
        assert np.all(np.diff(result) > 0)

    def test_get_ab_arrays_shape(self):
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        a, b = O94.get_ab_arrays(self.K_OPTICAL)
        assert a.shape == self.K_OPTICAL.shape
        assert b.shape == self.K_OPTICAL.shape

    def test_evaluate_equals_a_plus_b_over_rv(self):
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        Rv = 3.1
        a, b = O94.get_ab_arrays(self.K_OPTICAL)
        expected = a + b / Rv
        result = O94.evaluate(self.K_OPTICAL, Rv=Rv)
        assert_allclose(result, expected)

    def test_differs_from_ccm89_in_optical(self):
        """O94 updated the optical coefficients, so results must differ."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        result_ccm = CCM89.evaluate(self.K_OPTICAL, Rv=3.1)
        result_o94 = O94.evaluate(self.K_OPTICAL, Rv=3.1)
        assert not np.allclose(result_ccm, result_o94)

    def test_matches_ccm89_in_ir(self):
        """O94 uses the same IR coefficients as CCM89."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        k_ir = np.array([0.5, 0.8, 1.0], dtype=np.float64)
        assert_allclose(
            O94.evaluate(k_ir, Rv=3.1),
            CCM89.evaluate(k_ir, Rv=3.1),
        )

    def test_full_range(self):
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        k_full = np.linspace(0.3, 10.0, 200, dtype=np.float64)
        result = O94.evaluate(k_full, Rv=3.1)
        assert result.shape == k_full.shape
        assert np.all(np.isfinite(result))


class TestCaching:
    """Tests for the ab_cache mechanism."""

    def test_ccm89_caches_result(self):
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k = np.array([1.5, 2.0, 2.5], dtype=np.float64)
        CCM89.ab_cache.clear()
        CCM89.get_ab_arrays(k)
        cache_key = CCM89.get_cache_key(k)
        assert cache_key in CCM89.ab_cache

    def test_o94_caches_result(self):
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        k = np.array([1.5, 2.0, 2.5], dtype=np.float64)
        O94.ab_cache.clear()
        O94.get_ab_arrays(k)
        cache_key = O94.get_cache_key(k)
        assert cache_key in O94.ab_cache

    def test_cached_result_matches_fresh(self):
        """Calling get_ab_arrays twice returns identical arrays."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k = np.array([1.5, 2.0, 2.5], dtype=np.float64)
        CCM89.ab_cache.clear()
        a1, b1 = CCM89.get_ab_arrays(k)
        a2, b2 = CCM89.get_ab_arrays(k)
        assert_allclose(a1, a2)
        assert_allclose(b1, b2)

    def test_caches_are_independent(self):
        """CCM89.ab_cache and O94.ab_cache must not be the same object."""
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        from quasar_utils.dereddening.dust_extinction.o94 import O94
        assert CCM89.ab_cache is not O94.ab_cache

    def test_different_inputs_produce_different_keys(self):
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k1 = np.array([1.5, 2.0], dtype=np.float64)
        k2 = np.array([3.0, 4.0], dtype=np.float64)
        assert CCM89.get_cache_key(k1) != CCM89.get_cache_key(k2)


class TestFitDeriv:
    """Tests for the analytic derivative."""

    def test_fit_deriv_shape(self):
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k = np.array([1.5, 2.0, 2.5], dtype=np.float64)
        derivs = CCM89.fit_deriv(k, Rv=3.1)
        assert len(derivs) == 1
        assert derivs[0].shape == k.shape

    def test_fit_deriv_negative_b_gives_positive_deriv(self):
        """
        For the IR where b < 0: d(a + b/Rv)/dRv = -b/Rv² > 0 when b < 0.
        """
        from quasar_utils.dereddening.dust_extinction.ccm89 import CCM89
        k_ir = np.array([0.5, 0.8], dtype=np.float64)
        derivs = CCM89.fit_deriv(k_ir, Rv=3.1)
        assert np.all(derivs[0] > 0)
