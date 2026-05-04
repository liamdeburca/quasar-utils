"""Tests for the top-level dereddening functions."""
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord


# High-latitude coordinates to ensure nonzero but modest E(B-V)
SKY_COORD = SkyCoord(ra=180.0, dec=60.0, unit='deg', frame='icrs')

WAVELENGTHS = np.arange(3700, 9300, 1.25, dtype=np.float64)
FLUX = np.ones_like(WAVELENGTHS, dtype=np.float64)
ERROR = np.full_like(WAVELENGTHS, 0.1, dtype=np.float64)


class TestGetCorrection:
    """Tests for the get_correction function."""

    def test_returns_array(self):
        from quasar_utils.dereddening import get_correction
        correction = get_correction(WAVELENGTHS, SKY_COORD)
        assert isinstance(correction, np.ndarray)
        assert correction.shape == WAVELENGTHS.shape

    def test_correction_ge_one(self):
        """Dereddening correction must be >= 1 (adds flux back)."""
        from quasar_utils.dereddening import get_correction
        correction = get_correction(WAVELENGTHS, SKY_COORD)
        assert np.all(correction >= 1.0)

    def test_correction_with_sfd(self):
        from quasar_utils.dereddening import get_correction
        correction = get_correction(WAVELENGTHS, SKY_COORD, map_name='sfd')
        assert np.all(np.isfinite(correction))

    def test_correction_with_csfd(self):
        from quasar_utils.dereddening import get_correction
        correction = get_correction(WAVELENGTHS, SKY_COORD, map_name='csfd')
        assert np.all(np.isfinite(correction))

    def test_correction_with_ccm89(self):
        from quasar_utils.dereddening import get_correction
        correction = get_correction(WAVELENGTHS, SKY_COORD, law_name='ccm89')
        assert np.all(np.isfinite(correction))

    def test_correction_with_o94(self):
        from quasar_utils.dereddening import get_correction
        correction = get_correction(WAVELENGTHS, SKY_COORD, law_name='o94')
        assert np.all(np.isfinite(correction))

    def test_higher_rv_changes_correction(self):
        from quasar_utils.dereddening import get_correction
        c1 = get_correction(WAVELENGTHS, SKY_COORD, Rv=2.5)
        c2 = get_correction(WAVELENGTHS, SKY_COORD, Rv=4.0)
        assert not np.allclose(c1, c2)

    def test_blue_corrected_more_than_red(self):
        """Shorter wavelengths should have larger correction factors."""
        from quasar_utils.dereddening import get_correction
        correction = get_correction(WAVELENGTHS, SKY_COORD)
        assert correction[0] > correction[-1]


class TestDereddenSpectrum:
    """Tests for the deredden_spectrum function."""

    def test_returns_tuple_of_three(self):
        from quasar_utils.dereddening import deredden_spectrum
        result = deredden_spectrum((WAVELENGTHS, FLUX, ERROR), SKY_COORD)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_wavelengths_unchanged(self):
        from quasar_utils.dereddening import deredden_spectrum
        result = deredden_spectrum((WAVELENGTHS, FLUX, ERROR), SKY_COORD)
        assert_allclose(result[0], WAVELENGTHS)

    def test_flux_increases(self):
        """Dereddened flux should be >= observed flux."""
        from quasar_utils.dereddening import deredden_spectrum
        result = deredden_spectrum((WAVELENGTHS, FLUX, ERROR), SKY_COORD)
        assert np.all(result[1] >= FLUX)

    def test_error_increases(self):
        """Dereddened error should be >= observed error (same scaling)."""
        from quasar_utils.dereddening import deredden_spectrum
        result = deredden_spectrum((WAVELENGTHS, FLUX, ERROR), SKY_COORD)
        assert np.all(result[2] >= ERROR)

    def test_flux_and_error_same_scaling(self):
        """Flux and error must be scaled by the same correction factor."""
        from quasar_utils.dereddening import deredden_spectrum
        result = deredden_spectrum((WAVELENGTHS, FLUX, ERROR), SKY_COORD)
        flux_ratio = result[1] / FLUX
        error_ratio = result[2] / ERROR
        assert_allclose(flux_ratio, error_ratio)

    def test_all_map_curve_combinations(self):
        """Every (map, curve) combination should run without error."""
        from quasar_utils.dereddening import deredden_spectrum
        for map_name in ('sfd', 'csfd'):
            for law_name in ('ccm89', 'o94'):
                result = deredden_spectrum(
                    (WAVELENGTHS, FLUX, ERROR),
                    SKY_COORD,
                    map_name=map_name,
                    law_name=law_name,
                )
                assert np.all(np.isfinite(result[1]))
                assert np.all(np.isfinite(result[2]))


class TestRoundTrip:
    """
    Validate that reddening followed by dereddening recovers the original 
    spectrum.
    """

    def test_redden_then_deredden_recovers_flux(self):
        """
        Manually redden a flat spectrum with A(λ), then use deredden_spectrum
        to recover it. The round-trip should match to high precision.
        """
        from quasar_utils.dereddening import get_correction, deredden_spectrum

        correction = get_correction(WAVELENGTHS, SKY_COORD)

        # Simulate observed (reddened) spectrum: divide by correction
        flux_reddened = FLUX / correction
        error_reddened = ERROR / correction

        # Deredden the reddened spectrum
        result = deredden_spectrum(
            (WAVELENGTHS, flux_reddened, error_reddened), SKY_COORD,
        )

        assert_allclose(result[1], FLUX, rtol=1e-10)
        assert_allclose(result[2], ERROR, rtol=1e-10)

    def test_roundtrip_with_o94(self):
        from quasar_utils.dereddening import get_correction, deredden_spectrum

        correction = get_correction(
            WAVELENGTHS, SKY_COORD, law_name='o94',
        )
        flux_reddened = FLUX / correction
        error_reddened = ERROR / correction

        result = deredden_spectrum(
            (WAVELENGTHS, flux_reddened, error_reddened),
            SKY_COORD,
            law_name='o94',
        )

        assert_allclose(result[1], FLUX, rtol=1e-10)
        assert_allclose(result[2], ERROR, rtol=1e-10)
