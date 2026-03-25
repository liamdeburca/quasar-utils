"""Tests for the dustmaps submodule."""
import pytest
from dustmaps.sfd import SFDQuery
from dustmaps.csfd import CSFDQuery
from astropy.coordinates import SkyCoord
import astropy.units as u


class TestSetupDustmaps:
    """Tests for dustmaps cache setup."""

    def test_cache_directory_exists(self):
        from quasar_utils.dereddening.dustmaps import PATH_TO_CACHE
        assert PATH_TO_CACHE.exists()

    def test_sfd_cache_exists(self):
        from quasar_utils.dereddening.dustmaps import PATH_TO_CACHE
        assert (PATH_TO_CACHE / "sfd").exists()

    def test_csfd_cache_exists(self):
        from quasar_utils.dereddening.dustmaps import PATH_TO_CACHE
        assert (PATH_TO_CACHE / "csfd").exists()


class TestGetDustMap:
    """Tests for the get_dust_map factory function."""

    def test_returns_sfd_query(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        dust_map = get_dust_map('sfd')
        assert isinstance(dust_map, SFDQuery)

    def test_returns_csfd_query(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        dust_map = get_dust_map('csfd')
        assert isinstance(dust_map, CSFDQuery)

    def test_case_insensitive(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        assert isinstance(get_dust_map('SFD'), SFDQuery)
        assert isinstance(get_dust_map('CSFD'), CSFDQuery)

    def test_unsupported_map_raises(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        with pytest.raises(NotImplementedError):
            get_dust_map('planck')


class TestDustMapQuery:
    """Tests for querying dust maps with SkyCoord objects."""

    SKY_COORD = SkyCoord(ra=180.0, dec=45.0, unit='deg', frame='icrs')

    def test_sfd_query_returns_scalar(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        dust_map = get_dust_map('sfd')
        ebv = dust_map.query(self.SKY_COORD)
        assert isinstance(float(ebv), float)

    def test_csfd_query_returns_scalar(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        dust_map = get_dust_map('csfd')
        ebv = dust_map.query(self.SKY_COORD)
        assert isinstance(float(ebv), float)

    def test_sfd_ebv_is_nonnegative(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        dust_map = get_dust_map('sfd')
        ebv = dust_map.query(self.SKY_COORD)
        assert ebv >= 0

    def test_csfd_ebv_is_nonnegative(self):
        from quasar_utils.dereddening.dustmaps import get_dust_map
        dust_map = get_dust_map('csfd')
        ebv = dust_map.query(self.SKY_COORD)
        assert ebv >= 0
