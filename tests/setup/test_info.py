from astropy.constants import c
from numpy import isclose


def test_can_instantiate_info():
    from quasar_utils.setup.info import Info

    info = Info()
    assert info.is_updated
    assert isclose(info.loading.sigma_res, 69 / c.to("km/s").value)