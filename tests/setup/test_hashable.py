"""Test that all Info classes are hashable."""


def test_absorption_info_is_hashable():
    from quasar_utils.setup.absorption import AbsorptionInfo

    info = AbsorptionInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_balmer_info_is_hashable():
    from quasar_utils.setup.balmer import BalmerInfo

    info = BalmerInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_continuum_info_is_hashable():
    from quasar_utils.setup.continuum import ContinuumInfo

    info = ContinuumInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_convolution_info_is_hashable():
    from quasar_utils.setup.convolution import ConvolutionInfo

    info = ConvolutionInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_cosmology_info_is_hashable():
    from quasar_utils.setup.cosmology import CosmologyInfo

    info = CosmologyInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_error_info_is_hashable():
    from quasar_utils.setup.error import ErrorInfo

    info = ErrorInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_host_info_is_hashable():
    from quasar_utils.setup.host import HostInfo

    info = HostInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_iron_info_is_hashable():
    from quasar_utils.setup.iron import IronInfo

    info = IronInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_lines_info_is_hashable():
    from quasar_utils.setup.lines import LinesInfo

    info = LinesInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_loading_info_is_hashable():
    from quasar_utils.setup.loading import LoadingInfo

    info = LoadingInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_mcmc_info_is_hashable():
    from quasar_utils.setup.mcmc import MCMCInfo

    info = MCMCInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_nonlinear_info_is_hashable():
    from quasar_utils.setup.nonlinear import NonLinearInfo

    info = NonLinearInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_random_info_is_hashable():
    from quasar_utils.setup.random import RandomInfo

    info = RandomInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_units_info_is_hashable():
    from quasar_utils.setup.units import UnitsInfo

    info = UnitsInfo()
    hash_value = hash(info)
    assert isinstance(hash_value, int)


def test_info_is_hashable():
    from quasar_utils.setup.info import Info

    info = Info()
    assert hash(info) is not None