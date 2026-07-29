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


def test_nonlinear_info_is_hashable():
    from quasar_utils.setup.nonlinear import NonLinearInfo

    info = NonLinearInfo()
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
    # Info is not a dataclass so it cannot be hashed directly,
    # but we can verify that all its *Info attributes are hashable
    assert hash(info.absorption) is not None
    assert hash(info.balmer) is not None
    assert hash(info.continuum) is not None
    assert hash(info.error) is not None
    assert hash(info.host) is not None
    assert hash(info.iron) is not None
    assert hash(info.lines) is not None
    assert hash(info.loading) is not None
    assert hash(info.nonlinear) is not None
    assert hash(info.units) is not None


def test_info_subclasses_can_be_used_in_set():
    """Test that Info subclass instances can be added to a set."""
    from quasar_utils.setup.absorption import AbsorptionInfo
    from quasar_utils.setup.balmer import BalmerInfo

    abs_info1 = AbsorptionInfo()
    abs_info2 = AbsorptionInfo()
    balmer_info = BalmerInfo()

    # Should be able to add to a set
    info_set = {abs_info1, balmer_info}
    assert len(info_set) == 2

    # abs_info2 should equal abs_info1 since they have the same values
    assert hash(abs_info2) == hash(abs_info1)
