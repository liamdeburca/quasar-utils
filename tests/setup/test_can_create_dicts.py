def test_absorption():
    from quasar_utils.setup.absorption import AbsorptionInfo
    ainfo = AbsorptionInfo()
    ainfo.to_dict()

def test_balmer():
    from quasar_utils.setup.balmer import BalmerInfo
    binfo = BalmerInfo()
    binfo.to_dict()

def test_continuum():
    from quasar_utils.setup.continuum import ContinuumInfo
    cinfo = ContinuumInfo()
    cinfo.to_dict()

def test_convolution():
    from quasar_utils.setup.convolution import ConvolutionInfo
    cinfo = ConvolutionInfo()
    cinfo.to_dict()

def test_error():
    from quasar_utils.setup.error import ErrorInfo
    einfo = ErrorInfo()
    einfo.to_dict()

def test_host():
    from quasar_utils.setup.host import HostInfo
    hinfo = HostInfo()
    hinfo.to_dict()

def test_iron():
    from quasar_utils.setup.iron import IronInfo
    iinfo = IronInfo()
    iinfo.to_dict()

def test_lines():
    from quasar_utils.setup.lines import LinesInfo
    linfo = LinesInfo()
    linfo.to_dict()

def test_loading():
    from quasar_utils.setup.loading import LoadingInfo
    linfo = LoadingInfo()
    linfo.to_dict()

def test_nonlinear():
    from quasar_utils.setup.nonlinear import NonLinearInfo
    ninfo = NonLinearInfo()
    ninfo.to_dict()

def test_units():
    from quasar_utils.setup.units import UnitsInfo
    uinfo = UnitsInfo()
    uinfo.to_dict()

def test_info():
    from quasar_utils.setup import Info
    info = Info()
    info.to_dict()