import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "absorption": ["AbsorptionInfo"],
    "balmer": ["BalmerInfo"],
    "continuum": ["ContinuumInfo"],
    "convolution": ["ConvolutionInfo"],
    "cosmology": ["CosmologyInfo"],
    "error": ["ErrorInfo"],
    "host": ["HostInfo"],
    "info": ["Info"],
    "iron": ["IronInfo"],
    "lines": ["LinesInfo"],
    "loading": ["LoadingInfo"],
    "mcmc": ["MCMCKwargs", "MCMCInfo"],
    "nonlinear": ["NonLinearInfo", "FitterKwargs"],
    "random": ["RandomInfo"],
    "units": ["UnitsInfo"],
}

if TYPE_CHECKING:
    from .absorption import AbsorptionInfo  # noqa: F401
    from .balmer import BalmerInfo  # noqa: F401
    from .continuum import ContinuumInfo  # noqa: F401
    from .convolution import ConvolutionInfo  # noqa: F401
    from .cosmology import CosmologyInfo  # noqa: F401
    from .error import ErrorInfo  # noqa: F401
    from .host import HostInfo  # noqa: F401
    from .info import Info  # noqa: F401
    from .iron import IronInfo  # noqa: F401
    from .lines import LinesInfo  # noqa: F401
    from .loading import LoadingInfo  # noqa: F401
    from .mcmc import MCMCInfo, MCMCKwargs  # noqa: F401
    from .nonlinear import FitterKwargs, NonLinearInfo  # noqa: F401
    from .random import RandomInfo  # noqa: F401
    from .units import UnitsInfo  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
