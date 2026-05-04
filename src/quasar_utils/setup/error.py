from logging import getLogger
from typing import ClassVar, Self
from numpy.random import RandomState
from astropy.units import Unit
from dataclasses import field
from pydantic.dataclasses import dataclass

from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.numpy import RandomState_
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.misc import (
    Method, Scale, Variant, BootstrapType, FWHMStrategy, 
    VaryLines, OutLines, OutMeasures
)

logger = getLogger(__name__)

@dataclass
class ErrorInfo(_Info):
    method: Method = 'bootstrap'
    remodel: bool = False
    replace_missing: bool = True
    scale: Scale = 'global'
    variant: Variant = 'standard'
    bootstrap_type: BootstrapType = 'spectrum'
    fwhm_strategy: FWHMStrategy = 'average'
    iterations: int = 100
    random_state: RandomState_ = field(default_factory=lambda: RandomState(42))
    renew_rng: bool = True
    n_sigmas: float = 2.0
    res: int = 1000
    render_width: float | int = 5
    exact: bool = True
    _v_int: float | Quantity_ = 18_000 * Unit('km/s')
    ipv_int: float = 0
    _dx_int: float | Quantity_ = 50 * Unit('angstrom')
    vary_lines: VaryLines = field(default_factory=lambda: VaryLines({'all'}))
    out_lines: OutLines = field(default_factory=lambda: OutLines({'all'}))
    out_measures: OutMeasures = field(default_factory=lambda: OutMeasures({'all'}))
    percentiles: set[int] | None = field(default_factory=lambda: {5, 50, 95})
    tqdm_disable: bool = False
    tqdm_leave: bool = False

    v_int: float | None = field(default=None, init=False)
    dx_int: float | None = field(default=None, init=False)

    _keys: ClassVar[frozenset[str]] = frozenset([
        'method', 'scale', 'remodel', 'variant', 'replace_missing',
        'bootstrap_type', 
        'iterations', 'random_state', 'renew_rng', 'n_sigmas',
        'res', 'render_width', 'fwhm_strategy', 'exact', 
        '_v_int', 'ipv_int', '_dx_int',
        'vary_lines', 'out_lines', 'out_measures', 'percentiles', 
        'tqdm_disable', 'tqdm_leave',
        'v_int', 'dx_int',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'v_int': "to_velocity",
        'dx_int': "to_wavelength",
    }

    def update(self, info) -> None:
        super().update(info, logger)
        return 

    @classmethod
    @validate_call
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and path in cls._cache.keys():
            logger.debug(f"Using cached 'ErrorInfo' for '{path}'.")
            
            einfo = cls._cache[path]
            if create_copy: return einfo.copy()
            else:           return einfo

        einfo: ErrorInfo = ErrorInfo()
        if path is None:
            return einfo
        
        logger.debug(f"Configuring 'ErrorInfo' using '{path}'.")
        lines = get_lines_from_file.__wrapped__('ERROR', path, logger)

        for count, line in enumerate(lines, start=1):
            prefix = ''
            key = line[0].lower()

            match key:
                case 'method' | 'scale' | 'variant' | 'bootstrap_type' \
                    | 'fwhm_strategy':
                    val = line[1]

                case 'iterations' | 'res' | 'random_state':
                    val = parsing.as_int(line[1])
                    
                case 'remodel' | 'renew_rng' | 'tqdm_disable' | 'tqdm_leave' \
                    | 'replace_missing' | 'exact':
                    val = parsing.as_bool(line[1])

                case 'v_int' | 'dx_int':
                    prefix = '_'
                    val = parsing.as_scalar_or_quantity(line[1:])

                case 'out_measures':
                    val = parsing.as_set_of_measures(line[1])

                case 'out_waves' | 'vary_lines':
                    prefix = '_'
                    val = parsing.as_list_of_lines(line[1:])

                case 'percentiles':
                    val = set(parsing.as_list_of_ints(line[1]))
                                        
                case _:
                    val = float(line[1])

            einfo[prefix + key] = val  
            logger.debug(
                f">>> [{count}/{len(lines)}] '{key}': {val_and_type(val)}."
            ) 

        if einfo['_v_int'] != 0:
            einfo['ipv_int'] = 0
            einfo['_dx_int'] = 0
        elif einfo['ipv_int'] != 0:
            einfo['_dx_int'] = 0

        cls._cache[path] = einfo

        return einfo

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "error", logger)