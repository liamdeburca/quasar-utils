from logging import getLogger
from typing import ClassVar, Self
from astropy.units import Unit, Quantity

from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.misc.literals import Method, Scale, Variant, BootstrapType, \
    FWHMStrategy, OutMeasures, OutLines, VaryLines

logger = getLogger(__name__)

class ErrorInfo(_Info):
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

    @validate_call(validate_return=False)
    def __init__(
        self,
        method: Method = 'bootstrap',
        remodel: bool = False,
        replace_missing: bool = True,
        scale: Scale = 'global',
        variant: Variant = 'standard',
        bootstrap_type: BootstrapType = 'spectrum',
        fwhm_strategy: FWHMStrategy = 'average',
        iterations: int = 100,
        random_state: int = 42,
        renew_rng: bool = True,
        n_sigmas: float = 2.0,
        res: int = 1000,
        render_width: float | int = 5,
        exact: bool = True,
        _v_int: float | Quantity_ = 18_000 * Unit('km/s'),
        ipv_int: float = 0,
        _dx_int: float | Quantity_ = 50 * Unit('angstrom'),
        vary_lines: VaryLines = frozenset({'all'}),
        out_lines: OutLines = frozenset({'all'}),
        out_measures: OutMeasures = frozenset({'all'}),
        percentiles: set[int] | None = {5, 50, 95},
        tqdm_disable: bool = False,
        tqdm_leave: bool = False,
    ):
        self.method: Method = method
        self.remodel: bool = remodel
        self.replace_missing: bool = replace_missing
        self.scale: Scale = scale
        self.variant: Variant = variant
        self.bootstrap_type: BootstrapType = bootstrap_type
        self.fwhm_strategy: FWHMStrategy = fwhm_strategy
        self.iterations: int = iterations
        self.random_state: int = random_state
        self.renew_rng: bool = renew_rng
        self.n_sigmas: float = n_sigmas
        self.res: int = res
        self.render_width: float | int = render_width
        self.exact: bool = exact
        self._v_int: float | Quantity_ = _v_int
        self.ipv_int: float = ipv_int
        self._dx_int: float | Quantity_ = _dx_int
        self.vary_lines: VaryLines = vary_lines
        self.out_lines: OutLines = out_lines
        self.out_measures: OutMeasures = out_measures
        self.percentiles: set[int] | None = percentiles
        self.tqdm_disable: bool = tqdm_disable
        self.tqdm_leave: bool = tqdm_leave

        # Set by 'update' method
        self.v_int: float | None = None
        self.dx_int: float | None = None

    def update(self, info) -> None:
        super().update(info, logger)
        return 
        logger.debug("Updating 'ErrorInfo' class...")

        # Calculate dimensionless quantities
        msg = ">>> [1/4] 'v_int': "
        if isinstance(old := self['_v_int'], Quantity):
            self['v_int'] = new = info.units.getC(old)
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['v_int'] = float(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [2/4] 'dx_int': "
        if isinstance(old := self['_dx_int'], Quantity):
            self['dx_int'] = new = info.units.getWavelength(old)
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['dx_int'] = float(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [3/4] 'out_waves': "
        old = self['_out_waves']
        self['out_waves'] = set()
        for val in self['_out_waves']:
            if val == 'all':
                self['out_waves'].add('all')
            else:
                self['out_waves'].add(
                    info.units.getWavelength(val) \
                    if isinstance(val, Quantity) \
                    else float(val)
                )
        msg += f"{self['out_waves']}."
        logger.debug(msg)

        msg = ">>> [4/4] 'vary_lines': "
        self['vary_lines'] = set()
        for val in self['_vary_lines']:
            if val == 'all':
                self['vary_lines'].add('all')
            else:
                self['vary_lines'].add(
                    info.units.getWavelength(val) \
                    if isinstance(val, Quantity) \
                    else float(val)
                )

        msg += f"{self['vary_lines']}."
        logger.debug(msg)

        logger.debug("... finished updating 'ErrorInfo' class!")

    @classmethod
    @validate_call(validate_return=False)
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
    @validate_call(validate_return=False)
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "error", logger)