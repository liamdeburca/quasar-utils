from logging import getLogger
from typing import ClassVar, Self, Any
from astropy.units import Unit
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds
from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

DEFAULT_VALUES: dict[str, Any] = {
    '_x_limit': 1220 * Unit('angstrom'),
    '_v_sep': 10_000 * Unit('km/s'),
    '_v_off_bounds': (-5_000, 5_000) * Unit('km/s'),
    '_sigma_v_bounds': (250, 5_000) * Unit('km/s'),
    '_strength_bounds': (1e-16, 1) * Unit('erg/(s.cm2)'),
    '_w': 25,
    '_forced_splits': [1450, 1680, 2000] * Unit('angstrom'),
    'min_fittable_total': 50,
    'min_fittable_ratio': 0.6,
    'evaluate_initial': 3,
    'aggressive': False,
    'crop': False,
    'measure': 'getFluxSNR',
    'reverse': False,
    'snr': 10,
    'make_copies': False,
    'adapt_scale': True,
    'scale_init': 1.0,
    'scale_bounds': (0.0, 100.0),
}

@dataclass
class LinesInfo(_Info):
    fit: bool = True

    _x_limit: float | Quantity_ = field(default=DEFAULT_VALUES['_x_limit'])
    _v_sep: float | Quantity_ = field(default=DEFAULT_VALUES['_v_sep'])
    _v_off_bounds: AstropyBounds | Quantity_ = field(default=DEFAULT_VALUES['_v_off_bounds'])
    _sigma_v_bounds: AstropyBounds | Quantity_ = field(default=DEFAULT_VALUES['_sigma_v_bounds'])
    _strength_bounds: AstropyBounds | Quantity_ = field(default=DEFAULT_VALUES['_strength_bounds'])
    _w: int | Quantity_ = field(default=DEFAULT_VALUES['_w'])
    _forced_splits: list[float] | Quantity_ = field(default=DEFAULT_VALUES['_forced_splits'])
    min_fittable_total: int = field(default=DEFAULT_VALUES['min_fittable_total'])
    min_fittable_ratio: float = field(default=DEFAULT_VALUES['min_fittable_ratio'])
    evaluate_initial: int = field(default=DEFAULT_VALUES['evaluate_initial'])
    aggressive: bool = field(default=DEFAULT_VALUES['aggressive'])
    crop: bool = field(default=DEFAULT_VALUES['crop'])
    measure: str = field(default=DEFAULT_VALUES['measure'])
    reverse: bool = field(default=DEFAULT_VALUES['reverse'])
    snr: float | int = field(default=DEFAULT_VALUES['snr'])
    make_copies: bool = field(default=DEFAULT_VALUES['make_copies'])
    adapt_scale: bool = field(default=DEFAULT_VALUES['adapt_scale'])
    scale_init: float = field(default=DEFAULT_VALUES['scale_init'])
    scale_bounds: AstropyBounds = field(default=DEFAULT_VALUES['scale_bounds'])

    x_limit: float | None = field(default=None, init=False)
    v_sep: float | None = field(default=None, init=False)
    v_off_bounds: AstropyBounds | None = field(default=None, init=False)
    sigma_v_bounds: AstropyBounds | None = field(default=None, init=False)
    strength_bounds: AstropyBounds | None = field(default=None, init=False)
    w: int | None = field(default=None, init=False)
    forced_splits: list[float] | None = field(default=None, init=False)
    
    _keys: ClassVar[frozenset[str]] = frozenset([
        'fit',
        '_x_limit', 'x_limit',
        '_v_sep', 'v_sep',
        '_v_off_bounds', 'v_off_bounds',
        '_sigma_v_bounds', 'sigma_v_bounds',
        '_strength_bounds', 'strength_bounds',
        '_w', 'w',
        '_forced_splits', 'forced_splits',

        'min_fittable_total',
        'min_fittable_ratio',
        'evaluate_initial',
        'aggressive',
        'crop',
        'measure',
        'reverse',
        'snr',

        'make_copies',
        'adapt_scale',
        'scale_init',
        'scale_bounds',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'x_limit': "to_wavelength",
        'v_sep': "to_velocity",
        'v_off_bounds': "to_velocity_bounds",
        'sigma_v_bounds': "to_velocity_bounds",
        'strength_bounds': "to_strength_bounds",
        'w': "to_n_pixels",
        'forced_splits': "to_wavelength_array",
    }

    def update(self, info) -> None:
        """
        Converts parameters with units into their dimensionless equivalents.
        """
        super().update(info, logger)

    @classmethod
    @validate_call
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache.keys():
            logger.debug(f"Using cached 'LinesInfo' for '{path}'.")
            
            linfo = cls._cache[str(path)]
            if create_copy: return linfo.copy()
            else:           return linfo

        linfo: LinesInfo = LinesInfo()
        if path is None:
            return linfo

        logger.debug(f"Configuring 'LinesInfo' using '{path}'.")        
        lines = get_lines_from_file.__wrapped__('LINES', path, logger)

        for count, line in enumerate(lines, start=1):
            key = line[0].lower()
            
            match key:
                case 'min_fittable_total':
                    val = parsing.as_int(line[1])

                case 'refine' | 'crop' | 'reverse' | 'aggressive' | 'reverse' \
                    | 'make_copies' | 'adapt_scale':
                    val = parsing.as_bool(line[1])

                case 'v_sep' | 'w':
                    key = '_' + key
                    val = parsing.as_scalar_or_quantity(line[1:])

                case 'v_off_bounds' | 'sigma_v_bounds' | 'strength_bounds':
                    key = '_' + key
                    val = parsing.as_bounds_of_scalars_or_quantity(line[1:])

                case 'forced_splits':
                    key = '_' + key
                    val = parsing.as_list_of_scalars_or_quantity(line[1:])

                case 'measure':
                    val = line[1]  

                case 'x_limit':
                    key = '_' + key
                    val = parsing.as_quantity(line[1:])

                case 'min_fittable_ratio' | 'evaluate_initial' | 'snr' | \
                    'scale_init':
                    val = parsing.as_float(line[1])

                case 'scale_bounds':
                    val = parsing.as_bounds(line[1:])
                    
            linfo[key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] '{key}': {val_and_type(val)}."
            )

        cls._cache[str(path)] = linfo
                        
        return linfo
    
    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "lines", logger)