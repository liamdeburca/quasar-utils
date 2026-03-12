from logging import getLogger
from typing import ClassVar, Self
from numpy import array
from astropy.units import Unit, Quantity

from pydantic import validate_call

from ..utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds
from quasar_typing.pathlib import Path_, AbsoluteFilePath

logger = getLogger(__name__)

def _replace_nan_with_none(bounds: tuple):
    from numpy import isfinite
    return tuple([
        bound \
        if isfinite(bound) \
        else None \
        for bound in bounds
    ])

class LinesInfo(_Info):
    _keys: ClassVar[frozenset[str]] = frozenset([
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
    _cache: ClassVar[dict[Path_, Self]] = {}

    @validate_call(validate_return=False)
    def __init__(
        self,
        _x_limit: float | Quantity_ = 1220 * Unit('angstrom'),
        _v_sep: float | Quantity_ = 10_000 * Unit('km/s'),
        _v_off_bounds: AstropyBounds | Quantity_ = (-5_000, 5_000) * Unit('km/s'),
        _sigma_v_bounds: AstropyBounds | Quantity_ = (250, 5_000) * Unit('km/s'),
        _strength_bounds: AstropyBounds | Quantity_ = (1e-16, 1) * Unit('erg/(s.cm2)'),
        _w: int | Quantity_ = 25,
        _forced_splits: list[float] | Quantity_ = [1450, 1680, 2000] * Unit('angstrom'),
        min_fittable_total: int = 50,
        min_fittable_ratio: float = 0.6,
        evaluate_initial: int = 3,
        aggressive: bool = False,
        crop: bool = False,
        measure: str = 'getFluxSNR',
        reverse: bool = False,
        snr: float | int = 10,
        make_copies: bool = False,
        adapt_scale: bool = True,
        scale_init: float = 1.0,
        scale_bounds: AstropyBounds = (0.0, 100.0),
    ):
        self._x_limit: float | Quantity_ = _x_limit
        self._v_sep: float | Quantity_ = _v_sep
        self._v_off_bounds: AstropyBounds | Quantity_ = _v_off_bounds
        self._sigma_v_bounds: AstropyBounds | Quantity_ = _sigma_v_bounds
        self._strength_bounds: AstropyBounds | Quantity_ = _strength_bounds
        self._w: int | Quantity_ = _w
        self._forced_splits: list[float] | Quantity_ = _forced_splits
        self.min_fittable_total: int = min_fittable_total
        self.min_fittable_ratio: float = min_fittable_ratio
        self.evaluate_initial: int = evaluate_initial
        self.aggressive: bool = aggressive
        self.crop: bool = crop
        self.measure: str = measure
        self.reverse: bool = reverse
        self.snr: float | int = snr
        self.make_copies: bool = make_copies
        self.adapt_scale: bool = adapt_scale
        self.scale_init: float = scale_init
        self.scale_bounds: AstropyBounds = scale_bounds

        # Set by 'update' method
        self.x_limit: float | None = None
        self.v_sep: float | None = None
        self.v_off_bounds: AstropyBounds | None = None
        self.sigma_v_bounds: AstropyBounds | None = None
        self.strength_bounds: AstropyBounds | None = None
        self.w: int | None = None
        self.forced_splits: list[float] | None = None

    def update(self, info) -> None:
        """
        Converts parameters with units into their dimensionless equivalents.
        """
        logger.debug("Updating 'LinesInfo' class:")

        # Convert velocities to no. of pixels
        msg = ">>> [1/9] 'x_limit': "
        if isinstance(old := self['_x_limit'], Quantity):
            self['x_limit'] = new = info.units.getWavelength(old)
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['x_limit'] = float(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [2/9] 'v_sep': "
        if not isinstance(old := self['_v_sep'], Quantity):
            old *= info.units['velocity_unit']
        self['v_sep'] = new = info.units.getC(old)
        msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        logger.debug(msg)

        msg = ">>> [3/9] 'v_off_bounds': "
        if not isinstance(old := self['_v_off_bounds'], Quantity):
            old *= info.units['velocity_unit']
        self['v_off_bounds'] = new = _replace_nan_with_none(
            tuple(info.units.getC(old))
        )
        msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        logger.debug(msg)

        msg = ">>> [4/9] 'sigma_v_bounds': "
        if not isinstance(old := self['_sigma_v_bounds'], Quantity):
            old *= info.units['velocity_unit']
        self['sigma_v_bounds'] = new = _replace_nan_with_none(
            tuple(info.units.getC(old))
        )
        msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        logger.debug(msg)

        msg = ">>> [5/9] 'strength_bounds': "
        if isinstance(old := self['_strength_bounds'], Quantity):
            self['strength_bounds'] = new = _replace_nan_with_none(tuple(
                info.units.getStrength(old)
            ))
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['strength_bounds'] = _replace_nan_with_none(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [6/9] 'w': "
        if isinstance(old := self['_w'], Quantity):
            self['w'] = new = info.units.getC(old) // info.units['sigma_res']
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['w'] = int(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [7/9] 'forced_splits': "
        if isinstance(old := self['_forced_splits'], Quantity):
            self['forced_splits'] = new = info.units.getWavelength(old)
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['forced_splits'] = array(old, dtype=float)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        logger.debug("... finished updating 'LinesInfo' class!")

    @classmethod
    @validate_call(validate_return=False)
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and path in cls._cache.keys():
            logger.debug(f"Using cached 'LinesInfo' for '{path}'.")
            
            linfo = cls._cache[path]
            if create_copy: return linfo.copy()
            else:           return linfo

        linfo: LinesInfo = LinesInfo()
        if path is None:
            return linfo

        logger.debug(f"Configuring 'LinesInfo' using '{path}'.")        
        lines = get_lines_from_file.__wrapped__(logger, 'LINES', path)

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

        cls._cache[path] = linfo
                        
        return linfo