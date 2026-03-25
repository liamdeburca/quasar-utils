from logging import getLogger
logger = getLogger("quasar_utils.setup.continuum")

from typing import Iterable, ClassVar, Self
from astropy.units import Unit, Quantity

from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.pathlib import AbsoluteFilePath

class ContinuumInfo(_Info):
    _keys: ClassVar[frozenset[str]] = frozenset([
        '_x0', '_y0', '_windows', '_flux_bounds',
        'sigmas', 'x0', 'y0', 'windows', 'flux_bounds', 'alpha_bounds',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'x0': "to_wavelength",
        'y0': "to_flux",
        'windows': "to_wavelength_windows",
        'flux_bounds': "to_flux_bounds",
    }

    @validate_call(validate_return=False)
    def __init__(
        self,
        *,
        _x0: float | Quantity_ = 1450 * Unit('angstrom'),
        _y0: float | Quantity_ = 1e-17 * Unit('erg/(s.cm2.angstrom)'),
        _windows: Iterable[tuple[float, float]] | Quantity_ = [
            [1425, 1475], [1675, 1690], [1975, 2050], [2150, 2250]
        ] * Unit('angstrom'),
        _flux_bounds: Iterable[float] | Quantity_ = [1e-17, 1e-14] * Unit('erg/(s.cm2.angstrom)'),
        sigmas: list[float] = [3.00, 2.75, 2.50],
        alpha_bounds: AstropyBounds = (-5.0, 0.0),
    ):
        self._x0: float | Quantity_ = _x0
        self._y0: float | Quantity_ = _y0
        self._windows: Iterable[tuple[float, float]] | Quantity_ = _windows
        self._flux_bounds: Iterable[float] | Quantity_ = _flux_bounds
        self.sigmas: list[float] = sigmas
        self.alpha_bounds: AstropyBounds = alpha_bounds

        # Set by 'update' method
        self.x0: float | None = None
        self.y0: float | None = None
        self.windows: list[CoordBounds] | None = None
        self.flux_bounds: AstropyBounds | None = None

    def update(self, info) -> None:
        """
        Convert to unitless.
        """
        super().update(info, logger)
        return 
        logger.debug("Updating 'ContinuumInfo' class:")

        # Calculate dimensionless quantities
        msg = ">>> [1/4] 'x0': "
        if isinstance(old := self['_x0'], Quantity):
            self['x0'] = new = info.units.getWavelength(old)
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['x0'] = float(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)        

        msg = ">>> [2/4] 'y0': "
        if isinstance(old := self['_y0'], Quantity):
            self['y0'] = new = info.units.getFlux(old)
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['y0'] = float(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [3/4] 'windows': "
        if isinstance(old := self['_windows'], Quantity):
            self['windows'] = new = [
                tuple(w) for w in info.units.getWavelength(old)
            ]
            msg += f"\n{val_and_type(old)} -> \n{val_and_type(new)}."
        else:
            self['windows'] = old
            msg += f"\n{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [4/4] 'flux_bounds': "
        if isinstance(old := self['_flux_bounds'], Quantity):
            self['flux_bounds'] = new = tuple(info.units.getFlux(old))
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['flux_bounds'] = tuple(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        logger.debug("... finished updating 'ContinuumInfo' class!")

    @classmethod
    @validate_call(validate_return=False)
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache.keys():
            logger.debug(f"Using cached 'ContinuumInfo' for '{path}'.")
            
            cinfo = cls._cache[str(path)]
            if create_copy: return cinfo.copy()
            else:           return cinfo

        cinfo: ContinuumInfo = ContinuumInfo()
        if path is None:
            return cinfo
        
        logger.debug(f"Configuring 'ContinuumInfo' using '{path}':")  
        lines = get_lines_from_file.__wrapped__('CONTINUUM', path, logger)

        for count, line in enumerate(lines, start=1):
            key = line[0].lower()

            match key:
                case 'x0' | 'y0':
                    key = '_' + key
                    val = parsing.as_scalar_or_quantity(line[1:])

                case 'windows':
                    key = '_' + key
                    val = parsing.as_pairs_of_floats_or_quantity(line[1:])
                    
                case 'sigmas':
                    val = parsing.as_list_of_floats(line[1])

                case 'flux_bounds':
                    key = '_' + key
                    val = parsing.as_bounds_of_scalars_or_quantity(line[1:])

                case 'alpha_bounds':
                    val = parsing.as_bounds(line[1:])

            cinfo[key] = val
            msg = (
                '\n' if (key == '_windows') else ' '
            ).join([
                f">>> [{count}/{len(lines)}] {key}:",
                f"{val_and_type(val)}."
            ])
            logger.debug(msg)

        cls._cache[str(path)] = cinfo

        return cinfo

    @classmethod
    @validate_call(validate_return=False)
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "continuum", logger)