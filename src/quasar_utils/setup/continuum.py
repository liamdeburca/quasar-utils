from logging import getLogger
from typing import Iterable, ClassVar, Self
from astropy.units import Unit, Quantity

from pydantic import validate_call
from pydantic.dataclasses import dataclass

from . import parsing
from ..utils._info import _Info
from ..utils.utils import val_and_type
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.pathlib import Path_, AbsoluteFileLike

logger = getLogger(__name__)

@dataclass
class ContinuumInfo(_Info):
    _x0: float | Quantity_ = 1450 * Unit('angstrom')
    _y0: float | Quantity_ = 1e-17 * Unit('erg/(s.cm2.angstrom)')
    _windows: Iterable[float] | Quantity_ = [
        [1425, 1475],
        [1675, 1690],
        [1975, 2050],
        [2150, 2250],
    ] * Unit('angstrom')

    _flux_bounds: Iterable[float] | Quantity_ = [1e-17, 1e-14] * Unit('erg/(s.cm2.angstrom)')

    sigmas: list[float] = [3.00, 2.75, 2.50]
    
    x0: float | None = None
    y0: float | None = None
    windows: list[CoordBounds] | None = None
    flux_bounds: AstropyBounds | None = None
    alpha_bounds: AstropyBounds = (-5.0, 0.0)

    _keys: ClassVar[frozenset[str]] = frozenset([
        '_x0', '_y0', '_windows', '_flux_bounds',
        'sigmas', 'x0', 'y0', 'windows', 'flux_bounds', 'alpha_bounds',
    ])
    _cache: ClassVar[dict[Path_, Self]] = {}

    def update(self, info) -> None:
        """
        Convert to unitless.
        """
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

    @validate_call
    @classmethod
    def from_file(
        cls,
        path: AbsoluteFileLike | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and path in cls._cache.keys():
            logger.debug(f"Using cached 'ContinuumInfo' for '{path}'.")
            
            cinfo = cls._cache[path]
            if create_copy: return cinfo.copy()
            else:           return cinfo

        cinfo: ContinuumInfo = ContinuumInfo()
        if path is None:
            return cinfo
        
        logger.debug(f"Configuring 'ContinuumInfo' using '{path}':")  
        lines = get_lines_from_file.__wrapped__(logger, 'CONTINUUM', path)

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

        cls._cache[path] = cinfo

        return cinfo