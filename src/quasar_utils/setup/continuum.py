from logging import getLogger
from typing import Any, Iterable, ClassVar, Self
from astropy.units import Unit
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

DEFAULT_VALUES: dict[str, Any] = {
    '_x0': 1450 * Unit('angstrom'),
    '_y0': 1e-17 * Unit('erg/(s.cm2.angstrom)'),
    '_windows': [
        (1425, 1475), 
        (1675, 1690), 
        (1975, 2050), 
        (2150, 2250),
    ] * Unit('angstrom'),
    '_flux_bounds': [1e-17, 1e-14] * Unit('erg/(s.cm2.angstrom)'),
    'sigmas': [3.00, 2.75, 2.50],
    'alpha_bounds': (-5.0, 0.0),
}

@dataclass
class ContinuumInfo(_Info):
    fit: bool = True

    _x0: float | Quantity_ = field(default=DEFAULT_VALUES['_x0'])
    _y0: float | Quantity_ = field(default=DEFAULT_VALUES['_y0'])
    _windows: Iterable[CoordBounds] | Quantity_ = field(default_factory=lambda: DEFAULT_VALUES['_windows'])
    _flux_bounds: Iterable[float] | Quantity_ = field(default_factory=lambda: DEFAULT_VALUES['_flux_bounds'])
    sigmas: list[float] = field(default_factory=lambda: DEFAULT_VALUES['sigmas'])
    alpha_bounds: AstropyBounds = field(default_factory=lambda: DEFAULT_VALUES['alpha_bounds'])

    x0: float | None = field(default=None, init=False)
    y0: float | None = field(default=None, init=False)
    windows: list[CoordBounds] | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)

    _keys: ClassVar[frozenset[str]] = frozenset([
        'fit', '_x0', '_y0', '_windows', '_flux_bounds',
        'sigmas', 'x0', 'y0', 'windows', 'flux_bounds', 'alpha_bounds',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'x0': "to_wavelength",
        'y0': "to_flux",
        'windows': "to_wavelength_windows",
        'flux_bounds': "to_flux_bounds",
    }

    def update(self, info) -> None:
        """
        Convert to unitless.
        """
        super().update(info, logger)
        return 

    @classmethod
    @validate_call
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
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "continuum", logger)