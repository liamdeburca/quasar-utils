from logging import getLogger
from typing import Literal, ClassVar, Self
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
from quasar_typing.bounds import CoordBounds
from quasar_typing.misc.string_selection import StringSelection

logger = getLogger("quasar_utils.setup.balmer")

@dataclass
class BalmerInfo(_Info):
    fit: bool = True

    _windows: list[CoordBounds] | Quantity_ = [[3000, 4500]] * Unit('angstrom')
    source: Literal['SH1995'] = 'SH1995'
    allow_interp_fitting: bool = True
    n_u_min: int = 7
    n_u_max: int = 50
    _edge: float | Quantity_ = 3646 * Unit('angstrom')
    _dens: float | Quantity_ = 1e9 * Unit('cm^-3')
    _flux: float | Quantity_ = 1e-17 * Unit('erg/(s.cm2.angstrom)')
    _fwhm: float | Quantity_ = 5_000 * Unit('km/s')
    _temp: float | Quantity_ = 15_000 * Unit('K')
    tau: float = 1.0
    scale: float = 3.0
    ratio: float = 0.3
    _flux_bounds: AstropyBounds | Quantity_ = [1e-18, 1e-15] * Unit('erg/(s.cm2.angstrom)')
    _fwhm_bounds: AstropyBounds | Quantity_ = [1000, 20_000] * Unit('km/s')
    _temp_bounds: AstropyBounds | Quantity_ = [5_000, 30_000] * Unit('K')
    tau_bounds: AstropyBounds = (0.1, 10.0)
    scale_bounds: AstropyBounds = (1.0, 10.0)
    ratio_bounds: AstropyBounds = (0.5, 2.0)
    _fixed: StringSelection = field(default_factory=lambda: StringSelection({'temp', 'tau', 'scale', 'ratio'}))
    raster_n: int = 20
    min_fittable_ratio: float = 0.6
    min_fittable_total: int = 100

    windows: list[CoordBounds] | None = field(default=None, init=False)
    edge: float | None = field(default=None, init=False)
    dens: float | None = field(default=None, init=False)
    flux: float | None = field(default=None, init=False)
    fwhm: float | None = field(default=None, init=False)
    temp: float | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)
    fwhm_bounds: AstropyBounds | None = field(default=None, init=False)
    temp_bounds: AstropyBounds | None = field(default=None, init=False)
    fixed: dict[str, bool] | None = field(default=None, init=False)

    _keys: ClassVar[frozenset[str]] = frozenset([
        'fit',
        '_windows', 'windows',
        'source', 'allow_interp_fitting', 
        'n_u_min', 'n_u_max', '_edge', '_dens', 'edge', 'dens',
        '_flux', '_fwhm', '_temp', 'tau', 'scale', 'ratio',
        'flux', 'fwhm', 'temp',
        '_flux_bounds', '_fwhm_bounds', '_temp_bounds',
        'tau_bounds', 'scale_bounds', 'ratio_bounds',
        'flux_bounds', 'fwhm_bounds', 'temp_bounds',
        '_fixed', 'fixed',
        'raster_n',
        'min_fittable_ratio', 'min_fittable_total',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'windows': "to_wavelength_windows",
        'edge': "to_wavelength", 
        'dens': "to_density", 
        'flux': "to_flux", 
        'fwhm': "to_velocity", 
        'temp': "to_temperature",
        'flux_bounds': "to_flux_bounds", 
        'fwhm_bounds': "to_velocity_bounds", 
        'temp_bounds': "to_temperature_bounds",
        'fixed': "to_fixed",
    }

    def update(self, info) -> None:
        """
        Convert to unitsless.
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
            logger.debug(f"Using cached 'BalmerInfo' for '{path}'.")
            
            binfo = cls._cache[str(path)]
            if create_copy:
                return binfo.copy()
            else:
                return binfo

        binfo: BalmerInfo = BalmerInfo()
        if path is None:
            return binfo

        logger.debug(f"Configuring 'BalmerInfo' using '{path}':")
        lines = get_lines_from_file.__wrapped__('BALMER', path, logger)

        for count, line in enumerate(lines, start=1):
            prefix: str = ''
            key: str = line[0].lower()

            match key:
                case 'source':
                    val = parsing.as_str(line[1])

                case 'allow_interp_fitting':
                    val = parsing.as_bool(line[1])

                case 'n_u_min' | 'n_u_max' | 'min_fittable_total' | 'raster_n':
                    val = parsing.as_int(line[1])

                case 'min_fittable_ratio' | 'tau' | 'scale' | 'ratio':
                    val = parsing.as_float(line[1])

                case 'dens' | 'edge' | 'flux' | 'fwhm' | 'temp':
                    prefix: str = '_'
                    val = parsing.as_scalar_or_quantity(line[1:])

                case 'tau_bounds' | 'scale_bounds' | 'ratio_bounds':
                    val = parsing.as_bounds(line[1:])

                case 'flux_bounds' | 'fwhm_bounds' | 'temp_bounds':
                    prefix: str = '_'
                    val = parsing.as_bounds_of_scalars_or_quantity(line[1:])

            binfo[prefix + key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] '{prefix + key}': " \
                f"{val_and_type(val)}"
            )

        BalmerInfo._cache[str(path)] = binfo

        return binfo
    
    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "balmer", logger)