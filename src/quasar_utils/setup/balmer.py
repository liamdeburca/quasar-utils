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
from quasar_typing.misc import BalmerModelParams

logger = getLogger(__name__)

@dataclass
class BalmerInfo(_Info):
    fit: bool = True

    _windows: list[CoordBounds] | Quantity_ = [[3000, 4500]] * Unit('angstrom')
    
    _edge: float | Quantity_ = 3646 * Unit('angstrom')
    _fwhm_norm: float | Quantity_ = 5_000 * Unit('km/s')
    
    source: Literal['SH1995'] = 'SH1995'
    _temp: float | Quantity_ = 15_000 * Unit('K')
    _dens: float | Quantity_ = 1e9 * Unit('cm^-3')
    n_u_min: int = 7
    n_u_max: int = 50
    tau: float = 1.0
    scale: float = 3.0
    
    _flux: float | Quantity_ = 1e-17 * Unit('erg/(s.cm2.angstrom)')
    _fwhm: float | Quantity_ = 5_000 * Unit('km/s')
    _flux_bounds: AstropyBounds | Quantity_ = [1e-18, 1e-15] * Unit('erg/(s.cm2.angstrom)')
    _fwhm_bounds: AstropyBounds | Quantity_ = [1000, 20_000] * Unit('km/s')
    ratio: float = 0.3
    ratio_bounds: AstropyBounds = (0.5, 2.0)

    _fixed: BalmerModelParams = field(default_factory=lambda: BalmerModelParams({'ratio'}))
    raster_n: int = 20
    min_fittable_ratio: float = 0.6
    min_fittable_total: int = 100

    raster: bool = True
    fine_tune: bool = True

    windows: list[CoordBounds] | None = field(default=None, init=False)
    edge: float | None = field(default=None, init=False)
    fwhm_norm: float | None = field(default=None, init=False)
    temp: float | None = field(default=None, init=False)
    dens: float | None = field(default=None, init=False)
    flux: float | None = field(default=None, init=False)
    fwhm: float | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)
    fwhm_bounds: AstropyBounds | None = field(default=None, init=False)
    fixed: dict[str, bool] | None = field(default=None, init=False)


    _keys: ClassVar[frozenset[str]] = frozenset([
        'fit',
        '_windows', 'windows',
        '_edge', 'edge',
        '_fwhm_norm', 'fwhm_norm',
        'source',
        '_temp', 'temp',
        '_dens', 'dens',
        'n_u_min', 'n_u_max',
        'tau', 'scale', 
        '_flux', 'flux',
        '_fwhm', 'fwhm',
        'ratio',
        '_flux_bounds', 'flux_bounds',
        '_fwhm_bounds', 'fwhm_bounds',
        'ratio_bounds',
        '_fixed', 'fixed',
        'raster_n',
        'min_fittable_ratio', 'min_fittable_total',
        'raster', 'fine_tune',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'windows': "to_wavelength_windows",
        'edge': "to_wavelength", 
        'fwhm_norm': "to_velocity",
        'temp': "to_temperature",
        'dens': "to_density", 
        'flux': "to_flux", 
        'fwhm': "to_velocity", 
        'flux_bounds': "to_flux_bounds", 
        'fwhm_bounds': "to_velocity_bounds", 
        'fixed': "to_fixed",
    }

    def __hash__(self) -> int:
        return super().__hash__()

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