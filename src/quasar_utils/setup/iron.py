from logging import getLogger
from typing import ClassVar, Self
from pathlib import Path
from astropy.units import Unit, Quantity
from numpy import arange
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.numpy import SortedFloatVector, FittableFloatVector
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

DEFAULT_WINDOWS: Quantity = [[2050, 2700], [3000, 3500]] * Unit('angstrom')
DEFAULT_TEMPLATE_FILES: list[str] = ['vw_2001.fits', 'bw.fits', 'v_2003.fits']
DEFAULT_FWHM: Quantity = arange(1_000, 20_000+1, 250) * Unit('km/s')
DEFAULT_FLUX_BOUNDS: Quantity = [1e-17, 1e-14] * Unit('erg/(s.cm2.angstrom)')
DEFAULT_FWHM_BOUNDS: Quantity = [1_000, 20_000] * Unit('km/s')
DEFAULT_SPLIT: Quantity = [0, 0, 0] * Unit('angstrom')
DEFAULT_BIAS: list[str] = ['right', 'right', 'right']
DEFAULT_RATIO: list[float] = [1.0, 1.0, 1.0]
DEFAULT_FIXED: list[bool] = [True, True, True]

@dataclass
class IronInfo(_Info):
    fit: bool = True
    
    _windows: list[list] | Quantity_ = field(default_factory=lambda: DEFAULT_WINDOWS)
    template_files: list[str] = field(default_factory=lambda: DEFAULT_TEMPLATE_FILES)
    resample: bool = True
    _fwhm: list[float] | Quantity_ = field(default_factory=lambda: DEFAULT_FWHM)
    _flux_bounds: AstropyBounds | Quantity_ = field(default_factory=lambda: DEFAULT_FLUX_BOUNDS)
    _fwhm_bounds: AstropyBounds | Quantity_ = field(default_factory=lambda: DEFAULT_FWHM_BOUNDS)
    _split: list[float] | Quantity_ = field(default_factory=lambda: DEFAULT_SPLIT)
    bias: list[str] = field(default_factory=lambda: DEFAULT_BIAS)
    ratio: list[float] = field(default_factory=lambda: DEFAULT_RATIO)
    fixed: list[bool] = field(default_factory=lambda: DEFAULT_FIXED)
    _scale: float | Quantity_ = 140.0
    raster: bool = True
    fine_tune: bool = False
    allow_interp_fitting: bool = True

    windows: list[CoordBounds] | None = field(default=None, init=False)
    fwhm: SortedFloatVector | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)
    fwhm_bounds: AstropyBounds | None = field(default=None, init=False)
    split: FittableFloatVector | None = field(default=None, init=False)
    scale: float | None = field(default=None, init=False)

    _keys: ClassVar[frozenset[str]] = frozenset([
        'fit',
        'windows', '_windows',
        'template_files', 'resample', 
        'fwhm', '_fwhm', 
        'flux_bounds', '_flux_bounds', 
        'fwhm_bounds', '_fwhm_bounds',
        'split', '_split', 
        'bias', 'ratio', 'fixed', 
        '_scale', 'scale', 
        'raster', 'fine_tune',
        'allow_interp_fitting',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'windows': "to_wavelength_windows",
        'fwhm': "to_velocity",
        'flux_bounds': "to_flux_bounds",
        'fwhm_bounds': "to_velocity_bounds",
        'split': "to_wavelength_list",
        'scale': "to_scale",
    }

    def update(self, info) -> None:
        super().update(info, logger)

    @classmethod
    @validate_call
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache.keys():
            logger.debug(f"Using cached 'IronInfo' for '{path}'.")
            
            iinfo = cls._cache[str(path)]
            if create_copy: 
                return iinfo.copy()
            return iinfo
        
        iinfo: IronInfo = IronInfo()
        if path is None:
            return iinfo
                
        logger.debug(f"Configuring 'IronInfo' using '{path}'.")
        lines = get_lines_from_file.__wrapped__('IRON', path, logger)

        for count, line in enumerate(lines, start=1):
            prefix = ''
            key = line[0].lower()

            match key:
                case 'windows':
                    prefix = '_'
                    val = parsing.as_pairs_of_floats_or_quantity(line[1:])
                case 'bias':
                    val = parsing.as_list(line[1])
                case 'template_files':
                    val = list(map(Path, parsing.as_list(line[1])))
                case 'resample' | 'fine_tune' | 'allow_interp_fitting':
                    val = parsing.as_bool(line[1])
                case 'fwhm' | 'split':
                    prefix = '_'
                    val = (
                        parsing.as_range_of_quantities(line[1:]) \
                        if line[1].lower().startswith('range:') \
                        else parsing.as_list_of_scalars_or_quantity(line[1:])
                    )
                case 'ratio':
                    val = parsing.as_array_of_floats(line[1])
                case 'fixed':
                    val = parsing.as_array_of_bools(line[1])
                case 'flux_bounds':
                    prefix = '_'
                    val = parsing.as_bounds_of_scalars_or_quantity(line[1:])
                case 'scale':
                    prefix = '_'
                    val = parsing.as_scalar_or_quantity(line[1:])

            iinfo[prefix + key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] 'key': {val_and_type(val)}"
            )

        cls._cache[str(path)] = iinfo

        return iinfo
    
    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "iron", logger)