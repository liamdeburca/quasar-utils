from logging import getLogger
logger = getLogger("quasar_utils.setup.iron")

from typing import ClassVar, Self
from pathlib import Path
from astropy.units import Unit, Quantity
from numpy import arange, stack, array

from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.numpy import FittableFloatVector
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds
from quasar_typing.pathlib import AbsoluteFilePath, AbsoluteFITSPath


class IronInfo(_Info):
    _keys: ClassVar[frozenset[str]] = frozenset([
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
    
    @validate_call(validate_return=False)
    def __init__(
        self,
        _windows: list[list] | Quantity_ = [
            [2050, 2700],
            [3000, 3500],
        ] * Unit('angstrom'),
        template_files: list[str] = ['vw_2001.fits', 'bw.fits', 'v_2003.fits'],
        resample: bool = True,
        _fwhm: list[float] | Quantity_ = arange(1_000, 20_000+1, 250) * Unit('km/s'),
        _flux_bounds: AstropyBounds | Quantity_ = [1e-17, 1e-14] * Unit('erg/(s.cm2.angstrom)'),
        _fwhm_bounds: AstropyBounds | Quantity_ = [1_000, 20_000] * Unit('km/s'),
        _split: list[float] | Quantity_ = [0, 0, 0] * Unit('angstrom'),
        bias: list[str] = ['right', 'right', 'right'],
        ratio: list[float] = [1.0, 1.0, 1.0],
        fixed: list[bool] = [True, True, True],
        _scale: float | Quantity_ = 140.0,
        raster: bool = True,
        fine_tune: bool = False,
        allow_interp_fitting: bool = True,
    ):
        self._windows: list[list] | Quantity_ = _windows
        self.template_files: list[str] = template_files
        self.resample: bool = resample
        self._fwhm: list[float] | Quantity_ = _fwhm
        self._flux_bounds: AstropyBounds | Quantity_ = _flux_bounds
        self._fwhm_bounds: AstropyBounds | Quantity_ = _fwhm_bounds
        self._split: list[float] | Quantity_ = _split
        self.bias: list[str] = bias
        self.ratio: list[float] = ratio
        self.fixed: list[bool] = fixed
        self._scale: float | Quantity_ = _scale
        self.raster: bool = raster
        self.fine_tune: bool = fine_tune
        self.allow_interp_fitting: bool = allow_interp_fitting

        # Set by 'update' method
        self.windows: list[tuple[float, float]] | None = None
        self.fwhm: FittableFloatVector | None = None
        self.flux_bounds: AstropyBounds | None = None
        self.fwhm_bounds: AstropyBounds | None = None
        self.split: FittableFloatVector | None = None
        self.scale: float | None = None

    def update(self, info) -> None:
        super().update(info, logger)
        return

        logger.debug("Updating 'IronInfo' class...")

        msg = ">>> [1/6] 'windows': "
        if isinstance(old := self['_windows'], Quantity):
            self['windows'] = new = list(
                map(tuple, info.units.getWavelength(old))
            )
            msg = f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['windows'] = stack(old, axis=0)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [2/6] 'fwhm': "
        if not isinstance(old := self['_fwhm'], Quantity):
            old *= info.units['velocity_unit']
        self['fwhm'] = new = info.units.getC(old)
        msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        logger.debug(msg)

        msg = ">>> [3/6] 'flux_bounds': "
        if isinstance(old := self['_flux_bounds'], Quantity):
            self['flux_bounds'] = new = tuple(info.units.getFlux(old))
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['flux_bounds'] = tuple(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [4/6] 'fwhm_bounds': "
        if isinstance(old := self['_fwhm_bounds'], Quantity):
            self['fwhm_bounds'] = new = tuple(info.units.getC(old))
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['fwhm_bounds'] = tuple(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [5/6] 'split': "
        if isinstance(old := self['_split'], Quantity):
            self['split'] = new = info.units.getWavelength(old)
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['split'] = array(old, dtype=float)
            msg = f"{val_and_type(old)}."
        logger.debug(msg)

        msg = ">>> [6/6] 'scale': "
        if not isinstance(old := self['_scale'], Quantity):
            old *= info.units['velocity_unit']
        self['scale'] = new = info.units.getC(old) / info.loading['sigma_res']
        msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        logger.debug(msg)

    @classmethod
    @validate_call(validate_return=False)
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache.keys():
            logger.debug(f"Using cached 'IronInfo' for '{path}'.")
            
            iinfo = cls._cache[str(path)]
            if create_copy: return iinfo.copy()
            else:           return iinfo
        
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
    @validate_call(validate_return=False)
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "iron", logger)