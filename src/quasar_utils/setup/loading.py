from logging import getLogger
logger = getLogger("quasar_utils.setup.loading")

from typing import ClassVar, Self, Literal
from json import load as load_json
from pathlib import Path
from astropy.units import Unit, Quantity

from pydantic import validate_call

from .utils._info import _Info
from .utils.json_field import JSONField
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import CoordBounds
from quasar_typing.pathlib import Path_, AbsoluteFilePath


class LoadingInfo(_Info):
    _keys: ClassVar[frozenset[str]] = frozenset([
        'loader',
        'naming', 'deredden', 'rebin', 'load', 'conserve', 'covariance', 
        '_sigma_res', '_x_bounds',
        'sigma_res', 'x_bounds',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'sigma_res': "to_velocity", 
        'x_bounds': "to_wavelength_bounds",
    }

    @validate_call(validate_return=False)
    def __init__(
        self,
        loader: str = 'fits',
        naming: str = 'IGR',
        deredden: tuple[bool, Literal['sfd', 'csfd'] | None, Literal['ccm89', 'o94'] | None] = (True, 'sfd', 'ccm89'),
        rebin: bool = True,
        load: bool = True,
        conserve: bool = False,
        covariance: bool = False,
        _sigma_res: float | Quantity_ = 69 * Unit('km/s'),
        _x_bounds: CoordBounds | Quantity_ = (1000, 3000) * Unit('angstrom'),
        sigma_res: float | None = None,
        x_bounds: CoordBounds | None = None,
    ):
        self.loader: str = loader
        self.naming: str = naming
        self.deredden: tuple[bool, Literal['sfd', 'csfd'] | None, Literal['ccm89', 'o94'] | None] = deredden
        self.rebin: bool = rebin
        self.load: bool = load
        self.conserve: bool = conserve
        self.covariance: bool = covariance
        self._sigma_res: float | Quantity_ = _sigma_res
        self._x_bounds: CoordBounds | Quantity_ = _x_bounds
        self.sigma_res: float | None = sigma_res
        self.x_bounds: CoordBounds | None = x_bounds

    def update(self, info) -> None:
        """
        Calculate and assign the dimensionless velocity resolution, sigma_res,
        and the dimensionless rest-wavelength bounds, x_bounds.
        """
        super().update(info, logger)
        return 
        logger.debug("Updating 'LoadingInfo' class:")

        # Calculate velocity resolution (in units of the speed of light)
        msg = ">>> [1/2] 'sigma_res': "
        if not isinstance(old := self['_sigma_res'], Quantity):
            old *= info.units['velocity_unit']
        self['sigma_res'] = new = info.units.getC(old)
        msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        logger.debug(msg)

        msg = ">>> [2/2] 'x_bounds': "
        if isinstance(old := self['_x_bounds'], Quantity):
            self['x_bounds'] = new = tuple(info.units.getWavelength(old))
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['x_bounds'] = tuple(old)
            msg += f"{val_and_type(old)},"
        logger.debug(msg)

        logger.debug("... finished updating 'LoadingInfo' class!")

    @classmethod
    @validate_call(validate_return=False)
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache.keys():
            logger.debug(f"Using cached 'LoadingInfo' for '{path}'.")
            
            linfo = cls._cache[str(path)]
            if create_copy: return linfo.copy()
            else:           return linfo

        linfo: LoadingInfo = LoadingInfo()
        if path is None:
            return linfo

        logger.debug(f"Configuring 'LoadingInfo' using '{path}'.")        
        lines = get_lines_from_file.__wrapped__('LOADING', path, logger)

        for count, line in enumerate(lines, start=1):
            key: str = line[0].lower()

            match key.lower():
                case 'z':
                    val = parsing.as_loading_z(line[1:])

                case 'name' | 'ra' | 'dec':
                    val = parsing.as_loading_tuple(line[1:])
                
                case 'x' | 'y' | 'dy':
                    val = tuple(line[1:])

                case 'naming':
                    val = line[1].upper()

                case 'conserve' | 'covariance' | 'load' | 'rebin':
                    val = parsing.as_bool(line[1])

                case 'sigma_res':
                    key = '_' + key
                    val = parsing.as_scalar_or_quantity(line[1:])

                case 'x_bounds':
                    key = '_' + key
                    val = parsing.as_bounds_of_scalars_or_quantity(line[1:])
                    
                case _:
                    val = line[1]

            linfo[key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] '{key}': {val_and_type(val)}."
            )

        cls._cache[str(path)] = linfo

        return linfo
    
    @classmethod
    @validate_call(validate_return=False)
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "loading", logger)