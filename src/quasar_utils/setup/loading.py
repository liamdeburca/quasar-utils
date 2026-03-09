from logging import getLogger
from typing import ClassVar, Self
from astropy.units import Unit, Quantity

from pydantic import validate_call
from pydantic.dataclasses import dataclass

from . import parsing
from ..utils._info import _Info
from ..utils.utils import val_and_type
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import CoordBounds
from quasar_typing.pathlib import Path_, AbsoluteFileLike

logger = getLogger(__name__)

@dataclass
class LoadingInfo(_Info):
    loader: str = 'fits'
    x: tuple[int, str, str] = (1, 'ttype1', 'tunit1')
    y: tuple[int, str, str] = (1, 'ttype2', 'tunit2')
    dy: tuple[int, str, str] = (1, 'ttype3', 'tunit3')
    z: float | tuple[int, str] = (0, 'qzc_z')

    name: tuple[int, str] = (0, 'obj_name')
    ra: tuple[int, str] = (0, 'obj_ra')
    dec: tuple[int, str] = (0, 'obj_dec')

    plate: str = 'p'
    fiber: str = 'f'
    mjd: str = 'mjd'

    naming: str = 'IGR'
    rebin: bool = True
    load: bool = True
    conserve: bool = False
    covariance: bool = False
    _sigma_res: float | Quantity_ = 69 * Unit('km/s')
    _x_bounds: CoordBounds | Quantity_ = \
        (1000, 3000) * Unit('angstrom')
    
    sigma_res: float | None = None
    x_bounds: CoordBounds | None = None

    _keys: ClassVar[frozenset[str]] = frozenset([
        'loader', 'x', 'y', 'dy', 'z',
        'name', 'ra', 'dec',
        'plate', 'fiber', 'mjd',
        'naming', 'rebin', 'load', 'conserve', 'covariance', 
        '_sigma_res', '_x_bounds',
        'sigma_res', 'x_bounds',
    ])
    _cache: ClassVar[dict[Path_, Self]] = {}

    def update(self, info) -> None:
        """
        Calculate and assign the dimensionless velocity resolution, sigma_res,
        and the dimensionless rest-wavelength bounds, x_bounds.
        """
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

    @validate_call
    @classmethod
    def from_file(
        cls,
        path: AbsoluteFileLike | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and path in cls._cache.keys():
            logger.debug(f"Using cached 'LoadingInfo' for '{path}'.")
            
            linfo = cls._cache[path]
            if create_copy: return linfo.copy()
            else:           return linfo

        linfo: LoadingInfo = LoadingInfo()
        if path is None:
            return linfo

        logger.debug(f"Configuring 'LoadingInfo' using '{path}'.")        
        lines = get_lines_from_file.__wrapped__(logger, 'LOADING', path)

        for count, line in enumerate(lines, start=1):
            key: str = line[0].lower()

            match key.lower():
                case 'z':
                    val = parsing.as_loading_z(line[1:])

                case 'name' | 'ra' | 'dec' | 'x' | 'y' | 'dy':
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

        cls._cache[path] = linfo

        return linfo