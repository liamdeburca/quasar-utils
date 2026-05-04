from logging import getLogger
from typing import ClassVar, Self, Literal, Any
from astropy.units import Unit
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import CoordBounds
from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

DEFAULT_VALUES: dict[str, Any] = {
    'loader': 'fits',
    'naming': 'IGR',
    'deredden': (True, 'sfd', 'ccm89', 3.1),
    'rebin': True,
    'load': True,
    'conserve': False,
    'covariance': False,
    '_sigma_res': 69 * Unit('km/s'),
    '_x_bounds': (1000, 3000) * Unit('angstrom'),
}

@dataclass
class LoadingInfo(_Info):
    loader: str = field(default=DEFAULT_VALUES['loader'])
    naming: str = field(default=DEFAULT_VALUES['naming'])
    deredden: tuple[bool, Literal['sfd', 'csfd'], Literal['ccm89', 'o94'], float] = field(default=DEFAULT_VALUES['deredden'])
    rebin: bool = field(default=DEFAULT_VALUES['rebin'])
    load: bool = field(default=DEFAULT_VALUES['load'])
    conserve: bool = field(default=DEFAULT_VALUES['conserve'])
    covariance: bool = field(default=DEFAULT_VALUES['covariance'])
    _sigma_res: float | Quantity_ = field(default=DEFAULT_VALUES['_sigma_res'])
    _x_bounds: CoordBounds | Quantity_ = field(default=DEFAULT_VALUES['_x_bounds'])

    sigma_res: float | None = field(default=None, init=False)
    x_bounds: CoordBounds | None = field(default=None, init=False)

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

    @validate_call
    def __init__(
        self,
        loader: str = 'fits',
        naming: str = 'IGR',
        deredden: tuple[bool, Literal['sfd', 'csfd'], Literal['ccm89', 'o94'], float] = (True, 'sfd', 'ccm89', 3.1),
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
        self.deredden: tuple[bool, Literal['sfd', 'csfd'], Literal['ccm89', 'o94'], float] = deredden
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

    @classmethod
    @validate_call
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
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "loading", logger)