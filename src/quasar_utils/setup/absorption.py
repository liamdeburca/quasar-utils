from logging import getLogger
from typing import Self, ClassVar
from astropy.units import Quantity

from pydantic import validate_call
from pydantic.dataclasses import dataclass

from . import parsing
from ..utils._info import _Info
from ..utils.utils import val_and_type
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.pathlib import Path_, AbsoluteFileLike

logger = getLogger(__name__)

@dataclass
class AbsorptionInfo(_Info):
    p: int = 3
    p_crit: float = 0.01
    z_crit: float = -2
    refine: bool = True
    logspace: bool = True
    
    _w: int | Quantity_ = 25
    _join: int | Quantity_ = 3

    w: int | None = None
    join: int | None = None

    _keys: ClassVar[frozenset[str]] = frozenset([
        '_w', 'p', 'p_crit', 'z_crit', '_join', 'refine', 'logspace',
        'w', 'join',
    ])
    _cache: ClassVar[dict[Path_, Self]] = {}

    def update(self, info) -> None:
        """
        Converts parameters with units into their dimensionless equivalents. 
        """
        sigma_res: float = info.loading['sigma_res']

        logger.debug("Updating 'AbsorptionInfo' class:")

        msg = ">>> [1/2] 'w': "
        if isinstance(old := self['_w'], Quantity):
            self['w'] = new = info.units.getC(old) // sigma_res
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['w'] = int(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)
    
        msg = ">>> [2/2] 'join': "
        if isinstance(old := self['_join'], Quantity):
            self['join'] = new = info.units.getC(old) // sigma_res
            msg += f"{val_and_type(old)} -> {val_and_type(new)}."
        else:
            self['join'] = int(old)
            msg += f"{val_and_type(old)}."
        logger.debug(msg)

        logger.debug("... finished updating 'AbsorptionInfo' class!")
        
    @validate_call
    @classmethod
    def from_file(
        cls, 
        path: AbsoluteFileLike | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and path in cls._cache.keys():
            logger.debug(f"Using cached 'AbsorptionInfo' for '{path}'.")
            
            ainfo = cls._cache[path]
            if create_copy: return ainfo.copy()
            else:           return ainfo

        ainfo: AbsorptionInfo = AbsorptionInfo()
        if path is None: return ainfo
        
        logger.debug(f"Configuring 'AbsorptionInfo' using '{path}'.")
        lines = get_lines_from_file.__wrapped__(logger, 'ABSORPTION', path)

        for count, line in enumerate(lines, start=1):
            key = line[0].lower()

            match key:
                case 'w' | 'join':
                    key = '_' + key
                    val = parsing.as_scalar_or_quantity(line[1:])
                case 'p':
                    val = parsing.as_int(line[1])
                case 'refine' | 'logspace':
                    val = parsing.as_bool(line[1])
                case 'z_crit' | 'p_crit':
                    val = parsing.as_float(line[1])

            ainfo[key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] '{key}': {val_and_type(val)}."
            )

        cls._cache[path] = ainfo

        return ainfo