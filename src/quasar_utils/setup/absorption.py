from logging import getLogger
from typing import Self, ClassVar
from dataclasses import field
from pydantic.dataclasses import dataclass

from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Quantity_
from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

@dataclass
class AbsorptionInfo(_Info):
    fit: bool = True

    p: int = 3
    p_crit: float = 0.01
    z_crit: float = -2
    refine: bool = True
    logspace: bool = True
    _w: int | Quantity_ = 25
    _join: int | Quantity_ = 3

    w: int | None = field(default=None, init=False)
    join: int | None = field(default=None, init=False)

    _keys: ClassVar[frozenset[str]] = frozenset([
        'fit', '_w', 'p', 'p_crit', 'z_crit', '_join', 'refine', 'logspace',
        'w', 'join',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'w': "to_n_pixels", 
        'join': "to_n_pixels",
    }

    def update(self, info) -> None:
        """
        Converts parameters with units into their dimensionless equivalents. 
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
            logger.debug(f"Using cached 'AbsorptionInfo' for '{path}'.")
            
            ainfo = cls._cache[str(path)]
            if create_copy: return ainfo.copy()
            else:           return ainfo

        ainfo: AbsorptionInfo = AbsorptionInfo()
        if path is None: return ainfo
        
        logger.debug(f"Configuring 'AbsorptionInfo' using '{path}'.")
        lines = get_lines_from_file.__wrapped__('ABSORPTION', path, logger)

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

        AbsorptionInfo._cache[str(path)] = ainfo

        return ainfo
    
    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "absorption", logger)