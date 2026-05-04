__all__ = ['ASCIILoader']

from typing import Any
from pydantic import validate_call
from pandas import DataFrame
from numpy import stack
from astropy.units import Unit
from functools import partial

from quasar_typing.astropy import Quantity_
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.pandas import DataFrame_

from ..setup import Info
from .loader import _Loader
from ..naming import SDSS

from logging import getLogger
logger = getLogger(__name__)

@validate_call
def read_ascii(
    path: AbsoluteFilePath, 
    skip: int = 0,
) -> DataFrame_:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    with open(path, 'r') as f:
        all_lines = [line.strip().split() for line in f.readlines()[skip:]]

    col_names = all_lines[0]
    data = stack(
        [list(map(float, line)) for line in all_lines[1:]],
        axis=1,
    )
    return DataFrame({col: arr for col, arr in zip(col_names, data)})

@validate_call
def get_from_data(key: str, df: DataFrame_) -> float | Quantity_:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    match key:
        case 'x':
            unit = Unit("angstrom")
            col_name = 'restwl'
        case 'y':
            unit = Unit("1e-17 erg/(s.cm2.angstrom)")
            col_name = 'dredflux'
        case 'dy':
            unit = Unit("1e-17 erg/(s.cm2.angstrom)")
            col_name = 'err'

    return df[col_name].to_numpy() * unit

@validate_call
def get_from_fname(
    key: str,
    fname: str,
    default: Any,
) -> Any:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    return {
        s[0]: s[1:] for s in fname.split('_') if s[1:].isnumeric()
    }.get(key, default)

class ASCIILoader(_Loader):
    """
    Loader designed for reading ASCII files.
    """
    @validate_call
    def __init__(
        self,
        path: AbsoluteFilePath,
        *,
        info: Info = None,
        z: float | None = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        msg: str = "Initialising loader (ASCII): "
        msg += f"reading data from {path}, "

        from_fname = partial(
            get_from_fname.__wrapped__, 
            fname=path.name, 
            default=0,
        )
        from_data = partial(
            get_from_data.__wrapped__, 
            df=read_ascii(path, skip=1),
        )
        plate: int = int(from_fname('p'))
        fiber: int = int(from_fname('f'))
        mjd: int = int(from_fname('m'))

        msg += f"extracted metadata from filename ({plate=}, {fiber=}, {mjd=}), "

        title: str = SDSS.get_name(plate, fiber, mjd)
        msg += f"generated title from metadata ({title}), "

        if z is not None:
            msg += f"got redshift from argument ({z:.3f})."  
        elif isinstance(val := info.loading['z'], float):
            z = val
            msg += f"got redshift from Info instance ({z:.3f})."
        else:
            z = 0
            msg += f"using default redshift of z={z:.3f}."

        logger.debug(msg)

        super().__init__(
            from_data('x'),
            from_data('y'),
            from_data('dy'),
            z=z,
            title=title,
            path=path,
            info=info,
        )