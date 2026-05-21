__all__ = ['ASCIILoader']

from typing import Any
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from pandas import DataFrame
from numpy import stack, median, diff, full_like, float64
from astropy.units import Unit
from functools import partial

from quasar_typing.astropy import Quantity_
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.pandas import DataFrame_

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
        case 'dx':
            unit = Unit("angstrom")
            col_name = 'restwl'

            _x = df[col_name].to_numpy()
            return full_like(_x, median(diff(_x)), dtype=float64) * unit

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

@dataclass
class ASCIILoader(_Loader):
    """
    Loader designed for reading ASCII files.
    """
    def __post_init__(self):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        msg: str = "Initialising loader (ASCII): "
        msg += f"reading data from {self.path}, "

        from_fname = partial(
            get_from_fname.__wrapped__, 
            fname=self.path.name, 
            default=0,
        )
        from_data = partial(
            get_from_data.__wrapped__, 
            df=read_ascii(self.path, skip=1),
        )
        plate: int = int(from_fname('p'))
        fiber: int = int(from_fname('f'))
        mjd: int = int(from_fname('m'))

        msg += f"extracted metadata from filename ({plate=}, {fiber=}, {mjd=}), "

        self.title = SDSS.get_name(plate, fiber, mjd)
        msg += f"generated title from metadata: {self.title}, "

        if self.z != 0.0:
            msg += f"got redshift from argument ({self.z:.3f})."  
        else:
            self.z = 0.0
            msg += f"using default redshift of z={self.z:.3f}."

        logger.debug(msg)

        self.x = from_data('x')
        self.y = from_data('y')
        self.dy = from_data('dy')
        self.dx = from_data('dx')

        super().__post_init__()