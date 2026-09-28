__all__ = ["ASCIILoader"]

from functools import partial
from logging import getLogger
from typing import Any

from astropy.units import Quantity, Unit
from numpy import diff, float64, full_like, median, ones, stack
from pandas import DataFrame
from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import AbsoluteFilePath

from ..naming import SDSS
from .loader import _Loader

logger = getLogger(__name__)


def _read_ascii(path: AbsoluteFilePath, skip: int = 0) -> DataFrame:
    with open(path, "r") as f:
        all_lines = [line.strip().split() for line in f.readlines()[skip:]]

    col_names = all_lines[0]
    data = stack(
        [list(map(float, line)) for line in all_lines[1:]],
        axis=1,
    )
    return DataFrame({col: arr for col, arr in zip(col_names, data)})


def _get_from_data(key: str, df: DataFrame) -> float | Quantity:
    match key:
        case "x":
            unit = Unit("angstrom")
            col_name = "restwl"
        case "y":
            unit = Unit("1e-17 erg/(s.cm2.angstrom)")
            col_name = "dredflux"
        case "dy":
            unit = Unit("1e-17 erg/(s.cm2.angstrom)")
            col_name = "err"
        case "dx":
            unit = Unit("angstrom")
            col_name = "restwl"

            _x = df[col_name].to_numpy()
            return full_like(_x, median(diff(_x)), dtype=float64) * unit

    return df[col_name].to_numpy() * unit


def _get_from_fname(key: str, fname: str, default: Any) -> Any:
    return {s[0]: s[1:] for s in fname.split("_") if s[1:].isnumeric()}.get(
        key, default
    )


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
            _get_from_fname,
            fname=self.path.name,
            default=0,
        )
        from_data = partial(
            _get_from_data,
            df=_read_ascii(self.path, skip=1),
        )
        plate: int = int(from_fname("p"))
        fiber: int = int(from_fname("f"))
        mjd: int = int(from_fname("m"))

        msg += (
            f"extracted metadata from filename ({plate=}, {fiber=}, {mjd=}), "
        )

        self.title = SDSS.get_name(plate, fiber, mjd)
        msg += f"generated title from metadata: {self.title}, "

        if self.z != 0.0:
            msg += f"got redshift from argument ({self.z:.3f})."
        else:
            self.z = 0.0
            msg += f"using default redshift of z={self.z:.3f}."

        logger.debug(msg)

        self.x_original = from_data("x")
        self.y_original = from_data("y")
        self.dy_original = from_data("dy")
        self.dx_original = from_data("dx")
        self.res_kernels_original = ones(
            (1, self.x_original.size), 
            dtype=float64,
        )

        super().__post_init__()
