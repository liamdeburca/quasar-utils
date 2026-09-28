from functools import partial
from logging import getLogger
from typing import Any

from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.units import Quantity, Unit
from numpy import diff, float64, full_like, median, ones
from pydantic.dataclasses import dataclass

from ..naming import IGR, J2000
from ..setup import Info
from .loader import _Loader

logger = getLogger(__name__)


def _get_from_data(
    key: str,
    hdul: HDUList,
    info: Info,
) -> Quantity:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    if key == "dx":
        x = _get_from_data("x", hdul, info)
        return (
            full_like(x.value, median(diff(x.value)), dtype=float64) * x.unit
        )

    vals = info.loading[key]

    hdu = hdul[vals[0]]
    label = hdu.header[vals[1]]
    unit = hdu.header[vals[1]]

    return hdu.data[label].flatten() * Unit(unit)


def _get_from_header(
    key: str,
    hdul: HDUList,
    info: Info,
    default: Any,
) -> Any:
    ext, label = info.loading[key]
    return hdul[ext].header.get(label, default)


@dataclass
class FITSLoader(_Loader):
    """
    Loader designed for reading FITS (.fits) files.
    """

    def __post_init__(self):
        msg: str = "Initialising loader (FITS): "

        msg += f"(1) reading data from {self.path}, "
        with fits.open(self.path) as hdul:
            from_data: partial = partial(
                _get_from_data,
                hdul=hdul,
                info=self.info,
            )
            from_header: partial = partial(
                _get_from_header,
                hdul=hdul,
                info=self.info,
            )

            self.ra = float(from_header("ra", 0))
            self.dec = float(from_header("dec", 0))

            msg += f"(2) extracted coordinates from header (ra={self.ra}, dec={self.dec}), "

            name = from_header("name", "missing_name")
            if name == "missing_name":
                msg += "(3) no name in header, "
            else:
                msg += f"(3) extracted name from header ({name}), "

            match self.info.loading.naming.upper():
                case "IGR":
                    self.title = IGR.get_name(
                        self.ra * Unit("degree"),
                        self.dec * Unit("degree"),
                    )
                    msg += f"(4) generated IGR title from coordinates: {self.title}"
                case "J2000":
                    self.title = J2000.get_name(
                        self.ra * Unit("degree"),
                        self.dec * Unit("degree"),
                    )
                    msg += f"(4) generated J2000 title from coordinates: {self.title}"
                case _:
                    self.title = name
                    msg += "(4) no valid naming convention specified, "

            if self.z != 0.0:
                msg += f"(5) got redshift from argument ({self.z:.3f})."
            else:
                self.z = float(from_header("z", 0))
                msg += f"(5) got redshift from header ({self.z:.3f})."

            self.x_original = from_data("x")
            self.y_original = from_data("y")
            self.dy_original = from_data("dy")
            self.dx_original = from_data("dx")
            self.res_kernels_original = ones(
                (1, self.x_original.size), 
                dtype=float64,
            )

        logger.debug(msg)
        super().__post_init__()
