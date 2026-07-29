from functools import partial
from logging import getLogger
from typing import Any

from astropy.io import fits
from astropy.units import Unit
from numpy import diff, float64, full_like, median
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import HDUList_, Quantity_

from quasar_utils.decorators import validate_call
from quasar_utils.loading.loader import _Loader
from quasar_utils.naming import IGR, J2000
from quasar_utils.setup import Info

logger = getLogger(__name__)


@validate_call
def get_from_data(
    key: str,
    hdul: HDUList_,
    info: Info,
) -> Quantity_:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    if key == "dx":
        x = get_from_data.__wrapped__("x", hdul, info)
        return (
            full_like(x.value, median(diff(x.value)), dtype=float64) * x.unit
        )

    vals = info.loading[key]

    hdu = hdul[vals[0]]
    label = hdu.header[vals[1]]
    unit = hdu.header[vals[1]]

    return hdu.data[label].flatten() * Unit(unit)


@validate_call
def get_from_header(
    key: str,
    hdul: HDUList_,
    info: Info,
    default: Any,
) -> Any:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
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
                get_from_data.__wrapped__,
                hdul=hdul,
                info=self.info,
            )
            from_header: partial = partial(
                get_from_header.__wrapped__,
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

            self.x = from_data("x")
            self.y = from_data("y")
            self.dy = from_data("dy")
            self.dx = from_data("dx")

        logger.debug(msg)
        super().__post_init__()
