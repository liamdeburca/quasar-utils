from logging import getLogger
from typing import TypedDict

from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.units import Quantity, Unit
from numpy import float64, full
from quasar_typing.numpy import FloatVector

from ..naming import IGR, J2000
from .loader import _Loader

logger = getLogger(__name__)


class _PAQSDataDict(TypedDict):
    x: Quantity
    y: Quantity
    dy: Quantity
    dx: Quantity
    R: FloatVector

def _get_data(hdul: HDUList) -> _PAQSDataDict:
    """
    Loads PAQS data from a fits HDU list assuming: 

    - The wavelength array is stored in the 'WAVE' column.
    - The flux array is stored in the 'FLUX' column.
    - The flux error array is stored in the 'ERR_FLUX' column.
    - The wavelength unit is: 'angstrom'
    - The flux unit is: 'erg/(s.cm2.angstrom)'
    - The wavelength bin size is constant at 0.25 angstrom.
    """
    wave_unit = Unit("angstrom")
    flux_unit = Unit("erg/(s.cm2.angstrom)")

    data = hdul[1].data

    x = data["WAVE"].flatten().astype(float64)
    y = data["FLUX"].flatten().astype(float64)
    dy = data["ERR_FLUX"].flatten().astype(float64)
    dx = full(x.size, 0.25, dtype=float64)

    # 4MOST: R=4000 @ 3700 angstrom, R=7700 @ 9500 angstrom
    R = 3700.0 * (x - 3700.0) / (9500.0 - 3700.0)+ 4000.0

    return {
        "x": x * wave_unit,
        "y": y * flux_unit,
        "dy": dy * flux_unit,
        "dx": dx * wave_unit,
        "R": R,
    }


class PAQSLoader(_Loader):
    """
    Loader designed for reading PAQS files.
    """
    def __post_init__(self) -> None:
        msg: str = "Initialising loader (PAQS): "

        msg += f"(1) reading data from {self.path}, "
        with fits.open(self.path) as hdul:
            if self.ra is None:
                ra = hdul[0].header.get("RADEG", None)
                if ra is None:
                    ra = 0.0
                    msg += "(2) could not extract RA from header / "
                else:
                    ra = float(ra)
                    msg += f"(2) got RA ({ra:.1f}) from header / "
            else:
                ra = self.ra
                msg += f"(2) got RA from argument ({ra:.1f}) / "

            if self.dec is None:
                dec = hdul[0].header.get("DECDEG", None)
                if dec is None:
                    dec = 0.0
                    msg += "could not extract DEC from header, "
                else:
                    dec = float(dec)
                    msg += f"got DEC ({dec:.1f}) from header, "
            else:
                dec = self.dec
                msg += f"got DEC from argument ({dec:.1f}), "

            msg += f"proceeding with coordinates: (ra={ra:.3f}, "\
                f"(dec={dec:.3f}), "

            name = hdul[0].header.get("OBJ_NAME", None)
            if name is None:
                name = "missing_name"
                msg += "(3) no name in header, "
            else:
                msg += f"(3) extracted name from header ({name}), "

            match self.info.loading.naming.upper():
                case "IGR":
                    self.title = IGR.get_name(
                        ra * Unit("degree"),
                        dec * Unit("degree"),
                    )
                    msg += f"(4) generated IGR title from coordinates: {self.title}"
                case "J2000":
                    self.title = J2000.get_name(
                        ra * Unit("degree"),
                        dec * Unit("degree"),
                    )
                    msg += f"(4) generated J2000 title from coordinates: {self.title}"
                case _:
                    self.title = name
                    msg += "(4) no valid naming convention specified, "

            if self.z is None:
                z = hdul[0].header.get("OBJ_Z", None)
                if z is None:
                    self.z = 0.0
                    msg += "(5) no redshift in header, defaulting to 0.0."
                else:
                    self.z = float(z)
                    msg += f"(5) got redshift from header ({self.z:.3f})."
            else:
                msg += f"(5) got redshift from argument ({self.z:.3f})."

            data = _get_data(hdul)
            self.x_original = data["x"]
            self.y_original = data["y"]
            self.dy_original = data["dy"]
            self.dx_original = data["dx"]
            self.R_original = data["R"]

        logger.debug(msg)
        super().__post_init__()
