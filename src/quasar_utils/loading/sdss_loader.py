__all__ = ["SDSSLoader"]

from logging import getLogger

from astropy.constants import c
from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.units import Quantity, Unit
from numpy import float64, full_like, nan, ones
from pydantic.dataclasses import dataclass
from quasar_typing.numpy import FloatMatrix

from quasar_utils.loading.loader import _Loader
from quasar_utils.naming import IGR, J2000

logger = getLogger(__name__)
SIGMA_RES: float = 69.0 / c.to("km/s").value


def _get_data(hdul: HDUList) -> dict[str, Quantity | FloatMatrix]:
    """Return spectral arrays extracted from an SDSS HDUList.

    Parameters
    ----------
    hdul : HDUList_
        Opened FITS HDU list from an SDSS spectrum file.

    Raises
    ------
    ValidationError
        Pydantic type validation
    """
    x_unit = Unit("angstrom")
    flux_unit = Unit("1e-17 erg/(s.cm2.angstrom)")

    data = hdul[1].data

    x = 10 ** data["loglam"].astype(float64)
    dx = x * SIGMA_RES
    y = data["flux"].astype(float64)

    dy = full_like(x, nan, dtype=float64)
    mask = data["ivar"] > 0
    dy[mask] = (1 / data["ivar"][mask] ** 0.5).astype(float64)

    invalid_mask = data["and_mask"].astype(bool)
    y[invalid_mask] = nan
    dy[invalid_mask] = nan

    # No correction for instrumental resolution
    res_kernels = ones((1, x.size), dtype=float64)

    return {
        "x": x * x_unit,
        "y": y * flux_unit,
        "dy": dy * flux_unit,
        "dx": dx * x_unit,
        "res_kernels": res_kernels,
    }


@dataclass
class SDSSLoader(_Loader):
    """Loader for SDSS spectrum FITS files.

    This dataclass is specialised to read spectra produced by
    ``astroquery.sdss``. When RA/DEC, title or redshift are not
    provided the corresponding values are extracted from the FITS
    primary header. Spectral arrays are read from the first
    extension using :func:`get_data`.

    Parameters
    ----------
    path : str | AbsoluteFilePath
        Path to the SDSS FITS file.
    info : Info, optional
        Project configuration and unit helpers.
    z : float | None, optional
        Optional redshift override.
    title : str, optional
        Optional title override.
    ra : float | None, optional
        Right ascension in degrees.
    dec : float | None, optional
        Declination in degrees.
    """

    def __post_init__(self):
        """Post-initialise the SDSS loader and extract metadata.

        This opens the FITS file at ``self.path`` and, when not
        provided, extracts RA, DEC, object name and redshift from the
        primary header. The instance title is generated according to
        ``self.info.loading.naming`` when possible. Finally the
        spectral arrays are read via :func:`get_data` and assigned to
        ``self.x, self.y, self.dy, self.dx``.
        """

        msg: str = "Initialising loader (SDSS): "

        msg += f"(1) reading data from {self.path}, "
        with fits.open(self.path) as hdul:
            if self.ra is None:
                self.ra = float(hdul[0].header["RADEG"])
                msg += "(2) extracted 'ra' from header / "
            else:
                msg += "(2) got 'ra' from argument / "

            if self.dec is None:
                self.dec = float(hdul[0].header["DECDEG"])
                msg += "extracted 'dec' from header / "
            else:
                msg += "got 'dec' from argument / "

            msg += f"proceeding with coordinates: (ra={self.ra:.3f}, "\
                f"(dec={self.dec:.3f}), "

            if "OBJ_NAME" in hdul[0].header:
                name = hdul[0].header["OBJ_NAME"]
                msg += f"(3) extracted name from header ({name}), "
            else:
                name = "missing_name"
                msg += "(3) no name in header, "

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

            if self.z is None:
                try:
                    self.z = float(hdul[0].header["OBJ_Z"])
                    msg += f"(5) extracted redshift from header: {self.z:.3f}."
                except KeyError:
                    self.z = 0.0
                    msg += "(5) no redshift found in header, defaulting to 0.0."
            else:
                msg += f"(5) got redshift from argument: {self.z:.3f}."

            data = _get_data(hdul)
            self.x_original = data["x"]
            self.y_original = data["y"]
            self.dy_original = data["dy"]
            self.dx_original = data["dx"]
            self.res_kernels_original = data["res_kernels"]

        logger.debug(msg)
        super().__post_init__()
