__all__ = ["SDSSLoader"]

from logging import getLogger
from typing import Any, Self, TypedDict

import numpy as np
from astropy.constants import c
from astropy.io import fits
from astropy.units import Quantity, Unit
from pydantic import validate_call
from quasar_typing.pathlib import FITSPath

from quasar_utils.setup import Info

from .loader import _Loader
from .metadata import SDSSMeta

logger = getLogger(__name__)
SIGMA_RES: float = 69.0 / c.to("km/s").value


class _SDSSDataDict(TypedDict):
    x_original: Quantity
    y_original: Quantity
    dy_original: Quantity
    dx_original: Quantity
    R_original: None


def _get_data(hdul: fits.HDUList) -> _SDSSDataDict:
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

    x = 10 ** data["loglam"].astype(np.float64)
    dx = x * SIGMA_RES
    y = data["flux"].astype(np.float64)

    dy = np.full_like(x, np.nan, dtype=np.float64)
    mask = data["ivar"] > 0
    dy[mask] = (1 / data["ivar"][mask] ** 0.5).astype(np.float64)

    invalid_mask = data["and_mask"].astype(bool)
    y[invalid_mask] = np.nan
    dy[invalid_mask] = np.nan

    return {
        "x_original": x * x_unit,
        "y_original": y * flux_unit,
        "dy_original": dy * flux_unit,
        "dx_original": dx * x_unit,
        "R_original": None, # No instrumental correction
    }


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
    @classmethod
    @validate_call
    def fromPath(
        cls,
        path: FITSPath,
        info: Info | None = None,
        z: float | None = None,
        title: str | None = None,
        ra: float | None = None,
        dec: float | None = None,
    ) -> Self:
        from quasar_utils.naming import IGR, J2000

        info = info or Info()
        kwargs = {"path": path, "info": info}

        logger.debug("Creating SDSSLoader from path: %s", path)
        with fits.open(path) as hdul:
            if ra is None:
                ra = float(hdul[0].header["RADEG"])
                logger.debug("Extracted RA from header: %s", ra)
            else:
                logger.debug("Using provided RA: %s", ra)
            kwargs["ra"] = ra

            if dec is None:
                dec = float(hdul[0].header["DECDEG"])
                logger.debug("Extracted DEC from header: %s", dec)
            else:
                logger.debug("Using provided DEC: %s", dec)
            kwargs["dec"] = dec

            if info.loading.naming.upper() == "IGR":
                title = IGR.get_name(
                    ra=ra * Unit("deg"), 
                    dec=dec * Unit("deg"),
                )
                logger.debug("Generated title using IGR naming: %s", title)
            elif info.loading.naming.upper() == "J2000":
                title = J2000.get_name(
                    ra=ra * Unit("deg"), 
                    dec=dec * Unit("deg"),
                )
                logger.debug("Generated title using J2000 naming: %s", title)
            elif "OBJ_NAME" in hdul[0].header:
                title = hdul[0].header["OBJ_NAME"]
                logger.debug("Extracted title from header: %s", title)
            else:
                title = "missing_title"
                logger.debug("Title missing, using default: %s", title)
            kwargs["title"] = title

            if z is None:
                if "OBJ_Z" in hdul[0].header:
                    z = float(hdul[0].header["OBJ_Z"])
                    logger.debug("Extracted redshift from header: %s", z)
                else:
                    z = 0.0
                    logger.debug("Redshift missing, using default: %s", z)
            else:
                logger.debug("Using provided redshift: %s", z)
            kwargs["z"] = z

            data = _get_data(hdul)
            logger.debug(
                "Getting data: %s", 
                {k: None if v is None else v.shape for k, v in data.items()}
            )
            kwargs.update(data)

        return cls(**kwargs)

    ### Metadata

    def _get_metadata(self) -> dict[str, Any]:
        hdr = fits.getheader(self.path, ext=0)
        return {
            "path": self.path,
            "title": self.title,
            "ra": self.ra,
            "dec": self.dec,
            "z": self.z,
            "luminosity_distance_mpc": self.luminosity_distance_mpc or 0.0,
            "plate": int(hdr.get("PLATEID", -1)),
            "fiber": int(hdr.get("FIBERID", -1)),
            "mjd": int(hdr.get("MJD", -1)),
        }


    def get_metadata(self) -> SDSSMeta:
        return SDSSMeta(**self._get_metadata())