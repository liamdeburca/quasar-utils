__all__ = ['SDSSLoader']

from logging import getLogger
from typing import Any
from astropy.units import Unit
from astropy.io import fits
from numpy import float64, full_like, nan

from quasar_typing.astropy import HDUList_, Quantity_
from quasar_typing.pathlib import AbsoluteFITSPath

from .loader import _Loader
from ..setup import Info
from ..naming import IGR, J2000
from ..decorators import validate_call

logger = getLogger(__name__)

@validate_call
def get_data(
    hdul: HDUList_,
) -> tuple[Quantity_, Quantity_, Quantity_]:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    x_unit = Unit('angstrom')
    flux_unit = Unit('1e-17 erg/(s.cm2.angstrom)')

    data = hdul[1].data
    
    x = 10**data['loglam'].astype(float64)
    flux = data['flux'].astype(float64)

    dy = full_like(x, nan, dtype=float64)
    mask = data['ivar'] > 0
    dy[mask] = (1 / data['ivar'][mask]**0.5).astype(float64)

    invalid_mask = data['and_mask'].astype(bool)
    flux[invalid_mask] = nan
    dy[invalid_mask] = nan

    return (x * x_unit, flux * flux_unit, dy * flux_unit)

@validate_call
def get_from_header(
    hdul: HDUList_,
    info: Info,
    key: str,
    default: Any,
) -> Any:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    ext, label = info.loading[key]
    return hdul[ext].header.get(label, default)

class SDSSLoader(_Loader):
    """
    Loader designed for reading SDSS files, i.e. outputs from the 
    'astroquery.sdss' module.
    """
    @validate_call
    def __init__(
        self,
        path: AbsoluteFITSPath,
        *,
        info: Info = None,
        z: float | None = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        msg: str = "Initialising loader (SDSS): "

        msg += f"(1) reading data from {path}, "
        with fits.open(path) as hdul:
            ra = float(hdul[0].header['RADEG']) * Unit('degree')
            dec = float(hdul[0].header['DECDEG']) * Unit('degree')

            msg += f"(2) extracted coordinates from header ({ra=}, {dec=}), "

            if "OBJ_NAME" in hdul[0].header:
                name = hdul[0].header["OBJ_NAME"]
                msg += f"(3) extracted name from header ({name}), "
            else:
                name = "missing_name"
                msg += "(3) no name in header, "

            match convention := info.loading.naming.upper():
                case "IGR":
                    title = IGR.get_name(ra, dec)
                case "J2000":
                    title = J2000.get_name(ra, dec)
                case _:
                    title = name

            if title == name:
                msg += "(4) no valid naming convention specified, "
            else:
                msg += "(4) generated title from coordinates using " \
                    f"{convention} convention ({title}), "
                
            if z is not None:
                msg += f"(5) got redshift from argument ({z:.3f})."
            elif 'OBJ_Z' in hdul[0].header:
                z = float(hdul[0].header['OBJ_Z'])
                msg += f"(5) got redshift from header ({z:.3f})."
            else:
                z = 0
                msg = f"(5) no redshift found in header, defaulting to {z=}."

            super().__init__(
                *get_data(hdul),
                z=z,
                title=title,
                path=path,
                info=info,
            )
        
        logger.debug(msg)