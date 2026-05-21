__all__ = ['SDSSLoader']

from logging import getLogger
from typing import Any
from astropy.units import Unit
from astropy.constants import c
from astropy.io import fits
from numpy import float64, full_like, nan
from pydantic.dataclasses import dataclass

from quasar_typing.astropy import HDUList_, Quantity_

from quasar_utils.decorators import validate_call
from quasar_utils.loading.loader import _Loader
from quasar_utils.setup import Info
from quasar_utils.naming import IGR, J2000

logger = getLogger(__name__)
SIGMA_RES: float = 69.0 / c.to('km/s').value

@validate_call
def get_data(
    hdul: HDUList_,
) -> tuple[Quantity_, Quantity_, Quantity_, Quantity_]:
    x_unit = Unit('angstrom')
    flux_unit = Unit('1e-17 erg/(s.cm2.angstrom)')

    data = hdul[1].data
    
    x = 10**data['loglam'].astype(float64)
    dx = x * SIGMA_RES
    flux = data['flux'].astype(float64)

    dy = full_like(x, nan, dtype=float64)
    mask = data['ivar'] > 0
    dy[mask] = (1 / data['ivar'][mask]**0.5).astype(float64)

    invalid_mask = data['and_mask'].astype(bool)
    flux[invalid_mask] = nan
    dy[invalid_mask] = nan

    return (
        x * x_unit, 
        flux * flux_unit, 
        dy * flux_unit, 
        dx * x_unit,
    )

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

@dataclass
class SDSSLoader(_Loader):
    """
    Loader designed for reading SDSS files, i.e. outputs from the 
    'astroquery.sdss' module.
    """
    def __post_init__(self):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        msg: str = "Initialising loader (SDSS): "

        msg += f"(1) reading data from {self.path}, "
        with fits.open(self.path) as hdul:
            self.ra = float(hdul[0].header['RADEG'])
            self.dec = float(hdul[0].header['DECDEG'])

            msg += "(2) extracted coordinates from header (ra={}, dec={}), "\
                .format(self.ra, self.dec)

            if "OBJ_NAME" in hdul[0].header:
                name = hdul[0].header["OBJ_NAME"]
                msg += f"(3) extracted name from header ({name}), "
            else:
                name = "missing_name"
                msg += "(3) no name in header, "

            match self.info.loading.naming.upper():
                case "IGR":
                    self.title = IGR.get_name(
                        self.ra * Unit('degree'), 
                        self.dec * Unit('degree'),
                    )
                    msg += "(4) generated IGR title from coordinates: {}"\
                        .format(self.title)
                case "J2000":
                    self.title = J2000.get_name(
                        self.ra * Unit('degree'), 
                        self.dec * Unit('degree'),
                    )
                    msg += "(4) generated J2000 title from coordinates: {}"\
                        .format(self.title)
                case _:
                    self.title = name
                    msg += "(4) no valid naming convention specified, "
                
            if self.z != 0.0:
                msg += f"(5) got redshift from argument: {self.z:.3f}."
            elif 'OBJ_Z' in hdul[0].header:
                self.z = float(hdul[0].header['OBJ_Z'])
                msg += f"(5) got redshift from header: {self.z:.3f}."
            else:
                msg += f"(5) no redshift found in header, defaulting to: {self.z:.3f}."

            self.x, self.y, self.dy, self.dx = get_data(hdul)
        
        logger.debug(msg)
        super().__post_init__()