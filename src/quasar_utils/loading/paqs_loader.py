from logging import getLogger
logger = getLogger(__name__)

from typing import Any
from pydantic import validate_call
from astropy.units import Unit
from astropy.io import fits
from numpy import float64
from functools import partial

from quasar_typing.astropy import HDUList_, Quantity_
from quasar_typing.pathlib import AbsoluteFITSPath

from .loader import _Loader
from ..setup import Info
from ..naming import IGR, J2000

@validate_call
def get_data(
    hdul: HDUList_,
) -> tuple[Quantity_, Quantity_, Quantity_]:
    """
    ** PYDANTIC VALIDATED METHOD **
    """
    x_unit: str = 'angstrom'
    flux_unit: str = 'erg/(s.cm2.angstrom)'

    data = hdul[1].data

    x = data['wave'].flatten().astype(float64)
    flux = data['flux'].flatten().astype(float64)
    dy = data['err_flux'].flatten().astype(float64)

    x *= Unit(x_unit)
    flux *= Unit(flux_unit)
    dy *= Unit(flux_unit)

    return (x, flux, dy)

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

class PAQSLoader(_Loader):
    """
    Loader designed for reading PAQS files.
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
        msg: str = "Initialising loader (PAQS): "

        msg += f"(1) reading data from {path}, "
        with fits.open(path) as hdul:
            from_header: partial = partial(get_from_header, hdul, info)

            ra = float(from_header("ra", 0)) * Unit("degree")
            dec = float(from_header("dec", 0)) * Unit("degree")
            msg += f"(2) extracted coordinates from header \
                (ra={ra}, dec={dec}), "
            
            name = from_header("name", "missing_name")
            if name == "missing_name":
                msg += f"(3) no name in header, "
            else:
                msg += f"(3) extracted name from header ({name}), "

            match convention := info.loading['naming'].upper():
                case "IGR":
                    title = IGR.get_name(ra, dec)
                case "J2000":
                    title = J2000.get_name(ra, dec)
                case _:
                    title = name

            if title == name:
                msg += f"(4) no valid naming convention specified, "
            else:
                msg += f"(4) generated title from coordinates using \
                    {convention} convention ({title}), "
                
            if z is not None:
                msg += f"(5) got redshift from argument ({z:.3f})."  
            elif isinstance(val := info.loading['z'], float):
                z = val
                msg += f"(5) got redshift from Info instance ({z:.3f})."
            else:
                z: float = from_header('z', 0)
                msg += f"(5) got redshift from header ({z:.3f})."

            super().__init__(
                *get_data(hdul=hdul),
                z=z,
                title=title,
                path=path,
                info=info,
            )
        
        logger.debug(msg)