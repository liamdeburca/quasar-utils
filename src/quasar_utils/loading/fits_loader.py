from logging import getLogger
logger = getLogger(__name__)

from typing import Any
from astropy.units import Unit
from astropy.io import fits
from pydantic import validate_call
from functools import partial

from quasar_typing.astropy import HDUList_, Quantity_
from quasar_typing.pathlib import AbsoluteFilePath

from .loader import _Loader
from ..setup import Info
from ..naming import IGR, J2000

class FITSLoader(_Loader):
    """
    Loader designed for reading FITS (.fits) files. 
    """
    @validate_call
    def __init__(
        self,
        path: AbsoluteFilePath,
        *,
        info: Info = None,
        z: float | None = None
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        msg: str = "Initialising loader (FITS): "

        msg += f"(1) reading data from {path}, "
        with fits.open(path) as hdul:
            from_data: partial = partial(
                self.get_from_data.__wrapped__,
                cls=self.__class__,
                hdul=hdul,
                info=info,
            )
            from_header: partial = partial(
                self.get_from_header.__wrapped__,
                cls=self.__class__,
                hdul=hdul,
                info=info,
            )

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
                from_data('x'),
                from_data('y'),
                from_data('dy'),
                z=z,
                title=title,
                path=path,
                info=info,
            )
        
        logger.debug(msg)

    @classmethod
    @validate_call
    def get_from_data(
        cls,
        key: str,
        hdul: HDUList_,
        info: Info,
    ) -> Quantity_:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        vals = info.loading[key]
        
        hdu = hdul[vals[0]]
        label = hdu.header[vals[1]]
        unit = hdu.header[vals[1]]

        return hdu.data[label].flatten() * Unit(unit)
    
    @classmethod
    @validate_call
    def get_from_header(
        cls,
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
