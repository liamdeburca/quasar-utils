from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Self, TypedDict

from astropy.io import fits
from astropy.units import Quantity, Unit
from numpy import float64, full, interp, nan
from pydantic import StringConstraints, validate_call
from quasar_typing.astropy import TableType
from quasar_typing.numpy import FloatVector
from quasar_typing.pathlib import FITSPath

from quasar_utils.setup import Info

from .loader import _Loader
from .metadata import PAQSMeta

logger = getLogger(__name__)

_this_file: Path = Path(__file__).resolve()
PATH_TO_DATA: Path = _this_file.parent / "data/4most_low.csv"


class _PAQSDataDict(TypedDict):
    x_original: Quantity
    y_original: Quantity
    dy_original: Quantity
    dx_original: Quantity
    R_original: FloatVector


def _get_data(hdul: fits.HDUList) -> _PAQSDataDict:
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
    wave = data["WAVE"].flatten().astype(float64)
    flux = data["FLUX"].flatten().astype(float64)
    err = data["ERR"].flatten().astype(float64)
    qual = data["QUAL"].flatten().astype(float64)
    bin_size = full(wave.size, 0.25, dtype=float64)

    bad_pixels = (qual != 0.0)
    flux[bad_pixels] = nan
    err[bad_pixels] = nan

    # 4MOST: R=4000 @ 3700 angstrom, R=7700 @ 9500 angstrom
    R = interp(wave, *_get_Rs())

    return {
        "x_original": wave * wave_unit,
        "y_original": flux * flux_unit,
        "dy_original": err * flux_unit,
        "dx_original": bin_size * wave_unit,
        "R_original": R,
    }


@lru_cache(maxsize=1)
def _get_Rs() -> tuple[FloatVector, FloatVector]:
    """
    Return a tuple of wavelength-R pairs retrieved from the 'lambda' and 'R' 
    columns of the CSV file located at PATH_TO_DATA.

    Returns
    -------
    xs : FloatVector
        A read-only 1d array of wavelength values (angstrom)
    Rs : FloatVector
        A read_only 1d array of resolving power (R) values.
    """
    from pandas import read_csv
    
    df = read_csv(PATH_TO_DATA, usecols=["lambda", "R"])
    xs = df["lambda"].to_numpy(dtype=float64)
    Rs = df["R"].to_numpy(dtype=float64)

    xs.setflags(write=False)
    Rs.setflags(write=False)

    return xs, Rs
    

class PAQSLoader(_Loader):
    """
    Loader designed for reading PAQS files.
    """
    obj_nme: Annotated[str, StringConstraints(min_length=1)]
    obj_uid: int
    specuid: int

    @classmethod
    @validate_call
    def loadLookup(cls, path: FITSPath) -> TableType:
        """
        Read a FITS table from file. 
        """
        from astropy.table import Table
        return Table.read(path, format="fits")

    @classmethod
    @validate_call
    def fromPath(
        cls,
        path: FITSPath,
        lookup: TableType,
        info: Info | None = None,
        z: float | None = None,
        title: str | None = None,
        ra: float | None = None,
        dec: float | None = None,
    ) -> Self:
        from astropy.table import Row

        from quasar_utils.naming import IGR, J2000

        info = info or Info()
        kwargs = {"path": path, "info": info}
        lookup = lookup["OBJ_NME", "OBJ_UID", "SPECUID", "Z_QSO"]

        logger.debug("Creating PAQSLoader from path: %s", path)
        with fits.open(path) as hdul:
            obj_nme: str = str(hdul[0].header.get("OBJECT"))

            row = Row(lookup, (lookup["OBJ_NME"] == obj_nme).argmax())
            kwargs["obj_uid"] = int(row["OBJ_UID"])
            kwargs["specuid"] = int(row["SPECUID"])
            
            if ra is None:
                ra = float(hdul[0].header["RA"])
                logger.debug("Extracted RA from header: %s", ra)
            else:
                logger.debug("Using provided RA: %s", ra)
            kwargs["ra"] = ra

            if dec is None:
                dec = float(hdul[0].header["DEC"])
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
            else:
                title = obj_nme
                logger.debug("Extracted title from header: %s", title)
            kwargs["title"] = title

            if z is None:
                z = float(row["Z_QSO"])
                logger.debug("Using Z_QSO from lookup: %s", z)
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
        return {
            "path": self.path,
            "title": self.title,
            "ra": self.ra,
            "dec": self.dec,
            "z": self.z,
            "luminosity_distance_mpc": self.luminosity_distance_mpc or 0.0,
            "obj_nme": self.obj_nme,
            "obj_uid": self.obj_uid,
            "specuid": self.specuid,
        }

    def get_metadata(self) -> PAQSMeta:
        return PAQSMeta(**self._get_metadata())