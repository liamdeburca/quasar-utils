__all__ = ['_Loader']

from logging import getLogger

from typing import Self, Literal
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from numpy import diff, log, ascontiguousarray

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.numpy import FloatVector, CoordsTuple
from quasar_typing.astropy import Unit_, CompositeUnit_, Quantity_
from quasar_typing.pathlib import AbsoluteFilePath

from ..setup import Info
from ..binning import log_resample
from ..decorators import validate_call, validated_apply_info_to_method
from ..dereddening import deredden_spectrum

logger = getLogger(__name__)

class _Loader:
    @validate_call
    def __init__(
        self,
        x: FloatVector | Quantity_,
        y: FloatVector | Quantity_,
        dy: FloatVector | Quantity_,
        z: float = 0,
        ra: float | None = None,
        dec: float | None = None,
        title: str = "missing_title",
        path: str | AbsoluteFilePath = "missing_path",
        info: Info = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.x: FloatVector | Quantity_ = x
        self.y: FloatVector | Quantity_ = y
        self.dy: FloatVector | Quantity_ = dy

        self.z: float = z
        self.ra: float | None = ra
        self.dec: float | None = dec
        self.title: str = title
        self.path: AbsoluteFilePath | str = path
        self.info: Info = info

    @validated_apply_info_to_method(subjects=('loading',))
    def __call__(
        self,
        *,
        deredden: tuple[bool, Literal['sfd', 'csfd'], Literal['ccm89', 'o94'], float] | None = None,
        sigma_res: float | None = None,
        rebin: bool | None = None,
        conserve: bool | None = None,
        covariance: bool | None = None,
    ) -> dict:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        msg = f"Running {self.__class__.__name__} loading pipeline: "

        msg += "(1) creating unitless coordinates, "
        coords = self.transform_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            info=self.info,
        )

        if deredden[0]:
            if None in (self.ra, self.dec):
                msg += f"(2) skipping dereddening due to missing coordinates \
                    [RA={self.ra}, DEC={self.dec}], "
            else:
                msg += f"(2) applying dereddening correction \
                    [RA={self.ra}, DEC={self.dec}, map={deredden[1]}, \
                    law={deredden[2]}], Rv={deredden[3]}], "
                coords = self.deredden_coords.__wrapped__(
                    self.__class__,
                    coords,
                    self.ra,
                    self.dec,
                    deredden[1],
                    deredden[2],
                    self.info.units['wavelength_unit'],
                    deredden[3],
                )
        else:
            msg += "(2) skipping dereddening correction, "

        if rebin:
            msg += "(3) logarithmic re-binning, "
            coords = self.logbin_coords.__wrapped__(
                self.__class__, 
                coords, 
                sigma_res, 
                conserve,
                covariance,
            )
        else:
            _sigma_res = diff(log(coords[0]))

            msg += "(3) skipping logarithmic re-binning [assuming \
                'sigma_res'={:.2e}, actual 'sigma_res'={:.2e}±{:.2e}]".format(
                    sigma_res, _sigma_res.mean(), _sigma_res.std()
                )
            
        msg += "(4) applying redshift correction, "
        coords = self.redshift_correct_coords.__wrapped__(
            self.__class__, 
            coords, 
            self.z,
        )

        msg += "(5) ensuring coordinates are C-contiguous."
        coords = self.make_coords_contiguous.__wrapped__(
            self.__class__, 
            coords,
        )

        logger.debug(msg)

        return {
            'path': self.path,
            'title': self.title,
            'coords': coords,
            'info': self.info,
        }
    
    @classmethod
    def _validate(cls, value) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected a {cls.__name__} instance, \
                got {type(value).__name__}"
            raise PydanticCustomError("validation_error", msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
    
    @property
    def transformed_coords(self) -> CoordsTuple:
        return self.transform_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy), 
            info=self.info,
        )
    
    @classmethod
    @validate_call
    def transform_coords(
        cls,
        coords: CoordsTuple | tuple[Quantity_],
        info: Info = None,
    ) -> CoordsTuple:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        _x, _y, _dy = coords
        if isinstance(_x, Quantity):  
            _x  = info.units.getWavelength(_x)
        if isinstance(_y, Quantity): 
            _y  = info.units.getFlux(_y)
        if isinstance(_dy, Quantity):
            _dy = info.units.getFlux(_dy)

        return (_x, _y, _dy)
    
    @property
    def dereddenned_coords(self) -> CoordsTuple:
        assert self.ra is not None, "RA is required for dereddening"
        assert self.dec is not None, "DEC is required for dereddening"
        return self.deredden_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            self.ra,
            self.dec,
            self.info.loading['dereddening_map'],
            self.info.loading['dereddening_law'],
            self.info.units.getWavelengthUnit(),
            self.info.loading['Rv'],
        )    
    @classmethod
    @validate_call
    def deredden_coords(
        cls,
        coords: CoordsTuple,
        ra: float,
        dec: float,
        map_name: Literal['sfd', 'csfd'],
        law_name: Literal['ccm89', 'o94'],
        wavelength_unit: Unit_ | CompositeUnit_,
        Rv: float,
    ) -> CoordsTuple:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        return deredden_spectrum(
            coords,
            SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs'),
            map_name=map_name,
            law_name=law_name,
            wavelength_unit=wavelength_unit,
            Rv=Rv,
        )

    @property 
    def redshift_corrected_coords(self) -> CoordsTuple:
        return self.redshift_correct_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            self.z,
        )
    
    @classmethod
    @validate_call
    def redshift_correct_coords(
        cls,
        coords: CoordsTuple,
        z: float,
    ) -> CoordsTuple:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if z == 0: 
            return coords

        _x, _y, _dy = coords
        corr = 1 + z

        x_corr = _x / corr
        y_corr = _y * corr
        dy_corr = _dy * corr

        return (x_corr, y_corr, dy_corr)
    
    @property
    def logbinned_coords(self) -> CoordsTuple:
        return self.logbin_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            self.info.loading['sigma_res'],
            self.info.loading['conserve'],
            self.info.loading['covariance'],
        )
    
    @classmethod
    @validate_call
    def logbin_coords(
        cls,
        coords: CoordsTuple | tuple[Quantity_],
        sigma_res: float,
        conserve: bool,
        covariance: bool,
    ) -> CoordsTuple:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        return log_resample.__wrapped__(
            *coords,
            sigma_res,
            conserve=conserve,
            covariance=covariance,
        )

    @classmethod
    @validate_call
    def make_coords_contiguous(
        cls,
        coords: CoordsTuple,
    ) -> CoordsTuple:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        return tuple(
            arr if arr.flags['C_CONTIGUOUS'] else ascontiguousarray(arr)
            for arr in coords
        )