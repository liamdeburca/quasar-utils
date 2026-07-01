__all__ = ['_Loader']

from logging import getLogger
from dataclasses import field
from typing import Literal
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from numpy import diff, log, ascontiguousarray, full, float64

from pydantic.dataclasses import dataclass

from quasar_typing.numpy import FloatVector, CoordsTuple
from quasar_typing.astropy import Unit_, CompositeUnit_, Quantity_
from quasar_typing.pathlib import AbsoluteFilePath

from ..setup import Info
from ..binning import log_resample
from ..decorators import validate_call, validated_apply_info_to_method
from ..dereddening import deredden_spectrum

logger = getLogger(__name__)

@dataclass
class _Loader:
    path: str | AbsoluteFilePath
    info: Info = field(default_factory=Info)
    z: float = 0.0

    title: str = "missing_title"
    ra: float = 0.0
    dec: float = 0.0

    x: FloatVector | Quantity_ = field(init=False)
    y: FloatVector | Quantity_ = field(init=False)
    dy: FloatVector | Quantity_ = field(init=False)
    dx: float | FloatVector | Quantity_ = field(init=False)

    def __post_init__(self):
        if isinstance(self.dx, float):
            self.dx = full(len(self.x), self.dx, dtype=float64)

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
        coords, dx = self.transform_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
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
            coords, dx = self.logbin_coords.__wrapped__(
                self.__class__, 
                coords, 
                dx=dx,
                sigma_res=sigma_res, 
                conserve=conserve,
                covariance=covariance,
            )
        else:
            _sigma_res = diff(log(coords[0]))

            msg += "(3) skipping logarithmic re-binning [assuming \
                'sigma_res'={:.2e}, actual 'sigma_res'={:.2e}±{:.2e}]".format(
                    sigma_res, _sigma_res.mean(), _sigma_res.std()
                )
            
        msg += "(4) applying redshift correction, "
        coords, dx = self.redshift_correct_coords.__wrapped__(
            self.__class__, 
            coords, 
            dx=dx,
            z=self.z,
        )

        msg += "(5) ensuring coordinates are C-contiguous."
        x, y, dy, dx = self.make_coords_contiguous.__wrapped__(
            self.__class__, 
            *coords, dx,
        )

        logger.debug(msg)

        return {
            'path': self.path,
            'title': self.title,
            'x': x,
            'y': y,
            'dy': dy,
            'dx': dx,
            'info': self.info,
        }
    
    @property
    def transformed_coords(self) -> tuple[FloatVector,...]:
        return self.transform_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
            info=self.info,
        )
    
    @classmethod
    @validate_call
    def transform_coords(
        cls,
        coords: CoordsTuple,
        *,
        dx: FloatVector | Quantity_,
        info: Info,
    ) -> tuple[CoordsTuple, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        x, y, dy = coords
        if isinstance(x, Quantity):  
            x = info.units.getWavelength(x)
        if isinstance(y, Quantity): 
            y = info.units.getFlux(y)
        if isinstance(dy, Quantity):
            dy = info.units.getFlux(dy)
        if isinstance(dx, Quantity):
            dx = info.units.getWavelength(dx)

        return (x, y, dy), dx
    
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
    def redshift_corrected_coords(self) -> tuple[CoordsTuple, FloatVector]:
        return self.redshift_correct_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
            z=self.z,
        )
    
    @classmethod
    @validate_call
    def redshift_correct_coords(
        cls,
        coords: CoordsTuple,
        *,
        dx: FloatVector,
        z: float,
    ) -> tuple[CoordsTuple, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        if z == 0: 
            return coords, dx

        x, y, dy = coords
        corr = 1 + z

        x_corr = x / corr
        y_corr = y * corr
        dy_corr = dy * corr
        dx_corr = dx / corr

        return (x_corr, y_corr, dy_corr), dx_corr
    
    @property
    def logbinned_coords(self) -> tuple[CoordsTuple, FloatVector]:
        return self.logbin_coords.__wrapped__(
            self.__class__,
            (self.x, self.y, self.dy),
            dx=self.dx,
            sigma_res=self.info.loading['sigma_res'],
            conserve=self.info.loading['conserve'],
            covariance=self.info.loading['covariance'],
        )
    
    @classmethod
    @validate_call
    def logbin_coords(
        cls,
        coords: CoordsTuple,
        *,
        dx: FloatVector,
        sigma_res: float,
        conserve: bool,
        covariance: bool,
    ) -> tuple[CoordsTuple, FloatVector]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        xr, yr, dyr = log_resample.__wrapped__(
            *coords,
            sigma_res,
            dx=dx,
            conserve=conserve,
            covariance=covariance,
        ) 
        dxr = xr * sigma_res
        return (xr, yr, dyr), dxr

    @classmethod
    @validate_call
    def make_coords_contiguous(
        cls,
        *coords: FloatVector,
    ) -> tuple[FloatVector,...]:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        return tuple(
            arr if arr.flags['C_CONTIGUOUS'] else ascontiguousarray(arr)
            for arr in coords
        )