__all__ = [
    "LoaderOutput",
]

from typing import Literal, TypedDict, TypeVar

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from numpy import ascontiguousarray, interp, stack
from quasar_typing.astropy import CompositeUnit_, Quantity_, Unit_
from quasar_typing.numpy import CoordsTuple, FloatMatrix, FloatVector
from quasar_typing.pathlib import AbsoluteFilePath

from ..binning import log_resample
from ..dereddening import deredden_spectrum
from ..setup import Info

N = TypeVar("N", bound=int)
O = TypeVar("O", bound=int)
K = TypeVar("K", bound=int)
    

class LoaderOutput(TypedDict):
    path: str | AbsoluteFilePath
    title: str

    x_original: FloatVector
    y_original: FloatVector
    dy_original: FloatVector
    dx_original: FloatVector
    R_original: FloatVector | None

    x: FloatVector
    y: FloatVector
    dy: FloatVector
    dx: FloatVector
    R: FloatVector | None

    info: Info


def _transform_coords(
    coords: CoordsTuple,
    dx: FloatVector | Quantity_,
    info: Info,
) -> tuple[CoordsTuple, FloatVector]:
    """
    Transform the input coordinates to unitless numpy arrays.
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


def _deredden_coords(
    coords: CoordsTuple,
    ra: float,
    dec: float,
    map_name: Literal["sfd", "csfd"],
    law_name: Literal["ccm89", "o94"],
    wavelength_unit: Unit_ | CompositeUnit_,
    Rv: float,
) -> CoordsTuple:
    """
    Apply Galactic dereddening to the input spectrum coordinates.
    """
    return deredden_spectrum(
        coords,
        SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs"),
        map_name=map_name,
        law_name=law_name,
        wavelength_unit=wavelength_unit,
        Rv=Rv,
    )


def _redshift_correct_coords(
    coords: CoordsTuple,
    dx: FloatVector,
    z: float,
) -> tuple[CoordsTuple, FloatVector]:
    """
    Correct for cosmological redshift.
    """
    w_corr = 1.0 + z
    f_corr = w_corr ** 3

    x, y, dy = coords
    x_corr = x / w_corr
    dx_corr = dx / w_corr
    y_corr = y * f_corr
    dy_corr = dy * f_corr

    return (x_corr, y_corr, dy_corr), dx_corr


def _logbin_coords(
    coords: CoordsTuple,
    dx: FloatVector,
    sigma_res: float,
    conserve: bool,
    covariance: bool,
) -> tuple[CoordsTuple, FloatVector]:
    """
    Logarithmically bin the input spectrum coordinates.
    """
    if covariance:
        raise NotImplementedError(
            "Logarithmic binning with covariance propagation is not (yet) "
            "supported."
        )    

    xr, yr, dyr = log_resample.__wrapped__(
        *coords,
        sigma_res,
        dx=dx,
        conserve=conserve,
        covariance=False,
    )
    dxr = xr * sigma_res

    return (xr, yr, dyr), dxr


def _finalize_coords(*arr: FloatVector) -> FloatVector:
    """
    Ensure the input array is C-contiguous and read-only.
    """
    def inner(v: FloatVector) -> FloatVector:
        if not v.flags['C_CONTIGUOUS']:
            v = ascontiguousarray(v)
        v.setflags(write=False)
        return v
    
    return tuple(inner(v) for v in arr)

###

def interpolate_resolution_kernels[K, N, O](
    x: FloatVector[N],
    x_original: FloatVector[O],
    kernels_original: FloatMatrix[O, K]
) -> FloatMatrix[N, K]:
    kernels =  stack(
        [interp(x, x_original, ks) for ks in kernels_original],
        axis=1
    )
    return kernels / kernels.sum(axis=1, keepdims=True)
