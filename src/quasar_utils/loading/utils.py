from typing import TypeVar

import numpy as np
from quasar_typing.astropy import CompositeUnitType, QuantityType, UnitType
from quasar_typing.numpy import CoordsTuple, FloatVector

from quasar_utils.dereddening import LawName, MapName, _deredden_spectrum
from quasar_utils.setup import Info

N = TypeVar("N", bound=int)
O = TypeVar("O", bound=int)
K = TypeVar("K", bound=int)


def _transform_coords(
    coords: CoordsTuple,
    dx: FloatVector | QuantityType,
    info: Info,
) -> tuple[CoordsTuple, FloatVector]:
    """
    Transform the input coordinates to unitless numpy arrays.
    """
    from astropy.units import Quantity
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
    map_name: MapName,
    law_name: LawName,
    wavelength_unit: UnitType | CompositeUnitType,
    Rv: float,
) -> CoordsTuple:
    """
    Apply Galactic dereddening to the input spectrum coordinates.
    """
    from astropy.coordinates import SkyCoord
    return _deredden_spectrum(
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

    from quasar_utils.binning import _log_resample
    xr, yr, dyr = _log_resample(
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
            v = np.ascontiguousarray(v)
        v.setflags(write=False)
        return v
    
    return tuple(inner(v) for v in arr)
