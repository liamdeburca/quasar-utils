"""
Submodule containing utilities for dereddening spectra using the following 
modules:

- `dustmaps`
- `dust_extinction`
"""
__all__ = [
    "get_correction",
    "deredden_spectrum",
]

from typing import Literal
from numpy import exp
from astropy.units import Unit
from pydantic import validate_call

from quasar_typing.numpy import FloatVector
from quasar_typing.misc.coords_tuple import CoordsTuple
from quasar_typing.astropy import Unit_, CompositeUnit_, SkyCoord_

from .dustmaps import setup_dustmaps, get_dust_map
from .dust_extinction import get_dust_law

setup_dustmaps()

@validate_call(validate_return=False)
def get_correction(
    x: FloatVector,
    sky_coords: SkyCoord_,
    *,
    map_name: Literal['sfd', 'csfd'] = 'sfd',
    law_name: Literal['ccm89', 'o94'] = 'ccm89',
    wavelength_unit: Unit_ | CompositeUnit_ = Unit('angstrom'),
    Rv: float = 3.1,
) -> FloatVector:
    """
    ** PYDANTIC VALIDATED FUNCTION **

    Calculate the correction factor for a given spectrum using the specified
    dust map and dust curve.

    Parameters
    ----------
    x: FloatVector
        Wavelength array of the spectrum.
    sky_coords: SkyCoord_
        Sky coordinates of the source, used to query the dust map for the
        extinction value.
    map_name: Literal['sfd', 'csfd'], optional
        Name of the dust map to use for querying the extinction value.
        Default is 'sfd'.
    law_name: Literal['ccm89', 'o94'], optional
        Name of the dust law to use for dereddening the spectrum.
        Default is 'ccm89'.
    wavelength_unit: Unit_ | CompositeUnit_, optional
        Unit of the wavelength values in the input spectrum. Default is 'angstrom'.
    Rv: float, optional
        R(V) = A(V)/E(B-V) = total-to-selective extinction. Default is 3.1.

    Returns
    -------
    FloatVector
        Correction factor for the input spectrum.
    """
    dust_map = get_dust_map(map_name)
    dust_law = get_dust_law(law_name)

    k = (1 / (x * wavelength_unit)).to(dust_law.x_unit).value
    ebv = dust_map.query(sky_coords)
    ext = Rv * ebv * dust_law.evaluate(k, Rv=Rv)
    corr = exp(ext / 1.086) # equivalent to 10**(0.4 * ext)

    return corr

@validate_call(validate_return=False)
def deredden_spectrum(
    coords: CoordsTuple,
    sky_coords: SkyCoord_,
    *,
    map_name: Literal['sfd', 'csfd'] = 'sfd',
    law_name: Literal['ccm89', 'o94'] = 'ccm89',
    wavelength_unit: Unit_ | CompositeUnit_ = Unit('angstrom'),
    Rv: float = 3.1,
) -> CoordsTuple:
    """
    ** PYDANTIC VALIDATED FUNCTION **
    
    Deredden a spectrum using the specified dust map and dust curve.

    Parameters
    ----------
    coords: CoordsTuple
        Tuple of (x, y, dy) arrays representing the spectrum to be dereddened.
    sky_coords: SkyCoord
        Sky coordinates of the source, used to query the dust map for the 
        extinction value.
    map_name: Literal['sfd', 'csfd'], optional
        Name of the dust map to use for querying the extinction value. 
        Default is 'sfd'.
    law_name: Literal['ccm89', 'o94'], optional
        Name of the dust law to use for dereddening the spectrum. 
        Default is 'ccm89'.
    wavelength_unit: Unit | CompositeUnit, optional
        Unit of the wavelength values in the input spectrum. Default is 'angstrom'.
    Rv: float, optional
        R(V) = A(V)/E(B-V) = total-to-selective extinction. Default is 3.1.
    """
    correction = get_correction.__wrapped__(
        x=coords[0],
        sky_coords=sky_coords,
        map_name=map_name,
        law_name=law_name,
        wavelength_unit=wavelength_unit,
        Rv=Rv,
    )
    return (
        coords[0],
        coords[1] * correction,
        coords[2] * correction,
    )
