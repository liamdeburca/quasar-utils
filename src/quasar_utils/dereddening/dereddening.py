"""
Submodule containing utilities for dereddening spectra using the following
modules:

- `dustmaps`
- `dust_extinction`
"""
__all__ = [
    "deredden_spectrum",
    "get_correction",
]

from functools import wraps

import numpy as np
from astropy.units import Unit
from pydantic import validate_call
from quasar_typing.astropy.coordinates import SkyCoordType
from quasar_typing.astropy.units import CompositeUnitType, UnitType
from quasar_typing.literals import LawName, MapName
from quasar_typing.numpy import FloatVector, SpectrumCoordsTuple

### Public functions


@validate_call
def get_correction(
    x: FloatVector,
    sky_coords: SkyCoordType,
    *,
    map_name: MapName = "sfd",
    law_name: LawName = "ccm89",
    wavelength_unit: UnitType | CompositeUnitType | None = None,
    Rv: float = 3.1,
) -> FloatVector:
    """
    Calculate the correction factor for a given spectrum using the specified
    dust map and dust curve.

    Parameters
    ----------
    x: FloatVector
        Wavelength array of the spectrum.
    sky_coords: SkyCoordType
        Sky coordinates of the source, used to query the dust map for the
        extinction value.
    map_name: MapName, optional
        Name of the dust map to use for querying the extinction value.
        Default is 'sfd'.
    law_name: LawName, optional
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
    return _get_correction(
        x=x,
        sky_coords=sky_coords,
        map_name=map_name,
        law_name=law_name,
        wavelength_unit=wavelength_unit,
        Rv=Rv,
    )


@validate_call
def deredden_spectrum(
    coords: SpectrumCoordsTuple,
    sky_coords: SkyCoordType,
    *,
    map_name: MapName = "sfd",
    law_name: LawName = "ccm89",
    wavelength_unit: UnitType | CompositeUnitType | None = None,
    Rv: float = 3.1,
) -> SpectrumCoordsTuple:
    """
    Deredden a spectrum using the specified dust map and dust curve.

    Parameters
    ----------
    coords: SpectrumCoordsTuple
        Tuple of (x, y, dy) arrays representing the spectrum to be dereddened.
    sky_coords: SkyCoordType
        Sky coordinates of the source, used to query the dust map for the
        extinction value.
    map_name: MapName, optional
        Name of the dust map to use for querying the extinction value.
        Default is 'sfd'.
    law_name: LawName, optional
        Name of the dust law to use for dereddening the spectrum.
        Default is 'ccm89'.
    wavelength_unit: UnitType | CompositeUnitType | None, optional
        Unit of the wavelength values in the input spectrum.
    Rv: float, optional
        R(V) = A(V)/E(B-V) = total-to-selective extinction. Default is 3.1.
    """
    return _deredden_spectrum(
        coords, sky_coords, 
        map_name=map_name, 
        law_name=law_name, 
        wavelength_unit=wavelength_unit, 
        Rv=Rv,
    )


### Private functions


@wraps(get_correction)
def _get_correction(
    x: FloatVector,
    sky_coords: SkyCoordType,
    *,
    map_name: MapName = "sfd",
    law_name: LawName = "ccm89",
    wavelength_unit: UnitType | CompositeUnitType | None = None,
    Rv: float = 3.1,
) -> FloatVector:
    from .dust_extinction import get_dust_law
    from .dustmaps import get_dust_map
    
    if wavelength_unit is None:
        wavelength_unit = Unit("angstrom")

    dust_map = get_dust_map(map_name)
    dust_law = get_dust_law(law_name)

    k = (1 / (x * wavelength_unit)).to(dust_law.x_unit).value
    ebv = dust_map.query(sky_coords)
    ext = Rv * ebv * dust_law.evaluate(k, Rv=Rv)
    corr = np.exp(ext / 1.086)  # equivalent to 10**(0.4 * ext)

    return corr


@wraps(deredden_spectrum)
def _deredden_spectrum(
    coords: SpectrumCoordsTuple,
    sky_coords: SkyCoordType,
    *,
    map_name: MapName = "sfd",
    law_name: LawName = "ccm89",
    wavelength_unit: UnitType | CompositeUnitType | None = None,
    Rv: float = 3.1,
) -> SpectrumCoordsTuple:
    correction = _get_correction(
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