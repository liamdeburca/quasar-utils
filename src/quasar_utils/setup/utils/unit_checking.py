"""
This file contains utilities for checking the dimensions of units, i.e. whether 
a unit is a wavelength unit, a velocity unit, etc.
"""
from astropy.units import Unit

def is_wavelength_unit(unit: Unit) -> bool:
    return unit.is_equivalent("angstrom")

def check_is_wavelength_unit(unit: Unit) -> None:
    if not is_wavelength_unit(unit):
        raise ValueError(f"Expected a wavelength unit, but got {unit}.")
    
def is_velocity_unit(unit: Unit) -> bool:
    return unit.is_equivalent("m/s")

def check_is_velocity_unit(unit: Unit) -> None:
    if not is_velocity_unit(unit):
        raise ValueError(f"Expected a velocity unit, but got {unit}.")
    
def is_flux_unit(unit: Unit) -> bool:
    return unit.is_equivalent("erg/(s.cm2.angstrom)")

def check_is_flux_unit(unit: Unit) -> None:
    if not is_flux_unit(unit):
        raise ValueError(f"Expected a flux unit, but got {unit}.")
    
def is_density_unit(unit: Unit) -> bool:
    return unit.is_equivalent("1/cm3")
    
def check_is_density_unit(unit: Unit) -> None:
    if not is_density_unit(unit):
        raise ValueError(f"Expected a density unit, but got {unit}.")
    
def is_temperature_unit(unit: Unit) -> bool:
    return unit.is_equivalent("K")

def check_is_temperature_unit(unit: Unit) -> None:
    if not is_temperature_unit(unit):
        raise ValueError(f"Expected a temperature unit, but got {unit}.")