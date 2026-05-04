from typing import Any, Self
from numpy import arange, float64, finfo
from numpy.random import RandomState
from dataclasses import dataclass
from astropy.units import Unit, CompositeUnit

from quasar_typing.misc.string_selection import StringSelection

from . import unit_checking

MACHINE_PRECISION = finfo(float64).eps

@dataclass
class JSONField:
    field: str
    parent_field: str
    value: Any

    def __str__(self) -> str:
        return "JSONField({}-{}: value={})".format(
            self.parent_field, self.field, self.value,
        )

    @classmethod
    def load_from_json(
        cls, 
        d: dict,
        field: str,
        parent_field: str,
    ) -> Self:
        val = d[parent_field][field].get("value", None)
        unit_str = d[parent_field][field].get("unit", "")
        parse_as = d[parent_field][field].get("parse_as", None)

        if (val is not None) and (parse_as is not None):
            match parse_as:
                case "bool": 
                    val = bool(val)
                case "int": 
                    val = int(val)
                case "float": 
                    val = float(val)
                case "optional_float":
                    val = float(val) if (val is not None) else None
                case "float_bounds":
                    val = tuple(
                        float(v) if (v is not None) else None 
                        for v in val
                    )
                case "float_list":
                    val = [float(v) for v in val]
                case "str": 
                    val = str(val)
                case "str_list":
                    val = [str(v) for v in val]
                case "str_set":
                    val = set(str(v) for v in val)
                case "str_selection":
                    val = StringSelection(set(str(v) for v in val))
                case "unit":
                    val = Unit(val)
                case "composite_unit":
                    val = Unit(val)
                    if not isinstance(val, CompositeUnit):
                        val = Unit(f"1.0 {str(val)}") 
                case "n_pixels":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_velocity_unit(unit)
                        val *= unit
                    else:
                        val = int(val)
                case "arange":
                    val = arange(*val, dtype=float64)
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_velocity_unit(unit)
                        val *= unit        
                case "wavelength":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_wavelength_unit(unit)
                        val *= unit
                    else:
                        val = float(val)
                case "wavelength_list":
                    # Note: possible None values in the list
                    val = [float(v or 0) for v in val]
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_wavelength_unit(unit)
                        val *= unit
                case "wavelength_bounds":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_wavelength_unit(unit)
                        val = tuple(
                            v * unit if v is not None else v
                            for v in val
                        )
                    else:
                        val = tuple(
                            v if v is not None else v
                            for v in val
                        )
                case "wavelength_windows":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_wavelength_unit(unit)
                        val *= unit
                    else:
                        val = [[float(v) for v in pair] for pair in val]
                case "flux":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_flux_unit(unit)
                        val *= unit
                    else:
                        val = float(val)
                case "flux_bounds":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_flux_unit(unit)
                        val = tuple(
                            v * unit if v is not None else v
                            for v in val
                        )
                    else:
                        val = tuple(
                            float(v) if v is not None else v
                            for v in val
                        )
                case "strength":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_strength_unit(unit)
                        val *= unit
                    else:
                        val = float(val)
                case "strength_bounds":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_strength_unit(unit)
                        val = tuple(
                            v * unit if v is not None else v
                            for v in val
                        )
                    else:
                        val = tuple(
                            float(v) if v is not None else v
                            for v in val
                        )

                case "velocity":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_velocity_unit(unit)
                        val *= unit
                    else:
                        val = float(val)
                case "velocity_bounds":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_velocity_unit(unit)
                        val = tuple(
                            v * unit if v is not None else v
                            for v in val
                        )
                    else:
                        val = tuple(
                            float(v) if v is not None else v
                            for v in val
                        )
                case "density":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_density_unit(unit)
                        val *= unit
                    else:
                        val = float(val)
                case "density_bounds":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_density_unit(unit)
                        val = tuple(
                            v * unit if v is not None else v
                            for v in val
                        )
                    else:
                        val = tuple(
                            float(v) if v is not None else v
                            for v in val
                        )
                case "temperature":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_temperature_unit(unit)
                        val *= unit
                    else:
                        val = float(val)
                case "temperature_bounds":
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_temperature_unit(unit)
                        val = tuple(
                            v * unit if v is not None else v
                            for v in val
                        )
                    else:
                        val = tuple(
                            float(v) if v is not None else v
                            for v in val
                        )
                case "balmer_fixed_params":
                    possible_values = {'fwhm', 'temp', 'tau', 'scale', 'ratio'}
                    val = StringSelection(v for v in val if v in possible_values)
                case "loader":
                    possible_values = {'ascii', 'fits', 'paqs', 'sdss'}
                    val = val.strip().lower()
                    assert val in possible_values
                case "naming":
                    possible_values = {'j2000', 'igr', 'sdss'}
                    val = val.strip().lower()
                    assert val in possible_values
                case "deredden":
                    possible_maps = {'sfd', 'csfd'}
                    possible_laws = {'ccm89', 'o94'}
                    
                    b = val[0]
                    assert (map_ := val[1].strip().lower()) in possible_maps
                    assert (law_ := val[2].strip().lower()) in possible_laws
                    Rv = val[3]

                    val = (b, map_, law_, Rv)
                case "bias":
                    possible_values = {'left', 'right'}
                    val = [v.strip().lower() for v in val]
                    assert all(v in possible_values for v in val)
                case "split_scale":
                    val = float(val)
                    if unit_str:
                        unit = Unit(unit_str)
                        unit_checking.check_is_velocity_unit(unit)
                        val *= unit
                case "algo":
                    possible_values = {'trf', 'dogbox', 'lm'}
                    val = val.strip().lower()
                    assert val in possible_values
                case "tol":
                    val = max(val, MACHINE_PRECISION)
                case "random_state":
                    val = RandomState(int(val))
                case _:
                    msg = f"Parsing algorithm '{parse_as}' is not implemented!"
                    raise NotImplementedError(msg)

        return JSONField(field, parent_field, val)
    
    @classmethod
    def load_all_from_json(
        cls, 
        d: dict,
        parent_field: str,
    ) -> list[Self]:
        return [
            JSONField.load_from_json(d, field, parent_field) 
            for field in d[parent_field].keys()
        ]
