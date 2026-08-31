from collections.abc import Callable
from inspect import getmembers, ismethod
from typing import Any, TypedDict

from astropy.units import CompositeUnit, Quantity, Unit
from numpy import arange, finfo, float64
from numpy.random import RandomState
from quasar_typing.misc import (
    BalmerModelParams,
    HostGalaxyModelParams,
    OutLines,
    OutMeasures,
    StringSelection,
    VaryLines,
)
from quasar_typing.numpy import SortedFloatVector

from .unit_checking import (
    is_density_unit,
    is_flux_unit,
    is_temperature_unit,
    is_velocity_unit,
    is_wavelength_unit,
)

JSONDataType = str | int | float | bool | None
MACHINE_PRECISION = finfo(float64).eps

class JSONField(TypedDict):
    value: JSONDataType | list[JSONDataType] | dict[str, JSONDataType]
    unit: str

###

class Parser:
    @classmethod
    def as_bool(cls, field: JSONField) -> bool:
        return bool(field["value"])

    @classmethod
    def undo_as_bool(cls, value: bool) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_bool_list(cls, field: JSONField) -> list[bool]:
        return [bool(v) for v in field["value"]]

    @classmethod
    def undo_as_bool_list(cls, value: list[bool]) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_int(cls, field: JSONField) -> int:
        return int(field["value"])

    @classmethod
    def undo_as_int(cls, value: int) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_int_list(cls, field: JSONField) -> list[int]:
        return [int(v) for v in field["value"]]

    @classmethod
    def undo_as_int_list(cls, value: list[int]) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_float(cls, field: JSONField) -> float:
        return float(field["value"])

    @classmethod
    def undo_as_float(cls, value: float) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_optional_float(cls, field: JSONField) -> float | None:
        return None if field["value"] is None else cls.as_float(field)

    @classmethod
    def undo_as_optional_float(cls, value: float | None) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_float_list(cls, field: JSONField) -> list[float]:
        return [float(v) for v in field["value"]]

    @classmethod
    def undo_as_float_list(cls, value: list[float]) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_float_set(cls, field: JSONField) -> set[float]:
        return {float(v) for v in field["value"]}

    @classmethod
    def undo_as_float_set(cls, value: set[float]) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_float_bounds(cls, field: JSONField) -> tuple[float | None, float | None]:
        return (
            None if field["value"][0] is None else float(field["value"][0]),
            None if field["value"][1] is None else float(field["value"][1]),
        )

    @classmethod
    def undo_as_float_bounds(cls, value: tuple[float | None, float | None]) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_str(cls, field: JSONField) -> str:
        return str(field["value"])

    @classmethod
    def undo_as_str(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_str_list(cls, field: JSONField) -> list[str]:
        return [str(v) for v in field["value"]]

    @classmethod
    def undo_as_str_list(cls, value: list[str]) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_str_set(cls, field: JSONField) -> set[str]:
        return {str(v) for v in field["value"]}

    @classmethod
    def undo_as_str_set(cls, value: set[str]) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_str_selection(cls, field: JSONField) -> StringSelection[str]:
        return StringSelection(cls.as_str_set(field))

    @classmethod
    def undo_as_str_selection(cls, value: StringSelection[str]) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_unit(cls, field: JSONField) -> Unit:
        return Unit(field["value"])

    @classmethod
    def undo_as_unit(cls, value: Unit) -> JSONField:
        return {
            "value": str(value),
            "unit": None,
        }

    @classmethod
    def as_composite_unit(cls, field: JSONField) -> CompositeUnit:
        u = Unit(field["value"])
        assert isinstance(u, CompositeUnit)
        return u

    @classmethod
    def undo_as_composite_unit(cls, value: CompositeUnit) -> JSONField:
        s = value.to_string()
        if not s[0].isdigit():
            s = f"1 {s}"

        return {
            "value": s,
            "unit": None,
        }

    @classmethod
    def as_n_pixels(cls, field: JSONField) -> int | Quantity:
        if unit := field.get("unit", None):
            val = cls.as_float(field)
            unit = Unit(unit)
            assert is_velocity_unit(unit)
            return val * unit
        return cls.as_int(field)

    @classmethod
    def undo_as_n_pixels(cls, value: int | Quantity) -> JSONField:
        if isinstance(value, Quantity):
            val = float(value.value)
            unit = str(value.unit)
        else:
            val = int(value)
            unit = None
        return {
            "value": val,
            "unit": unit,
        }
        
            
    @classmethod
    def as_arange(cls, field: JSONField) -> Quantity | SortedFloatVector:
        val = arange(*field["value"], dtype=float64)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_velocity_unit(unit)
            return val * unit
        return val

    @classmethod
    def undo_as_arange(cls, value: Quantity | SortedFloatVector) -> JSONField:
        if isinstance(value, Quantity):
            arr = value.value
            unit = str(value.unit)
        else:
            arr = value
            unit = None
        return {
            "value": [float(arr[0]), float(arr[-1]), float(arr[1] - arr[0])],
            "unit": unit,
        }

    @classmethod
    def as_wavelength(cls, field: JSONField) -> Quantity:
        val = cls.as_float(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_wavelength_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_wavelength(cls, value: Quantity) -> JSONField:
        return {
            "value": float(value.value),
            "unit": str(value.unit),
        }

    @classmethod
    def as_wavelength_list(cls, field: JSONField) -> Quantity | list[float]:
        val = cls.as_float_list(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_wavelength_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_wavelength_list(cls, value: Quantity | list[float]) -> JSONField:
        if isinstance(value, Quantity):
            arr = value.value
            unit = str(value.unit)
        else:
            arr = value
            unit = None
        return {
            "value": [float(v) for v in arr],
            "unit": unit,
        }

    @classmethod
    def as_wavelength_bounds(cls, field: JSONField) -> Quantity | tuple[float, float]:
        val = cls.as_float_list(field)
        assert len(val) == 2 and val[0] < val[1]
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_wavelength_unit(unit)
            val *= unit
        return tuple(val)

    @classmethod
    def undo_as_wavelength_bounds(cls, value: Quantity | tuple[float, float]) -> JSONField:
        if isinstance(value, Quantity):
            arr = value.value
            unit = str(value.unit)
        else:
            arr = value
            unit = None
        return {
            "value": [float(v) for v in arr],
            "unit": unit,
        }

    @classmethod
    def as_wavelength_windows(cls, field: JSONField) -> Quantity | list[tuple[float, float]]:
        val = [[v[0], v[1]] for v in field["value"]]
        for v in val:
            assert v[0] < v[1]
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_wavelength_unit(unit)
            return val * unit
        return [tuple(v) for v in val]

    @classmethod
    def undo_as_wavelength_windows(cls, value: Quantity | list[tuple[float, float]]) -> JSONField:
        if isinstance(value, Quantity):
            arr = value.value
            unit = str(value.unit)
        else:
            arr = value
            unit = None
        return {
            "value": [[float(v[0]), float(v[1])] for v in arr],
            "unit": unit,
        }

    @classmethod
    def as_flux(cls, field: JSONField) -> Quantity | float:
        val = cls.as_float(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_flux_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_flux(cls, value: Quantity | float) -> JSONField:
        if isinstance(value, Quantity):
            val = float(value.value)
            unit = str(value.unit)
        else:
            val = float(value)
            unit = None
        return {
            "value": val,
            "unit": unit,
        }

    @classmethod
    def as_flux_bounds(cls, field: JSONField) -> tuple[Quantity | float | None, Quantity | float | None]:
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_flux_unit(unit)
            return tuple(
                None if v is None else float(v) * unit
                for v in field["value"]
            )
        return tuple(
            None if v is None else float(v)
            for v in field["value"]
        )

    @classmethod
    def undo_as_flux_bounds(cls, value: tuple[Quantity | float | None, Quantity | float | None]) -> JSONField:
        result = []
        unit = None
        for v in value:
            if v is None:
                result.append(None)
            elif isinstance(v, Quantity):
                result.append(float(v.value))
                if unit is None:
                    unit = str(v.unit)
            else:
                result.append(float(v))
        return {
            "value": result,
            "unit": unit,
        }

    @classmethod
    def as_strength(cls, field: JSONField) -> Quantity | float:
        val = cls.as_float(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_flux_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_strength(cls, value: Quantity | float) -> JSONField:
        if isinstance(value, Quantity):
            val = float(value.value)
            unit = str(value.unit)
        else:
            val = float(value)
            unit = None
        return {
            "value": val,
            "unit": unit,
        }

    @classmethod
    def as_strength_bounds(cls, field: JSONField) -> tuple[Quantity | float | None, Quantity | float | None]:
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_flux_unit(unit)
            return tuple(
                None if v is None else float(v) * unit
                for v in field["value"]
            )
        return tuple(
            None if v is None else float(v)
            for v in field["value"]
        )

    @classmethod
    def undo_as_strength_bounds(cls, value: tuple[Quantity | float | None, Quantity | float | None]) -> JSONField:
        result = []
        unit = None
        for v in value:
            if v is None:
                result.append(None)
            elif isinstance(v, Quantity):
                result.append(float(v.value))
                if unit is None:
                    unit = str(v.unit)
            else:
                result.append(float(v))
        return {
            "value": result,
            "unit": unit,
        }

    @classmethod
    def as_velocity(cls, field: JSONField) -> Quantity | float:
        val = cls.as_float(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_velocity_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_velocity(cls, value: Quantity | float) -> JSONField:
        if isinstance(value, Quantity):
            val = float(value.value)
            unit = str(value.unit)
        else:
            val = float(value)
            unit = None
        return {
            "value": val,
            "unit": unit,
        }

    @classmethod
    def as_velocity_list(cls, field: JSONField) -> Quantity | list[float]:
        val = cls.as_float_list(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_velocity_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_velocity_list(cls, value: Quantity | list[float]) -> JSONField:
        if isinstance(value, Quantity):
            arr = value.value
            unit = str(value.unit)
        else:
            arr = value
            unit = None
        return {
            "value": [float(v) for v in arr],
            "unit": unit,
        }

    @classmethod
    def as_velocity_bounds(cls, field: JSONField) -> tuple[Quantity | float | None, Quantity | float | None]:
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_velocity_unit(unit)
            return tuple(
                None if v is None else float(v) * unit
                for v in field["value"]
            )
        return tuple(
            None if v is None else float(v)
            for v in field["value"]
        )

    @classmethod
    def undo_as_velocity_bounds(cls, value: tuple[Quantity | float | None, Quantity | float | None]) -> JSONField:
        result = []
        unit = None
        for v in value:
            if v is None:
                result.append(None)
            elif isinstance(v, Quantity):
                result.append(float(v.value))
                if unit is None:
                    unit = str(v.unit)
            else:
                result.append(float(v))
        return {
            "value": result,
            "unit": unit,
        }

    @classmethod
    def as_density(cls, field: JSONField) -> Quantity:
        val = cls.as_float(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_density_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_density(cls, value: Quantity) -> JSONField:
        return {
            "value": float(value.value),
            "unit": str(value.unit),
        }

    @classmethod
    def as_density_bounds(cls, field: JSONField) -> tuple[Quantity | float | None, Quantity | float | None]:
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_density_unit(unit)
            return tuple(
                None if v is None else float(v) * unit
                for v in field["value"]
            )
        return tuple(
            None if v is None else float(v)
            for v in field["value"]
        )

    @classmethod
    def undo_as_density_bounds(cls, value: tuple[Quantity | float | None, Quantity | float | None]) -> JSONField:
        result = []
        unit = None
        for v in value:
            if v is None:
                result.append(None)
            elif isinstance(v, Quantity):
                result.append(float(v.value))
                if unit is None:
                    unit = str(v.unit)
            else:
                result.append(float(v))
        return {
            "value": result,
            "unit": unit,
        }

    @classmethod
    def as_temperature(cls, field: JSONField) -> Quantity | float:
        val = cls.as_float(field)
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_temperature_unit(unit)
            val *= unit
        return val

    @classmethod
    def undo_as_temperature(cls, value: Quantity | float) -> JSONField:
        if isinstance(value, Quantity):
            val = float(value.value)
            unit = str(value.unit)
        else:
            val = float(value)
            unit = None
        return {
            "value": val,
            "unit": unit,
        }

    @classmethod
    def as_temperature_bounds(cls, field: JSONField) -> tuple[Quantity | float | None, Quantity | float | None]:
        if unit := field.get("unit", None):
            unit = Unit(unit)
            assert is_temperature_unit(unit)
            return tuple(
                None if v is None else float(v) * unit
                for v in field["value"]
            )
        return tuple(
            None if v is None else float(v)
            for v in field["value"]
        )

    @classmethod
    def undo_as_temperature_bounds(cls, value: tuple[Quantity | float | None, Quantity | float | None]) -> JSONField:
        result = []
        unit = None
        for v in value:
            if v is None:
                result.append(None)
            elif isinstance(v, Quantity):
                result.append(float(v.value))
                if unit is None:
                    unit = str(v.unit)
            else:
                result.append(float(v))
        return {
            "value": result,
            "unit": unit,
        }

    @classmethod
    def as_balmer_params(cls, field: JSONField) -> BalmerModelParams:
        return BalmerModelParams(cls.as_str_set(field))

    @classmethod
    def undo_as_balmer_params(cls, value: BalmerModelParams) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_host_galaxy_params(cls, field: JSONField) -> HostGalaxyModelParams:
        return HostGalaxyModelParams(cls.as_str_set(field))

    @classmethod
    def undo_as_host_galaxy_params(cls, value: HostGalaxyModelParams) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_loader(cls, field: JSONField) -> str:
        assert field["value"] in {"fits", "ascii", "paqs", "sdss", "vito"}
        return cls.as_str(field)

    @classmethod
    def undo_as_loader(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_naming(cls, field: JSONField) -> str:
        val = cls.as_str(field).lower()
        assert val in {"igr", "j2000", "sdss"}
        return val

    @classmethod
    def undo_as_naming(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_deredden(cls, field: JSONField) -> tuple[bool, str, str, float]:
        perform = field["value"][0]
        _map = field["value"][1].lower()
        assert _map in {"sfd", "csfd"}
        _law = field["value"][2].lower()
        assert _law in {"ccm89", "o94"}
        _rv = float(field["value"][3])
        return (perform, _map, _law, _rv)

    @classmethod
    def undo_as_deredden(cls, value: tuple[bool, str, str, float]) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_bias(cls, field: JSONField) -> float:
        val = [v.lower() for v in cls.as_str_list(field)]
        assert all(v in {"left", "right"} for v in val)
        return val

    @classmethod
    def undo_as_bias(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_split_scale(cls, field: JSONField) -> float | Quantity:
        return cls.as_velocity(field)

    @classmethod
    def undo_as_split_scale(cls, value: float | Quantity) -> JSONField:
        return cls.undo_as_velocity(value)

    @classmethod
    def as_algo(cls, field: JSONField) -> str:
        val = cls.as_str(field).lower()
        assert val in {"trf", "dogbox", "lm"}
        return val

    @classmethod
    def undo_as_algo(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_tol(cls, field: JSONField) -> float:
        return max(cls.as_float(field), MACHINE_PRECISION)

    @classmethod
    def undo_as_tol(cls, value: float) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_random_state(cls, field: JSONField) -> RandomState:
        return RandomState(cls.as_int(field))

    @classmethod
    def undo_as_random_state(cls, value: RandomState) -> JSONField:
        return {
            "value": value.get_state()[2],
            "unit": None,
        }

    ### ErrorInfo

    @classmethod
    def as_method(cls, field: JSONField) -> str:
        val = cls.as_str(field).lower()
        assert val in {"bootstrap"}
        return val

    @classmethod
    def undo_as_method(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_scale(cls, field: JSONField) -> float:
        val = cls.as_str(field).lower()
        assert val in {"global", "semilocal", "local"}
        return val

    @classmethod
    def undo_as_scale(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_variant(cls, field: JSONField) -> str:
        val = cls.as_str(field).lower()
        assert val in {"spectrum", "standard", "flexible", "rigid"}
        return val

    @classmethod
    def undo_as_variant(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_bootstrap_type(cls, field: JSONField) -> str:
        val = cls.as_str(field).lower()
        assert val in {"spectrum", "model", "frequentist"}
        return val

    @classmethod
    def undo_as_bootstrap_type(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_fwhm_strategy(cls, field: JSONField) -> str:
        val = cls.as_str(field).lower()
        assert val in {"average", "narrowest", "widest"}
        return val

    @classmethod
    def undo_as_fwhm_strategy(cls, value: str) -> JSONField:
        return {
            "value": value,
            "unit": None,
        }

    @classmethod
    def as_vary_lines(cls, field: JSONField) -> VaryLines:
        return VaryLines(cls.as_str_set(field))

    @classmethod
    def undo_as_vary_lines(cls, value: VaryLines) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_out_lines(cls, field: JSONField) -> OutLines:
        return OutLines(cls.as_str_set(field))

    @classmethod
    def undo_as_out_lines(cls, value: OutLines) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    @classmethod
    def as_out_measures(cls, field: JSONField) -> OutMeasures:
        return OutMeasures(cls.as_str_set(field))

    @classmethod
    def undo_as_out_measures(cls, value: OutMeasures) -> JSONField:
        return {
            "value": list(value),
            "unit": None,
        }

    ###

    @classmethod
    def keys(cls) -> list[str]:
        return [
            name.removeprefix("as_")
            for name, _ in getmembers(cls, predicate=ismethod)
            if name.startswith("as_")
        ]

    @classmethod
    def undo_keys(cls) -> list[str]:
        return [
            name.removeprefix("undo_as_")
            for name, _ in getmembers(cls, predicate=ismethod)
            if name.startswith("undo_as_")
        ]

    ###

    @classmethod
    def _parse(cls, key: str) -> Callable[[JSONField], Any]:
        _key = key.removeprefix("as_")
        keys = cls.keys()
        if _key not in keys:
            raise ValueError(_key)
        return getattr(cls, f"as_{_key}")

    @classmethod
    def parse(cls, key: str, field: JSONField) -> Any:
        return cls._parse(key)(field)

    @classmethod
    def _undo(cls, key: str) -> Callable[[Any], JSONField]:
        _key = key.removeprefix("undo_as_")
        keys = cls.undo_keys()
        if _key not in keys:
            raise ValueError(_key)
        return getattr(cls, f"undo_as_{_key}")

    @classmethod
    def undo(cls, key: str, value: Any) -> JSONField:
        return cls._undo(key)(value)

PARSER_KEYS: list[str] = Parser.keys()
PARSER_UNDO_KEYS: list[str] = Parser.undo_keys()