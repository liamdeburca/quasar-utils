from collections.abc import Callable, Iterable
from inspect import getmembers, ismethod
from typing import Any

from astropy.units import Quantity
from numpy import array, float64, sort
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.misc import StringSelection
from quasar_typing.numpy import FloatVector, SortedFloatVector


class Updater:
    @classmethod
    def to_n_pixels(cls, info: object, value: int | Quantity) -> int:
        if isinstance(value, Quantity):
            value = int(info.units.getC(value) / info.loading.sigma_res)
        return value

    @classmethod
    def to_wavelength(cls, info: object, value: float | Quantity) -> float:
        if isinstance(value, Quantity):
            value = info.units.getWavelength(value)
        return value

    @classmethod
    def to_velocity(cls, info: object, value: float | Quantity) -> float:
        if not isinstance(value, Quantity):
            value *= info.units.velocity_unit
        return info.units.getC(value)

    @classmethod
    def to_temperature(cls, info: object, value: float | Quantity) -> float:
        if isinstance(value, Quantity):
            value = info.units.getTemperature(value)
        return value

    @classmethod
    def to_density(cls, info: object, value: float | Quantity) -> float:
        if isinstance(value, Quantity):
            value = info.units.getDensity(value)
        return value

    @classmethod
    def to_flux(cls, info: object, value: float | Quantity) -> float:
        if isinstance(value, Quantity):
            value = info.units.getFlux(value)
        return value

    @classmethod
    def to_velocity_bounds(cls, info: object, value: AstropyBounds | Iterable[Quantity | None]) -> AstropyBounds:
        tup = tuple(
            None 
                if b is None 
                else cls.to_velocity(info, b) 
            for b in value
        )
        assert len(tup) == 2
        return tup

    @classmethod
    def to_flux_bounds(cls, info: object, value: AstropyBounds | Iterable[Quantity | None]) -> AstropyBounds:
        tup = tuple(
            None 
                if b is None 
                else cls.to_flux(info, b) 
            for b in value
        )
        assert len(tup) == 2
        return tup

    @classmethod
    def to_fixed(cls, info: object, value: StringSelection) -> dict[str, bool]:
        return value.to_fixed()

    @classmethod
    def to_velocity_list(cls, info: object, value: Iterable[float | Quantity]) -> list[float]:
        return [cls.to_velocity(info, v) for v in value]

    @classmethod
    def to_sorted_velocity_array(cls, info: object, value: Iterable[float | Quantity]) -> SortedFloatVector:
        return sort(cls.to_velocity_list(info, value)).astype(float64, order='C')

    @classmethod
    def to_wavelength_list(cls, info: object, value: Iterable[float | Quantity]) -> list[float]:
        return [cls.to_wavelength(info, v) for v in value]

    @classmethod
    def to_wavelength_bounds(cls, info: object, value: Iterable[float | Quantity | None]) -> CoordBounds:
        tup = tuple(
            cls.to_wavelength(info, b) 
                if isinstance(b, Quantity)
                else b
            for b in value
        )
        assert len(tup) == 2
        return tup

    @classmethod
    def to_wavelength_array(cls, info: object, value: Iterable[float | Quantity]) -> FloatVector:
        return array(cls.to_wavelength_list(info, value)).astype(float64, order='C')

    @classmethod
    def to_sorted_wavelength_array(cls, info: object, value: Iterable[float | Quantity]) -> SortedFloatVector:
        return sort(cls.to_wavelength_list(info, value)).astype(float64, order='C')

    @classmethod
    def to_wavelength_windows(cls, info: object, value: Iterable[CoordBounds] | Quantity) -> list[CoordBounds]:
        if isinstance(value, Quantity):
            value = [
                tuple(window_bounds) 
                for window_bounds in info.units.getWavelength(value)
            ]
        return list(value)

    @classmethod
    def keys(cls) -> list[str]:
        return [
            name.removeprefix("to_")
            for name, _ in getmembers(cls, predicate=ismethod)
            if name.startswith("to_")
        ]

    @classmethod
    def __class_getitem__(cls, key: str) -> Callable[[object, Any], Any]:
        method_name = f"to_{key}"
        if hasattr(cls, method_name):
            return getattr(cls, method_name)
        raise KeyError(f"No updater method found for key '{key}'") 

UPDATER_KEYS: list[str] = Updater.keys()
