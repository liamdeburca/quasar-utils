from collections.abc import Iterable
from logging import getLogger
from typing import Any, Literal, Self

from astropy.constants import c, h, k_B
from astropy.units import Quantity, Unit
from numpy import array2string
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import CompositeUnit_, Quantity_, Unit_
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from ..utils.utils import check_val
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass(additional_keys=["strength_unit", "flux_unit"])
@dataclass
class UnitsInfo(_Info):
    wavelength_unit: CompositeUnit_ = field(
        default=Unit("1 angstrom"),
        desc="Choice of wavelength unit",
        dtype="str",
        parse_as="composite_unit",
    )
    energy_unit: CompositeUnit_ = field(
        default=Unit("1e-17 erg"),
        desc="Choice of energy unit",
        dtype="str",
        parse_as="composite_unit",
    )
    time_unit: CompositeUnit_ = field(
        default=Unit("1 s"),
        desc="Choice of time unit",
        dtype="str",
        parse_as="composite_unit",
    )
    area_unit: CompositeUnit_ = field(
        default=Unit("1 cm2"),
        desc="Choice of area unit",
        dtype="str",
        parse_as="composite_unit",
    )
    temp_unit: CompositeUnit_ = field(
        default=Unit("1 K"),
        desc="Choice of temperature unit",
        dtype="str",
        parse_as="composite_unit",
    )
    dens_unit: CompositeUnit_ = field(
        default=Unit("1 cm^-3"),
        desc="Choice of density unit",
        dtype="str",
        parse_as="composite_unit",
    )
    c_unit: CompositeUnit_ = field(
        default=Unit(c),
        desc="Choice of speed-of-light unit",
        dtype="str",
        parse_as="composite_unit",
    )
    velocity_unit: CompositeUnit_ = field(
        default=Unit("1 km/s"),
        desc="Choice of velocity unit",
        dtype="str",
        parse_as="composite_unit",
    )
    wavelength_format: str = field(
        default=".1f",
        desc="How to format wavelength values",
        dtype="str",
        parse_as="str",
    )
    velocity_format: str = field(
        default=".1f",
        desc="How to format velocity values",
        dtype="str",
        parse_as="str",
    )
    flux_format: str = field(
        default=".1f",
        desc="How to format flux density values",
        dtype="str",
        parse_as="str",
    )
    strength_format: str = field(
        default=".1f",
        desc="How to format strength (integrated flux density) values",
        dtype="str",
        parse_as="str",
    )
    other_format: str = field(
        default=".1f",
        desc="How to format other values",
        dtype="str",
        parse_as="str",
    )

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info) -> None:
        super().update(info, logger)

    @property
    def strength_unit(self) -> CompositeUnit_:
        return self.energy_unit / self.time_unit / self.area_unit

    @property
    def flux_unit(self) -> CompositeUnit_:
        return self.strength_unit / self.wavelength_unit

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["units"], dict[str, Any]]:
        return super().to_dict(
            "units", 
            blacklist=["strength_unit", "flux_unit"], 
            jsonify=jsonify,
        )

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "units", logger)

    ###

    def getFormat(
        self,
        dimension: str,
        power: float = 1,
    ) -> str:

        if power != 1:
            return self["other_format"]

        match dimension:
            case "wavelength":
                return self["wavelength_format"]
            case "velocity" | "c":
                return self["velocity_format"]
            case "flux":
                return self["flux_format"]
            case "strength":
                return self["strength_format"]
            case _:
                return self["other_format"]

    ###

    @staticmethod
    def _get_transformed_value(
        unit: CompositeUnit_,
        val: Quantity_ | float | Iterable,
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        Transforms a float or iterable of floats into a Quantity, or a Quantity
        into a float or numpy array of floats. 
        """
        val = check_val(val)
        unit = unit**power
        return val.to(unit).value \
            if isinstance(val, Quantity) \
            else (val * unit).to(unit)

    @staticmethod
    def _get_dimensioned_value(
        unit: CompositeUnit_,
        val: Quantity_ | float | Iterable,
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        Returns a Quantity-representation of the input in the correct units. If 
        the input is a scalar or iterable of scalars, it is multiplied by the 
        specified (powered) unit.
        """
        val = check_val(val)
        _unit = unit ** power
        if not isinstance(val, Quantity):
            val *= _unit
        return val.to(_unit)

    @staticmethod
    def _get_unitless_value(
        unit: CompositeUnit_,
        val: Quantity_ | float | Iterable,
        power: float = 1,
    ) -> float | Iterable[float]:
        """
        Returns the scalar value of a float, iterable of floats, or Quantity in 
        the correct units. 
        """
        val = check_val(val)
        if isinstance(val, Quantity):
            unit = unit**power
            return val.to(unit).value
        return val

    @staticmethod
    def _get_formatted_value(
        val: Quantity_ | float | Iterable[float],
        unit: CompositeUnit_,
        fmt: str,
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        """
        Get the formatted string representation of a value.

        Parameters
        ----------
        val : Quantity_ | float | Iterable[float]
            The value to format.
        unit : CompositeUnit_
            The unit of the value.
        fmt : str
            The format string.
        power : float, optional
            The power to raise the unit to, by default 1.0
        with_unit : bool, optional
            Whether to include the unit in the formatted string, by default True

        Returns
        -------
        s : str
            The formatted string representation of the value.
        """
        _unit = unit ** power
        if not isinstance(val, Quantity):
            val *= _unit

        if with_unit:
            return val.to_string(
                format="latex_inline",
                formatter=fmt,
            )

        if isinstance(val, Iterable):
            return array2string(
                val.to(_unit).value, 
                formatter={'all': lambda v: format(v, fmt)}, 
                sign="+",
            )
        return format(val.to(_unit).value, fmt)

    def formatUnitless(
        self,
        val: Quantity_ | float | Iterable[float],
        with_unit: bool = True,
    ) -> str:
        return self._get_formatted_value(
            val, 
            Unit(), 
            self.other_format, 
            with_unit=with_unit,
        )

    ###

    def getWavelength(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        ...
        """
        return self._get_transformed_value(self.wavelength_unit, val, power=power)

    def getUnitlessWavelength(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self._get_unitless_value(self.wavelength_unit, val, power=power)

    def formatWavelength(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        return self._get_formatted_value(
            val,
            self.wavelength_unit,
            self.wavelength_format,
            power=power,
            with_unit=with_unit,
        )

    ###

    def getC(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        ...
        """
        return self._get_transformed_value(self.c_unit, val, power=power)

    def getDimensionedC(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_:
        return self._get_dimensioned_value(self.c_unit, val, power=power)

    def getUnitlessC(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self._get_unitless_value(self.c_unit, val, power=power)

    def getUnitlessCFromVelocity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self.getUnitlessC(
            self.getDimensionedVelocity(val, power=power), 
            power=power,
        )

    def formatC(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        if not isinstance(val, Quantity):
            val *= c ** power
        return self.formatVelocity(val, power=power, with_unit=with_unit)

    ###

    def getVelocity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        ...
        """
        return self._get_transformed_value(self.velocity_unit, val, power=power)

    def getDimensionedVelocity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_:
        return self._get_dimensioned_value(self.velocity_unit, val, power=power)

    def getUnitlessVelocity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self._get_unitless_value(self.velocity_unit, val, power=power)

    def formatVelocity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        return self._get_formatted_value(
            val,
            self.velocity_unit,
            self.velocity_format,
            power=power,
            with_unit=with_unit,
        )

    ###

    def getFlux(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        ...
        """
        return self._get_transformed_value(self.flux_unit, val, power=power)

    def getDimensionedFlux(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_:
        return self._get_dimensioned_value(self.flux_unit, val, power=power)

    def getUnitlessFlux(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self._get_unitless_value(self.flux_unit, val, power=power)

    def formatFlux(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        return self._get_formatted_value(
            val,
            self.flux_unit,
            self.flux_format,
            power=power,
            with_unit=with_unit,
        )

    ###

    def getStrength(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        ...
        """
        return self._get_transformed_value(self.strength_unit, val, power=power)

    def getDimensionedStrength(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_:
        return self._get_dimensioned_value(self.strength_unit, val, power=power)

    def getUnitlessStrength(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self._get_unitless_value(self.strength_unit, val, power=power)

    def formatStrength(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        return self._get_formatted_value(
            val,
            self.strength_unit,
            self.strength_format,
            power=power,
            with_unit=with_unit,
        )

    ###

    def getDensity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        ...
        """
        return self._get_transformed_value(self.dens_unit, val, power=power)

    def getDimensionedDensity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_:
        return self._get_dimensioned_value(self.dens_unit, val, power=power)

    def getUnitlessDensity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self._get_unitless_value(self.dens_unit, val, power=power)

    def formatDensity(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        return self._get_formatted_value(
            val,
            self.dens_unit,
            self.other_format,
            power=power,
            with_unit=with_unit,
        )

    ###

    def getTemperature(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_ | float | Iterable[float]:
        """
        ...
        """
        return self._get_transformed_value(self.temp_unit, val, power=power)

    def getDimensionedTemperature(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> Quantity_:
        return self._get_dimensioned_value(self.temp_unit, val, power=power)

    def getUnitlessTemperature(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1,
    ) -> float | Iterable[float]:
        return self._get_unitless_value(self.temp_unit, val, power=power)

    def formatTemperature(
        self,
        val: Quantity_ | float | Iterable[float],
        power: float = 1.0,
        with_unit: bool = True,
    ) -> str:
        return self._get_formatted_value(
            val,
            self.temp_unit,
            self.other_format,
            power=power,
            with_unit=with_unit,
        )

    def getBoltzmannFactor(self) -> float:
        """
        Returns a unitless constant often used in a Planck function:
            boltz = h * c / k_B.

        Notes
        -----
        Although 'boltz' is unitless, its true units are:
            [boltz] = [temperature] x [wavelength]
        """
        return (h * c / k_B).to(self.temp_unit * self.wavelength_unit).value

    def getCorrespondingUnit(
        self,
        val: float | list,
        dimension_of: tuple[str, float | int] | None = None,
    ) -> tuple[float | list, Unit_ | None]:
        """
        Takes a single or multiple values, and convert them into output-ready
        formats, i.e. in the designated output units.

        Parameters
        ----------
        val : float, numpy.array
            Single value, or array of values, to transform (in dimensionless
            units).

        dimension_of : tuple, optional
            Tuple containing the true dimension (str) of the dimensionless
            value, and the power (int). If None, corresponds to the true value
            having no dimensions, e.g. kurtosis.

        Returns
        -------
        val : float, astropy.Quantity, numpy.array
            One value or array of values given in the designated output unit. Is
            identical to the input 'val' parameter, except when the true
            dimension is 'c' (speed of light), in which case the value is
            transformed to a dimensioned quantity (velocity) and transformed
            back, given in units of the designated (velocity) output unit.

        unit : astropy.Unit, optional
            The designated output unit. None of the 'val' is inherently
            dimensionless or scalar, such as the kurtosis.

        """
        if dimension_of is None:
            return val, None

        dimension, power = dimension_of
        if dimension == "c":
            val = self.getC(val, power=power)
            val = self.getVelocity(val, power=power)
            dimension = "velocity"

        unit = (
            self[f"{dimension}_unit"]
            if dimension in ["wavelength", "velocity"]
            else getattr(self, f"get{dimension.capitalize()}Unit")()
        )
        unit **= power

        return val, unit
