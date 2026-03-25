from logging import getLogger
logger = getLogger(__name__)

from typing import ClassVar, Self, Iterable
from astropy.units import Unit, Quantity
from astropy.constants import c, h, k_B

from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import check_val
from ..utils import parsing
from ..utils.parsing import get_lines_from_file

from quasar_typing.astropy import Unit_, Quantity_, CompositeUnit_
from quasar_typing.pathlib import AbsoluteFilePath

class UnitsInfo(_Info):
    _keys: ClassVar[frozenset[str]] = frozenset([
        'wavelength_unit', 'energy_unit', 'time_unit', 'area_unit', 
        'temp_unit', 'dens_unit','c_unit', 'velocity_unit', 
        'wavelength_format', 'velocity_format', 
        'flux_format', 'strength_format', 'other_format',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {}

    @validate_call(validate_return=False)
    def __init__(
        self,
        wavelength_unit: CompositeUnit_ = Unit('1 angstrom'),
        energy_unit: CompositeUnit_ = Unit('1e-17 erg'),
        time_unit: CompositeUnit_ = Unit('1 s'),
        area_unit: CompositeUnit_ = Unit('1 cm2'),
        temp_unit: CompositeUnit_ = Unit('1 K'),
        dens_unit: CompositeUnit_ = Unit('1 cm^-3'),
        c_unit: CompositeUnit_ = Unit(c),
        velocity_unit: CompositeUnit_ = Unit('1 km/s'),
        wavelength_format: str = '.3f',
        velocity_format: str = '.3f',
        flux_format: str = '.3e',
        strength_format: str = '.3e',
        other_format: str = '.3e',
    ):
        self.wavelength_unit: CompositeUnit_ = wavelength_unit
        self.energy_unit: CompositeUnit_ = energy_unit
        self.time_unit: CompositeUnit_ = time_unit
        self.area_unit: CompositeUnit_ = area_unit
        self.temp_unit: CompositeUnit_ = temp_unit
        self.dens_unit: CompositeUnit_ = dens_unit
        self.c_unit: CompositeUnit_ = c_unit
        self.velocity_unit: CompositeUnit_ = velocity_unit
        self.wavelength_format: str = wavelength_format
        self.velocity_format: str = velocity_format
        self.flux_format: str = flux_format
        self.strength_format: str = strength_format
        self.other_format: str = other_format

    def update(self, info) -> None:
        super().update(info, logger)
        # logger.debug("Updating 'UnitsInfo' class (does nothing).")
    
    def getFluxUnit(self) -> CompositeUnit_:
        return self.getStrengthUnit() / self['wavelength_unit']
    
    def getStrengthUnit(self) -> CompositeUnit_:
        return self['energy_unit'] / self['time_unit'] / self['area_unit']

    @classmethod
    @validate_call(validate_return=False)
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache.keys():
            logger.debug(f"Using cached 'UnitsInfo' for '{path}'.")
            
            uinfo = cls._cache[str(path)]
            if create_copy: return uinfo.copy()
            else:           return uinfo
        
        uinfo: UnitsInfo = UnitsInfo()
        if path is None:
            return uinfo
                
        logger.debug(f"Configuring 'UnitsInfo' using '{path}'.")
        lines = get_lines_from_file.__wrapped__('UNITS', path, logger)

        for count, line in enumerate(lines, start=1):
            key = line.pop(0).lower()
            match key.split('_')[1]:
                case 'unit':   val = parsing.as_composite_unit(line)
                case 'format': val = line[0]
            
            uinfo[key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] Configured '{key}' as '{val}'."
            )

        cls._cache[str(path)] = uinfo

        return uinfo
    
    @classmethod
    @validate_call(validate_return=False)
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "units", logger)

    ###

    def getFormat(
        self,
        dimension: str,
        power: float | int = 1,
    ) -> str:
        
        if power != 1: return self['other_format']

        match dimension:
            case 'wavelength':     return self['wavelength_format']
            case 'velocity' | 'c': return self['velocity_format']
            case 'flux':           return self['flux_format']
            case 'strength':       return self['strength_format']
            case _:                return self['other_format']
    ###

    @staticmethod
    def _get_transformed_value(
        unit: CompositeUnit_,
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | Iterable:
        """
        ...
        """        
        val = check_val(val)
        is_quantity = isinstance(val, Quantity)
        unit = unit**power

        if is_quantity: return val.to(unit).value
        else:           return (val * unit).to(unit)
    
    def getWavelength(
        self, 
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | Iterable:
        """
        ...
        """
        return self._get_transformed_value(
            self['wavelength_unit'], val, power=power
        )
    
    def getC(
        self,
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | Iterable:
        """
        ...
        """
        return self._get_transformed_value(
            self['c_unit'], val, power=power
        )
    
    def getVelocity(
        self, 
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | int | Iterable:
        """
        ...
        """
        return self._get_transformed_value(
            self['velocity_unit'], val, power=power
        )
    
    def getFlux(
        self, 
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | int | Iterable:
        """
        ...
        """
        return self._get_transformed_value(
            self.getFluxUnit(), val, power=power
        )
    
    def getStrength(
        self, 
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | int | Iterable:
        """
        ...
        """
        return self._get_transformed_value(
            self.getStrengthUnit(), val, power=power
        )

    def getDensity(
        self,
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | Iterable:
        """
        ...
        """
        return self._get_transformed_value(
            self['dens_unit'], val, power=power
        )

    def getTemperature(
        self,
        val: Quantity_ | float | int | Iterable,
        power: float | int = 1,
    ) -> Quantity_ | float | Iterable:
        """
        ...
        """
        return self._get_transformed_value(
            self['temp_unit'], val, power=power
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
        return (h * c / k_B) \
            .to(self['temp_unit'] * self['wavelength_unit']) \
            .value
    
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
        if dimension == 'c':
            val = self.getC(val, power=power)
            val = self.getVelocity(val, power=power)
            dimension = 'velocity'

        unit = self[f"{dimension}_unit"] \
            if dimension in ['wavelength', 'velocity'] \
            else getattr(self, f"get{dimension.capitalize()}Unit")()
        unit **= power

        return val, unit