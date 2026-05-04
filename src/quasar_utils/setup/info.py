__all__ = ['Info']

from logging import getLogger
from typing import Iterable, Any, Self, ClassVar
from json import load as load_json
from numpy import array, float64
from numpy.typing import NDArray
from astropy.units import Quantity
from dataclasses import field

from .absorption import AbsorptionInfo
from .balmer import BalmerInfo
from .continuum import ContinuumInfo
from .error import ErrorInfo
from .iron import IronInfo
from .lines import LinesInfo
from .loading import LoadingInfo
from .nonlinear import NonLinearInfo
from .units import UnitsInfo

from pydantic import validate_call
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

@dataclass
class Info:
    absorption: AbsorptionInfo = field(kw_only=True, default_factory=AbsorptionInfo)
    balmer: BalmerInfo = field(kw_only=True, default_factory=BalmerInfo)
    continuum: ContinuumInfo = field(kw_only=True, default_factory=ContinuumInfo)
    error: ErrorInfo = field(kw_only=True, default_factory=ErrorInfo)
    iron: IronInfo = field(kw_only=True, default_factory=IronInfo)
    lines: LinesInfo = field(kw_only=True, default_factory=LinesInfo)
    loading: LoadingInfo = field(kw_only=True, default_factory=LoadingInfo)
    nonlinear: NonLinearInfo = field(kw_only=True, default_factory=NonLinearInfo)
    units: UnitsInfo = field(kw_only=True, default_factory=UnitsInfo)

    _keys: ClassVar[frozenset[str]] = frozenset([
        'absorption', 'balmer', 'continuum', 'error', 'iron', 'lines', 
        'loading', 'nonlinear', 'units',
    ])

    def __post_init__(self) -> None:
        self.update()

    @classmethod
    @validate_call
    def from_file(
        cls, 
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        if path is None: 
            return Info()

        kwargs = {'path': path, 'create_copy': create_copy}
        def inner(cls):
            nonlocal kwargs
            return cls.from_file.__wrapped__(cls, **kwargs)

        return Info(
            absorption=inner(AbsorptionInfo),
            balmer=inner(BalmerInfo),
            continuum=inner(ContinuumInfo),
            error=inner(ErrorInfo),
            iron=inner(IronInfo),
            lines=inner(LinesInfo),
            loading=inner(LoadingInfo),
            nonlinear=inner(NonLinearInfo),
            units=inner(UnitsInfo),
        )
    
    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        
        if json is None:
            return Info()
        
        if not isinstance(json, dict):
            with open(json, 'r') as f:
                json = load_json(f)

        kwargs = {'json': json, 'create_copy': create_copy}
        def inner(cls):
            nonlocal kwargs
            return cls.from_json.__wrapped__(cls, **kwargs)

        return Info(
            absorption=inner(AbsorptionInfo),
            balmer=inner(BalmerInfo),
            continuum=inner(ContinuumInfo),
            error=inner(ErrorInfo),
            iron=inner(IronInfo),
            lines=inner(LinesInfo),
            loading=inner(LoadingInfo),
            nonlinear=inner(NonLinearInfo),
            units=inner(UnitsInfo),
        )

    def __getstate__(self) -> dict:
        state: dict = {'_keys': self._keys}
        state.update({key: getattr(self, key) for key in self._keys})
        return state
    
    def __setstate__(self, state: dict) -> None:
        self._keys: frozenset[str] = state.pop('_keys')
        for key, value in state.items():
            setattr(self, key, value)

    def update(self) -> None:
        if self.is_updated:
            logger.debug("'Info' class is already updated!")
        else:
            self.force_update()

    def force_update(self) -> None:
        logger.debug("Updating 'Info' class.")
        self.loading.update(self)
        for key in self._keys:
            getattr(self, key).update(self)

    @property
    def is_updated(self) -> bool:
        return all(getattr(self, key).is_updated for key in self._keys)

    def __bool__(self) -> bool:
        return self.is_updated
        
    def __getitem__(
        self, 
        key: str, 
        subjects: Iterable[str] = [
            'absorption', 'balmer', 'continuum', 'error', 'iron', 'lines', 
            'loading', 'nonlinear', 'units',
            # 'plotting', 
        ],
    ) -> Any:
        result = None
        for subinfo in (getattr(self, subject) for subject in subjects):
            if key in subinfo._keys: 
                result = getattr(subinfo, key)
                break

        return result
    
    def update_value(
        self,
        value: Any,
        conversion_name: str,
    ) -> Any:
        
        uinfo = self.units
        
        match conversion_name, isinstance(value, Quantity):
            case "to_n_pixels", True:
                result: int = int(uinfo.getC(value) / self.loading['sigma_res'])
            case "to_n_pixels", False:
                result: int = int(value)

            case "to_wavelength", True:
                result: float = uinfo.getWavelength(value)
            case "to_wavelength", False:
                result: float = float(value)

            case "to_wavelength_bounds", _:
                result: tuple[float | None, float | None] = tuple(
                    None if b is None else self.update_value(b, "to_wavelength")
                    for b in value
                )

            case "to_wavelength_bounds", _:
                result: tuple[float | None, float | None] = tuple(
                    None if b is None else float(b)
                    for b in value
                )

            case "to_wavelength_list", _:
                result: list[float] = [
                    self.update_value(v, "to_wavelength") for v in value
                ]

            case "to_wavelength_array", True:
                result: NDArray[float64] = uinfo.getWavelength(value)
            case "to_wavelength_array", False:
                result: NDArray[float64] = array(value, dtype=float64)

            case "to_wavelength_windows", True:
                result: list[tuple[float | None, float | None]] = list(map(
                    tuple, 
                    uinfo.getWavelength(value),
                ))
            case "to_wavelength_windows", False:
                result: list[tuple[float | None, float | None]] = list(map(
                    tuple, 
                    value,
                ))

            case "to_density", True:
                result: float = uinfo.getDensity(value)
            case "to_density", False:
                result: float = float(value)

            case "to_temperature", True:
                result: float = uinfo.getTemperature(value)
            case "to_temperature", False:
                result: float = float(value)

            case "to_temperature_bounds", _:
                result: tuple[float | None, float | None] = tuple(
                    None if b is None else self.update_value(b, "to_temperature")
                    for b in value
                )

            case "to_flux", True:
                result: float = uinfo.getFlux(value)
            case "to_flux", False: 
                result: float = float(value)

            case "to_flux_bounds", _:
                result: tuple[float | None, float | None] = tuple(
                    None if b is None else self.update_value(b, "to_flux")
                    for b in value
                )

            case "to_strength", True:
                result: float = uinfo.getStrength(value)
            case "to_strength", False:
                result: float = float(value)

            case "to_strength_bounds", _:
                result: tuple[float | None, float | None] = tuple(
                    None if b is None else self.update_value(b, "to_strength")
                    for b in value
                )

            case "to_velocity", _:
                if not isinstance(value, Quantity):
                    value *= uinfo["velocity_unit"]
                result: float = uinfo.getC(value)

            case "to_velocity_bounds", _:
                result: tuple[float | None, float | None] = tuple(
                    None if b is None else self.update_value(b, "to_velocity")
                    for b in value
                )

            case "to_scale", _:
                result: float = self.update_value(value, "to_velocity") \
                    / self.loading['sigma_res']
                
            case "to_scale_list", _:
                result: list[float] = [
                    self.update_value(v, "to_scale") 
                    for v in value
                ]

            case "to_fixed", _:
                result: dict[str, bool] = {
                    key: value(key)
                    for key in ('fwhm', 'temp', 'tau', 'scale', 'ratio')
                }

        return result