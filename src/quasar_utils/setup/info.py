__all__ = ['Info']

from logging import getLogger
from typing import Iterable, Any, Self, ClassVar
from json import load as load_json
from numpy import array, float64
from numpy.typing import NDArray
from astropy.units import Quantity

from .absorption import AbsorptionInfo
from .balmer import BalmerInfo
from .continuum import ContinuumInfo
from .error import ErrorInfo
from .iron import IronInfo
from .lines import LinesInfo
from .loading import LoadingInfo
from .nonlinear import NonLinearInfo
from .units import UnitsInfo
from ..fitting import TRFLSQFitter, DogBoxLSQFitter, LMLSQFitter

logger = getLogger(__name__)

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.pathlib import AbsoluteFilePath

class Info:
    _keys: ClassVar[frozenset[str]] = frozenset([
        'absorption', 'balmer', 'continuum', 'error', 'iron', 'lines', 
        'loading', 'nonlinear', 'units',
    ])

    @validate_call(validate_return=False)
    def __init__(
        self,
        *,
        absorption_info: AbsorptionInfo | None = None,
        balmer_info: BalmerInfo | None = None,
        continuum_info: ContinuumInfo | None = None,
        error_info: ErrorInfo | None = None,
        iron_info: IronInfo | None = None,
        lines_info: LinesInfo | None = None,
        loading_info: LoadingInfo | None = None,
        nonlinear_info: NonLinearInfo | None = None,
        units_info: UnitsInfo | None = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.absorption: AbsorptionInfo = (
            absorption_info
            if absorption_info is not None else
            AbsorptionInfo()
        )
        self.balmer: BalmerInfo = (
            balmer_info
            if balmer_info is not None else
            BalmerInfo()
        )
        self.continuum: ContinuumInfo = (
            continuum_info
            if continuum_info is not None else
            ContinuumInfo()
        )
        self.error: ErrorInfo = (
            error_info
            if error_info is not None else
            ErrorInfo()
        )
        self.iron: IronInfo = (
            iron_info
            if iron_info is not None else
            IronInfo()
        )
        self.lines: LinesInfo = (
            lines_info
            if lines_info is not None else
            LinesInfo()
        )
        self.loading: LoadingInfo = (
            loading_info
            if loading_info is not None else
            LoadingInfo()
        )
        self.nonlinear: NonLinearInfo = (
            nonlinear_info
            if nonlinear_info is not None else
            NonLinearInfo()
        )
        self.units: UnitsInfo = (
            units_info
            if units_info is not None else
            UnitsInfo()
        )
        self.update()

    @classmethod
    @validate_call(validate_return=False)
    def from_file(
        cls, 
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        if path is None: 
            return Info()

        kwargs = {'path': path, 'create_copy': create_copy}
        inner = lambda cls: cls.from_file.__wrapped__(cls, **kwargs)

        return Info(
            absorption_info=inner(AbsorptionInfo),
            balmer_info=inner(BalmerInfo),
            continuum_info=inner(ContinuumInfo),
            error_info=inner(ErrorInfo),
            iron_info=inner(IronInfo),
            lines_info=inner(LinesInfo),
            loading_info=inner(LoadingInfo),
            nonlinear_info=inner(NonLinearInfo),
            units_info=inner(UnitsInfo),
        )
    
    @classmethod
    @validate_call(validate_return=False)
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
        inner = lambda cls: cls.from_json.__wrapped__(cls, **kwargs)
        
        return Info(
            absorption_info=inner(AbsorptionInfo),
            balmer_info=inner(BalmerInfo),
            continuum_info=inner(ContinuumInfo),
            error_info=inner(ErrorInfo),
            iron_info=inner(IronInfo),
            lines_info=inner(LinesInfo),
            loading_info=inner(LoadingInfo),
            nonlinear_info=inner(NonLinearInfo),
            units_info=inner(UnitsInfo),
        )

    @classmethod
    def _validate(cls, value: object) -> Self:
        if value is None: value = Info()

        if not isinstance(value, Info):
            msg = f"Expected an Info instance, got {type(value).__name__}"
            raise PydanticCustomError('validation_error', msg)

        if not value.is_updated:
            value.update()

        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)

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
                result = subinfo[key]
                break

        return result
    
    def update_value(
        self,
        value: Any,
        conversion_name: str,
    ) -> Any:
        
        uinfo = self.units
        
        match conversion_name:
            case "to_n_pixels":
                result: int = (
                    int(uinfo.getC(value) / self.loading['sigma_res'])
                    if isinstance(value, Quantity) else 
                    int(value)
                )
            case "to_wavelength":
                result: float = (
                    uinfo.getWavelength(value) 
                    if isinstance(value, Quantity) else 
                    float(value)
                )
            case "to_wavelength_bounds":
                result: tuple[float | None, float | None] = (
                    None if b is None else self.update_value(b, "to_wavelength")
                    for b in value
                )
            case "to_wavelength_list":
                result: list[float] = [
                    self.update_value(v, "to_wavelength") for v in value
                ]
            case "to_wavelength_array":
                result: NDArray[float64] = (
                    uinfo.getWavelength(value)
                    if isinstance(value, Quantity) else
                    array(value, dtype=float64)
                )
            case "to_wavelength_windows":
                result: list[tuple[float | None, float | None]] = list(map(
                    tuple, 
                    uinfo.getWavelength(value) if isinstance(value, Quantity) else value
                ))
            case "to_density":
                result: float = (
                    uinfo.getDensity(value)
                    if isinstance(value, Quantity) else 
                    float(value)
                )
            case "to_temperature":
                result: float = (
                    uinfo.getTemperature(value)
                    if isinstance(value, Quantity) else
                    float(value)
                )
            case "to_temperature_bounds":
                result: tuple[float | None, float | None] = (
                    None if b is None else self.update_value(b, "to_temperature")
                    for b in value
                )
            case "to_flux":
                result: float = (
                    uinfo.getFlux(value)
                    if isinstance(value, Quantity) else 
                    float(value)
                )
            case "to_flux_bounds":
                result: tuple[float | None, float | None] = (
                    None if b is None else self.update_value(b, "to_flux")
                    for b in value
                )
            case "to_strength":
                result: float = (
                    uinfo.getStrength(value)
                    if isinstance(value, Quantity) else 
                    float(value)
                )
            case "to_strength_bounds":
                result: tuple[float | None, float | None] = (
                    None if b is None else self.update_value(b, "to_strength")
                    for b in value
                )
            case "to_velocity":
                if not isinstance(value, Quantity):
                    value *= uinfo["velocity_unit"]
                result: float = uinfo.getC(value)

            case "to_velocity_bounds":
                result: tuple[float | None, float | None] = (
                    None if b is None else self.update_value(b, "to_velocity")
                    for b in value
                )
            case "to_scale":
                result: float = self.update_value(value, "to_velocity") \
                    / self.loading['sigma_res']
            case "to_scale_list":
                result: list[float] = [
                    self.update_value(v, "to_scale") 
                    for v in value
                ]
            case "to_fixed":
                result: dict[str, bool] = {
                    'fwhm': 'fwhm' in value,
                    'temp': 'temp' in value,
                    'tau': 'tau' in value,
                    'scale': 'scale' in value,
                    'ratio': 'ratio' in value,
                }
            case "to_algo":
                match value:
                    case "trf": result = TRFLSQFitter
                    case "dogbox": result = DogBoxLSQFitter
                    case "lm": result = LMLSQFitter
                    case _: raise ValueError(f"Unknown algorithm: {value}")
        return result