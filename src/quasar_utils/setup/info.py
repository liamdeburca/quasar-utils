__all__ = ['Info']

from logging import getLogger
from typing import Iterable, Any, Self, ClassVar

from .absorption import AbsorptionInfo
from .balmer import BalmerInfo
from .continuum import ContinuumInfo
from .error import ErrorInfo
from .iron import IronInfo
from .lines import LinesInfo
from .loading import LoadingInfo
from .nonlinear import NonLinearInfo
from .units import UnitsInfo

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
        absorption_info: AbsorptionInfo | None = None,
        balmer_info: BalmerInfo | None = None,
        continuum_info: ContinuumInfo | None = None,
        error_info: ErrorInfo | None = None,
        iron_info: IronInfo | None = None,
        lines_info: LinesInfo | None = None,
        loading_info: LoadingInfo | None = None,
        nonlinear_info: NonLinearInfo | None = None,
        units_info: UnitsInfo | None = None,
        *,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.absorption: AbsorptionInfo = (
            AbsorptionInfo.from_file(path=path, create_copy=create_copy)
            if absorption_info is None \
            else absorption_info \
        )
        self.balmer: BalmerInfo = (
            BalmerInfo.from_file(path=path, create_copy=create_copy)
            if balmer_info is None \
            else balmer_info \
        )
        self.continuum: ContinuumInfo = (
            ContinuumInfo.from_file(path=path, create_copy=create_copy)
            if continuum_info is None \
            else continuum_info \
        )
        self.error: ErrorInfo = (
            ErrorInfo.from_file(path=path, create_copy=create_copy)
            if error_info is None \
            else error_info \
        )
        self.iron: IronInfo = (
            IronInfo.from_file(path=path, create_copy=create_copy)
            if iron_info is None \
            else iron_info \
        )
        self.lines: LinesInfo = (
            LinesInfo.from_file(path=path, create_copy=create_copy)
            if lines_info is None \
            else lines_info \
        )
        self.loading: LoadingInfo = (
            LoadingInfo.from_file(path=path, create_copy=create_copy)
            if loading_info is None \
            else loading_info \
        )
        self.nonlinear: NonLinearInfo = (
            NonLinearInfo.from_file(path=path, create_copy=create_copy)
            if nonlinear_info is None \
            else nonlinear_info \
        )
        self.units: UnitsInfo = (
            UnitsInfo.from_file(path=path, create_copy=create_copy)
            if units_info is None \
            else units_info \
        )
        self.update()

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