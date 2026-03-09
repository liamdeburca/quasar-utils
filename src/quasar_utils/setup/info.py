__all__ = ['Info']

from logging import getLogger
from typing import Iterable, Any, Self

from .absorption import AbsorptionInfo
from .balmer import BalmerInfo
from .continuum import ContinuumInfo
from .error import ErrorInfo
from .iron import IronInfo
from .lines import LinesInfo
from .loading import LoadingInfoß
from .nonlinear import NonLinearInfo
from .units import UnitsInfo

logger = getLogger(__name__)

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function
from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import AbsoluteFilePath

@dataclass
class Info:
    absorption:  AbsorptionInfo
    balmer:      BalmerInfo
    continuum:   ContinuumInfo
    error:       ErrorInfo
    iron:        IronInfo
    lines:       LinesInfo
    loading:     LoadingInfo
    nonlinear:   NonLinearInfo
    units:       UnitsInfo

    _keys: frozenset[str] = frozenset([
        'absorption', 'balmer', 'continuum', 'error', 'iron', 'lines', 
        'loading', 'nonlinear', 'units',
    ])
    
    @validate_call
    def __init__(
        self, 
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ):
        self.absorption = AbsorptionInfo.from_file.__wrapped__(
            AbsorptionInfo, path=path, create_copy=create_copy,
        )
        self.balmer = BalmerInfo.from_file.__wrapped__(
            BalmerInfo, path=path, create_copy=create_copy,
        )
        self.continuum = ContinuumInfo.from_file.__wrapped__(
            ContinuumInfo, path=path, create_copy=create_copy,
        )
        self.error = ErrorInfo.from_file.__wrapped__(
            ErrorInfo, path=path, create_copy=create_copy,
        )
        self.iron = IronInfo.from_file.__wrapped__(
            IronInfo, path=path, create_copy=create_copy,
        )
        self.lines = LinesInfo.from_file.__wrapped__(
            LinesInfo, path=path, create_copy=create_copy,
        )
        self.loading = LoadingInfo.from_file.__wrapped__(
            LoadingInfo, path=path, create_copy=create_copy,
        )
        self.nonlinear = NonLinearInfo.from_file.__wrapped__(
            NonLinearInfo, path=path, create_copy=create_copy,
        )
        self.units = UnitsInfo.from_file.__wrapped__(
            UnitsInfo, path=path, create_copy=create_copy,
        )

        self.update()

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
    
    @classmethod
    def _validate(cls, value: object) -> Self:
        if value is None: 
            value = Info()
        
        elif not isinstance(value, Info):
            msg = "Expected 'Info' or 'None', but got '{}'!".format(type(value))
            raise PydanticCustomError('validation_error', msg)
        
        if not value.is_updated: 
            value.update()

        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)