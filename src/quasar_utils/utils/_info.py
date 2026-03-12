from abc import ABC, abstractmethod
from typing import Any, Callable, Self

from quasar_typing.pathlib import AbsoluteFilePath

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

class _Info(ABC):
    """
    Class for method inheritance.
    """
    def __str__(self, simple: bool = False) -> str:
        s = "'{}' class".format(self.__class__.__name__)
        if not simple:
            s += "w/ "
            for key in self._keys:
                s += "({}) {}, ".format(key, self[key])
        
        return s.removesuffix(', ') + '.'
    
    def __getitem__(self, key: str) -> Any:
        if key in self._keys: return getattr(self, key)
        else:                 raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._keys: return setattr(self, key, value)
        else:                 raise KeyError(key)

    def __bool__(self) -> bool:
        return self.is_updated
    
    def __getstate__(self) -> dict:
        state: dict = {'_keys': self._keys}
        state.update({key: getattr(self, key) for key in self._keys})
        return state
    
    @classmethod
    def __setstate__(cls, state: dict) -> None:
        cls._keys = frozenset(state.pop('_keys'))

        for key in cls._keys:
            setattr(cls, key, state[key])

    def copy(self) -> Self:
        """
        Creates a copy of the class instance.
        """
        new = self.__class__()
        for key in self._keys: new[key] = self[key]
        return new
    
    def or_default(self, kwargs: dict) -> Callable[[str], Any]:
        def _or_default(key: str) -> Any:
            if key in kwargs: return kwargs[key]
            else:             return self[key]
        return _or_default

    def __enter__(self) -> Callable[[str], Any]:
        return self.f
    
    def __exit__(self, type, value, traceback) -> None:
        del self.f

    @property
    def is_updated(self) -> bool:
        """
        Whether all parameters have been updated.
        """
        return not any(getattr(self, key) is None for key in self._keys)
    
    # ----- Pydantic ----- #

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = "Expected a {} instance, got {}".format(
                cls.__name__, 
                type(value).__name__,
            )
            raise PydanticCustomError('validation_error', msg)
        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)

    # ----- Abstract Methods ----- #

    @abstractmethod
    def update(self, info) -> None:
        """
        Updates all parameters. 
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, path: AbsoluteFilePath = None):
        """
        Creates and configures an instance from a file.
        """
        pass