from collections.abc import ABC, abstractmethod
from typing import Any, Callable, Self

from quasar_typing.pathlib import AbsoluteFileLike

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
    
    @classmethod
    def __getitem__(cls, key: str) -> Any:
        if key in cls._keys: return getattr(cls, key)
        else:                 raise KeyError(key)

    @classmethod
    def __setitem__(cls, key: str, value: Any) -> None:
        if key in cls._keys: return setattr(cls, key, value)
        else:                raise KeyError(key)

    @classmethod
    def __bool__(cls) -> bool:
        return cls.is_updated
    
    @classmethod
    def __getstate__(cls) -> dict:
        state: dict = {'_keys': cls._keys}
        state.update({key: getattr(cls, key) for key in cls._keys})
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
            msg = "Expected '{}', but got '{}'!".format(
                cls.__name__, type(value),
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
    def from_file(cls, path: AbsoluteFileLike = None):
        """
        Creates and configures an instance from a file.
        """
        pass