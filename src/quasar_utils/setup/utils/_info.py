from logging import getLogger, Logger
from abc import ABC, abstractmethod
from typing import Any, Callable, Self, ClassVar
from pathlib import Path
from json import load as load_json

from .json_field import JSONField
from ...utils.utils import val_and_type

from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

class _Info(ABC):
    """
    Class for method inheritance.
    """
    _keys: ClassVar[frozenset[str]] = frozenset()
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {}

    def __str__(self, simple: bool = False) -> str:
        s = "'{}' class".format(self.__class__.__name__)
        if not simple:
            s += "w/ "
            for key in self._keys:
                s += "({}) {}, ".format(key, self[key])
        
        return s.removesuffix(', ') + '.'
    
    def __getitem__(self, key: str) -> Any:
        if key in self._keys: 
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._keys: 
            return setattr(self, key, value)
        raise KeyError(key)

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
        return all(getattr(self, key) is not None for key in self._keys)
    
    # ----- Pydantic ----- #

    # @classmethod
    # def _validate(cls, value: object) -> Self:
    #     if not isinstance(value, cls):
    #         msg = "(TEST) Expected a {} instance, got {}".format(
    #             cls.__name__, 
    #             type(value).__name__,
    #         )
    #         raise PydanticCustomError('validation_error', msg)
    #     return value
    
    # @classmethod
    # def __get_pydantic_core_schema__(cls, source_type, handler):
    #     return no_info_plain_validator_function(cls._validate)

    # ----- Abstract Methods ----- #

    @classmethod
    @abstractmethod
    def from_file(cls, path: AbsoluteFilePath = None):
        """
        Creates and configures an instance from a file.
        """
        pass

    def update(
        self, 
        info, 
        logger: Logger,
    ) -> None:
        """
        Updates all parameters. 
        """
        msg = f"Updating '{self.__class__.__name__}' class: "
        if not self._values_to_update: 
            msg += "(nothing to update)."

        logger.debug(msg)
        n_values = len(self._values_to_update)
        count: int = 1
        for key, conversion_name in self._values_to_update.items():
            old = self["_" + key]
            self[key] = new = info.update_value(old, conversion_name)

            msg = f"[{count:<2}/{n_values:<2}] '{key}': "
            msg += (
                # f"{val_and_type(old)} (no change)." 
                # if old == new else
                f"{val_and_type(old)} -> {val_and_type(new)}."
            )
            logger.debug(msg)
            count += 1

        assert self.is_updated

    @classmethod
    def from_json(
        cls,
        json: dict[str, dict] | Path | None,
        create_copy: bool,
        parent_field: str,
        logger: Logger,
    ) -> Self:
        
        if isinstance(json, Path) and str(json) in cls._cache.keys():
            logger.debug(f"Using cached '{cls.__name__}' for '{json}'.")
            
            info = cls._cache[str(json)]
            return info.copy() if create_copy else info

        info: _Info = cls()
        if json is None: 
            return info

        if (add_to_cache := isinstance(json, Path)):
            cache_key = str(json)
            with open(json, 'r') as f:
                json = load_json(f)

        fields = JSONField.load_all_from_json(json, parent_field)
        for count, field in enumerate(fields, start=1):
            key = field.field
            assert key in cls._keys, \
                f"Invalid key '{key}' in '{cls.__name__}' JSON configuration."
            
            if key in cls._values_to_update:
                key = '_' + key

            info[key] = val = field.value
            logger.debug(
                f">>> [{count}/{len(fields)}] '{key}': {val_and_type(val)}."
            )

        if add_to_cache:
            cls._cache[cache_key] = info

        return info