import hashlib
import json as json_pkg
from abc import ABC
from collections.abc import Callable
from dataclasses import fields
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
from astropy import units

from .dataclasses import field_to_dict, get_field_metadata
from .parsing import Parser
from .updating import Updater

logger = getLogger(__name__)


def _make_hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_make_hashable(v) for v in value)
    elif isinstance(value, dict):
        return frozenset((k, _make_hashable(v)) for k, v in value.items())
    elif isinstance(value, set):
        return frozenset(_make_hashable(v) for v in value)
    elif isinstance(value, np.ndarray):
        return value.tobytes()
    elif isinstance(value, (units.Unit, units.CompositeUnit, Path)):
        return str(value)
    elif isinstance(value, np.random.RandomState):
        return value.get_state()
    else:
        return value


class _Info(ABC):
    """
    Class for method inheritance.
    """
    _keys: ClassVar[frozenset[str]]
    _cache: ClassVar[dict[str, Self]]
    _values_to_update: ClassVar[dict[str, str]]

    def __str__(self) -> str:
        return object.__str__(self)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._keys:
            return setattr(self, key, value)
        raise KeyError(key)

    def __bool__(self) -> bool:
        return self.is_updated

    def _hash_helper(self) -> str:
        d = {
            k2: {k1: v1["value"] for k1, v1 in v2.items()}
            for k2, v2 in self.to_dict(jsonify=True).items()
        }
        return json_pkg.dumps(
            d, 
            sort_keys=True, 
            separators=(",", ":"),
            indent=None,
        )

    def create_hash(self, algo: str = "sha256") -> str:
        return hashlib.new(
            algo, 
            self._hash_helper().encode("utf-8"),
        ).hexdigest()

    def __hash__(self) -> int:
        return hash(self._hash_helper())

    def __getstate__(self) -> dict:
        return {key: getattr(self, key) for key in self._keys}

    def __setstate__(self, state: dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def copy(self) -> Self:
        """
        Creates a copy of the class instance.
        """
        new = self.__class__()
        for key in self._keys:
            setattr(new, key, getattr(self, key))
        return new

    def or_default(self, kwargs: dict) -> Callable[[str], Any]:
        def _or_default(key: str) -> Any:
            return kwargs.get(key, getattr(self, key))
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
        return all(
            getattr(self, key) is not None
            for key in self._keys
            if key in self._values_to_update
        )

    def update(self, info: object, logger: Logger) -> None:
        """
        Updates all parameters.
        """
        msg = f"Updating '{self.__class__.__name__}' class: "
        if not self._values_to_update:
            msg += "(nothing to update)."

        logger.debug(msg)
        n_values = len(self._values_to_update)
        count: int = 1
        for name, method_name in self._values_to_update.items():
            old = getattr(self, f"_{name}")

            updater = Updater[method_name]
            new = updater(info, old)
            setattr(self, name, new)

            msg = f"[{count:<2}/{n_values:<2}] '{name}': {old=} -> {new=}"
            logger.debug(msg)
            count += 1

        assert self.is_updated

    def to_dict(
        self, 
        parent_field: str, 
        blacklist: list[str] | None = None,
        jsonify: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Creates a dictionary representation of this instance.

        This function is designed for writing JSON/YAML config files.
        """
        blacklist = blacklist or []
        out = {}
        for field in fields(self):
            if field.name in self._values_to_update:
                continue
            if field.name in blacklist:
                continue
            out.update(field_to_dict(self, field, jsonify=jsonify))

        return {parent_field: out}

    @classmethod
    def _from_json(
        cls,
        json: dict[str, dict] | Path | str | None,
        create_copy: bool,
        parent_field: str,
        logger: Logger,
    ) -> Self:

        if isinstance(json, Path) and str(json) in cls._cache:
            logger.debug(f"Using cached '{cls.__name__}' for '{json}'.")

            info = cls._cache[str(json)]
            return info.copy() if create_copy else info

        info: _Info = cls()
        if json is None:
            return info

        if add_to_cache := isinstance(json, (str, Path)):
            cache_key = str(json)
            with open(json, "r") as f:
                json = json_pkg.load(f)

        json = json.get(parent_field, json)

        # Loop through all fields that: appear in the JSON file AND are not 
        # updated versions of existing fields (i.e., not in _values_to_update).
        for name, metadata in (
            item
            for item in get_field_metadata(cls).items()
            if (item[0].removeprefix('_') in json)
                and (item[0] not in cls._values_to_update)
        ):
            parser = Parser._parse(metadata["parse_as"])
            field = json[name.removeprefix('_')]
            try:
                value = parser(field)
            except AssertionError as e:
                msg = f"Failed to parse '{parent_field}::{name}': {field}."
                raise AssertionError(msg) from e
            setattr(info, name, value)

            logger.debug(
                f">>> [{cls.__name__}] '{name}' ({metadata['dtype']}): {value}."
            )

        if add_to_cache:
            cls._cache[cache_key] = info

        return info