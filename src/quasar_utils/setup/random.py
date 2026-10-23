from logging import getLogger
from typing import Any, Literal, Self

import pydantic
from quasar_typing.numpy import RandomStateType
from quasar_typing.pathlib import JSONPath

from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)
    

@finalise_dataclass(additional_keys=["random_state"])
@pydantic.dataclasses.dataclass
class RandomInfo(_Info):
    seed: int = field(
        default=42,
        dtype="int",
        parse_as="int",
    )
    renew_rng: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )

    @property
    def random_state(self) -> RandomStateType:
        from numpy.random import RandomState
        if self.renew_rng:
            return RandomState(self.seed)

        # Use cached value
        if hasattr(self, "_random_state"):
            return self._random_state

        self._random_state = RandomState(self.seed)
        return self._random_state

    def __hash__(self) -> int:
        return super().__hash__()

    def __getitem__(self, key: str) -> Any:
        if key == "random_state":
            return self.random_state
        return super().__getitem__(key)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("random_state", None)
        state.pop("_random_state", None)
        return state
    
    def update(self, info) -> None:
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["random"], dict[str, Any]]:
        return super().to_dict("random", jsonify=jsonify)

    @classmethod
    @pydantic.validate_call
    def from_json(
        cls,
        json: dict[str, dict] | JSONPath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "random", logger)
