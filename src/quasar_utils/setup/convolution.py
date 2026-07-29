from logging import getLogger
from typing import ClassVar, Self

from pydantic import validate_call
from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import AbsoluteFilePath

from .utils._info import _Info

logger = getLogger(__name__)


@dataclass
class ConvolutionInfo(_Info):
    allow_interp_fitting: bool = True
    n_scales: float = 3.0

    _keys: ClassVar[frozenset[str]] = frozenset(
        [
            "allow_interp_fitting",
            "n_scales",
        ]
    )
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {}

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info) -> None:
        super().update(info, logger)

    @classmethod
    @validate_call
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "convolution", logger)
