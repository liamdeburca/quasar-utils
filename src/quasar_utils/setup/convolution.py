from logging import getLogger
from typing import Any, Literal, Self

from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@dataclass
class ConvolutionInfo(_Info):
    allow_interp_fitting: bool = field(
        default=False,
        desc="Whether to use fitting by interpolation instead of convolution",
        dtype="bool",
        parse_as="bool",
    )
    n_scales: float = field(
        default=3.0,
        desc="Half-width of convolution kernel in units of sigma",
        dtype="float",
        parse_as="float",
    )

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info: object) -> None:
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["convolution"], dict[str, Any]]:
        return super().to_dict("convolution", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "convolution", logger)
