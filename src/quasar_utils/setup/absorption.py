from logging import getLogger
from typing import Any, Literal, Self

import pydantic
from quasar_typing.astropy import QuantityType
from quasar_typing.pathlib import JSONPath

from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@pydantic.dataclasses.dataclass
class AbsorptionInfo(_Info):
    fit: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    _w: int | QuantityType = field(
        default=25,
        dtype="int | float",
        parse_as="n_pixels",
        update_to="n_pixels",
        has_unit=True,
    )
    p: int = field(
        default=2,
        dtype="int",
        parse_as="int",
    )
    p_crit: float = field(
        default=0.01,
        dtype="float",
        parse_as="float",
    )
    z_crit: float = field(
        default=-2.0,
        dtype="float",
        parse_as="float",
    )
    _join: int | QuantityType = field(
        default=3,
        dtype="int | float",
        parse_as="n_pixels",
        update_to="n_pixels",
        has_unit=True,
    )
    refine: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    logspace: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )

    w: int | None = field(default=None, init=False)
    join: int | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info: object) -> None:
        super().update(info, logger)

    def to_dict(
        self, 
        jsonify: bool = False,
    ) -> dict[Literal["absorption"], dict[str, Any]]:
        return super().to_dict("absorption", jsonify=jsonify)

    @classmethod
    @pydantic.validate_call
    def from_json(
        cls,
        json: dict[str, dict] | JSONPath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "absorption", logger)
