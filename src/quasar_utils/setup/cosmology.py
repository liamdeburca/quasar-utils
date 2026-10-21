from logging import getLogger
from typing import Any, Literal, Self

import pydantic
from quasar_typing.astropy.cosmology import CosmologyName
from quasar_typing.pathlib import JSONPath

from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@pydantic.dataclasses.dataclass
class CosmologyInfo(_Info):
    cosmology: CosmologyName = field(
        default="Planck18",
        desc="Name of the cosmology",
        dtype="str",
        parse_as="str",
    )

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info: object) -> None:
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["cosmology"], dict[str, Any]]:
        return super().to_dict("cosmology", jsonify=jsonify)

    @classmethod
    @pydantic.validate_call
    def from_json(
        cls,
        json: dict[str, dict] | JSONPath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "cosmology", logger)
