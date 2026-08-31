from logging import getLogger
from typing import Any, Literal, Self, TypedDict

from numpy import finfo
from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)

MACHINE_PRECISION = finfo(float).eps

class FitterKwargs(TypedDict):
    method: Literal["trf", "dogbox", "lm"]
    loss: str
    max_nfev: int
    ftol: float
    xtol: float
    gtol: float
    f_scale: float


@finalise_dataclass(additional_keys=["fitter_kwargs"])
@dataclass
class NonLinearInfo(_Info):
    method: Literal["trf", "dogbox", "lm"] = field(
        default="trf",
        dtype="str",
        parse_as="algo",
    )
    loss: str = field(
        default="linear",
        dtype="str",
        parse_as="str",
    )
    max_nfev: int = field(
        default=100,
        dtype="int",
        parse_as="int",
    )
    ftol: float = field(
        default=1e-8,
        dtype="float",
        parse_as="float",
    )
    xtol: float = field(
        default_factory=lambda: float(finfo(float).eps),
        dtype="float",
        parse_as="float",
    )
    gtol: float = field(
        default_factory=lambda: float(finfo(float).eps),
        dtype="float",
        parse_as="float",
    )
    f_scale: float = field(
        default=1.0,
        dtype="float",
        parse_as="float",
    )

    def __hash__(self) -> int:
        return super().__hash__()

    def __getitem__(self, key: str) -> Any:
        if key == "fitter_kwargs":
            return self.fitter_kwargs
        return super().__getitem__(key)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("fitter_kwargs")
        return state

    def update(self, info) -> None:
        super().update(info, logger)

    @property
    def fitter_kwargs(self) -> FitterKwargs:
        """
        Returns a dictionary of keyword arguments fully compatible with 
        `scipy.optimize.least_squares`.
        """
        return {
            'method': self.method,
            'loss': self.loss,
            'max_nfev': self.max_nfev,
            'ftol': self.ftol,
            'xtol': self.xtol,
            'gtol': self.gtol,
            'f_scale': self.f_scale,
        }

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["nonlinear"], dict[str, Any]]:
        return super().to_dict(
            "nonlinear", 
            blacklist=["fitter_kwargs"], 
            jsonify=jsonify,
        )

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "nonlinear", logger)
