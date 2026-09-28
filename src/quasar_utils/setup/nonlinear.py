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
    ftol: float | None
    xtol: float | None
    gtol: float | None
    f_scale: float
    calc_jac: bool


@finalise_dataclass(additional_keys=["fitter_kwargs"])
@dataclass
class NonLinearInfo(_Info):
    method: Literal["trf", "dogbox", "lm"] = field(
        desc="Choice of optimization algorithm: 'trf' (Trust Region Reflective), 'dogbox' (Dogleg), or 'lm' (Levenberg-Marquardt).",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default="trf",
        dtype="str",
        parse_as="algo",
    )
    loss: str = field(
        desc="Specifies the the loss function. The default of 'linear' corresponds to a standard least-squares loss.",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default="linear",
        dtype="str",
        parse_as="str",
    )
    max_nfev: int = field(
        desc="Maximum number of function evaluations allowed during the optimization.",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default=100,
        dtype="int",
        parse_as="int",
    )
    ftol: float | None = field(
        desc="Tolerance for termination by the change of the cost function. If None, termination by 'ftol' is disabled.",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default=1e-8,
        dtype="float | None",
        parse_as="optional_float",
    )
    xtol: float | None= field(
        desc="Tolerance for termination by the change of the solution vector. If None, termination by 'xtol' is disabled.",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default=None,
        dtype="float | None",
        parse_as="optional_float",
    )
    gtol: float | None = field(
        desc="Tolerance for termination by the norm of the gradient. If None, termination by 'gtol' is disabled.",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default=None,
        dtype="float | None",
        parse_as="optional_float",
    )
    f_scale: float = field(
        desc="Scaling for residuals. This parameter has no effect when the loss function is 'linear'.",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default=1.0,
        dtype="float",
        parse_as="float",
    )
    calc_jac: bool = field(
        desc="If true, the Jacobian is calculated analytically. Otherwise a '2-point' scheme is used.",
        comment="See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares",
        default=True,
        dtype="bool",
        parse_as="bool",
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
            'calc_jac': self.calc_jac,
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
