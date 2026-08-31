from logging import getLogger
from typing import Any, Literal, Self

from astropy.units import Unit
from numpy.random import RandomState
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.misc import (
    BootstrapType,
    FWHMStrategy,
    Method,
    OutLines,
    OutMeasures,
    Scale,
    Variant,
    VaryLines,
)
from quasar_typing.numpy import RandomState_
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@dataclass
class ErrorInfo(_Info):
    method: Method = field(
        default="bootstrap",
        desc="Method used for error estimation",
        dtype="str",
        parse_as="str",
    )
    remodel: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    replace_missing: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    scale: Scale = field(
        default="global",
        dtype="str",
        parse_as="str",
    )
    variant: Variant = field(
        default="standard",
        dtype="str",
        parse_as="str",
    )
    bootstrap_type: BootstrapType = field(
        default="spectrum",
        dtype="str",
        parse_as="str",
    )
    fwhm_strategy: FWHMStrategy = field(
        default="average",
        dtype="str",
        parse_as="str",
    )
    iterations: int = field(
        default=100,
        dtype="int",
        parse_as="int",
    )
    random_state: RandomState_ = field(
        default_factory=lambda: RandomState(42),
        dtype="int",
        parse_as="random_state",
    )
    renew_rng: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    n_sigmas: float = field(
        default=2.0,
        dtype="float",
        parse_as="float",
    )
    res: int = field(
        default=1000,
        dtype="int",
        parse_as="int",
    )
    render_width: float = field(
        default=5,
        dtype="float",
        parse_as="float",
    )
    # ! render_width: float | int = 5
    exact: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    _v_int: float | Quantity_ = field(
        default=18_000 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity",
        has_unit=True,
    )
    ipv_int: float = field(
        default=0,
        dtype="float",
        parse_as="float",
    )
    _dx_int: float | Quantity_ = field(
        default=50 * Unit("angstrom"),
        dtype="float",
        parse_as="wavelength",
        update_to="wavelength",
        has_unit=True,
    )
    vary_lines: VaryLines = field(
        default_factory=lambda: VaryLines({"all"}),
        dtype="list[str]",
        parse_as="vary_lines",
    )
    out_lines: OutLines = field(
        default_factory=lambda: OutLines({"all"}),
        dtype="list[str]",
        parse_as="out_lines",
    )
    out_measures: OutMeasures = field(
        default_factory=lambda: OutMeasures({"all"}),
        dtype="list[str]",
        parse_as="out_measures",
    )
    percentiles: set[float] = field(
        default_factory=lambda: {5, 50, 95},
        dtype="list[float]",
        parse_as="float_set",
    )
    # ! Should these fields be removed?
    tqdm_disable: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    tqdm_leave: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )

    v_int: float | None = field(default=None, init=False)
    dx_int: float | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info) -> None:
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["error"], dict[str, Any]]:
        return super().to_dict("error", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "error", logger)
