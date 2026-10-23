from logging import getLogger
from typing import Any, Literal, Self

import pydantic
from astropy.units import Unit
from quasar_typing.astropy import QuantityType
from quasar_typing.bounds import CoordBounds
from quasar_typing.literals import LawName, MapName
from quasar_typing.pathlib import JSONPath

from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@pydantic.dataclasses.dataclass
class LoadingInfo(_Info):
    loader: str = field(
        default="fits",
        dtype="str",
        parse_as="loader",
    )
    naming: str = field(
        default="igr",
        dtype="str",
        parse_as="naming",
    )
    deredden: tuple[
        bool, 
        MapName, 
        LawName, 
        float,
    ] = field(
        default=(True, "sfd", "ccm89", 3.1),
        dtype="list[bool, str, str, float]",
        parse_as="deredden",
    )
    rebin: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    load: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    conserve: bool = field(
        desc="If True, conserves the integrated flux density during rebinning",
        default=True,
        dtype="bool",
        parse_as="bool",
        comment="Do not touch this unless you know what you are doing",
    )
    covariance: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    _sigma_res: float | QuantityType = field(
        default=69 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity_c",
        has_unit=True,
    )
    _x_bounds: CoordBounds | QuantityType = field(
        default=(1200, 10_000) * Unit("angstrom"),
        dtype="list[float, float]",
        parse_as="wavelength_bounds",
        update_to="wavelength_bounds",
        has_unit=True,
    )

    kernel_w: int = field(
        desc="Maximum size (pix) of the kernels used to correct for instrumental resolution.",
        default=15,
        dtype="int",
        parse_as="int",
    )
    kernel_n_sigma: float = field(
        desc="Number of standard deviations to include when constructing the resolution kernels.",
        default=3.0,
        dtype="float",
        parse_as="float",
    )
    kernel_normalize: bool = field(
        desc="Whether to normalize the resolution kernels s.t. they sum to 1.",
        default=True,
        dtype="bool",
        parse_as="bool",
    )

    sigma_res: float | None = field(default=None, init=False)
    x_bounds: CoordBounds | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info) -> None:
        """
        Calculate and assign the dimensionless velocity resolution, sigma_res,
        and the dimensionless rest-wavelength bounds, x_bounds.
        """
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["loading"], dict[str, Any]]:
        return super().to_dict("loading", jsonify=jsonify)

    @classmethod
    @pydantic.validate_call
    def from_json(
        cls,
        json: dict[str, dict] | JSONPath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "loading", logger)
