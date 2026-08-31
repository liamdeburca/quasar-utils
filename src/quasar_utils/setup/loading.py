from logging import getLogger
from typing import Any, Literal, Self

from astropy.units import Unit
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import CoordBounds
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@dataclass
class LoadingInfo(_Info):
    loader: str = field(
        default="fits",
        dtype="str",
        parse_as="loader",
    )
    naming: str = field(
        default="IGR",
        dtype="str",
        parse_as="naming",
    )
    deredden: tuple[
        bool, Literal["sfd", "csfd"], Literal["ccm89", "o94"], float
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
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    covariance: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    _sigma_res: float | Quantity_ = field(
        default=69 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity",
        has_unit=True,
    )
    _x_bounds: CoordBounds | Quantity_ = field(
        default=(1000, 10_000) * Unit("angstrom"),
        dtype="list[float, float]",
        parse_as="wavelength_bounds",
        update_to="wavelength_bounds",
        has_unit=True,
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
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "loading", logger)
