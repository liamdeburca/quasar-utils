from logging import getLogger
from typing import Any, Literal, Self

from astropy.units import Unit
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@dataclass
class ContinuumInfo(_Info):
    fit: bool = field(
        default=True,
        desc="Whether to fit the power law continuum",
        dtype="bool",
        parse_as="bool",
    )
    _x0: float | Quantity_ = field(
        default=1450.0 * Unit("angstrom"),
        desc="Reference wavelength used for power law model",
        dtype="float",
        parse_as="wavelength",
        update_to="wavelength",
        has_unit=True,
    )
    _y0: float | Quantity_ = field(
        default=1.0,
        desc="Reference flux density used for power law linear fitting",
        dtype="float",
        parse_as="flux",
        update_to="flux",
        has_unit=True,
    )
    _windows: list[CoordBounds] | Quantity_ = field(
        default=[
            [1425.0, 1475.0],
            [1675.0, 1690.0],
            [1975.0, 2050.0],
            [2150.0, 2250.0],
            [5400.0, 5800.0],
            [7000.0, 9000.0],
        ] * Unit("angstrom"),
        desc="List of wavelength windows used for continuum fitting",
        dtype="list[list[float, float]]",
        parse_as="wavelength_windows",
        update_to="wavelength_windows",
        has_unit=True,
    )
    _flux_bounds: AstropyBounds | Quantity_ = field(
        default=(1.0, 10_000.0),
        desc="Lower and upper bounds for the flux density at the reference wavelength",
        dtype="list[float | None]",
        parse_as="flux_bounds",
        update_to="flux_bounds",
        has_unit=True,
    )
    sigmas: list[float] = field(
        default_factory=lambda: [3.00, 2.75, 2.50],
        desc="Sequence of sigma-clipping thresholds",
        dtype="list[float]",
        parse_as="float_list",
    )
    alpha_bounds: AstropyBounds = field(
        default=(-3.0, 0.0),
        desc="Lower and upper bounds for the power law index",
        dtype="list[float | None]",
        parse_as="float_bounds",
    )
    min_fittable_total: int = field(
        default=10,
        desc="Minimum total number of fittable pixels",
        dtype="int",
        parse_as="int",
    )

    x0: float | None = field(default=None, init=False)
    y0: float | None = field(default=None, init=False)
    windows: list[CoordBounds] | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info: object) -> None:
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["continuum"], dict[str, Any]]:
        return super().to_dict("continuum", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "continuum", logger)
