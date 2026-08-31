from logging import getLogger
from typing import Any, Literal, Self

from astropy.units import Unit
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds
from quasar_typing.numpy import SortedFloatVector
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@dataclass
class LinesInfo(_Info):
    fit: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    _x_limit: float | Quantity_ = field(
        default=1220 * Unit("angstrom"),
        dtype="float",
        parse_as="wavelength",
        update_to="wavelength",
        has_unit=True,
    )
    _v_sep: float | Quantity_ = field(
        default=10_000 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity",
        has_unit=True,
    )
    _v_off_bounds: AstropyBounds | Quantity_ = field(
        default=(-1_000, 1_000) * Unit("km/s"),
        dtype="list[float | None]",
        parse_as="velocity_bounds",
        update_to="velocity_bounds",
        has_unit=True,
    )
    _fwhm_v_bounds: AstropyBounds | Quantity_ = field(
        default=(1000, 10_000) * Unit("km/s"),
        dtype="list[float | None]",
        parse_as="velocity_bounds",
        update_to="velocity_bounds",
        has_unit=True,
    )
    _strength_bounds: AstropyBounds | Quantity_ = field(
        default=(0.0, 100_000.0),
        dtype="list[float | None]",
        parse_as="flux_bounds",
        update_to="flux_bounds",
        has_unit=True,
    )
    _w: int | Quantity_ = field(
        default=25,
        dtype="int",
        parse_as="n_pixels",
        update_to="n_pixels",
        has_unit=True,
    )
    _forced_splits: list[float] | Quantity_ = field(
        default=[1450, 1680, 2000] * Unit("angstrom"),
        dtype="list[float]",
        parse_as="wavelength_list",
        update_to="sorted_wavelength_array",
        has_unit=True,
    )
    min_fittable_total: int = field(
        default=50,
        dtype="int",
        parse_as="int",
    )
    min_fittable_ratio: float = field(
        default=0.6,
        dtype="float",
        parse_as="float",
    )
    evaluate_initial: float = field(
        default=3.0,
        dtype="float",
        parse_as="float",
    )
    aggressive: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    crop: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    measure: str = field(
        default="getFluxSNR",
        dtype="str",
        parse_as="str",
    )
    reverse: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    snr: float | int = field(
        default=10,
        dtype="float",
        parse_as="float",
    )
    make_copies: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    adapt_scale: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    scale_init: float = field(
        default=1.0,
        dtype="float",
        parse_as="float",
    )
    scale_bounds: AstropyBounds = field(
        default=(0.0, 10.0),
        dtype="list[float | None]",
        parse_as="float_bounds",
    )
    scale_fixed: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )

    x_limit: float | None = field(default=None, init=False)
    v_sep: float | None = field(default=None, init=False)
    v_off_bounds: AstropyBounds | None = field(default=None, init=False)
    fwhm_v_bounds: AstropyBounds | None = field(default=None, init=False)
    strength_bounds: AstropyBounds | None = field(default=None, init=False)
    w: int | None = field(default=None, init=False)
    forced_splits: SortedFloatVector | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info) -> None:
        """
        Converts parameters with units into their dimensionless equivalents.
        """
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["lines"], dict[str, Any]]:
        return super().to_dict("lines", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "lines", logger)
