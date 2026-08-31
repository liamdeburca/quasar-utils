from logging import getLogger
from typing import Any, Literal, Self

from astropy.units import Unit
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.misc import BalmerModelParams
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)

@finalise_dataclass
@dataclass
class BalmerInfo(_Info):
    fit: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    _windows: list[CoordBounds] | Quantity_ = field(
        default=[[3000, 4500]] * Unit("angstrom"),
        dtype="list[list[float, float]]",
        parse_as="wavelength_windows",
        update_to="wavelength_windows",
        has_unit=True,
    )
    _edge: float | Quantity_ = field(
        default=3646 * Unit("angstrom"),
        dtype="float",
        parse_as="wavelength",
        update_to="wavelength",
        has_unit=True,
    )
    _fwhm_norm: float | Quantity_ = field(
        default=5_000 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity",
        has_unit=True,
    )
    source: Literal["SH1995"] = field(
        default="SH1995",
        dtype="str",
        parse_as="str",
    )
    _temp: float | Quantity_ = field(
        default=15_000 * Unit("K"),
        dtype="float",
        parse_as="temperature",
        update_to="temperature",
        has_unit=True,
    )
    _dens: float | Quantity_ = field(
        default=1e9 * Unit("cm^-3"),
        dtype="float",
        parse_as="density",
        update_to="density",
        has_unit=True,
    )
    n_u_min: int = field(
        default=7,
        dtype="int",
        parse_as="int",
    )
    n_u_max: int = field(
        default=50,
        dtype="int",
        parse_as="int",
    )
    tau: float = field(
        default=1.0,
        dtype="float",
        parse_as="float",
    )
    scale: float = field(
        default=3.0,
        dtype="float",
        parse_as="float",
    )
    _flux: float | Quantity_ = field(
        default=1.0,
        dtype="float",
        parse_as="flux",
        update_to="flux",
        has_unit=True,
    )
    _fwhm: float | Quantity_ = field(
        default=5_000 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity",
        has_unit=True,
    )
    _flux_bounds: AstropyBounds | Quantity_ = field(
        default=(0.0, 1000.0),
        dtype="list[float]",
        parse_as="flux_bounds",
        update_to="flux_bounds",
        has_unit=True,
    )
    _fwhm_bounds: AstropyBounds | Quantity_ = field(
        default=[1000.0, 10_000.0] * Unit("km/s"),
        dtype="list[float]",
        parse_as="velocity_bounds",
        update_to="velocity_bounds",
        has_unit=True,
    )
    ratio: float = field(
        default=0.3,
        dtype="float",
        parse_as="float",
    )
    ratio_bounds: AstropyBounds = field(
        default=(0.5, 2.0),
        dtype="list[float]",
        parse_as="float_bounds",
    )
    _fixed: BalmerModelParams = field(
        default=BalmerModelParams({"ratio"}),
        dtype="list[str]",
        parse_as="balmer_params",
        update_to="fixed",
    )
    raster_n: int = field(
        default=20,
        dtype="int",
        parse_as="int",
    )
    min_fittable_ratio: float = field(
        default=0.6,
        dtype="float",
        parse_as="float",
    )
    min_fittable_total: int = field(
        default=100,
        dtype="int",
        parse_as="int",
    )
    raster: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    fine_tune: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    windows: list[CoordBounds] | None = field(default=None, init=False)
    edge: float | None = field(default=None, init=False)
    fwhm_norm: float | None = field(default=None, init=False)
    temp: float | None = field(default=None, init=False)
    dens: float | None = field(default=None, init=False)
    flux: float | None = field(default=None, init=False)
    fwhm: float | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)
    fwhm_bounds: AstropyBounds | None = field(default=None, init=False)
    fixed: dict[str, bool] | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info: object) -> None:
        super().update(info, logger)

    def to_dict(
        self, 
        jsonify: bool = False,
    ) -> dict[Literal["balmer"], dict[str, Any]]:
        return super().to_dict("balmer", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return cls._from_json(json, create_copy, "balmer", logger)
