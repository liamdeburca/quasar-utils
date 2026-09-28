from logging import getLogger
from typing import Any, Literal, Self

from astropy.units import Unit
from pydantic.dataclasses import dataclass
from pydantic_core import PydanticCustomError
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.misc import HostGalaxyModelParams
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@dataclass
class HostInfo(_Info):
    fit: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )

    _x_min: float | Quantity_ = field(
        desc="If the spectrum's largest wavelength is below this value, host galaxy emission will not be modelled.",
        default=4000 * Unit("angstrom"),
        dtype="float",
        parse_as="wavelength",
        update_to="wavelength",
        has_unit=True,
    )
    _windows: list[CoordBounds] | Quantity_ = field(
        default=[
            [5400.0, 5750.0],
            [7000.0, 9000.0],
        ] * Unit("angstrom"),
        dtype="list[list[float, float]]",
        parse_as="wavelength_windows",
        update_to="wavelength_windows",
        has_unit=True,
    )
    _x_norm: float | Quantity_ = field(
        default=6000 * Unit("angstrom"),
        dtype="float",
        parse_as="wavelength",
        update_to="wavelength",
        has_unit=True,
    )
    _fwhm_norm: float | Quantity_ = field(
        default=0 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity_kms",
        has_unit=True,
    )
    _flux: float | Quantity_ = field(
        default=1.0,
        dtype="float",
        parse_as="flux",
        update_to="flux",
        has_unit=True,
    )
    _fwhm: float | Quantity_ = field(
        default=0 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity_kms",
        has_unit=True,
    )
    _flux_bounds: AstropyBounds | Quantity_ = field(
        default=(0.0, 1000.0),
        dtype="list[float | None]",
        parse_as="flux_bounds",
        update_to="flux_bounds",
        has_unit=True,
    )
    _fwhm_bounds: AstropyBounds | Quantity_ = field(
        default=[0, 1000] * Unit("km/s"),
        dtype="list[float | None]",
        parse_as="velocity_bounds",
        update_to="velocity_kms_bounds",
        has_unit=True,
    )
    _fixed: HostGalaxyModelParams = field(
        default_factory=lambda: HostGalaxyModelParams({"fwhm"}),
        dtype="list[str]",
        parse_as="host_galaxy_params",
        update_to="fixed",
    )
    sources: list[Literal["bc2003"]] = field(
        default_factory=lambda: [
            "bc2003",
            "bc2003",
            "bc2003",
            # "bc2003",
            # "bc2003",
            # "bc2003",
            # "bc2003",
        ],
        dtype="list[str]",
        parse_as="str_list",
    )
    ages: list[int] = field(
        desc="Ages of the host galaxy templates in Gyr.",
        default_factory=lambda: [
            # int(1.015 * 1e9),
            int(2.500 * 1e9),
            int(4.500 * 1e9),
            #  int(5.000 * 1e9),
            #  int(6.000 * 1e9),
            #  int(8.000 * 1e9),
            int(12.000 * 1e9),
        ],
        dtype="list[float | int]",
        parse_as="gyr_list",
        comment="Older ages are all quite similar in shape. It may be best to include one young and one older template.",
    )
    default_source: Literal["bc2003"] = field(
        default="bc2003",
        dtype="str",
        parse_as="str",
    )
    default_age: int = field(
        default=int(4.5 * 1e9),
        dtype="float | int",
        parse_as="gyr",
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
    min_fittable_ratio: float = field(
        comment="Not currently used",
        default=0.6,
        dtype="float",
        parse_as="float",
    )
    min_fittable_total: int = field(
        comment="Not currently used",
        default=100,
        dtype="int",
        parse_as="int",
    )

    x_min: float | None = field(default=None, init=False)
    windows: list[CoordBounds] | None = field(default=None, init=False)
    x_norm: float | None = field(default=None, init=False)
    fwhm_norm: float | None = field(default=None, init=False)
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
    ) -> dict[Literal["host"], dict[str, Any]]:
        return super().to_dict("host", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        out = cls._from_json(json, create_copy, "host", logger)

        options: list[tuple[str, int]] = list(zip(out.sources, out.ages))
        if (out.default_source, out.default_age) not in options:
            raise PydanticCustomError(
                "validation_error",
                "Default source-age combination is not in the list of "
                "available source-age options: "
                f"({out.default_source}, {out.default_age}), {options=}"
            )

        return out
