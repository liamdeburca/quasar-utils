from logging import getLogger
from typing import Any, Literal, Self

from astropy.units import Unit
from numpy import arange
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.numpy import FloatVector, SortedFloatVector
from quasar_typing.pathlib import AbsoluteFilePath

from ..decorators import validate_call
from .utils import _Info, field, finalise_dataclass

logger = getLogger(__name__)


@finalise_dataclass
@dataclass
class IronInfo(_Info):
    fit: bool = field(
        default=True,
        dtype="bool",
        parse_as="bool",
    )
    _windows: list[CoordBounds] | Quantity_ = field(
        default=[
            [2050.0, 2700.0],
            [3000.0, 3500.0],
            [4500.0, 4700.0],
            [5100.0, 5400.0],
        ] * Unit("angstrom"),
        dtype="list[list[float, float]]",
        parse_as="wavelength_windows",
        update_to="wavelength_windows",
        has_unit=True,
    )
    template_files: list[str] = field(
        default_factory=lambda: [
            "vw2001_blue", "vw2001_red", 
            "bw", 
            "v2003_blue", "v2003_red",
        ],
        dtype="list[str]",
        parse_as="str_list",
    )
    resample: bool = field(
        default=False,
        dtype="bool",
        parse_as="bool",
    )
    _fwhm: list[float] | Quantity_ = field(
        default=arange(1_000, 20_000 + 1, 250) * Unit("km/s"),
        dtype="list[float]",
        parse_as="velocity_list",
        update_to="sorted_velocity_kms_array",
        has_unit=True,
    )
    _flux_bounds: AstropyBounds | Quantity_ = field(
        default=(1.0, 1000.0),
        dtype="list[float | None]",
        parse_as="flux_bounds",
        update_to="flux_bounds",
        has_unit=True,
    )
    _fwhm_bounds: AstropyBounds | Quantity_ = field(
        default=[1_000, 10_000] * Unit("km/s"),
        dtype="list[float | None]",
        parse_as="velocity_bounds",
        update_to="velocity_kms_bounds",
        has_unit=True,
    )
    _split: list[float] | Quantity_ = field(
        default=[0, 0, 0, 0, 0] * Unit("angstrom"),
        dtype="list[float]",
        parse_as="wavelength_list",
        update_to="wavelength_array",
        has_unit=True,
    )
    bias: list[str] = field(
        default_factory=lambda: 5 * ["right"],
        dtype="list[str]",
        parse_as="bias",
    )
    ratio: list[float] = field(
        default_factory=lambda: 5 * [1.0],
        dtype="list[float]",
        parse_as="float_list",
    )
    fixed: list[bool] = field(
        default_factory=lambda: 5 * [True],
        dtype="list[bool]",
        parse_as="bool_list",
    )
    _scale: float | Quantity_ = field(
        default=140.0 * Unit("angstrom"),
        dtype="float",
        parse_as="wavelength",
        update_to="wavelength",
        has_unit=True,
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
    _fwhm_norm: float | Quantity_ = field(
        default=5_000 * Unit("km/s"),
        dtype="float",
        parse_as="velocity",
        update_to="velocity_kms",
        has_unit=True,
    )
    tie_similar: bool = field(
        desc="If true, `IronModel` instances whose templates share the same prefix have their FWHM tied.",
        default=True,
        parse_as="bool",
        dtype="bool",
    )

    windows: list[CoordBounds] | None = field(default=None, init=False)
    fwhm: SortedFloatVector | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)
    fwhm_bounds: AstropyBounds | None = field(default=None, init=False)
    split: FloatVector | None = field(default=None, init=False)
    scale: float | None = field(default=None, init=False)
    fwhm_norm: float | None = field(default=None, init=False)

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info) -> None:
        super().update(info, logger)

    def to_dict(
        self,
        jsonify: bool = False,
    ) -> dict[Literal["iron"], dict[str, Any]]:
        return super().to_dict("iron", jsonify=jsonify)

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        out: IronInfo = cls._from_json(json, create_copy, "iron", logger)
        n_templates = len(out.template_files)
        if n_templates != len(out._split):
            raise ValueError(
                "Number of splits does not match the number of templates: "
                f"{n_templates=} != {len(out._split)}"
            )
        if n_templates != len(out.ratio):
            raise ValueError(
                "Number of ratios does not match the number of templates: "
                f"{n_templates=} != {len(out.ratio)}"
            )
        if n_templates != len(out.bias):
            raise ValueError(
                "Number of biases does not match the number of templates: "
                f"{n_templates=} != {len(out.bias)}"
            )
        if n_templates != len(out.fixed):
            raise ValueError(
                "Number of fixed flags does not match the number of templates: "
                f"{n_templates=} != {len(out.fixed)}"
            )

        return out