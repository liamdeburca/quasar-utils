from dataclasses import field
from logging import getLogger
from typing import ClassVar, Literal, Self

from astropy.units import Unit
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Quantity_
from quasar_typing.bounds import AstropyBounds, CoordBounds
from quasar_typing.misc import HostGalaxyModelParams
from quasar_typing.pathlib import AbsoluteFilePath

from .utils._info import _Info

logger = getLogger(__name__)


@dataclass
class HostInfo(_Info):
    fit: bool = True

    _windows: list[CoordBounds] | Quantity_ = [[3000, 4500]] * Unit("angstrom")
    _x_norm: float | Quantity_ = 6000 * Unit("angstrom")
    _fwhm_norm: float | Quantity_ = 0 * Unit("km/s")
    _flux: float | Quantity_ = 1e-17 * Unit("erg/(s.cm2.angstrom)")
    _fwhm: float | Quantity_ = 0 * Unit("km/s")
    _flux_bounds: AstropyBounds | Quantity_ = [1e-18, 1e-15] * Unit(
        "erg/(s.cm2.angstrom)"
    )
    _fwhm_bounds: AstropyBounds | Quantity_ = [0, 1000] * Unit("km/s")
    _fixed: HostGalaxyModelParams = field(
        default_factory=lambda: HostGalaxyModelParams({"fwhm"})
    )

    template_files: list[str | AbsoluteFilePath] = field(default_factory=list)
    sources: list[Literal["bc2003"]] = field(
        default_factory=lambda: ["bc2003"]
    )
    ages: list[int] = field(
        default_factory=lambda: [
            1_015_190_000,
            2_500_000_000,
            4_500_000_000,
            5_000_000_000,
            6_000_000_000,
            8_000_000_000,
            12_000_000_000,
        ]
    )

    raster: bool = True
    fine_tune: bool = True
    only_model: bool = False

    min_fittable_ratio: float = 0.6
    min_fittable_total: int = 100

    windows: list[CoordBounds] | None = field(default=None, init=False)
    x_norm: float | None = field(default=None, init=False)
    fwhm_norm: float | None = field(default=None, init=False)
    flux: float | None = field(default=None, init=False)
    fwhm: float | None = field(default=None, init=False)
    flux_bounds: AstropyBounds | None = field(default=None, init=False)
    fwhm_bounds: AstropyBounds | None = field(default=None, init=False)
    fixed: dict[str, bool] | None = field(default=None, init=False)

    min_fittable_ratio: float = 0.6
    min_fittable_total: int = 100

    _keys: ClassVar[frozenset[str]] = frozenset(
        [
            "fit",
            "_windows",
            "windows",
            "_x_norm",
            "x_norm",
            "_fwhm_norm",
            "fwhm_norm",
            "_flux",
            "flux",
            "_fwhm",
            "fwhm",
            "_flux_bounds",
            "flux_bounds",
            "_fwhm_bounds",
            "fwhm_bounds",
            "_fixed",
            "fixed",
            "template_files",
            "sources",
            "ages",
            "raster",
            "fine_tune",
            "only_model",
            "min_fittable_ratio",
            "min_fittable_total",
        ]
    )
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        "windows": "to_wavelength_windows",
        "x_norm": "to_wavelength",
        "fwhm_norm": "to_velocity",
        "flux": "to_flux",
        "fwhm": "to_velocity",
        "flux_bounds": "to_flux_bounds",
        "fwhm_bounds": "to_velocity_bounds",
        "fixed": "to_fixed",
    }

    def __hash__(self) -> int:
        return super().__hash__()

    def update(self, info) -> None:
        """
        Convert to unitsless.
        """
        super().update(info, logger)

    @classmethod
    @validate_call
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        raise NotImplementedError()

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "host", logger)
