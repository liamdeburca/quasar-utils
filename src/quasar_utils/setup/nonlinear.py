__all__ = [
    "FitterKwargs",
    "NonLinearInfo",
]

from dataclasses import field
from logging import getLogger
from typing import Any, ClassVar, Literal, Self, TypedDict

from numpy import finfo
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import AbsoluteFilePath

from quasar_utils.setup.utils._info import _Info
from quasar_utils.utils import parsing
from quasar_utils.utils.utils import val_and_type

logger = getLogger(__name__)

MACHINE_PRECISION = finfo(float).eps

class FitterKwargs(TypedDict):
    method: Literal["trf", "dogbox", "lm"]
    loss: str
    max_nfev: int
    ftol: float
    xtol: float
    gtol: float
    f_scale: float


@dataclass
class NonLinearInfo(_Info):
    method: Literal["trf", "dogbox", "lm"] = 'trf'
    loss: str = field(default='linear')
    max_nfev: int = field(default=100)
    ftol: float = field(default=1e-8)
    xtol: float = field(default=MACHINE_PRECISION)
    gtol: float = field(default=MACHINE_PRECISION)
    f_scale: float = field(default=1.0)

    _keys: ClassVar[frozenset[str]] = frozenset(
        [
            "method",
            "loss",
            "max_nfev",
            "ftol",
            "xtol",
            "gtol",
            "f_scale",
            "fitter_kwargs",
        ]
    )
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {}

    def __hash__(self) -> int:
        return super().__hash__()

    def __getitem__(self, key: str) -> Any:
        if key == "fitter_kwargs":
            return self.fitter_kwargs
        return super().__getitem__(key)

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
        }

    @classmethod
    @validate_call
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache:
            logger.debug(f"Using cached 'NonLinearInfo' for '{path}'.")

            ninfo = cls._cache[str(path)]
            if create_copy:
                return ninfo.copy()
            else:
                return ninfo

        ninfo: NonLinearInfo = NonLinearInfo()
        if path is None:
            return ninfo

        logger.debug(f"Configuring 'NonLinearInfo' using '{path}':")
        lines = parsing.get_lines_from_file.__wrapped__(
            "NONLINEAR", path, logger
        )

        for count, line in enumerate(lines, start=1):
            key = line[0].lower()

            match key:
                case "method":
                    val = parsing.as_str(line[1])
                case "loss":
                    val = line[1]
                case "max_nfev":
                    val = max([parsing.as_int(line[1]), 1])
                case "ftol" | "xtol" | "gtol" | "f_scale":
                    val = max([parsing.as_float(line[1]), MACHINE_PRECISION])

            ninfo[key] = val
            logger.debug(
                f">>> [{count}/{len(lines)}] '{key}': {val_and_type(val)}."
            )

        cls._cache[str(path)] = ninfo

        return ninfo

    @classmethod
    @validate_call
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "nonlinear", logger)
