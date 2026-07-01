from logging import getLogger
from functools import cached_property
from typing import ClassVar, Self, Any, Literal
from numpy import finfo
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import validate_call

from quasar_utils.fitting import FitterInstance
from quasar_utils.setup.utils._info import _Info
from quasar_utils.utils import parsing
from quasar_utils.utils.utils import val_and_type
from quasar_typing.pathlib import AbsoluteFilePath

logger = getLogger(__name__)

MACHINE_PRECISION = finfo(float).eps

DEFAULT_VALUES: dict[str, Any] = {
    'algo': 'trf',
    'loss': 'linear',
    'maxiter': 100,
    'ftol': 1e-8,
    'xtol': MACHINE_PRECISION,
    'gtol': MACHINE_PRECISION,
    'f_scale': 1.0,
}

@dataclass
class NonLinearInfo(_Info):
    algo: Literal['trf', 'dogbox', 'lm'] | None = field(default=DEFAULT_VALUES['algo'], init=False)
    loss: str = field(default=DEFAULT_VALUES['loss'])
    maxiter: int = field(default=DEFAULT_VALUES['maxiter'])
    ftol: float = field(default=DEFAULT_VALUES['ftol'])
    xtol: float = field(default=DEFAULT_VALUES['xtol'])
    gtol: float = field(default=DEFAULT_VALUES['gtol'])
    f_scale: float = field(default=DEFAULT_VALUES['f_scale'])

    _keys: ClassVar[frozenset[str]] = frozenset([
        'algo', 'loss', 'maxiter', 'ftol', 'xtol', 'gtol', 'f_scale', 'fitter',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {}

    def __hash__(self) -> int:
        return super().__hash__()

    # def __getstate__(self) -> dict:
    #     state = super().__getstate__()
    #     state.pop('fitter')
    #     return state
    
    # def __setstate__(self, state: dict) -> None:
    #     super().__setstate__(state)

    def __getitem__(self, key: str) -> Any:
        if key == 'fitter': 
            return self.fitter
        return super().__getitem__(key)

    def update(self, info) -> None:
        super().update(info, logger)
    
    @cached_property
    def fitter(self) -> FitterInstance:
        return FitterInstance(
            self.algo,
            loss=self.loss,
            maxiter=self.maxiter,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            f_scale=self.f_scale,
        )

    @classmethod
    @validate_call
    def from_file(
        cls,
        path: AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and str(path) in cls._cache.keys():
            logger.debug(f"Using cached 'NonLinearInfo' for '{path}'.")
            
            ninfo = cls._cache[str(path)]
            if create_copy: return ninfo.copy()
            else:           return ninfo

        ninfo: NonLinearInfo = NonLinearInfo()
        if path is None:
            return ninfo

        logger.debug(f"Configuring 'NonLinearInfo' using '{path}':")        
        lines = parsing.get_lines_from_file.__wrapped__('NONLINEAR', path, logger)

        for count, line in enumerate(lines, start=1):
            key = line[0].lower()

            match key:
                case 'algo':
                    key = "_" + key
                    val = parsing.as_str(line[1])
                case 'loss':
                    val = line[1]
                case 'maxiter':
                    val = max([parsing.as_int(line[1]), 1])
                case 'ftol' | 'xtol' | 'gtol' | 'f_scale':
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