from logging import getLogger
from functools import cached_property
from typing import ClassVar, Self, Any, Literal
from numpy import finfo
from dataclasses import field
from pydantic.dataclasses import dataclass
from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file
from ..fitting import TRFLSQFitter, DogBoxLSQFitter, LMLSQFitter

from quasar_typing.numpy import FittableFloatVector
from quasar_typing.astropy import Fitter_, FitterInstance, Model_, FitInfo
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
    'f_scale': 1,
}

@dataclass
class NonLinearInfo(_Info):
    algo: Literal['trf', 'dogbox', 'lm'] | None = field(default=DEFAULT_VALUES['algo'], init=False)
    loss: str = field(default=DEFAULT_VALUES['loss'])
    maxiter: int = field(default=DEFAULT_VALUES['maxiter'])
    ftol: float = field(default=DEFAULT_VALUES['ftol'])
    xtol: float = field(default=DEFAULT_VALUES['xtol'])
    gtol: float = field(default=DEFAULT_VALUES['gtol'])
    f_scale: float = field(default=1)

    _keys: ClassVar[frozenset[str]] = frozenset([
        'algo', 'algo', 'loss', 'maxiter', 'ftol', 'xtol', 'gtol', 'f_scale',
        'algorithm', 'fitter',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {}
    
    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.pop('algorithm')
        state.pop('fitter')
        return state
    
    @classmethod
    def __setstate__(cls, state: dict):
        cls._keys: frozenset[str] = frozenset(state.pop('_keys'))
        for key in filter(lambda key: key not in ('fitter', 'algorithm'), cls._keys):
            setattr(cls, key, state[key])

    def __getitem__(self, key: str) -> Any:
        if key == 'fitter': 
            return self.fitter
        return super().__getitem__(key)

    def update(self, info) -> None:
        super().update(info, logger)

    @cached_property
    def kwargs(self) -> dict[str, float | int | str]:
        return dict(
            loss = self['loss'],
            maxiter = self['maxiter'],
            ftol = self['ftol'],
            xtol = self['xtol'],
            gtol = self['gtol'],
            f_scale = self['f_scale'],
        )
    
    @cached_property
    def algorithm(self) -> Fitter_:
        return {
            'trf': TRFLSQFitter,
            'dogbox': DogBoxLSQFitter,
            'lm': LMLSQFitter,
        }[self['algo']]
    
    @cached_property
    def fitter(self) -> FitterInstance:        
        algo = self.algorithm(calc_uncertainties=True)
        kwargs = self.kwargs

        def fitter_instance(
            model: Model_,
            x: FittableFloatVector,
            y: FittableFloatVector,
            dy: FittableFloatVector,
            inplace: bool = False,
        ) -> tuple[Model_, FitInfo]:
            nonlocal algo, kwargs
            fit = algo(
                model,
                x,
                y,
                weights = 1 / dy,
                inplace = inplace,
                **kwargs,
            )
            return fit, algo.fit_info
        
        return fitter_instance

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
        lines = get_lines_from_file.__wrapped__('NONLINEAR', path, logger)

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