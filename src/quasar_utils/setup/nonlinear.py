from logging import getLogger
from typing import ClassVar, Self, Any
from numpy import finfo

from pydantic import validate_call
from pydantic.dataclasses import dataclass

from . import parsing
from ..utils._info import _Info
from ..utils.utils import val_and_type
from ..utils.parsing import get_lines_from_file
from .. import fitting

from quasar_typing.numpy import FittableFloatVector
from quasar_typing.astropy import Fitter_, FitterInstance, Fittable1DModel_, \
    CompoundModel_, FitInfo_
from quasar_typing.pathlib import Path_, AbsoluteFileLike

logger = getLogger(__name__)

MACHINE_PRECISION = finfo(float).eps

@dataclass
class NonLinearInfo(_Info):
    algo: Fitter_ = fitting.TRFLSQFitter
    loss: str = 'linear'

    maxiter: int = 100
    ftol: float = 1e-8
    xtol: float = MACHINE_PRECISION
    gtol: float = MACHINE_PRECISION
    f_scale: float = 1

    _keys: ClassVar[frozenset[str]] = frozenset([
        'algo', 'loss', 'maxiter', 'ftol', 'xtol', 'gtol', 'f_scale',
        'fitter',
    ])
    _cache: ClassVar[dict[Path_, Self]] = {}

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('fitter') # Fitter cannot be pickled
        return state
    
    @classmethod
    def __setstate__(cls, state: dict):
        cls._keys: frozenset[str] = frozenset(state.pop('_keys'))
        for key in filter(lambda key: key != 'fitter', cls._keys):
            setattr(cls, key, state[key])

    def __getitem__(self, key: str) -> Any:
        if key == 'fitter': return self.fitter
        else:               return super().__getitem__(key)

    def update(self, info) -> None:
        logger.debug("Updating 'NonLinearInfo' class (does nothing):")

    @property
    def kwargs(self) -> dict[str, float | int | str]:
        return dict(
            loss = self['loss'],
            maxiter = self['maxiter'],
            ftol = self['ftol'],
            xtol = self['xtol'],
            gtol = self['gtol'],
            f_scale = self['f_scale'],
        )
    
    @property
    def fitter(self) -> FitterInstance:        
        algo = self['algo'](calc_uncertainties=True)
        kwargs = self.kwargs

        @validate_call
        def fitter(
            model: Fittable1DModel_ | CompoundModel_,
            x: FittableFloatVector,
            y: FittableFloatVector,
            dy: FittableFloatVector,
            inplace: bool = False,
        ) -> tuple[Fittable1DModel_ | CompoundModel_, FitInfo_]:
            fit = algo(
                model,
                x,
                y,
                weights = 1 / dy,
                inplace = inplace,
                **kwargs,
            )
            return fit, algo.fit_info
        
        return fitter

    @validate_call
    @classmethod
    def from_file(
        cls,
        path: AbsoluteFileLike | None = None,
        create_copy: bool = True,
    ) -> Self:

        if path is not None and path in cls._cache.keys():
            logger.debug(f"Using cached 'NonLinearInfo' for '{path}'.")
            
            ninfo = cls._cache[path]
            if create_copy: return ninfo.copy()
            else:           return ninfo

        ninfo: NonLinearInfo = NonLinearInfo()
        if path is None:
            return ninfo

        logger.debug(f"Configuring 'NonLinearInfo' using '{path}':")        
        lines = get_lines_from_file.__wrapped__(logger, 'NONLINEAR', path)

        for count, line in enumerate(lines, start=1):
            key = line[0].lower()

            match key:
                case 'algo':
                    val = getattr(fitting, line[1])
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

        cls._cache[path] = ninfo

        return ninfo