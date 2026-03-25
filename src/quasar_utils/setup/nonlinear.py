from logging import getLogger
logger = getLogger(__name__)

from typing import ClassVar, Self, Any, Literal
from numpy import finfo

from pydantic import validate_call

from .utils._info import _Info
from ..utils.utils import val_and_type
from ..utils import parsing
from ..utils.parsing import get_lines_from_file
from ..fitting import TRFLSQFitter, DogBoxLSQFitter, LMLSQFitter

from quasar_typing.numpy import FittableFloatVector
from quasar_typing.astropy import Fitter_, FitterInstance, Model_, FitInfo
from quasar_typing.pathlib import AbsoluteFilePath


MACHINE_PRECISION = finfo(float).eps

class NonLinearInfo(_Info):
    _keys: ClassVar[frozenset[str]] = frozenset([
        '_algo', 'algo', 'loss', 'maxiter', 'ftol', 'xtol', 'gtol', 'f_scale',
        'fitter',
    ])
    _cache: ClassVar[dict[str, Self]] = {}
    _values_to_update: ClassVar[dict[str, str]] = {
        'algo': "to_algo",
    }

    @validate_call(validate_return=False)
    def __init__(
        self,
        _algo: Literal['trf', 'dogbox', 'lm'] | None = 'trf',
        loss: str = 'linear',
        maxiter: int = 100,
        ftol: float = 1e-8,
        xtol: float = MACHINE_PRECISION,
        gtol: float = MACHINE_PRECISION,
        f_scale: float = 1,
    ):
        self._algo: Literal['trf', 'dogbox', 'lm'] = _algo
        self.loss: str = loss
        self.maxiter: int = maxiter
        self.ftol: float = ftol
        self.xtol: float = xtol
        self.gtol: float = gtol
        self.f_scale: float = f_scale

        self.algo: Fitter_ | None = None

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
        super().update(info, logger)

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

        def fitter(
            model: Model_,
            x: FittableFloatVector,
            y: FittableFloatVector,
            dy: FittableFloatVector,
            inplace: bool = False,
        ) -> tuple[Model_, FitInfo]:
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

    @classmethod
    @validate_call(validate_return=False)
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
    @validate_call(validate_return=False)
    def from_json(
        cls,
        json: dict[str, dict] | AbsoluteFilePath | None = None,
        create_copy: bool = True,
    ) -> Self:
        return super().from_json(json, create_copy, "nonlinear", logger)