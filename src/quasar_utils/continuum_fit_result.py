__all__ = ['ContinuumFitResult']

from typing import Self
from numpy import diag, zeros_like, isclose
from numpy.linalg import det
from scipy.stats._multivariate import multivariate_normal, multivariate_normal_frozen
from itertools import repeat

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.numpy import FloatVector, FloatMatrix
from quasar_models.continuum import PowerLawModel

class ContinuumFitResult:
    @validate_call
    def __init__(
        self,
        mean: FloatVector,
        cov: FloatMatrix,
        x0: float | None = None,
        y0: float | None = None,
    ):
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        assert mean.ndim == 1, 'Mean vector must be 1D.'
        assert mean.size == 2, 'Mean vector must have size 2.'
        assert cov.ndim == 2, 'Covariance matrix must be 2D.'
        assert cov.shape == (2, 2), 'Covariance matrix must be 2x2.'

        self.mean: FloatVector = mean
        self.cov:  FloatMatrix = cov
        self.std:  FloatVector = diag(cov)**0.5

        _var = self.std[:,None] * self.std[None,:]
        is_valid = ~isclose(_var, 0)
        
        self.corr = zeros_like(cov)
        self.corr[is_valid] = self.cov[is_valid] / _var[is_valid]

        _det = det(cov)
        if isclose(_det, 0):
            cov = diag(diag(cov))

        self.dist: multivariate_normal_frozen = multivariate_normal(mean, cov)

        # Rerefence wavelength used for PLWiggle model
        self.x0: float = x0 or 1450
        self.y0: float = y0 or 1

    @classmethod
    def _validate(cls, value) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected {cls} instance, \
                got {type(value).__name__}"
            raise PydanticCustomError("validation_error", msg)
        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler):
        return no_info_plain_validator_function(cls._validate)

    def __call__(
        self,
        method_type: str,
        iterations: int,
        random_state,
    ) -> None:
        match method_type:
            case 'default' | 'wiggle' | 'fixed':
                out = self.mean \
                    if iterations == 1 \
                    else repeat(self.mean, iterations)
            case 'full' | 'continuum':
                out = self.dist.rvs(
                    size = iterations,
                    random_state = random_state
                )
            case 'ellipse' | 'square':
                msg = f"Please implement '{method_type}' method type."
                raise NotImplementedError(msg)

        return out
    
    def toPowerLawModel(
        self,
        n_sigmas: float | None = None,
    ) -> PowerLawModel:
        powerlaw_model = PowerLawModel(self.x0, self.y0, *self.mean)        
        if n_sigmas is not None:
            powerlaw_model.flux.bounds = (
                self.mean[0] - n_sigmas * self.std[0],
                self.mean[0] + n_sigmas * self.std[0],
            )
            powerlaw_model.alpha_bounds = (
                self.mean[1] - n_sigmas * self.std[1],
                self.mean[1] + n_sigmas * self.std[1],
            )

        return powerlaw_model

    # def toWiggleModel(
    #     self,
    #     n_sigmas: Optional[float],
    # ) -> PLWiggleLike:
    #     from ..models.pl_wiggle import PLWiggle

    #     plwiggle_model = PLWiggle(self.x0, self.y0, *self.mean, *self.mean)
    #     if n_sigmas is not None:
    #         plwiggle_model.flux.bounds = (
    #             self.mean[0] - n_sigmas * self.std[0],
    #             self.mean[0] + n_sigmas * self.std[0]
    #         )
    #         plwiggle_model.alpha.bounds = (
    #             self.mean[1] - n_sigmas * self.std[1],
    #             self.mean[1] + n_sigmas * self.std[1]
    #         )

    #     return plwiggle_model
