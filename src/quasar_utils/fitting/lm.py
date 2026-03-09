import warnings
from astropy.modeling.fitting import fitter_unit_support
from astropy.utils.exceptions import AstropyDeprecationWarning

from .baseclasses import _BaseClass, DEFAULT_MAXITER, DEFAULT_FTOL, \
    DEFAULT_XTOL, DEFAULT_GTOL, DEFAULT_EPS

from quasar_typing.numpy import FittableFloatVector
from quasar_typing.astropy import Fittable1DModel_, CompoundModel_

class LMLSQFitter(_BaseClass):
    """
    `scipy.optimize.least_squares` Levenberg-Marquardt algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """

    def __init__(self, calc_uncertainties=False):
        super().__init__("lm", calc_uncertainties, True)

    @fitter_unit_support
    def __call__(
        self,
        model: Fittable1DModel_ | CompoundModel_,
        x: FittableFloatVector,
        y: FittableFloatVector,
        z: FittableFloatVector = None,
        weights: FittableFloatVector = None,
        maxiter: int = DEFAULT_MAXITER,
        ftol: float = DEFAULT_FTOL,
        xtol: float = DEFAULT_XTOL,
        gtol: float = DEFAULT_GTOL,
        f_scale: float = 1,
        epsilon: float = DEFAULT_EPS,
        estimate_jacobian: bool = False,
        filter_non_finite: bool = False,
        *,
        inplace: bool = False,
        warn_me: bool = False,
        verbose: bool = False,
    ):
        # Since there are several fitters with proper support for bounds, it
        # is not a good idea to keep supporting the hacky bounds algorithm
        # from LevMarLSQFitter here, and better to communicate with users
        # that they should use another fitter. Once we remove the deprecation,
        # we should update ``supported_constraints`` and change ``True`` to
        # ``False`` in the call to ``super().__init__`` above.
        if model.has_bounds:
            warnings.warn(
                "Using LMLSQFitter for models with bounds is now "
                "deprecated since astropy 7.0. We recommend you use another non-linear "
                "fitter such as TRFLSQFitter or DogBoxLSQFitter instead "
                "as these have full support for fitting models with "
                "bounds",
                AstropyDeprecationWarning,
                stacklevel=2,
            )
        return super().__call__(
            model,
            x,
            y,
            z = z,
            weights = weights,
            maxiter = maxiter,
            ftol = ftol,
            xtol = xtol,
            gtol = gtol,
            loss = 'linear',
            f_scale = f_scale,
            epsilon = epsilon,
            estimate_jacobian = estimate_jacobian,
            filter_non_finite = filter_non_finite,
            inplace = inplace,
            warn_me = warn_me,
            verbose = verbose,
        )