import warnings
from collections.abc import Callable

from astropy.modeling.fitting import (
    _convert_input,
    _NonLinearLSQFitter,
    _validate_model,
    fitter_to_model_params,
    fitter_unit_support,
    model_to_fit_params,
)
from astropy.utils.exceptions import AstropyUserWarning
from numpy import dot, finfo, float64, inf, transpose
from quasar_typing.astropy import Model_
from quasar_typing.numpy import FittableFloatVector
from scipy import optimize
from scipy.linalg import svd

LOSSES: dict = {
    "linear": "linear",
    "soft_l1": "soft_l1",
    "huber": "huber",
    "cauchy": "cauchy",
    "arctan": "arctan",
}
PRECISION: float = finfo(float).eps

DEFAULT_MAXITER: int = 100
DEFAULT_FTOL: float = 1e-8
DEFAULT_XTOL: float = PRECISION
DEFAULT_GTOL: float = PRECISION
DEFAULT_EPS: float = float64(1.4901161193847656e-08)


class _NonLinearLSQFitter(_NonLinearLSQFitter):
    """
    (Modified) Base class for Non-Linear least-squares fitters.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds : bool
        If set, the parameter bounds for a model will be enforced for each given
        parameter while fitting via a simple min/max condition.
        Default: True
    """

    supported_constraints = ["fixed", "tied", "bounds"]
    """
    The constraint types supported by this fitter type.
    """

    @staticmethod
    def _wrap_deriv(
        params, model, weights, x, y, z=None, fit_param_indices=None
    ):
        """
        Wraps the method calculating the Jacobian of the function to account
        for model constraints.
        `scipy.optimize.leastsq` expects the function derivative to have the
        above signature (parlist, (argtuple)). In order to accommodate model
        constraints, instead of using p directly, we set the parameter list in
        this function.
        """
        import numpy as np

        if weights is None:
            weights = 1.0

        # print('----- _wrap_deriv -----')
        # print('has_fixed: ', model.has_fixed)
        # if model.has_fixed:
        # print('>>>', [item[0] for item in model.fixed.items() if item[1]])

        # print('has_tied: ', model.has_tied)
        # if model.has_tied:
        # print('>>>', [item[0] for item in model.tied.items() if item[1]])

        if model.has_fixed or model.has_tied:
            # update the parameters with the current values from the fitter
            fitter_to_model_params(model, params)
            # print('z is None:', z is None)
            if z is None:
                # print('model.parameters:', model.parameters.shape, model.parameters)
                full = np.array(model.fit_deriv(x, *model.parameters))
                # print('full shape:', full.shape)
                if not model.col_fit_deriv:
                    full_deriv = np.ravel(weights) * full.T
                else:
                    full_deriv = np.ravel(weights) * full
            else:
                full = np.array(
                    [
                        np.ravel(_)
                        for _ in model.fit_deriv(x, y, *model.parameters)
                    ]
                )
                if not model.col_fit_deriv:
                    full_deriv = np.ravel(weights) * full.T
                else:
                    full_deriv = np.ravel(weights) * full

            # print('full_deriv shape:', full_deriv.shape)

            pars = [getattr(model, name) for name in model.param_names]
            fixed = [par.fixed for par in pars]
            # print('fixed:', fixed)
            tied = [par.tied for par in pars]
            tied = list(
                np.where([par.tied is not False for par in pars], True, tied)
            )
            # print('tied:', tied)
            fix_and_tie = np.logical_or(fixed, tied)
            # print('fix_and_tie:', fix_and_tie)
            ind = np.logical_not(fix_and_tie)
            # print('ind:', ind)
            # print('nonzero ind:', np.nonzero(ind))

            if not model.col_fit_deriv:
                residues = np.asarray(full_deriv[np.nonzero(ind)]).T
            else:
                residues = full_deriv[np.nonzero(ind)]

            # print('residues shape:', residues.shape)
            return [np.ravel(_) for _ in residues]
        else:
            if z is None:
                fit_deriv = np.array(model.fit_deriv(x, *params))
                # print('fit_deriv shape:', fit_deriv.shape)
                try:
                    output = np.array(
                        [np.ravel(_) for _ in np.array(weights) * fit_deriv]
                    )
                    if output.shape != fit_deriv.shape:
                        output = np.array(
                            [
                                np.ravel(_)
                                for _ in np.atleast_2d(weights).T * fit_deriv
                            ]
                        )
                    return output
                except ValueError:
                    return np.array(
                        [
                            np.ravel(_)
                            for _ in np.array(weights)
                            * np.moveaxis(fit_deriv, -1, 0)
                        ]
                    ).transpose()
            else:
                if not model.col_fit_deriv:
                    return [
                        np.ravel(_)
                        for _ in (
                            np.ravel(weights)
                            * np.array(model.fit_deriv(x, y, *params)).T
                        ).T
                    ]
                return [
                    np.ravel(_)
                    for _ in weights * np.array(model.fit_deriv(x, y, *params))
                ]

    @fitter_unit_support
    def __call__(
        self,
        model: Model_,
        x: FittableFloatVector,
        y: FittableFloatVector,
        z: FittableFloatVector = None,
        weights: FittableFloatVector = None,
        maxiter: int = DEFAULT_MAXITER,
        ftol: float = DEFAULT_FTOL,
        xtol: float = DEFAULT_XTOL,
        gtol: float = DEFAULT_GTOL,
        loss: str = "linear",
        f_scale: float = 1,
        epsilon: float = DEFAULT_EPS,
        estimate_jacobian: bool = False,
        filter_non_finite: bool = False,
        *,
        inplace: bool = False,
        warn_me: bool = False,
        verbose: bool = False,
    ) -> Model_:
        """
        Fit data to this model.

        Parameters
        ----------
        model : `~astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array
           input coordinates
        y : array
           input coordinates
        z : array, optional
           input coordinates
        weights : array, optional
            Weights for fitting. For data with Gaussian uncertainties, the weights
            should be 1/sigma.

            .. versionchanged:: 5.3
                Calculate parameter covariances while accounting for ``weights``
                as "absolute" inverse uncertainties. To recover the old behavior,
                choose ``weights=None``.

        maxiter : int
            maximum number of iterations
        ftol : float
            Relative error desired in the approximate solution
        xtol : float
            Relative error desired in the approximate solution
        gtol : float
            Relative error desired in the approximate solution
        epsilon : float
            A suitable step length for the forward-difference
            approximation of the Jacobian (if model.fjac=None). If
            epsfcn is less than the machine precision, it is
            assumed that the relative errors in the functions are
            of the order of the machine precision.
        estimate_jacobian : bool
            If False (default) and if the model has a fit_deriv method,
            it will be used. Otherwise the Jacobian will be estimated.
            If True, the Jacobian will be estimated in any case.
        equivalencies : list or None, optional, keyword-only
            List of *additional* equivalencies that are should be applied in
            case x, y and/or z have units. Default is None.
        filter_non_finite : bool, optional
            Whether or not to filter data with non-finite values. Default is False
        inplace : bool, optional
            If `False` (the default), a copy of the model with the fitted
            parameters set will be returned. If `True`, the returned model will
            be the same instance as the model passed in, and the parameter
            values will be changed inplace.

        Returns
        -------
        fitted_model : `~astropy.modeling.FittableModel`
            If ``inplace`` is `False` (the default), this is a copy of the
            input model with parameters set by the fitter. If ``inplace`` is
            `True`, this is the same model as the input model, with parameters
            updated to be those set by the fitter.

        """
        model_copy: Model_ = _validate_model(
            model,
            self.supported_constraints,
            copy=not inplace,
        )
        model_copy.sync_constraints = False
        _, fit_param_indices, _ = model_to_fit_params(model_copy)

        if filter_non_finite:
            x, y, z, weights = self._filter_non_finite(x, y, z, weights)

        farg = (
            model_copy,
            weights,
        ) + _convert_input(x, y, z)
        fkwarg = {"fit_param_indices": set(fit_param_indices)}

        # import numpy as np

        # print(model)
        # print(getattr(model, 'op', None))

        # x0 = model.parameters
        # print("x0 shape:", x0.shape)
        # print("weights:", None if weights is None else weights.shape, None if weights is None else weights.dtype)
        # test_jac = self._wrap_deriv(x0, model_copy, weights, *farg[2:], fit_param_indices=None)
        # print("test_jac shape:", np.shape(test_jac))
        # fit_deriv = model.fit_deriv(x, *x0)  # or however params are unpacked
        # print("fit_deriv raw output shape:", np.shape(fit_deriv))
        # print("fit_deriv raw output type:", type(fit_deriv))

        init_values, fitparams, cov_x = self._run_fitter(
            model_copy,
            farg,
            fkwarg,
            maxiter,
            ftol,
            xtol,
            gtol,
            loss,
            f_scale,
            epsilon,
            estimate_jacobian,
            warn_me,
            verbose,
        )
        self._compute_param_cov(
            model_copy, y, init_values, cov_x, fitparams, farg, fkwarg, weights
        )

        model_copy.sync_constraints = True
        return model_copy


class _BaseClass(_NonLinearLSQFitter):
    """
    (Modified) Wrapper class for `scipy.optimize.least_squares` method, which
    provides:
        - Trust Region Reflective
        - dogbox
        - Levenberg-Marquardt
    algorithms using the least squares statistic.

    Parameters
    ----------
    method : str
        ‘trf’ :  Trust Region Reflective algorithm, particularly suitable
            for large sparse problems with bounds. Generally robust method.
        ‘dogbox’ : dogleg algorithm with rectangular trust regions, typical
            use case is small problems with bounds. Not recommended for
            problems with rank-deficient Jacobian.
        ‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK.
            Doesn’t handle bounds and sparse Jacobians. Usually the most
            efficient method for small unconstrained problems.
    calc_uncertainties : bool
        If the covariance matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds: bool
        If set, the parameter bounds for a model will be enforced for each given
        parameter while fitting via a simple min/max condition. A True setting
        will replicate how LevMarLSQFitter enforces bounds.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """

    def __init__(
        self,
        method: str,
        calc_uncertainties: bool = False,
        use_min_max_bounds: bool = False,
    ):
        super().__init__(calc_uncertainties, use_min_max_bounds)
        self._method: str = method

    def _run_fitter(
        self,
        model: Model_,
        farg,
        fkwarg,
        maxiter: int,
        ftol: float,
        xtol: float,
        gtol: float,
        loss: str | Callable,
        f_scale: float,
        epsilon: float,
        estimate_jacobian: bool,
        warn_me: bool,
        verbose: bool,
    ):

        if model.fit_deriv is None or estimate_jacobian:
            dfunc = "2-point"
        else:

            def _dfunc(params, model: Model_, weights, *args, **context):
                out = self._wrap_deriv(
                    params,
                    model,
                    weights,
                    *args,
                    fit_param_indices=None,
                )
                return transpose(out) if model.col_fit_deriv else out

            dfunc = _dfunc

        init_values, _, bounds = model_to_fit_params(model)

        # Note, if use_min_max_bounds is True we are defaulting to enforcing bounds
        # using the old method employed by LevMarLSQFitter, this is different
        # from the method that optimize.least_squares employs to enforce bounds
        # thus we override the bounds being passed to optimize.least_squares so
        # that it will not enforce any bounding.
        if self._use_min_max_bounds:
            bounds = (-inf, inf)

        self.fit_info = optimize.least_squares(
            self.objective_function,  # fun
            init_values,  # x0
            jac=dfunc,
            bounds=bounds,
            method=self._method,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            loss=LOSSES.get(loss, "linear"),
            f_scale=f_scale,
            max_nfev=maxiter,
            diff_step=epsilon**0.5,
            verbose=verbose,
            args=farg,
            kwargs=fkwarg,
        )

        # Adapted from ~scipy.optimize.minpack, see:
        # https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/optimize/minpack.py#L795-L816
        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(self.fit_info.jac, full_matrices=False)
        threshold = PRECISION * max(self.fit_info.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[: s.size]
        cov_x = dot(VT.T / s**2, VT)

        fitter_to_model_params(model, self.fit_info.x, False)
        if not self.fit_info.success and warn_me:
            warnings.warn(
                f"The fit may be unsuccessful; check: \n    {self.fit_info.message}",
                AstropyUserWarning,
            )

        return init_values, self.fit_info.x, cov_x
