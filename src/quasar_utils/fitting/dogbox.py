from astropy.modeling.fitting import deprecated_renamed_argument
from .baseclasses import _BaseClass

class DogBoxLSQFitter(_BaseClass):
    """
    (Modified) DogBox algorithm and least squares statistic.

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

    @deprecated_renamed_argument("use_min_max_bounds", None, "7.0")
    def __init__(self, calc_uncertainties=False, use_min_max_bounds=False):
        super().__init__("dogbox", calc_uncertainties, use_min_max_bounds)