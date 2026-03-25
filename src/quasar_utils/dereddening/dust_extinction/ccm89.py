from numpy import zeros_like, polyval, float64
from astropy.units import Unit
from quasar_typing.numpy import FloatVector
from .basemodel import BaseModel

class CCM89(BaseModel):
    r"""
    ** MODIFIED CLASS **

    Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model

    Parameters
    ----------
    Rv: float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Raises
    ------
    InputParameterError
       Input Rv values outside of defined range

    Notes
    -----
    From Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245)
    """
    x_unit = Unit('1/micron')
    Rv_range = [2.0, 6.0]
    x_range = [0.3, 10.0]
    ab_cache: dict[str, tuple[FloatVector, FloatVector]] = {}

    @classmethod
    def get_ab_arrays(cls, k: FloatVector) -> tuple[FloatVector, FloatVector]:
        """
        Calculates the `a` and `b` arrays for the CCM89 extinction curve based 
        on the input `k` values.

        Notes
        -----
        This method assumes that the input `k` values are in units of micron^-1.
        """
        cache_key: str = cls.get_cache_key(k)
        if cache_key not in cls.ab_cache.keys():        
            a = zeros_like(k, dtype=float64)
            b = zeros_like(k, dtype=float64)

            ir_indxs   = (0.3 <= k) & (k <  1.1)
            opt_indxs  = (1.1 <= k) & (k <  3.3)
            nuv_indxs  = (3.3 <= k) & (k <= 8.0)
            fnuv_indxs = (5.9 <= k) & (k <= 8)
            fuv_indxs  = (8   <  k) & (k <= 10)

            # Infrared
            y = k[ir_indxs] ** 1.61
            a[ir_indxs] = 0.574 * y
            b[ir_indxs] = -0.527 * y

            # NIR/optical
            y = k[opt_indxs] - 1.82
            a[opt_indxs] = polyval(
                (0.32999, -0.7753, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1),
                y,
            )
            b[opt_indxs] = polyval(
                (-2.09002, 5.3026, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0), 
                y,
            )

            # NUV
            y = k[nuv_indxs]
            a[nuv_indxs] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341)
            b[nuv_indxs] = -3.09 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263)

            # far-NUV
            y = k[fnuv_indxs] - 5.9
            a[fnuv_indxs] += y**2 * (-0.04473 - 0.009779 * y)
            b[fnuv_indxs] += y**2 * (0.2130 + 0.1207 * y)

            # FUV
            y = k[fuv_indxs] - 8.0
            a[fuv_indxs] = polyval(
                (-0.070, 0.137, -0.628, -1.073), 
                y,
            )
            b[fuv_indxs] = polyval(
                (0.374, -0.42, 4.257, 13.67), 
                y,
            )

            cls.ab_cache[cache_key] = (a, b)

        return cls.ab_cache[cache_key]