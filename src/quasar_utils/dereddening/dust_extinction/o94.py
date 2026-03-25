from numpy import zeros_like, polyval, float64
from astropy.units import Unit
from quasar_typing.numpy import FloatVector
from .basemodel import BaseModel

class O94(BaseModel):
    r"""
    ** MODIFIED CLASS **

    O'Donnell (1994) Milky Way R(V) dependent model

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
    From O'Donnell (1994, ApJ, 422, 158)
      Updates/improves the optical portion of the CCM89 model
    """
    x_unit = Unit('1/micron')
    Rv_range = [2.0, 6.0]
    x_range = [0.3, 10.0]
    ab_cache: dict[str, tuple[FloatVector, FloatVector]] = {}

    @classmethod
    def get_ab_arrays(cls, k: FloatVector) -> tuple[FloatVector, FloatVector]:
        """
        Calculates the `a` and `b` arrays for the O94 extinction curve based 
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
                (-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1), 
                y,
            )
            b[opt_indxs] = polyval(
                (3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0), 
                y,
            )

            # NUV
            y = k[nuv_indxs]
            a[nuv_indxs] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341)
            b[nuv_indxs] = -3.09 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263)

            # far-NUV
            y = k[fnuv_indxs] - 5.9
            a[fnuv_indxs] += -0.04473 * y**2 - 0.009779 * y**3
            b[fnuv_indxs] += 0.2130 * y**2 + 0.1207 * y**3

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