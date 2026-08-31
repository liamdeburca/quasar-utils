__all__ = [
    "fft_approach",
    "fft_test",
    "join_regions",
    "log_fft_test",
    "refine_regions",
    "remove_single",
]

from logging import getLogger
from math import log as math_log
from typing import Literal

from numpy import (
    bool_,
    complex128,
    exp,
    float64,
    invert,
    maximum,
    ones_like,
)
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from quasar_typing.numpy import (
    BoolArray,
    BoolVector,
    FittableFloatArray,
    FloatArray,
    FloatVector,
)
from scipy.fft import fft as scipy_fft
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import gamma

from ..decorators import validate_call
from .smoothing import interpolate_missing
from .utils import nan_residuals

logger = getLogger(__name__)

@validate_call
def remove_single(mask: BoolArray) -> BoolArray:
    left: NDArray[bool_] = ones_like(mask, dtype=bool_)
    right: NDArray[bool_] = left.copy()

    left[..., :-1] = invert(mask[..., 1:])
    right[..., 1:] = invert(mask[..., :-1])

    return mask & ~(left & mask & right)


@validate_call
def log_fft_test(
    z: FittableFloatArray,
    k: NonNegativeInt | None = None,
    base: PositiveFloat | None = None,
) -> tuple[float | FloatArray, float | FloatArray]:
    """Perform the FFT-based normality test on the residuals.

    Parameters
    ----------
    z : numpy.array
        Array of residuals. If not 1-dimensional, the last axis runs along the
        spectral axis.
    k : int, optional
        Index of frequency to consider. If not defined, all frequencies are
        used.
    base : float, optional
        Base of the logarithm used for the statistical significance. If not
        defined, the natural logarithm is used.

    Returns
    -------
    stat : float
        Statistical measure.
    log_p : float
        Statistical significance of the measure in logarithmic form. If `base`
        is specified, this is the logarithm of the statistical significance with
        the given base.

    Raises
    ------
    ValueError
        - If the input array is shorter than `N=6` elements.
        - If the frequency index `k` is out of range for the input array: `k>=N`
        - If the logarithm base is less than or equal to 1.
    """
    N: int = z.shape[-1]
    x: NDArray[complex128] = scipy_fft(z, axis=-1)

    # Validate small-N cases for aggregate statistic
    if k is None and N < 6:
        raise ValueError(f"fft: insufficient input length N={N} for aggregate-frequency statistic; require N>=6")

    if k is None:
        stat: float = (abs(x[..., 1 : N // 2]) ** 2).sum(axis=-1) / N
        m: int = int(N // 2 - 2)
        log_p: float = gamma(m, 1).logsf(stat)
    else:
        # validate provided frequency index
        if k >= N:
            raise ValueError(f"fft: frequency index k={k} out of range for length N={N}")
        stat: float = abs(x[..., k]) ** 2 / N
        log_p: float = -stat

    if base is not None:
        if base <= 1:
            msg = f"The logarithm base must be greater than 1: {base=}"
            logger.critical(msg)
            raise ValueError(msg)
        log_p /= math_log(base)

    return stat, log_p

@validate_call
def fft_test(
    z: FittableFloatArray,
    k: NonNegativeInt | None = None,
) -> tuple[float | FloatArray, float | FloatArray]:
    """Perform the FFT-based normality test on the residuals.

    See `log_fft_test` for details.

    Parameters
    ----------
    z : numpy.array
        Array of residuals. If not 1-dimensional, the last axis runs along the
        spectral axis.
    k : int, optional
        Index of frequency to consider. If not defined, all frequencies are
        used.

    Returns
    -------
    stat : float
        Statistical measure.
    p : float
        Statistical significance of the measure.

    Raises
    ------
    ValueError
        See `log_fft_test` for details.
    """
    stat, log_p = log_fft_test.__wrapped__(z, k=k, base=None)
    return stat, exp(log_p)

@validate_call
def fft_approach(
    z: FittableFloatArray, 
    p_crit: PositiveFloat,
    z_crit: float, 
    w: PositiveInt,
) -> tuple[FloatArray, BoolArray]:
    """
    Applies an absorption-identification approach based on a normality test
    based on the discrete Fourier transform. Residuals are assumed to be sampled
    from a (standard) normal distribution.

    Parameters
    ----------
    z : numpy.array
        Array of residuals.
    p_crit : float
        Critical statistical likelihood used for identifying outlying pixels.
    z_crit : float
        Critical residual value used for identifying outlying pixels.
    w : int
        Window size used when scanning the residual array.

    Returns
    -------
    mask : numpy.array
        Boolean array with True when pixel is considered anomalous, False when
        not.

    Notes
    -----
    I suggest:
    >   p_crit = 1e-2
    >   z_crit = -2
    >   w = 25
    """
    l: int = int(w // 2)

    z_slices: NDArray[float64] = sliding_window_view(
        z, w, axis=-1
    )  # (N-w+1, w)

    ps: NDArray[float64] = ones_like(z, dtype=float64, order="C")
    # Request actual p-values (not logarithms) from fft for the sliding windows.
    ps[..., l:-l] = fft_test.__wrapped__(z_slices)[1]

    not_edge: NDArray[bool_] = ones_like(z, dtype=bool_, order="C")
    not_edge[:w] = False
    not_edge[-w:] = False

    mask: NDArray[bool_] = binary_dilation(
        remove_single.__wrapped__((ps < p_crit) & (z < z_crit)),
        iterations=(w // 4),
        mask=not_edge,
    )

    return ps, mask


@validate_call
def join_regions(
    mask: BoolVector,
    iterations: PositiveInt | Literal[0],
) -> BoolVector:
    """Joins nearby highlighted regions.

    Parameters
    ----------
    mask : numpy.array
        Boolean array where pixels with True values are potentially joined.
    iterations : int
        Number of iterations to perform. Using n iterations, all True pixels
        separated by at most 2n False pixels will be joined.

    Returns
    -------
    mask : numpy.array
        Modification of the input mask with True pixels potentially joined.
    """
    new_mask: NDArray[bool_] = binary_erosion(
        binary_dilation(mask, iterations=iterations),
        iterations=iterations,
    )
    return mask | new_mask


@validate_call
def refine_regions(
    mask: BoolVector,
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    y_smooth: FloatVector,
    y_bg: FloatVector,
    valid_pixels: BoolVector | None = None,
) -> tuple[BoolVector, FloatVector]:
    """
    Refines the selection of outlying pixels using linear interpolation.

    Parameters
    ----------
    mask : numpy.array
        Boolean array where anomalous pixels are True.
    x : numpy.array
        Rest wavelength array.
    y : numpy.array
        Flux density array.
    dy : numpy.array
        Flux density uncertainty array.
    y_smooth : numpy.array
        Initial smoothed flux density array.

    Returns
    -------
    new_mask : numpy.array
        Boolean array of the refined selection of anomalous pixels.
    y_smooth : numpy.array
        Refined smoothed flux density array, with rejected pixels' values
        replaced using a linear interpolator.
    """
    y_smooth = interpolate_missing.__wrapped__(x, y_smooth, invert(mask))
    z = nan_residuals(y, maximum(y_smooth, y_bg), dy, z_fill=0, mask=valid_pixels)
    new_mask = binary_erosion(mask, iterations=0, mask=(z > 0))
    binary_dilation(new_mask, iterations=0, mask=(z < 0), output=new_mask)

    return new_mask, y_smooth
