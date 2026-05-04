__all__ = [
    "remove_single",
    "fft",
    "fft_approach",
    "join_regions",
    "refine_regions",
]

from logging import getLogger
from math import log as math_log
from numpy import (
    ones_like, invert, exp, maximum, float64, complex128, bool_,
)
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import fft as scipy_fft
from scipy.stats import gamma
from scipy.ndimage import binary_dilation, binary_erosion

logger = getLogger(__name__)

from pydantic import validate_call
from quasar_typing.numpy import FloatArray, BoolArray, BoolVector, FloatVector

from .smoothing import interpolate_missing
from .utils import nan_residuals

@validate_call
def remove_single(mask: BoolArray) -> BoolArray:
    left: NDArray[bool_] = ones_like(mask, dtype=bool_)
    right: NDArray[bool_] = left.copy()

    left[...,:-1] = invert(mask[...,1:])
    right[...,1:] = invert(mask[...,:-1])

    return mask & ~(left & mask & right)

@validate_call
def fft(
    z: FloatArray, 
    k: int | None = None, 
    log: bool | float | None = 10,
) -> tuple[float | FloatArray, float | FloatArray]:
    """
    Lorem ipsum...
    
    Parameters
    ----------
    z : numpy.array
        Array of residuals. If not 1-dimensional, the last axis runs along the 
        spectral axis.
    k : int, optional
        Index of frequency to consider. If not defined, all frequencies are 
        used. 
    log : bool or float, optional
        Whether to calculate the logarithm of the statistical significance, and
        optionally, what base to use.If True, the natural logarithm is used. If 
        a number is given, the specified base is used. If None, no logarithm is 
        calculated. Default: 10.

    Returns
    -------
    stat : float
        Statistical measure. 
    p : float
        Statistical significance of the measure. 
    """
    N: int = z.shape[-1]
    x: NDArray[complex128] = scipy_fft(z, axis=-1)

    if isinstance(k, int):
        stat: float = abs(x[...,k])**2 / N
        log_p: float = -stat

    else:
        stat: float = (abs(x[...,1:N//2])**2).sum(axis=-1) / N
        m: int = int(N // 2 - 2)
        log_p: float = gamma(m, 1).logsf(stat)

    if isinstance(log, bool) and log is True: 
        p = log_p
    elif isinstance(log, (int, float)) and log > 1: 
        p = log_p / math_log(log)
    else:
        p = exp(log_p)

    return stat, p

@validate_call
def fft_approach(
    z: FloatArray, 
    p_crit: float, 
    z_crit: float, 
    w: int
) -> tuple[FloatArray, BoolArray]:
    """
    ** PYDANTIC VALIDATED FUNCTION **

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

    z_slices: NDArray[float64] = sliding_window_view(z, w, axis=-1) # (N-w+1, w)

    ps: NDArray[float64] = ones_like(z, dtype=float64)
    ps[...,l:-l] = fft.__wrapped__(z_slices, log=False)[1]

    not_edge: NDArray[bool_] = ones_like(z, dtype=bool_)
    not_edge[:w]  = False
    not_edge[-w:] = False

    mask: NDArray[bool_] = binary_dilation(
        remove_single.__wrapped__((ps < p_crit) & (z < z_crit)),
        iterations = (w // 4),
        mask = not_edge,
    )

    return ps, mask

@validate_call
def join_regions(
    mask: BoolArray,
    iterations: int,
) -> BoolArray:
    """
    ** PYDANTIC VALIDATED FUNCTION **

    Joins nearby highlighted regions. 

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
    _mask: NDArray[bool_] = binary_erosion(
        binary_dilation(mask, iterations=iterations),
        iterations = iterations,
    )
    return mask | _mask

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
    ** PYDANTIC VALIDATED FUNCTION **

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
    mask : numpy.array
        Boolean array of the refined selection of anomalous pixels.
    y_smooth : numpy.array
        Refined smoothed flux density array, with rejected pixels' values 
        replaced using a linear interpolator. 
    """
    y_smooth: NDArray[float64] = interpolate_missing.__wrapped__(
        x, y_smooth, invert(mask),
    )
    z: NDArray[float64] = nan_residuals(
        y, 
        maximum(y_smooth,  y_bg), 
        dy, 
        z_fill = 0,
        mask = valid_pixels,
    )
    mask: NDArray[bool_] = binary_dilation(
        binary_erosion(
            mask,
            iterations = 0,
            mask = (z > 0),
        ),
        iterations = 0,
        mask = (z < 0),
    )
    
    return mask, y_smooth