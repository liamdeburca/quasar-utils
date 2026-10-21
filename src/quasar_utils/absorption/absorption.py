__all__ = [
    "fft_approach",
    "fft_test",
    "join_regions",
    "log_fft_test",
    "refine_regions",
    "remove_single",
]

import math
from functools import wraps
from logging import getLogger
from typing import Literal, TypeVar, final, overload

import numpy as np
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt, validate_call
from quasar_typing.misc import WindowSize
from quasar_typing.numpy import (
    BoolArray,
    BoolVector,
    FittableFloatMatrix,
    FittableFloatVector,
    FloatVector,
)
from scipy.fft import fft as scipy_fft
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import gamma

logger = getLogger(__name__)

Mask = TypeVar("Mask", bound=BoolArray)

### Public functions (w/ validation)

@validate_call
def remove_single[Mask](mask: Mask) -> Mask:
    """Remove isolated True pixels from the mask.
    
    A pixel is considered isolated if it is surrounded by False values
    on both sides.
    
    Parameters
    ----------
    mask : numpy.array
        Boolean array to process.
    
    Returns
    -------
    mask : numpy.array
        Modified mask with isolated True pixels removed.
    """
    return _remove_single(mask)


@overload
def log_fft_test(
    z: FittableFloatVector,
    k: NonNegativeInt | None = None,
    base: PositiveFloat | None = None,
) -> tuple[float, float]:
    ...

@overload
def log_fft_test(
    z: FittableFloatMatrix,
    k: NonNegativeInt | None = None,
    base: PositiveFloat | None = None,
) -> tuple[FloatVector, FloatVector]:
    ...

@final
@validate_call
def log_fft_test(
    z: FittableFloatVector | FittableFloatMatrix,
    k: NonNegativeInt | None = None,
    base: PositiveFloat | None = None,
) -> tuple[float | FloatVector, float | FloatVector]:
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
    return _log_fft_test(z, k=k, base=base)



@overload
def fft_test(
    z: FittableFloatVector,
    k: NonNegativeInt | None = None,
) -> tuple[float, float]:
    ...

@overload
def fft_test(
    z: FittableFloatMatrix,
    k: NonNegativeInt | None = None,
) -> tuple[FloatVector, FloatVector]:
    ...

@final
@validate_call
def fft_test(
    z: FittableFloatVector | FittableFloatMatrix,
    k: NonNegativeInt | None = None,
) -> tuple[float | FloatVector, float | FloatVector]:
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
    return _fft_test(z, k=k)


@validate_call
def fft_approach(
    z: FittableFloatVector, 
    p_crit: PositiveFloat,
    z_crit: float, 
    w: WindowSize,
) -> tuple[FloatVector, BoolVector]:
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
    w : WindowSize
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
    return _fft_approach(z, p_crit=p_crit, z_crit=z_crit, w=w)


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
    return _join_regions(mask, iterations=iterations)


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
    return _refine_regions(
        mask,
        x,
        y,
        dy,
        y_smooth,
        y_bg,
        valid_pixels=valid_pixels,
    )


### Private functions (w/o validation, use functools.wraps to inherit docstrings)

@wraps(remove_single)
def _remove_single[Mask](mask: Mask) -> Mask:
    left = np.ones_like(mask, dtype=np.bool_)
    right = left.copy()

    left[..., :-1] = np.invert(mask[..., 1:])
    right[..., 1:] = np.invert(mask[..., :-1])

    return mask & ~(left & mask & right)


@wraps(log_fft_test)
def _log_fft_test(
    z: FittableFloatVector | FittableFloatMatrix,
    k: NonNegativeInt | None = None,
    base: PositiveFloat | None = None,
) -> tuple[float | FloatVector, float | FloatVector]:
    N = z.shape[-1]
    x = scipy_fft(z)

    if k is None and N < 6:
        raise ValueError(f"fft: insufficient input length N={N} for aggregate-frequency statistic; require N>=6")

    if k is None:
        stat = (abs(x[..., 1 : N // 2]) ** 2).sum(axis=-1) / N
        m = int(N // 2 - 2)
        log_p = gamma(m, 1).logsf(stat)
    else:
        if k >= N:
            raise ValueError(f"fft: frequency index k={k} out of range for length N={N}")
        stat = abs(x[..., k]) ** 2 / N
        log_p = -stat

    if base is not None:
        if base <= 1:
            msg = f"The logarithm base must be greater than 1: {base=}"
            logger.critical(msg)
            raise ValueError(msg)
        log_p /= math.log(base)

    return stat, log_p


@wraps(fft_test)
def _fft_test(
    z: FittableFloatVector | FittableFloatMatrix,
    k: NonNegativeInt | None = None,
) -> tuple[float | FloatVector, float | FloatVector]:
    stat, log_p = _log_fft_test(z, k=k, base=None)
    return stat, np.exp(log_p)


@wraps(fft_approach)
def _fft_approach(
    z: FittableFloatVector, 
    p_crit: PositiveFloat,
    z_crit: float, 
    w: WindowSize,
) -> tuple[FloatVector, BoolVector]:
    l: int = int(w // 2)

    z_slices = np.lib.stride_tricks.sliding_window_view(z, w, axis=-1)
    ps = np.ones_like(z, dtype=np.float64, order="C")
    ps[l:-l] = _fft_test(z_slices)[1]

    not_edge = np.ones_like(z, dtype=np.bool_, order="C")
    not_edge[:w] = False
    not_edge[-w:] = False

    mask = binary_dilation(
        _remove_single((ps < p_crit) & (z < z_crit)),
        iterations=(w // 4),
        mask=not_edge,
    )

    return ps, mask


@wraps(join_regions)
def _join_regions(
    mask: BoolVector,
    iterations: PositiveInt | Literal[0],
) -> BoolVector:
    new_mask = binary_erosion(
        binary_dilation(mask, iterations=iterations),
        iterations=iterations,
    )
    return mask | new_mask


@wraps(refine_regions)
def _refine_regions(
    mask: BoolVector,
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    y_smooth: FloatVector,
    y_bg: FloatVector,
    valid_pixels: BoolVector | None = None,
) -> tuple[BoolVector, FloatVector]:
    from .smoothing import _interpolate_missing
    y_smooth = _interpolate_missing(x, y_smooth, np.invert(mask))
    z = _nan_residuals(
        y, 
        np.maximum(y_smooth, y_bg), 
        dy, 
        z_fill=0.0, 
        mask=valid_pixels,
    )
    new_mask = binary_erosion(mask, iterations=0, mask=(z > 0))
    binary_dilation(new_mask, iterations=0, mask=(z < 0), output=new_mask)

    return new_mask, y_smooth

### Utility functions

def _nan_residuals(
    y: FloatVector,
    f: FloatVector,
    dy: FloatVector,
    z_fill: float = 0,
    mask: BoolVector | None = None,
) -> FloatVector:

    if mask is None:
        mask = np.isfinite([y, f]).all(axis=0) & (dy > 0)

    z = np.full_like(y, z_fill)
    z[mask] = (y[mask] - f[mask]) / dy[mask]

    return z