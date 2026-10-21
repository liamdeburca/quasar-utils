__all__ = [
    "remove_absorption",
    "smooth_spectrum",
]
from functools import wraps

from numpy import arange, invert, isfinite, maximum
from pydantic import validate_call
from quasar_typing.numpy import BoolVector, FloatVector

### Public functions (w/ validation)

@validate_call
def remove_absorption(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    y_smooth: FloatVector,
    y_bg: FloatVector,
    valid_pixels: BoolVector,
    w: int,
    p_crit: float,
    z_crit: float,
    join: int,
    refine: bool,
) -> tuple[FloatVector, BoolVector, FloatVector]:
    """
    Identify and remove absorption features from a spectrum.

    Applies an FFT-based absorption-detection algorithm to identify anomalous
    pixels in the flux density array, optionally joins nearby regions, and
    refines the selection.

    Parameters
    ----------
    x : numpy.array
        Rest wavelength array.
    y : numpy.array
        Flux density array.
    dy : numpy.array
        Flux density uncertainty array.
    y_smooth : numpy.array
        Smoothed flux density array.
    y_bg : numpy.array
        Background flux density array.
    valid_pixels : numpy.array
        Boolean array designating valid pixels.
    w : int
        Window size used in absorption detection.
    p_crit : float
        Critical statistical likelihood for identifying outliers.
    z_crit : float
        Critical residual value for identifying outliers.
    join : int
        Number of iterations for joining nearby regions.
    refine : bool
        Whether to refine the absorption region selection.

    Returns
    -------
    p_absorbed : numpy.array
        P-values from the absorption detection.
    absorbed_pixels : numpy.array
        Boolean array indicating detected absorption pixels.
    y_smooth : numpy.array
        Smoothed flux density array with absorption removed.
    """
    return _remove_absorption(
        x, y, dy, y_smooth, y_bg, valid_pixels, w, p_crit, z_crit, join, refine
    )


@validate_call
def smooth_spectrum(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    valid_pixels: BoolVector,
    w: int,
    p: int,
    logspace: bool,
) -> FloatVector:
    """
    Smooth a spectrum using weighted Savitzky-Golay filtering.

    Applies weighted polynomial smoothing to the flux density array, optionally
    working in log-space.

    Parameters
    ----------
    x : numpy.array
        Rest wavelength array.
    y : numpy.array
        Flux density array.
    dy : numpy.array
        Flux density uncertainty array.
    valid_pixels : numpy.array
        Boolean array designating valid pixels to use in smoothing.
    w : int
        Window size for the Savitzky-Golay filter (must be odd).
    p : int
        Polynomial order for local polynomial fitting.
    logspace : bool
        If True, smoothing is performed in log-space using indices instead of
        wavelengths.

    Returns
    -------
    y_smooth : numpy.array
        Smoothed flux density array.
    """
    return _smooth_spectrum(x, y, dy, valid_pixels, w, p, logspace)


### Private functions (w/o validation, use functools.wraps to inherit docstrings)

@wraps(remove_absorption)
def _remove_absorption(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    y_smooth: FloatVector,
    y_bg: FloatVector,
    valid_pixels: BoolVector,
    w: int,
    p_crit: float,
    z_crit: float,
    join: int,
    refine: bool,
) -> tuple[FloatVector, BoolVector, FloatVector]:
    from .absorption import (
        _fft_approach,
        _join_regions,
        _nan_residuals,
        _refine_regions,
    )
    from .smoothing import _interpolate_missing

    if valid_pixels is None:
        valid_pixels = isfinite([x, y, dy]).all(axis=0) & (dy > 0)

    z = _nan_residuals(
        y,
        maximum(y_smooth, y_bg),
        dy,
        z_fill=0,
        mask=valid_pixels,
    )
    p_absorbed, absorbed_pixels = _fft_approach(
        z,
        p_crit,
        z_crit,
        w,
    )
    if isinstance(join, int):
        absorbed_pixels = _join_regions(
            absorbed_pixels,
            join,
        )

    if refine:
        absorbed_pixels, y_smooth = _refine_regions(
            absorbed_pixels,
            x,
            y,
            dy,
            y_smooth,
            y_bg,
            valid_pixels=valid_pixels,
        )
    else:
        y_smooth = _interpolate_missing(
            x,
            y_smooth,
            invert(absorbed_pixels),
        )

    return p_absorbed, absorbed_pixels, y_smooth


@wraps(smooth_spectrum)
def _smooth_spectrum(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    valid_pixels: BoolVector,
    w: int,
    p: int,
    logspace: bool,
) -> FloatVector:
    from .smoothing import _weighted_savgol_filter
    return _weighted_savgol_filter(
        arange(len(x)) if logspace else x,
        y,
        dy,
        w=w,
        p=p,
        mask=valid_pixels,
    )
