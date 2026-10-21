__all__ = [
    "create_slides",
    "get_gap_sizes",
    "get_valid_indices",
    "interpolate_missing",
    "solve_weighted_poly",
    "weighted_savgol_filter",
]

from functools import wraps
from itertools import batched
from logging import getLogger
from typing import Literal

import numpy as np
from pydantic import PositiveInt, validate_call
from quasar_typing.errors import SmoothingError
from quasar_typing.misc import PolynomialOrder, WindowSize
from quasar_typing.numpy import (
    BoolVector,
    FloatMatrix,
    FloatVector,
    SortedFloatVector,
)

logger = getLogger(__name__)


### Public functions (w/ validation)

@validate_call
def get_gap_sizes(mask: BoolVector) -> list[tuple[int, int, int]]:
    """
    Calculates the sizes of gaps in the boolean array, i.e. a list of left
    index, right index, and size triplets.

    A gap is defined as a contiguous sequence of False values.

    Notes
    -----
    Edge cases are handled as follows:

    * If the array is empty, an empty list is returned.
    * If the array has one element, a gap is returned if that element is False.
    * If the array has more than one element:,
        * And all values are False, a single gap covering the entire array is
        returned.
        * And the first and last values are both True, the gaps are calculated
        between the first and last True values.
        * Otherwise, the gaps are calculated between the first and last True
        values, and additional gaps are added for the left and right edges as
        necessary.
    """
    return _get_gap_sizes(mask)


@validate_call
def interpolate_missing(
    x: SortedFloatVector,
    y: FloatVector,
    mask: BoolVector,
    out: FloatVector | None = None,
) -> FloatVector | None:
    """
    Linearly interpolate y-values based on the mask.

    Parameters
    ----------
    x : numpy.array
        Array along the first axis.
    y : numpy.array
        Array along the second axis. Assigned pixels' values will be linearly
        interpolated.
    mask : numpy.array
        Boolean array designating which values to linearly interpolate. A value
        of False replaces the original value.
    out : numpy.array, optional
        Optional output array to store the interpolated values. If None, a new
        array is created and returned.

    Returns
    -------
    numpy.array or None
        Array along the second axis with designated pixels' values interpolated.
        If `out` is provided, the function returns None and modifies `out` in 
        place.

    Raises
    ------
    ValueError
        If the mask does not contain at least one True value for interpolation.
    """
    return _interpolate_missing(x, y, mask, out)


@validate_call
def get_valid_indices(
    mask: BoolVector,
    w: WindowSize,
    p: PolynomialOrder,
    side: Literal["left", "right"] | None = None,
    mode: Literal["flexible", "rigid", "semi-rigid", "standard"] = "standard",
) -> BoolVector:
    """Calculate where the flux density array can be smoothed adequately.

    Parameters
    ----------
    mask : numpy.array
        Boolean array with True values signifying existant pixels.
    w : int
        Window size used for smoothing.
    p : int
        Polynomial order used for local polynomial fitting when smoothing.
    side : Literal['left', 'right'] | None, optional
        Whether to use a centred, or left- or right-biased window kernel.
    mode : Literal['flexible', 'rigid', 'semi-rigid', 'standard']
        Determines how many pixels are required to adequately smooth a pixel in
        the flux density array. Default is 'standard'.

        Within the sliding window:
        * 'flexible': exactly avoids over-fitting.
        * 'rigid': all pixels must be valid.
        * 'semi-rigid': more than 3/4 pixels must be valid.
        * 'standard': more than 1/2 pixels must be valid.

    Returns
    -------
    points_are_covered : numpy.array
        Boolean array with True when the window centred on the pixel has an
        adequate number of valid pixels.
    """
    return _get_valid_indices(mask, w, p, side, mode)


@validate_call
def create_slides(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: WindowSize,
    mask: BoolVector | None = None,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    """
    Takes input array and creates slides used for smoothing.

    Parameters
    ----------
    x : numpy.array (1d)
        Rest wavelength array.
    y : numpy.array (1d)
        Flux density array.
    dy : numpy.array (1d)
        Flux density uncertainty array.
    w : PositiveInt
        Window size used for smoothing.
    mask : numpy.array
        Boolean array with False in pixels whose smoothed values aren't
        calculated using Savitzky-Golay smoothing.

    Returns
    -------
    x_slides : numpy.array (2d)
        Slides of the rest wavelength array.
    y_slides : numpy.array (2d)
        Slides of the flux density array.
    dy_slides : numpy.array (2d)
        Slides of the flux density uncertainty array.

    Notes
    -----
    The returned 2d numpy.arrays have shapes of (# of slides, window size).
    """
    return _create_slides(x, y, dy, w, mask)


@validate_call
def solve_weighted_poly(
    x_slides: FloatMatrix,
    y_slides: FloatMatrix,
    dy_slides: FloatMatrix,
    p: PolynomialOrder,
    full: bool = False,
) -> FloatVector | FloatMatrix:
    """
    Fits p-order polynomials for all coordinate-slides in parallel.

    Parameters
    ----------
    x_slides : numpy.array
        Slides of the rest wavelength array.
    y_slides : numpy.array
        Slides of the flux density array.
    dy_slides : numpy.array
        Slides of the flux density error array.
    p : int
        Order of the polynomial to fit at each slide.
    full : bool
        Whether to return all polynomial coefficients (True) or first
        coefficient (False). Default is False.

    Returns
    -------
    solutions : numpy.array
        Fitted polynomial coefficients for all slides. If full is False, only
        the first coefficients are returned, which may be directly used as the
        smoothed flux density array.
    """
    return _solve_weighted_poly(x_slides, y_slides, dy_slides, p, full)


@validate_call
def weighted_savgol_filter(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: WindowSize,
    p: PolynomialOrder,
    mask: BoolVector | None = None,
    interpolate: bool = True,
    mode: str = "standard",
) -> FloatVector:
    """
    Performs weighted Savitzky-Golay smoothing on the input spectrum.

    Parameters
    ----------
    x : numpy.array
        Rest wavelength array.
    y : numpy.array
        Flux density array.
    dy : numpy.array
        Flux density uncertainty array.
    w : WindowSize
        Window size used for smoothing.
    p : PolynomialOrder
        Order of the polynomial to fit at each slide.
    mask : numpy.array, optional
        Boolean array designating which pixels' values NOT to smooth (when
        False).
    interpolate : bool, True
        Whether to linearly interpolate pixels' smoothed values when the mask is
        False or the pixel is invalid. If False, the original flux density value
        is used instead of interpolating. Default is True.
    mode : {'flexible', 'rigid', 'semi-rigid', 'standard'}
        Determines how many pixels are required to adequately smooth a pixel in
        the flux density array. Default is 'standard'.

        Within the sliding window:
         * 'flexible': exactly avoids over-fitting.
         * 'rigid': all pixels must be valid.
         * 'semi-rigid': more than 3/4 pixels must be valid.
         * 'standard': more than 1/2 pixels must be valid.

    Returns
    -------
    y_smooth : numpy.array
        Array of smoothed (and possible interpolated) flux density values.

    Raises
    ------
    ValidationError
        If Pydantic validation fails for the input parameters.
    SmoothingError
        If window and polynomial parameter validation fails.
    """
    return _weighted_savgol_filter(x, y, dy, w, p, mask, interpolate, mode)


### Private functions (w/o validation, use functools.wraps to inherit docstrings)

@wraps(get_gap_sizes)
def _get_gap_sizes(mask: BoolVector) -> list[tuple[int, int, int]]:
    n_pix = mask.size
    match n_pix:
        case 0:
            return []
        case 1:
            return [(0, 0, 1)] if not mask[0] else []
        case _:
            if not mask.any():
                return [(0, n_pix - 1, n_pix)]
            if not (mask[0] and mask[-1]):
                true_indices = np.argwhere(mask).flatten()
                _mask = mask[true_indices[0] : true_indices[-1] + 1]

                out = _get_gap_sizes(_mask)
                if not mask[0]:
                    out = [
                        (
                            left_idx + true_indices[0],
                            right_idx + true_indices[0],
                            size,
                        )
                        for (left_idx, right_idx, size) in out
                    ]

                    left = 0
                    right = true_indices[0] - 1
                    out.insert(0, (left, right, right - left + 1))
                if not mask[-1]:
                    left = true_indices[-1] + 1
                    right = len(mask) - 1
                    out.append((left, right, right - left + 1))

                return out

            edges = np.argwhere(mask[:-1] ^ mask[1:]).flatten()

            return [
                (left + 1, right, right - left)
                for (left, right) in batched(edges, 2)
            ]


@wraps(interpolate_missing)
def _interpolate_missing(
    x: SortedFloatVector,
    y: FloatVector,
    mask: BoolVector,
    out: FloatVector | None = None,
) -> FloatVector | None:
    _x = x[mask]
    _y = y[mask]
    if _x.size == 0:
        raise ValueError("'mask' must contain at least one True value for interpolation.")
    elif _x.size == 1:
        if out is None:
            return np.full_like(y, _y[0])
        out.fill(_y[0])
    else:
        _out = np.interp(x, _x, _y, left=_y[0], right=_y[-1])
        if out is None:
            return _out
        out[:] = _out

    return None


@wraps(get_valid_indices)
def _get_valid_indices(
    mask: BoolVector,
    w: WindowSize,
    p: PolynomialOrder,
    side: Literal["left", "right"] | None = None,
    mode: Literal["flexible", "rigid", "semi-rigid", "standard"] = "standard",
) -> BoolVector:
    match side:
        case "left":
            kernel = np.append(np.ones(w), np.zeros(w - 1))
        case "right":
            kernel = np.append(np.zeros(w - 1), np.ones(w))
        case _:
            kernel = np.ones(w)

    n_points_covered = np.convolve(mask.astype(int), kernel, mode="same")

    match mode:
        case "flexible":
            n_points_minimum = p + 2
        case "rigid":
            n_points_minimum = w
        case "semi-rigid":
            n_points_minimum = 3 * (w // 4)
        case "standard":
            n_points_minimum = (w // 2) + 1

    return n_points_covered >= n_points_minimum


@wraps(create_slides)
def _create_slides(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: WindowSize,
    mask: BoolVector | None = None,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    l = w // 2
    filler = np.nan * np.ones(l)
    _fill = lambda arr: np.concatenate([filler, arr, filler], axis=0)

    f = lambda arr: np.lib.stride_tricks.sliding_window_view(
        _fill(arr), 
        window_shape=w,
    )

    x_slides = f(x) - x[:, None]
    y_slides = f(y)
    dy_slides = f(dy)

    if mask is not None:
        x_slides = x_slides[mask]
        y_slides = y_slides[mask]
        dy_slides = dy_slides[mask]

    return x_slides, y_slides, dy_slides


@wraps(solve_weighted_poly)
def _solve_weighted_poly(
    x_slides: FloatMatrix,
    y_slides: FloatMatrix,
    dy_slides: FloatMatrix,
    p: PositiveInt,
    full: bool = False,
) -> FloatVector | FloatMatrix:
    n_slides = y_slides.shape[0]
    coeffs = np.zeros((n_slides, p + 1), dtype=np.float64)

    for i in range(n_slides):
        x_s = x_slides[i]
        y_s = y_slides[i]
        dy_s = dy_slides[i]

        valid = np.isfinite([x_s, y_s, dy_s]).all(axis=0) &  (dy_s > 0)
        if valid.sum() == 0:
            coeffs[i, :] = np.nan
            continue

        X = np.polynomial.polynomial.polyvander(x_s[valid], deg=p)

        sqrt_w = 1.0 / dy_s[valid]
        Xw = X * sqrt_w[:, None]
        yw = y_s[valid] * sqrt_w

        sol, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        coeffs[i, :] = sol

    return coeffs if full else coeffs[:, 0]


@wraps(weighted_savgol_filter)
def _weighted_savgol_filter(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: WindowSize,
    p: PolynomialOrder,
    mask: BoolVector | None = None,
    interpolate: bool = True,
    mode: str = "standard",
) -> FloatVector:
    if w <= 0:
        raise SmoothingError("w must be a positive integer.")
    if w % 2 == 0:
        raise SmoothingError("w must be odd.")
    if w <= p:
        raise SmoothingError("w must be greater than polynomial order p.")
    if x.shape[0] < w:
        raise SmoothingError(
            f"Input array length ({x.shape[0]}) must be at least "
            f"window size w ({w})."
        )

    y_smooth = y.copy()

    x_valid = np.isfinite(x)
    if mask is None:
        mask = x_valid & np.isfinite([y, dy]).all(axis=0) & (dy > 0)

    gap_sizes = _get_gap_sizes(mask)
    if gap_sizes:
        msg = f"Identified the following gaps ('left', 'right', 'size'): {gap_sizes}."
        logger.debug(msg)

    valid_indices = _get_valid_indices(mask, w, p, mode=mode)
    valid_indices &= x_valid

    if not valid_indices.any():
        msg = f"Cannot smooth the spectrum w/ {w=}, {p=}, and {mode=}: "
        msg += "returning copy of original flux density array."
        logger.info(msg)
        return y_smooth

    slides = _create_slides(
        np.where(x_valid, x, np.nan),
        np.where(mask, y, np.nan),
        np.where(mask, dy, np.nan),
        w,
        mask=valid_indices,
    )
    y_smooth[valid_indices] = _solve_weighted_poly(*slides, p)

    if interpolate:
        _interpolate_missing(x, y_smooth, valid_indices, out=y_smooth)
    return y_smooth
