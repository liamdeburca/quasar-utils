__all__ = [
    "create_slides",
    "get_gap_sizes",
    "get_valid_indices",
    "interpolate_missing",
    "solve_weighted_poly",
    "weighted_savgol_filter",
]

from itertools import batched
from logging import getLogger
from typing import Literal

from numpy import (
    append,
    argwhere,
    concatenate,
    convolve,
    float64,
    full_like,
    interp,
    isfinite,
    nan,
    ones,
    where,
    zeros,
)
from numpy.lib.stride_tricks import sliding_window_view
from numpy.linalg import lstsq
from numpy.polynomial.polynomial import polyvander
from pydantic import PositiveInt
from quasar_typing.errors import SmoothingError
from quasar_typing.numpy import (
    BoolVector,
    FloatMatrix,
    FloatVector,
    SortedFloatVector,
)

from ..decorators import validate_call

logger = getLogger(__name__)


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
                true_indices = argwhere(mask).flatten()
                _mask = mask[true_indices[0] : true_indices[-1] + 1]

                out = get_gap_sizes.__wrapped__(_mask)
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

            edges = argwhere(mask[:-1] ^ mask[1:]).flatten()

            return [
                (left + 1, right, right - left)
                for (left, right) in batched(edges, 2)
            ]


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
    _x = x[mask]
    _y = y[mask]
    if _x.size == 0:
        raise ValueError("'mask' must contain at least one True value for interpolation.")
    elif _x.size == 1:
        if out is None:
            return full_like(y, _y[0], dtype=float64)
        out.fill(_y[0])
    else:
        _out = interp(x, _x, _y, left=_y[0], right=_y[-1])
        if out is None:
            return _out
        out[:] = _out

    return None

@validate_call
def get_valid_indices(
    mask: BoolVector,
    w: PositiveInt,
    p: PositiveInt,
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
    match side:
        case "left":
            kernel = append(ones(w), zeros(w - 1))
        case "right":
            kernel = append(zeros(w - 1), ones(w))
        case _:
            kernel = ones(w)

    # Calculate the number of points covered by the kernel at each location
    n_points_covered = convolve(mask.astype(int), kernel, mode="same")

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


@validate_call
def create_slides(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: PositiveInt,
    mask: BoolVector | None = None,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    """
    ** PYDANTIC VALIDATED FUNCTION **

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
    l = w // 2
    filler = nan * ones(l)
    _fill = lambda arr: concatenate([filler, arr, filler], axis=0)

    f = lambda arr: sliding_window_view(_fill(arr), window_shape=w)

    x_slides = f(x) - x[:, None]  # (n_slides, w)
    y_slides = f(y)
    dy_slides = f(dy)

    if mask is not None:
        x_slides = x_slides[mask]
        y_slides = y_slides[mask]
        dy_slides = dy_slides[mask]

    return x_slides, y_slides, dy_slides


@validate_call
def solve_weighted_poly(
    x_slides: FloatMatrix,
    y_slides: FloatMatrix,
    dy_slides: FloatMatrix,
    p: PositiveInt,
    full: bool = False,
) -> FloatVector | FloatMatrix:
    """
    ** PYDANTIC VALIDATED FUNCTION **

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
    # Use a numerically stable per-slide weighted least-squares solver.
    # For each slide, form the design matrix X (columns 1, x, x^2, ..., x^p),
    # scale rows by sqrt(weights)=1/dy and call lstsq. This avoids explicit
    # formation/inversion of normal-equation matrices and is more robust.
    n_slides = y_slides.shape[0]
    coeffs = zeros((n_slides, p + 1), dtype=float64)

    for i in range(n_slides):
        x_s = x_slides[i]
        y_s = y_slides[i]
        dy_s = dy_slides[i]

        # Select usable rows: finite x,y,dy and positive dy
        valid = isfinite(x_s) & isfinite(y_s) & isfinite(dy_s) & (dy_s > 0)
        if valid.sum() == 0:
            coeffs[i, :] = nan
            continue

        # Design matrix (1, x, x^2, ..., x^p)
        X = polyvander(x_s[valid], deg=p)

        # Row-scale by sqrt(weights): sqrt_w = 1/dy
        sqrt_w = 1.0 / dy_s[valid]
        Xw = X * sqrt_w[:, None]
        yw = y_s[valid] * sqrt_w

        sol, *_ = lstsq(Xw, yw, rcond=None)
        coeffs[i, :] = sol

    return coeffs if full else coeffs[:, 0]


@validate_call
def weighted_savgol_filter(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: PositiveInt,
    p: PositiveInt,
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
    w : int
        Window size used for smoothing.
    p : int
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

    """        
    # Validate window and polynomial parameters at runtime -- pydantic
    # enforces some checks but relational invariants (w > p) are enforced here.
    if w <= 0:
        raise SmoothingError("w must be a positive integer.")
    if w % 2 == 0:
        raise SmoothingError("w must be odd.")
    if w <= p:
        raise SmoothingError("w must be greater than polynomial order p.")
    if x.shape[0] < w:
        raise SmoothingError(f"Input array length ({x.shape[0]}) must be at least window size w ({w}).")

    y_smooth = y.copy()

    x_valid = isfinite(x)
    if mask is None:
        mask = x_valid & isfinite(y) & isfinite(dy) & (dy > 0)

    gap_sizes = get_gap_sizes.__wrapped__(mask)
    if gap_sizes:
        msg = f"Identified the following gaps ('left', 'right', 'size'): {gap_sizes}."
        logger.debug(msg)

    valid_indices = get_valid_indices.__wrapped__(mask, w, p, mode=mode)
    valid_indices &= x_valid  # Require defined value of centre of window.

    if not valid_indices.any():
        msg = f"Cannot smooth the spectrum w/ {w=}, {p=}, and {mode=}: "
        msg += "returning copy of original flux density array."
        logger.warning(msg)
        return y_smooth

    slides = create_slides.__wrapped__(
        where(x_valid, x, nan),
        where(mask, y, nan),
        where(mask, dy, nan),
        w,
        mask=valid_indices,
    )
    y_smooth[valid_indices] = solve_weighted_poly.__wrapped__(*slides, p)

    if interpolate:
        interpolate_missing.__wrapped__(x, y_smooth, valid_indices, out=y_smooth)
    return y_smooth