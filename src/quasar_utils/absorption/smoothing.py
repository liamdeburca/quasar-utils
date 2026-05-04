__all__ = [
    'get_gap_sizes',
    'interpolate_missing',
    'get_valid_indices',
    'create_slides',
    'solve_weighted_poly',
    'weighted_savgol_filter',
]

from logging import getLogger
from typing import Literal

logger = getLogger(__name__)

from numpy import (
    nan, ones, concatenate, where, interp, argwhere, append, zeros, convolve, 
    nansum, matmul, arange, isfinite, float64
)
from numpy.polynomial.polynomial import polyvander
from numpy.linalg import inv
from numpy.lib.stride_tricks import sliding_window_view

from itertools import batched

from pydantic import validate_call
from quasar_typing.numpy import FloatVector, BoolVector, FloatMatrix

@validate_call
def get_gap_sizes(mask: BoolVector) -> list[tuple[int, int, int]]:
    """
    ** PYDANTIC VALIDATED FUNCTION **
    """
    if not (mask[0] and mask[-1]):
        true_indices = argwhere(mask).flatten()
        _mask = mask[true_indices[0]:true_indices[-1]+1]

        out = get_gap_sizes.__wrapped__(_mask)
        if not mask[0]:
            left = 0
            right = true_indices[0]-1
            out.insert(
                0,
                (left, right, right - left + 1)
            )
        if not mask[-1]:
            left = true_indices[-1] + 1
            right = len(mask) - 1
            out.append((
                left,
                right,
                right - left + 1,
            ))

        return out
    
    edges = argwhere(mask[:-1] ^ mask[1:]).flatten()
    
    return [
        (left + 1, right, right - left) \
        for (left, right) \
        in batched(edges, 2)
    ]

@validate_call
def interpolate_missing(
    x: FloatVector, 
    y: FloatVector, 
    mask: BoolVector,
) -> FloatVector:
    """
    ** PYDANTIC VALIDATED FUNCTION **

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

    Returns
    -------
    y_int : numpy.array
        Array along the second axis with designated pixels' values interpolated.
    """
    _x = x[mask]
    _y = y[mask]
    return where(
        mask,
        y,
        interp(x, _x, _y, left=_y[0], right=_y[-1])
    )

@validate_call
def get_valid_indices(
    mask: BoolVector, 
    w: int, 
    p: int, 
    side: Literal['left', 'right'] | None = None,
    mode: Literal['flexible', 'rigid', 'semi-rigid', 'standard'] = 'standard',
) -> BoolVector:
    """
    ** PYDANTIC VALIDATED FUNCTION **

    Calculated the pixels whose flux density values can be smoothed adequately. 

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
        case 'left':  kernel = append(ones(w), zeros(w-1))
        case 'right': kernel = append(zeros(w-1), ones(w))
        case _:       kernel = ones(w)

    # Calculate the number of points covered by the kernel at each location
    n_points_covered = convolve(mask.astype(int), kernel, mode='same')

    match mode:
        case 'flexible':   n_points_minimum = p + 2
        case 'rigid':      n_points_minimum = w + 1
        case 'semi-rigid': n_points_minimum = 3 * (w // 4)
        case 'standard':   n_points_minimum = (w // 2) + 1

    return (n_points_covered >= n_points_minimum)

@validate_call
def create_slides(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: int,
    mask: BoolVector | None = None
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
    w : int
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

    x_slides = f(x) - x[:,None]# (n_slides, w)
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
    p: int,
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
    n_slides = y_slides.shape[0] # Number of slides. 
    sq_weight_slides = 1 / dy_slides**2 # Slides of squared weights. 
    y_sq_weight_slides = y_slides * sq_weight_slides

    vandermonde_matrix = polyvander(x_slides, deg=2*p) # (n_slides, 2p+1)
    elements = nansum(vandermonde_matrix * sq_weight_slides[...,None], axis=1) # (n_slides, p+1)

    # Build matrix for left-hand-side of equation
    LHS = ones(shape=(n_slides, 2*p+1, 2*p+1), dtype=float64)
    for i, element in enumerate(elements.T, start=1):
        indices = arange(i)
        LHS[...,indices,indices[::-1]] = element[...,None]

    # Crop the matrix, and invert it
    LHS_inverted = inv(LHS[...,:p+1,:p+1])

    # Create the RHS vector
    RHS = nansum(
        vandermonde_matrix[...,:p+1] * y_sq_weight_slides[...,None],
        axis = 1
    ) # (n_slides, p+1)
    
    # Solve linear problems in parallel
    solutions = matmul(LHS_inverted, RHS[...,None])

    return solutions if full else solutions[...,0].T[0]

@validate_call
def weighted_savgol_filter(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    w: int,
    p: int,
    mask: BoolVector | None = None,
    interpolate: bool = True,
    mode: str = 'standard',
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
    """
    y_smooth = y.copy()

    x_valid = isfinite(x)
    mask = mask \
        if mask is not None \
        else x_valid & isfinite(y) & (dy > 0)
    
    gap_sizes = get_gap_sizes.__wrapped__(mask)
    if len(gap_sizes) > 1:
        msg = f"Identified the following gaps ('left', 'right', 'size'): {gap_sizes}."
        logger.debug(msg)
    
    valid_indices = get_valid_indices.__wrapped__(
        mask, 
        w, 
        p, 
        mode = mode,
    ) & x_valid # Require defined value of centre of window.

    slides = create_slides.__wrapped__(
        where(x_valid, x, nan),
        where(mask, y, nan),
        where(mask, dy, nan),
        w,
        mask = valid_indices,
    )
    y_smooth[valid_indices] = solve_weighted_poly.__wrapped__(*slides, p)

    return (
        interpolate_missing.__wrapped__(x, y_smooth, valid_indices) \
        if interpolate \
        else y_smooth
    )