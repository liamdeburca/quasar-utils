__all__ = [
    'remove_absorption',
    'smooth_spectrum',
]

from numpy import isfinite, invert, maximum, arange

from pydantic import validate_call
from quasar_typing.numpy import FloatVector, BoolVector

from .absorption import fft_approach, join_regions, refine_regions
from .smoothing import interpolate_missing, weighted_savgol_filter
from .utils import nan_residuals

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
    ** PYDANTIC VALIDATED FUNCTION **
    """
    if valid_pixels is None:
        valid_pixels = isfinite([x, y, dy]).all(axis=0) & (dy > 0)

    z = nan_residuals(
        y, 
        maximum(y_smooth, y_bg), 
        dy, 
        z_fill = 0, 
        mask = valid_pixels,
    )
    p_absorbed, absorbed_pixels = fft_approach.__wrapped__(
        z, p_crit, z_crit, w,
    )
    if isinstance(join, int): 
        absorbed_pixels = join_regions.__wrapped__(
            absorbed_pixels, join,
        )
    
    if refine:
        absorbed_pixels, y_smooth = refine_regions.__wrapped__(
            absorbed_pixels, x, y, dy, y_smooth, y_bg, 
            valid_pixels=valid_pixels,
        )
    else:
        y_smooth = interpolate_missing.__wrapped__(
            x, y_smooth, invert(absorbed_pixels),
        )

    return p_absorbed, absorbed_pixels, y_smooth
    
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
    ** PYDANTIC VALIDATED FUNCTION **
    """
    _x = arange(len(x)) if logspace else x
    return weighted_savgol_filter.__wrapped__(
        _x, y, dy, w=w, p=p, mask=valid_pixels,
    )
