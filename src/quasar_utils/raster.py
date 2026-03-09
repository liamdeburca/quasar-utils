"""
File containing utility functions for performing raster fits of templates.
"""

__all__ = ["rasterise"]

from numpy import float64, einsum, clip, inf, full_like, nan
from numpy.typing import NDArray

from pydantic import validate_call
from pydantic_core import PydanticCustomError
from quasar_typing.numpy import FloatVector, FittableFloatVector, FloatMatrix
from quasar_typing.bounds import AstropyBounds

def _validate_rasterise_inputs(
    y: NDArray[float64],
    dy: NDArray[float64],
    fwhm: NDArray[float64],
    data: NDArray[float64],
) -> None:
    if not y.size == dy.size == data.shape[1]:
        msg = "Input arrays y ({}), dy ({}) must have the same shape, and \
            match the second dimension of data ({})!".format(
                y.shape, dy.shape, data.shape[1]
            )
        raise PydanticCustomError("validation_error", msg)
    
    if not fwhm.size == data.shape[0]:
        msg = "Input array fwhm ({}) must have the same shape as the first \
            dimension of data ({})!".format(fwhm.shape, data.shape[0])
        raise PydanticCustomError("validation_error", msg)

# Basic rasterisation
@validate_call(validate_return=False)
def rasterise(
    y: FittableFloatVector,
    dy: FittableFloatVector,
    fwhm: FloatVector,
    data: FloatMatrix,
    *,
    flux_bounds: AstropyBounds = (None, None),
    fwhm_bounds: AstropyBounds = (None, None),
) -> tuple[FloatVector, FloatVector]:
    """
    Basic rasterisation:

    Compares the input data and each template row, finding the optimal flux 
    which minimises the chi-square.
    """
    _validate_rasterise_inputs(y, dy, fwhm, data)

    fwhm_lb: float = fwhm_bounds[0] if fwhm_bounds[0] is not None else 0
    fwhm_ub: float = fwhm_bounds[1] if fwhm_bounds[1] is not None else inf 

    mask = (fwhm_lb <= fwhm) & (fwhm <= fwhm_ub)
    assert mask.any(), "No templates within FWHM bounds!"

    chi2s = full_like(fwhm, nan, dtype=float64)
    fluxs = chi2s.copy()

    _data = data[mask]

    w2 = 1 / dy**2
    _fluxs = clip(
        einsum("ij,...j,...j->i", _data, y, w2) \
        / einsum("ij,ij,...j->i", _data, _data, w2),
        a_min = flux_bounds[0],
        a_max = flux_bounds[1],
    )
    diffs = y[None,:] -  fluxs[:,None] * _data
    _chi2s = einsum("ij,ij,...j->i", diffs, diffs, w2) / y.size

    fluxs[mask] = _fluxs
    chi2s[mask] = _chi2s

    return chi2s, fluxs