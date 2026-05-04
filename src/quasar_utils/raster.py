"""
File containing utility functions for performing raster fits of templates.
"""
__all__ = ["rasterise"]

from numpy import float64, einsum, clip, inf, full_like, nan, isnan
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
@validate_call
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
    **PYDANTIC VALIDATED FUNCTION**

    Compares the input data and each template row, finding the optimal flux 
    which minimises the chi-square.
    """
    _validate_rasterise_inputs(y, dy, fwhm, data)

    fwhm_lb: float = fwhm_bounds[0] or fwhm[0]
    fwhm_ub: float = fwhm_bounds[1] or fwhm[-1]

    mask = (fwhm_lb <= fwhm) & (fwhm <= fwhm_ub)
    assert mask.any(), "No templates within FWHM bounds!"

    chi2s = full_like(fwhm, nan, dtype=float64)
    fluxs = chi2s.copy()

    # If template is not covered by data
    if (data[mask] == 0).all(): return chi2s, fluxs

    w2 = 1 / dy**2
    
    den = einsum("ij,ij,...j->i", data[mask], data[mask], w2)
    num = einsum("ij,...j,...j->i", data[mask], y, w2)
    fluxs[mask] = clip(
        num / den,
        a_min=flux_bounds[0] or 0,
        a_max=flux_bounds[1] or inf, 
    )
    _d = y[None,:] - (fluxs[:,None] * data)[mask]
    chi2s[mask] = einsum("ij,ij,...j->i", _d, _d, w2) / y.size

    return chi2s, fluxs