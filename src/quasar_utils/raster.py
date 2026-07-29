"""
File containing utility functions for performing raster fits of templates.
"""

__all__ = ["rasterise"]

from logging import getLogger

from numpy import clip, einsum, float64, full, inf, nan, zeros
from pydantic_core import PydanticCustomError
from quasar_typing.bounds import AstropyBounds
from quasar_typing.numpy import FittableFloatVector, FloatMatrix, FloatVector

from quasar_utils.decorators import validate_call

logger = getLogger(__name__)


def _validate_rasterise_inputs(
    y: FloatVector,
    dy: FloatVector,
    fwhm: FloatVector,
    data: FloatVector,
) -> None:
    if not y.size == dy.size == data.shape[1]:
        msg = (
            f"Input arrays y ({y.shape}), dy ({dy.shape}) must have the same shape, and \
            match the second dimension of data ({data.shape[1]})!"
        )
        raise PydanticCustomError("validation_error", msg)

    if not fwhm.size == data.shape[0]:
        msg = (
            f"Input array fwhm ({fwhm.shape}) must have the same shape as the first \
            dimension of data ({data.shape[0]})!"
        )
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
    Compares the input data and each template row, finding the optimal flux
    which minimises the chi-square.

    Returns the chi-square and flux for each template row. Rows outside the FWHM
    bounds and rows with vanishing denominators are ignored, i.e.
    `chi-square=np.inf`.

    To access the best-fit row, use `np.argmin(chi2s)`.
    """
    _validate_rasterise_inputs(y, dy, fwhm, data)

    fwhm_lb: float = fwhm_bounds[0] or fwhm[0]
    fwhm_ub: float = fwhm_bounds[1] or fwhm[-1]

    chi2s = full(fwhm.shape, inf, dtype=float64)
    fluxs = full(fwhm.shape, nan, dtype=float64)

    mask = (fwhm_lb <= fwhm) & (fwhm <= fwhm_ub)
    if not mask.any():
        msg = f"No template rows within FWHM bounds ({fwhm_lb:.1e}, {fwhm_ub:.1e})"
        logger.info(msg)
        return chi2s, fluxs

    # If template is not covered by data
    if (data[mask] == 0).all():
        msg = (
            "All template rows are zero suggesting that data does not cover "
            "the template."
        )
        logger.info(msg)
        return chi2s, fluxs

    den = zeros(fwhm.shape, dtype=float64)
    num = zeros(fwhm.shape, dtype=float64)

    w2 = 1 / dy**2
    den[mask] = einsum("ij,ij,...j->i", data[mask], data[mask], w2)

    mask = den != 0
    if not mask.any():
        msg = "All template rows have vanishing denominators!"
        logger.info(msg)
        return chi2s, fluxs

    num[mask] = einsum("ij,...j,...j->i", data[mask], y, w2)

    fluxs[mask] = clip(
        num[mask] / den[mask],
        a_min=-inf if flux_bounds[0] is None else flux_bounds[0],
        a_max=inf if flux_bounds[1] is None else flux_bounds[1],
    )
    _d = y[None, :] - (fluxs[:, None] * data)[mask]
    chi2s[mask] = einsum("ij,ij,...j->i", _d, _d, w2) / y.size

    return chi2s, fluxs
