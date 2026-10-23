"""
File containing utility functions for performing raster fits of templates.
"""
__all__ = ["rasterise"]

from functools import wraps
from logging import getLogger

import numpy as np
import pydantic
from quasar_typing.bounds import AstropyBounds
from quasar_typing.numpy import FittableFloatVector, FloatMatrix, FloatVector

logger = getLogger(__name__)


def _validate_rasterise_inputs(
    y: FloatVector,
    dy: FloatVector,
    fwhm: FloatVector,
    data: FloatVector,
) -> None:
    if y.shape != dy.shape:
        raise ValueError(
            f"Input arrays y ({y.shape}) and dy ({dy.shape}) must have the same "
            "shape."
        )
    if y.size != data.shape[1]:
        raise ValueError(
            f"Input array y ({y.shape}) must match the second dimension of data "
            f"({data.shape[1]})!"
        )
    if fwhm.size != data.shape[0]:
        raise ValueError(
            f"Input array fwhm ({fwhm.shape}) must match the first dimension "
            f"of data ({data.shape[0]})!"
        )


# Public functions

@pydantic.validate_call
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
    return _rasterise(
        y, dy, fwhm, data,
        flux_bounds=flux_bounds,
        fwhm_bounds=fwhm_bounds,
    )


### Private functions

@wraps(rasterise)
def _rasterise(
    y: FittableFloatVector,
    dy: FittableFloatVector,
    fwhm: FloatVector,
    data: FloatMatrix,
    *,
    flux_bounds: AstropyBounds = (None, None),
    fwhm_bounds: AstropyBounds = (None, None),
) -> tuple[FloatVector, FloatVector]:
    fwhm_lb: float = fwhm_bounds[0] or fwhm[0]
    fwhm_ub: float = fwhm_bounds[1] or fwhm[-1]

    chi2s = np.full(fwhm.shape, np.inf, dtype=np.float64)
    fluxs = np.full(fwhm.shape, np.nan, dtype=np.float64)

    mask = (fwhm_lb <= fwhm) & (fwhm <= fwhm_ub)
    if not mask.any():
        msg = f"No template rows within FWHM bounds ({fwhm_lb:.1e}, {fwhm_ub:.1e})"
        logger.info(msg)
        raise ValueError(msg)

    # If template is not covered by data
    if (data[mask] == 0).all():
        msg = "All template rows are zero suggesting that data does not cover "\
            "the template."
        logger.info(msg)
        raise ValueError(msg)

    den = np.zeros(fwhm.shape, dtype=np.float64)
    num = np.zeros(fwhm.shape, dtype=np.float64)

    w2 = 1 / dy**2
    den[mask] = np.einsum("ij,ij,...j->i", data[mask], data[mask], w2)

    mask = den != 0
    if not mask.any():
        msg = "All template rows have vanishing denominators!"
        logger.info(msg)
        raise ValueError(msg)

    num[mask] = np.einsum("ij,...j,...j->i", data[mask], y, w2)

    fluxs[mask] = np.clip(
        num[mask] / den[mask],
        a_min=-np.inf if flux_bounds[0] is None else flux_bounds[0],
        a_max=np.inf if flux_bounds[1] is None else flux_bounds[1],
    )
    _d = y[None, :] - (fluxs[:, None] * data)[mask]
    chi2s[mask] = np.einsum("ij,ij,...j->i", _d, _d, w2) / y.size

    return chi2s, fluxs