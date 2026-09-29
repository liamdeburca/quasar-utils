"""
Utility functions for correcting for the inherent resolving power of 
spectroscopic instruments.

These functions assume that the resolving power is defined as: R = l / dl,
where l is the (observed wavelength) and dl is the FWHM of a delta function 
response.
"""
from math import inf, log, sqrt

from numpy import einsum, empty, float64, pad
from numpy.lib.stride_tricks import sliding_window_view
from quasar_typing.numpy import FloatArray, FloatMatrix, FloatVector
from scipy.stats import norm

SIGMA_TO_FWHM: float = sqrt(8 * log(2))
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM


def R_to_sigma(R: float, x: float) -> float:
    """
    Convert a resolving power, R, to the corresponding Gaussian sigma at a 
    given wavelength, x.
    """
    dx = x / R
    return dx * FWHM_TO_SIGMA


def create_kernel(
    R: float, 
    mu: float,
    x: FloatVector, 
    dx: FloatVector, 
    *,
    n_sigma: float,
) -> FloatVector:
    """
    Generate a Gaussian kernel for a given resolving power, R. The kernel is 
    placed at the center of the wavelength array, x, and is cropped by the 
    n_sigma parameter (all points beyond n_sigma * sigma are not included in the 
    kernel).
    """
    n = x.size
    assert n % 2 == 1, x.shape
    
    sigma = R_to_sigma(R, mu)
    kernel = norm.pdf(x, loc=mu, scale=sigma) * dx
    
    # Crop the kernel to n_sigma
    kernel[x < mu - n_sigma * sigma] = 0
    kernel[x > mu + n_sigma * sigma] = 0

    return kernel


def create_kernels(
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    *,
    w: float,
    n_sigma: float,
) -> FloatMatrix:
    """
    Create a matrix of Gaussian kernels. 

    Each column of the resulting matrix corresponds to a Gaussian kernel 
    centered at the respective wavelength in x.
    """
    _x = pad(
        x,
        (w // 2, w // 2),
        mode="constant",
        constant_values=(-inf, inf),
    )
    _dx = pad(
        dx,
        (w // 2, w // 2),
        mode="constant",
        constant_values=0,
    )
    windows = sliding_window_view([_x, _dx], window_shape=w, axis=1)

    kernels = empty((x.size, w), dtype=float64)
    for i, mu in enumerate(x):
        kernels[i,:] = create_kernel(
            R[i], 
            mu, 
            *windows[:,i],
            n_sigma=n_sigma,
        )

    return kernels


def apply_resolution(
    y: FloatArray, 
    kernels: FloatMatrix,
    out: FloatArray | None = None,
) -> FloatArray:
    """
    Apply a set of resolution kernels to the input array, y.

    Parameters
    ----------
    y : FloatArray
        The input array to be convolved. Shape (..., n). 
    kernels : FloatMatrix
        The matrix of kernels. Shape (n, w), where w is the width of each 
        kernel.
    out : FloatArray | None, optional
        An optional array to store the output. If None, a new array is created.

    Returns
    -------
    FloatArray
        The convolved array. Shape (..., n).
    """
    if y.shape[-1] != kernels.shape[0]:
        msg = f"'y' has shape (...,{y.shape[-1]}) but 'kernels' has shape "\
            f"({kernels.shape[0]},...)."
        raise ValueError(msg)

    w = kernels.shape[1]
    # Only expand along last axis
    pad_width = [(0, 0)] * (y.ndim - 1) + [(w // 2, w // 2)]
    windows = sliding_window_view(
        pad(y, pad_width, mode="constant", constant_values=0),
        window_shape=w,
        axis=-1,
    )
    return einsum("...ij,ij->...i", windows, kernels, out=out)