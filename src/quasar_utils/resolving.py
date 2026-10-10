"""
Utility functions for correcting for the inherent resolving power of 
spectroscopic instruments.

These functions assume that the resolving power is defined as: R = l / dl,
where l is the (observed wavelength) and dl is the FWHM of a delta function 
response.
"""
from math import inf, log, pi, sqrt
from typing import Literal, final, overload

from numpy import einsum, exp, float64, pad, prod, where, zeros_like
from numpy.lib.stride_tricks import sliding_window_view
from quasar_typing.numpy import (
    BoolVector,
    FloatArray,
    FloatMatrix,
    FloatVector,
)
from quasar_typing.scipy import csr_matrix_
from scipy.sparse import csr_matrix
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
    normalize: bool,
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

    if normalize:
        kernel /= kernel.sum()
    return kernel


@overload
def create_kernels(
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    *,
    w: int,
    n_sigma: float,
    normalize: bool,
    sparse: Literal[False] = False,
) -> FloatMatrix:
    ...

@overload
def create_kernels(
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    *,
    w: int,
    n_sigma: float,
    normalize: bool,
    sparse: Literal[True],
) -> csr_matrix_:
    ...

@final
def create_kernels(
    R: FloatVector,
    x: FloatVector,
    dx: FloatVector,
    *,
    w: int,
    n_sigma: float,
    normalize: bool,
    sparse: bool = False,
) -> FloatMatrix | csr_matrix_:
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
    xs, dxs = sliding_window_view([_x, _dx], window_shape=w, axis=1)

    sigmas = x / R * FWHM_TO_SIGMA
    zs = (xs - x[:,None]) / sigmas[:,None]
    zs[zs < -n_sigma] = -inf
    zs[zs >  n_sigma] = inf

    z2s = zs ** 2
    kernels = exp(-0.5 * z2s) / sigmas[:,None] / sqrt(2 * pi)
    kernels *= dxs    

    if normalize:
        kernels /= kernels.sum(axis=1, keepdims=True)
    return csr_matrix(kernels) if sparse else kernels


@overload
def mask_kernels(
    kernels: FloatMatrix,
    mask: BoolVector,
) -> FloatMatrix:
    ...

@overload
def mask_kernels(
    kernels: csr_matrix_,
    mask: BoolVector,
) -> csr_matrix_:
    ...

@final
def mask_kernels(
    kernels: FloatMatrix | csr_matrix_,
    mask: BoolVector,
) -> FloatMatrix | csr_matrix_:
    """
    Apply a boolean mask to an array of kernels.

    Parameters
    ----------
    kernels : FloatMatrix
        The matrix of kernels. Shape (n, w)
    mask : BoolVector
        The boolean mask to apply. Shape (n,). 

    Returns
    -------
    FloatMatrix
        The masked array of kernels.
    """
    if mask.all():
        return kernels.copy()
    
    w = kernels.shape[1]
    if w == 1:
        return kernels[mask]
    
    local_masks = sliding_window_view(
        pad(mask, (w // 2, w // 2), mode="constant", constant_values=True), 
        window_shape=w, 
        axis=0,
    )
    if (is_sparse := isinstance(kernels, csr_matrix)):
        kernels = kernels.toarray()
    
    new_kernels = where(local_masks[mask], kernels[mask], 0)
    return csr_matrix(new_kernels) if is_sparse else new_kernels


def limit_loss(
    kernels: FloatMatrix | csr_matrix_,
    mask: BoolVector,
    shrink: bool,
) -> BoolVector:
    """
    Limit loss at the edges of the spectral windows selected by the mask by 
    constructing a new mask.

    If shrink is True, the mask is shrunk -- only pixels with negligible loss 
    are selected. Otherwise, the mask is expanded -- include pixels outside the 
    mask to compensate for loss inside the original mask.
    """
    signal = mask.astype(float64)
    if shrink:
        signal = 1.0 - signal
    apply_resolution(signal, kernels, out=signal)
    return (signal == 0) if shrink else (signal > 0)


def apply_resolution(
    y: FloatArray, 
    kernels: FloatMatrix | csr_matrix_,
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
    
    pad_width = [(0, 0)] * (y.ndim - 1) + [(w // 2, w // 2)]
    windows = sliding_window_view(
        pad(y, pad_width, mode="constant", constant_values=0),
        window_shape=w,
        axis=-1,
    )
    if not isinstance(kernels, csr_matrix):
        return einsum("...ij,ij->...i", windows, kernels, out=out)

    if out is None:
        out = zeros_like(y, float64)
    
    batch_prod = int(prod(y.shape[:-1]))
    win_reshaped = windows.reshape(
        batch_prod, 
        kernels.shape[0], 
        kernels.shape[1],
    )
    out_reshaped = out.reshape(
        batch_prod, 
        kernels.shape[0],
    )    
    for i in range(kernels.shape[0]):
        out_reshaped[:, i] = (win_reshaped[:, i, :] @ kernels[i].T).ravel()
    
    return out

