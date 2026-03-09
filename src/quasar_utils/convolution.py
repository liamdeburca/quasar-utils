from functools import lru_cache
from numpy import log, arange, exp, pad, searchsorted, int_
from scipy.signal import fftconvolve
from math import pi

from pydantic import validate_call
from quasar_typing.numpy import IntVector, FloatVector, FloatMatrix

GAUSS_AMP = 1 / (2 * pi)**0.5
SIGMA_TO_FWHM: float = 2 * (2 * log(2))**0.5
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM
N_SCALES: float = 5.0

def _scale(
    fwhm: float, 
    sigma_res: float,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
) -> float:
    """
    Converts a FWHM (velocity) to a standard deviation (pixels).
    """
    return fwhm * fwhm_to_sigma / sigma_res

def _pixels(
    scale: float,
    n_scales: float = N_SCALES,
) -> IntVector:
    """
    Creates a symmetric array of pixels given a scale. 
    """
    l: int = int(n_scales * scale)
    return arange(-l, l+1, dtype=int_)

@lru_cache
def kernel(
    fwhm: float,
    sigma_res: float,
    gauss_amp: float = GAUSS_AMP
) -> FloatVector:
    """
    Creates a Gaussian kernel with the appropriate FWHM. 
    """
    s = _scale(fwhm, sigma_res)
    z = _pixels(s) / s
    k = gauss_amp * exp(-0.5 * z**2) / s

    return k / k.sum()

@lru_cache
def kernel_deriv(
    fwhm: float,
    sigma_res: float,
    sigma_to_fwhm: float = SIGMA_TO_FWHM
) -> FloatVector:
    """
    Creates a kernel equivalent to a Gaussian derivative w.r.t. its FWHM. 
    """
    k = kernel(fwhm, sigma_res)
    s = _scale(fwhm, sigma_res)
    z = _pixels(s) / s

    return sigma_to_fwhm * k * (z**2 - 1)

@validate_call(validate_return=False)
def convolve_signal(
    signal: FloatVector, 
    kernel: FloatVector,
) -> FloatVector:
    """
    Convolves (fft) a given signal with a kernel. The signal is padded using its
    edge values. 
    """    
    assert len(kernel) % 2 == 1, "Kernel length must be odd."
    assert signal.size >= kernel.size, "Signal length must be >= kernel length."

    return fftconvolve(
        pad(signal, pad_width=len(kernel) // 2, mode='edge'),
        kernel,
        mode='valid',
    )

def _identify_closest_idx(
    fwhm: FloatVector,
    fwhm_final: float,
) -> int:
    """
    Identifies the index of the closest (but smaller) FWHM in the template's 
    FWHM array. 
    """
    if fwhm_final >= fwhm[-1]: return fwhm.size - 1
    else:                      return searchsorted(fwhm, fwhm_final) - 1

def _identify_closest(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
) -> tuple[float, FloatVector]:
    """
    Identifies the closest (but smaller) FWHM in the template's FWHM array,
    and returns its corresponding spectrum. 
    """
    idx = _identify_closest_idx(fwhm, fwhm_final)
    return fwhm[idx], data[idx]

@validate_call(validate_return=False)
def convolve(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    sigma_res: float,
) -> FloatVector:
    """
    Convolves a template, defined by spectra and FWHM values, to a desired FWHM
    value:

    1.  Identifies if the desired FWHM is smaller than the template's available
        FWHM values. If so, a zero-array is returned. 
    2.  If the desired FWHM is already covered by the template, the 
        corresponding spectrum is returned. 
    3.  Finally, the closest (but smaller) FWHM is identified, and its 
        corresponding spectrum is broadened using a Gaussian kernel with FWHM:

            FWHM_kernel = sqrt(FWHM_final^2 - FWHM_init^2)
    """
    assert fwhm_final >= fwhm[0], \
        "Desired FWHM is smaller than template's minimum FWHM."
    
    fwhm_init, signal = _identify_closest(data, fwhm, fwhm_final)
    
    fwhm_kernel: float = (fwhm_final**2 - fwhm_init**2)**0.5
    k: FloatVector = kernel(fwhm_kernel, sigma_res)

    return convolve_signal.__wrapped__(signal, k)

@validate_call(validate_return=False)
def convolve_deriv(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    sigma_res: float,
) -> FloatVector:
    """
    Computes the derivative of a convolved template w.r.t. the final FWHM.
    """
    assert fwhm_final >= fwhm[0], \
        "Desired FWHM is smaller than template's minimum FWHM."
    
    fwhm_init, signal = _identify_closest(data, fwhm, fwhm_final)

    fwhm_kernel: float = (fwhm_final**2 - fwhm_init**2)**0.5
    dk: FloatVector = kernel_deriv(fwhm_kernel, sigma_res)

    return convolve_signal.__wrapped__(signal, dk)