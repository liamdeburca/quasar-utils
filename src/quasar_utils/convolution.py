import math
from functools import wraps

import numpy as np
import pydantic
from quasar_typing.numpy import FloatMatrix, FloatVector, IntVector
from scipy.signal import fftconvolve

GAUSS_AMP = 1 / (2 * math.pi)**0.5
SIGMA_TO_FWHM: float = 2 * (2 * math.log(2))**0.5
FWHM_TO_SIGMA: float = 1 / SIGMA_TO_FWHM
N_SCALES: float = 5.0


### Public functions


@pydantic.validate_call
def scale(
    fwhm: float, 
    sigma_res: float,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
) -> float:
    """
    Converts a FWHM (velocity) to a standard deviation (pixels).
    """
    return _scale(fwhm, sigma_res, fwhm_to_sigma)


@pydantic.validate_call
def pixels(
    scale: float,
    n_scales: float = N_SCALES,
) -> IntVector:
    """
    Creates a symmetric array of pixels given a scale. 
    """
    return _pixels(scale, n_scales=n_scales)


@pydantic.validate_call
def kernel(
    fwhm: float,
    sigma_res: float,
    gauss_amp: float = GAUSS_AMP
) -> FloatVector:
    """
    Creates a Gaussian kernel with the appropriate FWHM. 
    """
    return _kernel(fwhm, sigma_res, gauss_amp=gauss_amp)


@pydantic.validate_call
def kernel_deriv(
    fwhm: float,
    sigma_res: float,
    sigma_to_fwhm: float = SIGMA_TO_FWHM
) -> FloatVector:
    """
    Creates a kernel equivalent to a Gaussian derivative w.r.t. its FWHM. 
    """
    return _kernel_deriv(fwhm, sigma_res, sigma_to_fwhm=sigma_to_fwhm)


@pydantic.validate_call
def convolve_signal(
    signal: FloatVector, 
    kernel: FloatVector,
) -> FloatVector:
    """    
    Convolves (fft) a given signal with a kernel. The signal is padded using its
    edge values. 
    """
    return _convolve_signal(signal, kernel)


@pydantic.validate_call
def identify_closest_idx(
    fwhm: FloatVector,
    fwhm_final: float,
    for_deriv: bool = False,
) -> int:
    """
    Identifies the index of the closest FWHM in the template's FWHM array. 
    """
    return _identify_closest_idx(fwhm, fwhm_final, for_deriv=for_deriv)


@pydantic.validate_call
def identify_closest(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    for_deriv: bool = False,
) -> tuple[float, FloatVector]:
    """
    Identifies the closest (but smaller) FWHM in the template's FWHM array,
    and returns its corresponding spectrum. 
    """
    return _identify_closest(data, fwhm, fwhm_final, for_deriv=for_deriv)


@pydantic.validate_call
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
    return _convolve(data, fwhm, fwhm_final, sigma_res)


@pydantic.validate_call
def convolve_deriv(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    sigma_res: float,
) -> FloatVector:
    """    
    Computes the derivative of a convolved template w.r.t. the final FWHM.
    """
    return _convolve_deriv(data, fwhm, fwhm_final, sigma_res)


### Private functions


@wraps(scale)
def _scale(
    fwhm: float, 
    sigma_res: float,
    fwhm_to_sigma: float = FWHM_TO_SIGMA,
) -> float:
    return fwhm * fwhm_to_sigma / sigma_res


@wraps(pixels)
def _pixels(scale: float, n_scales: float = N_SCALES) -> IntVector:
    l = int(n_scales * scale)
    return np.arange(-l, l+1, dtype=np.int_)


@wraps(kernel)
def _kernel(
    fwhm: float,
    sigma_res: float,
    gauss_amp: float = GAUSS_AMP
) -> FloatVector:
    if fwhm == 0: 
        return np.ones(1, dtype=np.float64)

    s = _scale(fwhm, sigma_res)
    z = _pixels(s) / s
    k = gauss_amp * np.exp(-0.5 * z**2) / s
    return k / k.sum()


@wraps(kernel_deriv)
def _kernel_deriv(
    fwhm: float,
    sigma_res: float,
    sigma_to_fwhm: float = SIGMA_TO_FWHM
) -> FloatVector:
    if fwhm == 0: 
        return -sigma_to_fwhm * np.ones(1, dtype=np.float64)

    k = _kernel(fwhm, sigma_res)
    s = _scale(fwhm, sigma_res)
    z = _pixels(s) / s
    return sigma_to_fwhm * k * (z**2 - 1)


@wraps(convolve_signal)
def _convolve_signal(
    signal: FloatVector, 
    kernel: FloatVector,
) -> FloatVector:
    assert kernel.size % 2 == 1, "Kernel length must be odd."
    return fftconvolve(
        np.pad(signal, pad_width=kernel.size // 2, mode='edge'),
        kernel,
        mode='valid',
    )


@wraps(identify_closest_idx)
def _identify_closest_idx(
    fwhm: FloatVector,
    fwhm_final: float,
    for_deriv: bool = False,
) -> int:
    if fwhm_final >= fwhm[-1]: 
        idx = fwhm.size - 1
    else:
        idx = np.searchsorted(fwhm, fwhm_final, side='right') - 1

    if for_deriv and idx != 0 and fwhm[idx] == fwhm_final:
        idx -= 1

    return idx

@wraps(identify_closest)
def _identify_closest(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    for_deriv: bool = False,
) -> tuple[float, FloatVector]:
    idx = _identify_closest_idx(fwhm, fwhm_final, for_deriv=for_deriv)
    return fwhm[idx], data[idx]    


@wraps(convolve)
def _convolve(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    sigma_res: float,
) -> FloatVector:
    assert fwhm_final >= fwhm[0], \
        "Desired FWHM is smaller than template's minimum FWHM."
        
    fwhm_init, signal = _identify_closest(data, fwhm, fwhm_final)
    if fwhm_init == fwhm_final: 
        return signal

    fwhm_kernel = (fwhm_final**2 - fwhm_init**2)**0.5
    k = _kernel(fwhm_kernel, sigma_res)

    return _convolve_signal(signal, k)


@wraps(convolve_deriv)
def _convolve_deriv(
    data: FloatMatrix,
    fwhm: FloatVector,
    fwhm_final: float,
    sigma_res: float,
) -> FloatVector:
    assert fwhm_final >= fwhm[0], \
        "Desired FWHM is smaller than template's minimum FWHM."
    
    # If: fwhm_final == fwhm[0] -> no convolutions is possible!
    if fwhm_final == fwhm[0]: 
        return -SIGMA_TO_FWHM * data[0]
    
    fwhm_init, signal = _identify_closest(
        data, fwhm, fwhm_final, 
        for_deriv=True,
    )
    
    fwhm_kernel = (fwhm_final**2 - fwhm_init**2)**0.5
    dk = _kernel_deriv(fwhm_kernel, sigma_res)
    return _convolve_signal(signal, dk)