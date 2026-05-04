from numpy import float64, bool_, isfinite, full_like
from numpy.typing import NDArray

def nan_residuals(
    y: NDArray[float64], 
    f: NDArray[float64], 
    dy: NDArray[float64],
    z_fill: float = 0,
    mask: NDArray[bool_] | None = None,
) -> NDArray[float64]:
    
    if mask is None:
        mask = isfinite(y) & isfinite(f) & (dy > 0)
    
    z = full_like(y, z_fill, dtype=float64)
    z[mask] = (y[mask] - f[mask]) / dy[mask]

    return z