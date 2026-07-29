from numpy import searchsorted

cdef int _get_index(
    const double val,
    const double[::1] x,
    const int mode,
):
    """
    Assumes linearly spaced `x` with constant spacing `dx`.
    - `mode` == 0: returns the index s.t. `x[index] <= val`
    - `mode` == 1: returns the index s.t. `val <= x[index]`
    """
    cdef double x0 = x[0]

    if val < x0:
        return 0
    elif val >= x[x.shape[0] - 1]:
        return x.shape[0] - 1
    else:
        if mode == 0:
            return searchsorted(x, val, side="right") - 1
        elif mode == 1:
            return searchsorted(x, val, side="left")

def get_lower_index(
    double val,
    double[::1] x,
) -> int:
    return _get_index(val, x, 0)

def get_upper_index(
    double val,
    double[::1] x,
) -> int:
    return _get_index(val, x, 1)