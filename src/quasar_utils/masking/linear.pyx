from libc.math cimport floor, ceil

cdef int _get_index(
    const double val,
    const double[::1] x,
    const double dx,
    const int mode,
):
    """
    Assumes linearly spaced `x` with constant spacing `dx`.
    - `mode` == 0: returns the index s.t. `x[index] <= val`
    - `mode` == 1: returns the index s.t. `val <= x[index]`
    """
    cdef double x0 = x[0]
    cdef int idx

    if val < x0:
        return 0
    elif val >= x[x.shape[0] - 1]:
        return x.shape[0] - 1
    else:
        if mode == 0:
            idx = int(ceil((val - x0) / dx))
            if val <= x[idx - 1]:
                idx -= 1
        elif mode == 1:
            idx = int(floor((val - x0) / dx))
            if x[idx + 1] <= val:
                idx += 1

    return idx

def get_lower_index(
    double val,
    double[::1] x,
    double dx,
) -> int:
    return _get_index(val, x, dx, 0)

def get_upper_index(
    double val,
    double[::1] x,
    double dx,
) -> int:
    return _get_index(val, x, dx, 1)