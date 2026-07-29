from libc.math cimport log, floor, ceil

cdef int _get_index(
    const double val,
    const double[::1] x,
    const double log_x0,
    const double denom,
    const int mode,
):
    """
    Assumes logarithmically spaced `x` with spacing `dx[n] = v_res * x[n]`.
    - `mode` == 0: returns the index s.t. `x[index] <= val`
    - `mode` == 1: returns the index s.t. `val <= x[index]`
    """
    cdef double x0 = x[0]
    cdef int idx

    if val < x0:
        idx = 0
    elif val >= x[x.shape[0] - 1]:
        idx = x.shape[0] - 1
    else:
        if mode == 0:
            idx = int(ceil((log(val) - log_x0) / denom))
            if val <= x[idx - 1]:
                idx -= 1
        elif mode == 1:
            idx = int(floor((log(val) - log_x0) / denom))
            if x[idx + 1] <= val:
                idx += 1

    return idx

def get_lower_index(
    double val,
    double[::1] x,
    double log_x0,
    double denom,
) -> int:
    return _get_index(val, x, log_x0, denom, 0)

def get_upper_index(
    double val,
    double[::1] x,
    double log_x0,
    double denom,
) -> int:
    return _get_index(val, x, log_x0, denom, 1)