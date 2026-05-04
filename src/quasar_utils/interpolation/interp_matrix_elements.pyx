# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

cdef inline int _interp_matrix_elements_no_bias_c(
    double[::1] x,
    double[::1] xb,
    int[::1] indices,
    int[::1] i_indices,
    int[::1] j_indices,
    double[::1] vals,
):
    cdef int n_in = len(x)
    cdef int n_out = len(xb)
    cdef int i, j, count, idx
    cdef double xb_i, x0, x1, w0, w1

    j = 0
    for i in range(n_out):
        while j < n_in and x[j] < xb[i]:
            j += 1
        indices[i] = j

    count = 0
    for i in range(n_out):
        xb_i = xb[i]
        idx = indices[i]
        if idx == 0:
            if xb_i == x[0]:
                i_indices[count] = i
                j_indices[count] = 0
                vals[count] = 1.0
                count += 1
            else:
                continue
        elif idx >= n_in:
            continue
        else:
            x0 = x[idx - 1]
            x1 = x[idx]
            w0 = (x1 - xb_i) / (x1 - x0)
            w1 = 1 - w0

            if w0 != 0.0:
                i_indices[count] = i
                j_indices[count] = idx - 1
                vals[count] = w0
                count += 1
            if w1 != 0.0:
                i_indices[count] = i
                j_indices[count] = idx
                vals[count] = w1
                count += 1

    return count

cdef inline int _interp_matrix_elements_c(
    double[::1] x,
    double[::1] xb,
    int[::1] indices,
    int[::1] i_indices,
    int[::1] j_indices,
    double[::1] vals,
    double[::1] bias,
    double left_val,
    double right_val,
):
    cdef int n_in = len(x)
    cdef int n_out = len(xb)
    cdef int i, j, count, idx
    cdef double xb_i, x0, x1, w0, w1

    j = 0
    for i in range(n_out):
        while j < n_in and x[j] < xb[i]:
            j += 1
        indices[i] = j

    count = 0
    for i in range(n_out):
        xb_i = xb[i]
        idx = indices[i]
        if idx == 0:
            if xb_i == x[0]:
                i_indices[count] = i
                j_indices[count] = 0
                vals[count] = 1.0
                count += 1
            else:
                bias[i] = left_val
        elif idx >= n_in:
            bias[i] = right_val
            continue
        else:
            x0 = x[idx - 1]
            x1 = x[idx]
            w0 = (x1 - xb_i) / (x1 - x0)
            w1 = 1 - w0

            if w0 != 0.0:
                i_indices[count] = i
                j_indices[count] = idx - 1
                vals[count] = w0
                count += 1
            if w1 != 0.0:
                i_indices[count] = i
                j_indices[count] = idx
                vals[count] = w1
                count += 1

    return count

def _interp_matrix_elements_no_bias(
    double[::1] x,
    double[::1] xb,
    int[::1] indices,
    int[::1] i_indices,
    int[::1] j_indices,
    double[::1] vals,
):
    return _interp_matrix_elements_no_bias_c(
        x, xb, 
        indices, i_indices, j_indices, vals,
    )

def _interp_matrix_elements(
    double[::1] x,
    double[::1] xb,
    int[::1] indices,
    int[::1] i_indices,
    int[::1] j_indices,
    double[::1] vals,
    double[::1] bias,
    double left_val,
    double right_val,
):
    return _interp_matrix_elements_c(
        x, xb, 
        indices, i_indices, j_indices, vals, 
        bias, left_val, right_val,
    )