cdef inline int _alpha_matrix_elements_c(
    double[::1] x,
    double[::1] xr,
    int[::1] i_indices,
    int[::1] j_indices,
    double[::1] vals,
):
    """
    Modifys the input arrays ('i_indices', 'j_indices', 'vals') in-place, 
    returning the number of changed elements. 

    Once cropped, these arrays may be used to construct a sparse matrix.
    """
    cdef int i, j
    cdef int nx = len(x) - 1
    cdef int nxr = len(xr) - 1
    cdef int count = 0
    cdef int j_start = 0

    for i in range(nxr):
        for j in range(j_start, nx):
            if x[j+1] <= xr[i]: 
                continue
            if x[j] >= xr[i+1]:
                break

            i_indices[count] = i
            j_indices[count] = j
            vals[count] = (min(x[j+1], xr[i+1]) - max(x[j], xr[i])) / (x[j+1] - x[j])
            count += 1

            if x[j+1] == xr[i+1]:
                j_start = j + 1
                break
            if x[j+1] > xr[i+1]:
                j_start = j
                break

    return count

def _alpha_matrix_elements(
    double[::1] x,
    double[::1] xr,
    int[::1] i_indices,
    int[::1] j_indices,
    double[::1] vals,
):
    return _alpha_matrix_elements_c(x, xr, i_indices, j_indices, vals)