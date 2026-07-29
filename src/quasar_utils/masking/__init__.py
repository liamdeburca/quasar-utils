from math import log
from typing import Literal

from numpy import bool_, zeros_like
from quasar_typing.numpy import BoolVector, FloatVector

from quasar_utils.decorators import validate_call

from . import irregular, linear, logarithmic


def get_slice_from_irregular(
    arr: FloatVector,
    bounds: tuple[float, float],
) -> slice:
    idx1 = irregular.get_lower_index(bounds[0], arr)
    idx2 = idx1 + irregular.get_upper_index(bounds[1], arr[idx1:])

    return slice(idx1, idx2 + 1)


def get_slice_from_linear(
    arr: FloatVector,
    bounds: tuple[float, float],
    *,
    dx: float | None = None,
) -> slice:
    if dx is None:
        dx = arr[1] - arr[0]

    idx1 = linear.get_lower_index(bounds[0], arr, dx)
    idx2 = idx1 + linear.get_upper_index(bounds[1], arr[idx1:], dx)

    return slice(idx1, idx2 + 1)


def get_slice_from_logarithmic(
    arr: FloatVector,
    bounds: tuple[float, float],
    *,
    log_x0: float | None = None,
    v_res: float | None = None,
    denom: float | None = None,
) -> slice:
    if log_x0 is None:
        log_x0 = log(arr[0])
    if denom is None:
        if v_res is None:
            v_res = (arr[1] / arr[0]) - 1
        denom = log(1 + v_res)

    idx1 = logarithmic.get_lower_index(bounds[0], arr, log_x0, denom)
    idx2 = logarithmic.get_upper_index(bounds[1], arr, log_x0, denom)

    return slice(idx1, idx2 + 1)


@validate_call
def get_mask(
    arr: FloatVector,
    bounds: tuple[float, float],
    *,
    array_type: Literal["irregular", "linear", "logarithmic"] | None = None,
    where: BoolVector | None = None,
    dx: float | None = None,
    log_x0: float | None = None,
    v_res: float | None = None,
    denom: float | None = None,
) -> BoolVector:

    if array_type is None:
        # Standard numpy indexing
        lb, ub = bounds
        mask = (lb <= arr) & (arr <= ub)
    else:
        mask = zeros_like(arr, dtype=bool_)
        match array_type:
            case "irregular":
                sel = get_slice_from_irregular(arr, bounds)
            case "linear":
                sel = get_slice_from_linear(arr, bounds, dx=dx)

            case "logarithmic":
                sel = get_slice_from_logarithmic(
                    arr, bounds, log_x0=log_x0, v_res=v_res, denom=denom
                )

        mask[sel] = True

    return mask if where is None else mask & where
