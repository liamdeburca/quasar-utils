from collections.abc import Iterable
from typing import Any

from astropy.units import Quantity, Unit


def check_if_comment(string: str):
    return string[0] == "#"


def trim_line(line: str):

    out = []
    for string in line.strip().split():
        if check_if_comment(string):
            break

        out.append(string)

    return out


def check_val(
    val: Quantity | float | Iterable[float],
) -> Quantity | float | Iterable[float]:
    """
    Returns an Quantity if the input is a non-unitless Quantity. Otherwise 
    returns the scalar value of the input.
    """
    if isinstance(val, Quantity) and val.unit == Unit():
        return val.value
    else:
        return val


def _or_default(ref: dict, kwargs: dict, key: str) -> Any:
    value = kwargs.get(key, None)
    default = ref[key]

    return default if (value is None) else value


def val_and_type(val: Any) -> str:
    return f"{val} ({type(val).__name__})"
