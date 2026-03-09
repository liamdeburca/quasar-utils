from typing import Any, Union
from astropy.units import Unit, Quantity
from astropy.units.quantity import Quantity

def check_if_comment(string:str):
    return string[0] == '#'

def trim_line(line:str):
    
    out = []
    for string in line.strip().split():
        if check_if_comment(string):
            break

        out.append(string)

    return out

def check_val(
    val: Union[Any, Quantity],
) -> Union[Any, Quantity]: 
    """
    Lorem ipsum...
    """
    if isinstance(val, Quantity) and val.unit == Unit(): return val.value
    else:                                                return val

def _or_default(ref: dict, kwargs: dict, key: str) -> Any:
    value = kwargs.get(key, None)
    default = ref[key]

    return default if (value is None) else value

def val_and_type(val: Any) -> str:
    return f"{val} ({type(val).__name__})"