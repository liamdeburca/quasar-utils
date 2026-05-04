from typing import Optional, Union, Callable, Iterable
from numpy import ndarray, array, nan_to_num, inf
from astropy.units import Unit, Quantity, CompositeUnit
from itertools import batched

from pydantic import validate_call
from quasar_typing.pathlib import AbsoluteFilePath
from quasar_typing.logging import Logger_

def check_if_comment(string:str):
    return string[0] == '#'

def trim_line(line:str):
    out = []
    for string in line.strip().split():
        if check_if_comment(string): break
        out.append(string)
    return out

@validate_call
def get_lines_from_file(
    hdr: str,
    path: AbsoluteFilePath,
    logger: Logger_ | None = None,
) -> list[list[str]]:

    if logger is not None:
        logger.debug(f"Reading block '{hdr}' of path '{path}':")

    with open(path) as f:
        all_lines = [
            line \
            for line in map(trim_line, f.readlines()) \
            if len(line) > 0
        ]

    if logger is not None:
        logger.debug(f">>> Found {len(all_lines)} lines in total.")        

    lines = []
    active = False
    for line in all_lines:
        if line[0] == hdr.upper():
            active = True
            continue

        if active and line[0].isupper():
            break
        elif active and len(line) >= 2:
            lines.append(line)

    if logger is not None:
        logger.debug(f">>> Found {len(lines)} relevant lines.")

    return lines

###

def _map(f: Callable, l: Iterable) -> list:
    return list(map(f, l))

###

def as_str(s: str) -> str:
    return s

def as_float(s: str) -> Optional[float]:
    try:               return float(s)
    except ValueError: return None

def as_int(s: str) -> Optional[int]:
    return int(as_float(s)) if s.isnumeric() else None

def as_bool(s: str) -> Optional[bool]:
    if not s.isalpha():
        raise ValueError('this line cannot be parsed as a boolean!')
    
    match s.lower():
        case 'true' | 't' | 'yes' | 'y': return True
        case 'false' | 'f' | 'no' | 'n': return False
        case _:                          return None

def as_float_or_int(s: str) -> Union[float, int, None]:
    f = as_float(s)
    if (f is None) or ('.' in s): return f
    else:                         return int(f)

###

def as_list(s: str) -> list[str]:
    return s.split('/')

###

def as_list_of_floats(s: str) -> list[Optional[float]]:
    return _map(as_float, as_list(s))

def as_list_of_ints(s: str) -> list[Optional[int]]:
    return _map(as_int, as_list(s))

def as_list_of_bools(s: str) -> list[Optional[bool]]:
    return _map(as_bool, as_list(s))

###

def as_array_of_floats(s: str) -> ndarray[float]:
    return array(as_list_of_floats(s), dtype=float)

def as_array_of_ints(s: str) -> ndarray[int]:
    return array(as_list_of_ints(s), dtype=int)

def as_array_of_bools(s: str) -> ndarray[bool]:
    return array(as_list_of_bools(s), dtype=bool)

### 

def as_range_of_floats(s: str) -> ndarray[float]:
    from numpy import arange
    
    assert s.lower().startswith('range:')
    s = s.lower().removeprefix('range:')

    start, stop, step = as_list_of_floats(s)[:3]

    return arange(start, stop + step/2, step, dtype=float)

def as_range_of_ints(s: str) -> ndarray[int]:
    from numpy import arange

    assert s.lower().startswith('range:')
    s = s.lower().removeprefix('range:')

    start, stop, step = as_list_of_ints(s)[:3]

    return arange(start, stop + step/2, step, dtype=int)

def as_range_of_quantities(l: list[str]) -> Quantity:
    return as_range_of_floats(l[0]) * Unit(l[1])

###

def as_unit(s: str) -> Unit:
    return Unit(s)

def as_composite_unit(l: list[str]) -> CompositeUnit:
    return Unit(''.join(l))

def as_quantity(l: list[str, str]) -> Quantity:
    return as_float(l[0]) * as_unit(l[1])

def as_scalar_or_quantity(l: list[str]) -> Union[float, int, Quantity]:
    if len(l) == 1: return as_float_or_int(l[0])
    else:           return as_quantity(l)

def as_list_of_quantity(l: list[str, str]) -> Quantity:
    return as_list_of_floats(l[0]) * as_unit(l[1])

def as_list_of_scalars_or_quantity(l: list[str]) -> Union[list, Quantity]:
    if len(l) == 1: return _map(as_float_or_int, as_list(l[0]))
    else:           return as_list_of_quantity(l)

###

def as_bounds(l: list[str]) -> tuple:
    x1, x2 = _map(as_float, l)
    return (
        nan_to_num(x1, nan=-inf),
        nan_to_num(x2, nan=inf),
    )    

def as_bounds_of_quantity(l: list[str]) -> Quantity:
    return as_bounds(l[:2]) * as_unit(l[2])

def as_bounds_of_scalars_or_quantity(l: list[str]) -> Union[tuple, Quantity]:
    if len(l) == 2: return as_bounds(l)
    else:           return as_bounds_of_quantity(l)

### 

def as_pairs_of_floats(s: str) -> list:
    return _map(list, batched(as_list_of_floats(s), 2))

def as_pairs_of_quantity(l: list[str]) -> Quantity:
    return as_pairs_of_floats(l[0]) * as_unit(l[1])

def as_pairs_of_floats_or_quantity(l: list[str]) -> Union[list, Quantity]:
    if len(l) == 1: return as_pairs_of_floats(l)
    else:           return as_pairs_of_quantity(l)

### SPECIAL CASES

def as_loading_z(l: list[str]) -> Union[float, tuple]:
    return as_float(l[0]) if len(l) == 1 else as_loading_tuple(l)

def as_loading_tuple(l: list[str]) -> tuple:
    return (as_int(l[0]), *l[1:])

def as_set_of_measures(s: str) -> set[str]:
    return set(_map(str.lower, as_list(s)))

def as_list_of_lines(l: list[str]) -> list[Union[str, float, Quantity]]:
    unit = 1.0 if len(l) == 1 else as_unit(l[1])
    return [
        as_float(s) * unit if s.isnumeric() else 'all' \
        for s in as_list(l[0])
    ]