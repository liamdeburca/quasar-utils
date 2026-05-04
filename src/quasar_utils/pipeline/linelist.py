__all__ = ['read_linelist', 'DEFAULT_LINE_LIST_PATH']

from astropy.units import Quantity
from pandas import read_csv
from functools import partial
from pathlib import Path

from quasar_typing.pathlib import AbsoluteCSVPath
from quasar_typing.pandas import LineList

from ..decorators import validate_call
from ..setup import Info

_this_file: Path = Path(__file__).resolve()
DEFAULT_LINE_LIST_PATH: AbsoluteCSVPath \
    = _this_file.parent / 'defaults/line_list.csv'

def n_max_converter(s: str) -> int:
    return int(s) if s else 1

def line_converter(info: Info, s: str) -> float:
    assert len(s) > 0
    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getWavelength(Quantity(s))
    )

def needs_line_converter(s: str) -> float | None:
    return s or None

def strength_lower_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['strength_bounds'][0]
    
    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getStrength(Quantity(s))
    )

def strength_upper_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['strength_bounds'][1]

    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getStrength(Quantity(s))
    )

def sigma_v_lower_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['sigma_v_bounds'][0]

    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getC(Quantity(s))
    )

def sigma_v_upper_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['sigma_v_bounds'][1]

    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getC(Quantity(s))
    )

def v_off_lower_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['v_off_bounds'][0]

    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getC(Quantity(s))
    )

def v_off_upper_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['v_off_bounds'][1]

    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getC(Quantity(s))
    )

def is_copy_of_converter(s: str) -> str | None:
    return s or None

def scale_init_converter(info: Info, s: str) -> float:
    return float(s) if s else info.lines['scale_init']

def scale_lower_converter(info: Info, s: str) -> float:
    return float(s) if s else info.lines['scale_bounds'][0]

def scale_upper_converter(info: Info, s: str) -> float:
    return float(s) if s else info.lines['scale_bounds'][1]

@validate_call
def read_linelist(
    *,
    path: AbsoluteCSVPath = DEFAULT_LINE_LIST_PATH,
    info: Info = None,
) -> LineList:
    """
    ** PYDANTIC VALIDATED FUNCTION **
    """
    df = read_csv(
        path,
        skipinitialspace = True,
        usecols = LineList.REQUIRED_COLUMNS,
        converters = dict(
            n_max          = n_max_converter,
            needs_line     = needs_line_converter,
            is_copy_of     = is_copy_of_converter,
            line           = partial(line_converter,           info),
            strength_lower = partial(strength_lower_converter, info),
            strength_upper = partial(strength_upper_converter, info),
            sigma_v_lower  = partial(sigma_v_lower_converter,  info),
            sigma_v_upper  = partial(sigma_v_upper_converter,  info),
            v_off_lower    = partial(v_off_lower_converter,    info),
            v_off_upper    = partial(v_off_upper_converter,    info),
            scale_init     = partial(scale_init_converter,     info),
            scale_lower    = partial(scale_lower_converter,    info),
            scale_upper    = partial(scale_upper_converter,    info),
        )
    )
    df.sort_values('line', inplace=True)
    return df