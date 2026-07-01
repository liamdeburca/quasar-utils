__all__ = ['LineList', 'DEFAULT_LINE_LIST_PATH']

from typing import Self
from astropy.units import Quantity
from pandas import DataFrame, read_csv
from functools import partial
from pathlib import Path

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from quasar_typing.pathlib import AbsoluteCSVPath

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

def fwhm_v_lower_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['fwhm_v_bounds'][0]

    return (
        float(s) 
        if len(s.split(' ')) == 1 else 
        info.units.getC(Quantity(s))
    )

def fwhm_v_upper_converter(info: Info, s: str) -> float:
    if not s: 
        return info.lines['fwhm_v_bounds'][1]

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

def scale_fixed_converter(info: Info, s: str) -> bool:
    return bool(s) if s else info.lines['scale_fixed']

class LineList(DataFrame):
    """
    pandas.DataFrame
    """
    REQUIRED_COLUMNS = [
        'name', 
        'complex',
        'n_max', 
        'needs_line',
        'line',
        'strength_lower', 
        'strength_upper',
        'v_off_lower', 
        'v_off_upper',
        'fwhm_v_lower', 
        'fwhm_v_upper',
        'is_copy_of',
        'scale_init', 
        'scale_lower', 
        'scale_upper',
        'scale_fixed',
    ]

    @classmethod
    @validate_call
    def read_csv(
        cls,
        *,
        path: AbsoluteCSVPath = DEFAULT_LINE_LIST_PATH,
        info: Info = None,
    ) -> Self:
        df = read_csv(
            path,
            skipinitialspace = True,
            usecols = cls.REQUIRED_COLUMNS,
            converters = dict(
                n_max          = n_max_converter,
                needs_line     = needs_line_converter,
                is_copy_of     = is_copy_of_converter,
                line           = partial(line_converter,           info),
                strength_lower = partial(strength_lower_converter, info),
                strength_upper = partial(strength_upper_converter, info),
                fwhm_v_lower  = partial(fwhm_v_lower_converter,    info),
                fwhm_v_upper  = partial(fwhm_v_upper_converter,    info),
                v_off_lower    = partial(v_off_lower_converter,    info),
                v_off_upper    = partial(v_off_upper_converter,    info),
                scale_init     = partial(scale_init_converter,     info),
                scale_lower    = partial(scale_lower_converter,    info),
                scale_upper    = partial(scale_upper_converter,    info),
                scale_fixed    = partial(scale_fixed_converter,    info),
            )
        )
        df.sort_values('line', inplace=True)
        return df[df['n_max'] != 0]
    
    @classmethod
    def _validate(cls, value: object) -> Self:

        if not isinstance(value, DataFrame):
            msg = f"Expected a 'pandas.DataFrame', got {type(value).__name__}"
            raise PydanticCustomError('validation_error', msg)
        
        missing_columns = [
            col for col in cls.REQUIRED_COLUMNS
            if col not in value.columns
        ]
        if len(missing_columns) > 0:
            cols = ", ".join(missing_columns)
            msg = f"Line list DataFrame is missing required columns: {cols}"
            raise PydanticCustomError('validation_error', msg)

        return value
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)