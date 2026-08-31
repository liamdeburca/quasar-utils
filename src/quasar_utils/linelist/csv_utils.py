__all__ = ["read_csv", "to_csv"]

from collections import defaultdict
from functools import partial

from pandas import (
    DataFrame,
)
from pandas import (
    read_csv as pd_read_csv,
)
from quasar_typing.pathlib import (
    AbsoluteCSVPath,
    NewAbsoluteCSVPath,
)

from quasar_utils.setup import Info

from .utils import (
    REQUIRED_COLUMNS,
    complex_converter,
    df_to_dict,
    fwhm_v_lower_converter,
    fwhm_v_upper_converter,
    is_copy_of_converter,
    line_converter,
    n_max_converter,
    name_converter,
    needs_line_converter,
    scale_fixed_converter,
    scale_init_converter,
    scale_lower_converter,
    scale_upper_converter,
    strength_lower_converter,
    strength_upper_converter,
    v_off_lower_converter,
    v_off_upper_converter,
)


def read_csv_header(
    path: AbsoluteCSVPath,
) -> DataFrame:
    df: DataFrame = pd_read_csv(
        path,
        skipinitialspace=True,
        nrows=0,
    )
    df.rename(
        columns=lambda c: c.strip().lower(), 
        inplace=True,
    )
    return df


def read_csv(
    path: AbsoluteCSVPath,
    info: Info,
) -> DataFrame:
    names = list(read_csv_header(path).columns)
    df = pd_read_csv(
        path,
        header=0,
        skipinitialspace=True,
        names=names,
        usecols=[names.index(c) for c in REQUIRED_COLUMNS],
        converters={
            "name": name_converter,
            "complex": complex_converter,
            "n_max": n_max_converter,
            "needs_line": needs_line_converter,
            "is_copy_of": is_copy_of_converter,
            "line": partial(line_converter, info),
            "strength_lower": partial(strength_lower_converter, info),
            "strength_upper": partial(strength_upper_converter, info),
            "fwhm_v_lower": partial(fwhm_v_lower_converter, info),
            "fwhm_v_upper": partial(fwhm_v_upper_converter, info),
            "v_off_lower": partial(v_off_lower_converter, info),
            "v_off_upper": partial(v_off_upper_converter, info),
            "scale_init": partial(scale_init_converter, info),
            "scale_lower": partial(scale_lower_converter, info),
            "scale_upper": partial(scale_upper_converter, info),
            "scale_fixed": partial(scale_fixed_converter, info),
        },
    )
    df.sort_values("line", inplace=True)
    return df[df["n_max"] != 0]


def to_csv(
    df: DataFrame,
    path: NewAbsoluteCSVPath,
    info: Info,
) -> None:
    _output_dict = df_to_dict(df=df, info=info)
    output_dict = defaultdict(list)

    for name, field in _output_dict.items():
        output_dict[name] = field
        for cname in REQUIRED_COLUMNS:
            if cname == "name":
                continue
            output_dict[cname].append(field.get(cname, ""))

    output_df = DataFrame.from_dict(output_dict)
    output_df.to_csv(path, index=False)
