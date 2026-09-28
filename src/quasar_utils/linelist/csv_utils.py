__all__ = ["read_csv", "to_csv"]

from collections import defaultdict
from functools import partial

from pandas import DataFrame, Series
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
    BroadConverter,
    Converter,
    NarrowConverter,
    df_to_dict,
)


def _get_source_row(
    *,
    df: DataFrame,
    is_copy_of: str,
) -> Series:
    source_row = df.loc[df["name"] == is_copy_of].iloc[0]
    if source_row["is_copy_of"]:
        # Remove the current row from the dataframe to avoid circular references
        return _get_source_row(
            df=df[df["name"] != is_copy_of],
            is_copy_of=source_row["is_copy_of"],
        )
    return source_row


def _change_element(
    *,
    df: DataFrame, 
    i: int, 
    source_row: Series, 
    col_name: str, 
    multiply_by: float | None = None,
) -> bool:
    """
    Change a specified element of a dataframe based on a source row. Returns 
    True if the element was changed.
    """
    val = source_row[col_name]
    if multiply_by is not None:
        val *= multiply_by

    if df.at[i, col_name] != val:
        df.at[i, col_name] = val
        return True
    return False


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
            "name": Converter.name_converter,
            "linetype": Converter.linetype_converter,
            "complex": Converter.complex_converter,
            "n_max": Converter.n_max_converter,
            "needs_line": Converter.needs_line_converter,
            "is_copy_of": Converter.is_copy_of_converter,
            "wave": partial(Converter.wave_converter, info),
            "strength_lower": partial(Converter.strength_lower_converter, info),
            "strength_upper": partial(Converter.strength_upper_converter, info),
            "fwhm_v_lower": partial(Converter.fwhm_v_lower_converter, info),
            "fwhm_v_upper": partial(Converter.fwhm_v_upper_converter, info),
            "v_off_lower": partial(Converter.v_off_lower_converter, info),
            "v_off_upper": partial(Converter.v_off_upper_converter, info),
            "scale_init": partial(Converter.scale_init_converter, info),
            "scale_lower": partial(Converter.scale_lower_converter, info),
            "scale_upper": partial(Converter.scale_upper_converter, info),
            "scale_fixed": partial(Converter.scale_fixed_converter, info),
        },
    )
    df.sort_values("wave", inplace=True)
    df = df[df["n_max"] != 0]

    for i, row in df.iterrows():
        converter = NarrowConverter \
            if row["linetype"] == "n" \
            else BroadConverter

        df.at[i, "strength_lower"] = str(converter.strength_lower_converter(info, row["strength_lower"]))
        df.at[i, "strength_upper"] = str(converter.strength_upper_converter(info, row["strength_upper"]))
        df.at[i, "fwhm_v_lower"] = str(converter.fwhm_v_lower_converter(info, row["fwhm_v_lower"]))
        df.at[i, "fwhm_v_upper"] = str(converter.fwhm_v_upper_converter(info, row["fwhm_v_upper"]))
        df.at[i, "v_off_lower"] = str(converter.v_off_lower_converter(info, row["v_off_lower"]))
        df.at[i, "v_off_upper"] = str(converter.v_off_upper_converter(info, row["v_off_upper"]))

    df = df.astype({
        "strength_lower": float,
        "strength_upper": float,
        "fwhm_v_lower": float,
        "fwhm_v_upper": float,
        "v_off_lower": float,
        "v_off_upper": float,
    })

    for i, row in filter(
        lambda tup: tup[1]["is_copy_of"] in df["name"].values,
        df.iterrows(),
    ):
        # Copy velocity bounds and type
        source_row = _get_source_row(df=df, is_copy_of=row["is_copy_of"])
        f = partial(_change_element, df=df, i=i, source_row=source_row)

        f(col_name="n_max")
        f(col_name="linetype")
        f(col_name="fwhm_v_lower")
        f(col_name="fwhm_v_upper")
        f(col_name="v_off_lower")
        f(col_name="v_off_upper")

        # Update strength bounds            
        if row["scale_fixed"]:
            scale_lower = scale_upper= row["scale_init"]
        else:
            scale_lower = row["scale_lower"]
            scale_upper = row["scale_upper"]

        f(col_name="strength_lower", multiply_by=scale_lower)
        f(col_name="strength_upper", multiply_by=scale_upper)

    return df


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
