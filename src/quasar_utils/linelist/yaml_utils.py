__all__ = ["read_yaml", "to_yaml"]

from collections import defaultdict

from pandas import DataFrame
from quasar_typing.pathlib import AbsoluteYAMLPath, NewAbsoluteYAMLPath
from yaml import safe_dump, safe_load

from quasar_utils.setup.info import Info

from .utils import (
    BroadConverter,
    Converter,
    NarrowConverter,
    df_to_dict,
)


def read_yaml(
    path: AbsoluteYAMLPath,
    info: Info,
) -> DataFrame:
    with open(path, "r") as f:
        raw_data = safe_load(f)

    # Dictionary of lists to hold the processed data
    processed_data = defaultdict(list)
    for name, f in filter(
        lambda item: "line" in item[1] and item[1].get("n_max", 1) != 0,
        raw_data.items(),
    ):
        processed_data["name"].append(name)

        processed_data["linetype"].append(
            linetype := Converter.linetype_converter(f["linetype"])
        )
        processed_data["complex"].append(
            Converter.complex_converter(f["complex"]),
        )
        processed_data["n_max"].append(
            Converter.n_max_converter(f.get("n_max", "1"))
        )
        processed_data["needs_line"].append(
            Converter.needs_line_converter(f.get("needs_line", ""))
        )
        processed_data["wave"].append(
            Converter.wave_converter(info, f["wave"])
        )

        _Converter = NarrowConverter if linetype == "n" else BroadConverter

        processed_data["strength_lower"].append(
            _Converter.strength_lower_converter(info, f.get("strength_lower", ""))
        )
        processed_data["strength_upper"].append(
            _Converter.strength_upper_converter(info, f.get("strength_upper", ""))
        )
        processed_data["fwhm_v_lower"].append(
            _Converter.fwhm_v_lower_converter(info, f.get("fwhm_v_lower", ""))
        )
        processed_data["fwhm_v_upper"].append(
            _Converter.fwhm_v_upper_converter(info, f.get("fwhm_v_upper", ""))
        )
        processed_data["v_off_lower"].append(
            _Converter.v_off_lower_converter(info, f.get("v_off_lower", ""))
        )
        processed_data["v_off_upper"].append(
            _Converter.v_off_upper_converter(info, f.get("v_off_upper", ""))
        )
        del _Converter

        processed_data["is_copy_of"].append(
            Converter.is_copy_of_converter(f.get("is_copy_of", ""))
        )
        processed_data["scale_init"].append(
            Converter.scale_init_converter(info, f.get("scale_init", ""))
        )
        processed_data["scale_lower"].append(
            Converter.scale_lower_converter(info, f.get("scale_lower", ""))
        )
        processed_data["scale_upper"].append(
            Converter.scale_upper_converter(info, f.get("scale_upper", ""))
        )
        processed_data["scale_fixed"].append(
            Converter.scale_fixed_converter(info, f.get("scale_fixed", ""))
        )

    df = DataFrame.from_dict(processed_data)
    df.sort_values("wave", inplace=True)
    return df


def to_yaml(
    df: DataFrame,
    path: NewAbsoluteYAMLPath,
    info: Info,
) -> None:
    output_dict = df_to_dict(df=df, info=info)
    with open(path, "w") as f:
        safe_dump(output_dict, f, sort_keys=False)
