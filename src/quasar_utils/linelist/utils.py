from astropy.units import Quantity
from pandas import DataFrame

from quasar_utils.setup import Info

REQUIRED_COLUMNS = [
    "name",
    "complex",
    "n_max",
    "needs_line",
    "line",
    "strength_lower",
    "strength_upper",
    "v_off_lower",
    "v_off_upper",
    "fwhm_v_lower",
    "fwhm_v_upper",
    "is_copy_of",
    "scale_init",
    "scale_lower",
    "scale_upper",
    "scale_fixed",
]


def n_max_converter(s: str) -> int:
    return int(s) if s else 1


def line_converter(info: Info, s: str) -> float:
    assert len(s) > 0
    return (
        float(s)
        if len(s.split(" ")) == 1
        else info.units.getWavelength(Quantity(s))
    )


def needs_line_converter(s: str) -> str | None:
    return s or None


def strength_lower_converter(info: Info, s: str) -> float:
    if not s:
        return info.lines.strength_bounds[0]

    return (
        float(s)
        if len(s.split(" ")) == 1
        else info.units.getStrength(Quantity(s))
    )


def strength_upper_converter(info: Info, s: str) -> float:
    if not s:
        return info.lines.strength_bounds[1]

    return (
        float(s)
        if len(s.split(" ")) == 1
        else info.units.getStrength(Quantity(s))
    )


def fwhm_v_lower_converter(info: Info, s: str) -> float:
    if not s:
        return info.lines.fwhm_v_bounds[0]

    return float(s) if len(s.split(" ")) == 1 else info.units.getC(Quantity(s))


def fwhm_v_upper_converter(info: Info, s: str) -> float:
    if not s:
        return info.lines.fwhm_v_bounds[1]

    return float(s) if len(s.split(" ")) == 1 else info.units.getC(Quantity(s))


def v_off_lower_converter(info: Info, s: str) -> float:
    if not s:
        return info.lines.v_off_bounds[0]

    return float(s) if len(s.split(" ")) == 1 else info.units.getC(Quantity(s))


def v_off_upper_converter(info: Info, s: str) -> float:
    if not s:
        return info.lines.v_off_bounds[1]

    return float(s) if len(s.split(" ")) == 1 else info.units.getC(Quantity(s))


def is_copy_of_converter(s: str) -> str | None:
    return s or None


def scale_init_converter(info: Info, s: str) -> float:
    return float(s) if s else info.lines.scale_init


def scale_lower_converter(info: Info, s: str) -> float:
    return float(s) if s else info.lines.scale_bounds[0]


def scale_upper_converter(info: Info, s: str) -> float:
    return float(s) if s else info.lines.scale_bounds[1]


def scale_fixed_converter(info: Info, s: str) -> bool:
    return bool(s) if s else info.lines.scale_fixed


###


def line_reverser(line: float, info: Info) -> str:
    return str(info.units.getWavelength(line))


def strength_reverser(strength: float, info: Info) -> str:
    return str(info.units.getStrength(strength))


def v_reverser(fwhm_v: float, info: Info) -> str:
    return str(info.units.getC(fwhm_v))


###


def df_to_dict(
    *,
    df: DataFrame,
    info: Info,
) -> dict:
    output_dict = {}
    for _, row in df.iterrows():
        name = row["name"]
        output_dict[name] = field = {}

        if row["complex"]:
            field["complex"] = row["complex"]

        field["n_max"] = row["n_max"]
        field["line"] = line_reverser(row["line"], info)

        if row["needs_line"]:
            field["needs_line"] = row["needs_line"]

        field["strength_lower"] = strength_reverser(
            info, row["strength_lower"]
        )
        field["strength_upper"] = strength_reverser(
            info, row["strength_upper"]
        )

        field["v_off_lower"] = v_reverser(info, row["v_off_lower"])
        field["v_off_upper"] = v_reverser(info, row["v_off_upper"])

        field["fwhm_v_lower"] = v_reverser(info, row["fwhm_v_lower"])
        field["fwhm_v_upper"] = v_reverser(info, row["fwhm_v_upper"])

        if row["is_copy_of"]:
            field["is_copy_of"] = row["is_copy_of"]

        field["scale_init"] = row["scale_init"]
        field["scale_lower"] = row["scale_lower"]
        field["scale_upper"] = row["scale_upper"]
        field["scale_fixed"] = row["scale_fixed"]

    return output_dict
