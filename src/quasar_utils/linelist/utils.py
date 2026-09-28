from astropy.units import Quantity
from pandas import DataFrame

from quasar_utils.setup import Info

REQUIRED_COLUMNS: frozenset[str] = frozenset([
    "name",
    "linetype", 
    "complex",
    "n_max",
    "wave",
    "needs_line",
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
])

class Converter:
    @classmethod
    def strength_lower_converter(cls, info: Info, s: str) -> float:
        if not hasattr(cls, "strength_bounds"):
            return s
        
        _s = s.strip()
        if _s:
            val = (
                float(_s)
                if len(_s.split(" ")) == 1
                else info.units.getStrength(Quantity(_s))
            )
        else:
            val = cls.strength_bounds(info)[0]

        assert val >= 0.0
        return val

    @classmethod
    def strength_upper_converter(cls, info: Info, s: str) -> float:
        if not hasattr(cls, "strength_bounds"):
            return s
        
        _s = s.strip()
        if _s:
            val = (
                float(_s)
                if len(_s.split(" ")) == 1
                else info.units.getStrength(Quantity(_s))
            )
        else:
            val = cls.strength_bounds(info)[1]

        assert val >= 0.0
        return val

    @classmethod
    def fwhm_v_lower_converter(cls, info: Info, s: str) -> float:
        if not hasattr(cls, "fwhm_v_bounds"):
            return s
        
        _s = s.strip()
        if _s:
            val = (
                float(_s)
                if len(_s.split(" ")) == 1
                else info.units.getKMS(Quantity(_s))
            )
        else:
            val = cls.fwhm_v_bounds(info)[0]

        assert val > 0.0
        return val

    @classmethod
    def fwhm_v_upper_converter(cls, info: Info, s: str) -> float:
        if not hasattr(cls, "fwhm_v_bounds"):
            return s

        _s = s.strip()
        if _s:
            val = (
                float(_s)
                if len(_s.split(" ")) == 1
                else info.units.getKMS(Quantity(_s))
            )
        else:
            val = cls.fwhm_v_bounds(info)[1]

        assert val > 0.0
        return val

    @classmethod
    def v_off_lower_converter(cls, info: Info, s: str) -> float:
        if not hasattr(cls, "v_off_bounds"):
            return s
        
        _s = s.strip()
        if _s:
            val = (
                float(_s)
                if len(_s.split(" ")) == 1
                else info.units.getKMS(Quantity(_s))
            )
        else:
            val = cls.v_off_bounds(info)[0]

        return val

    @classmethod
    def v_off_upper_converter(cls, info: Info, s: str) -> float:
        if not hasattr(cls, "v_off_bounds"):
            return s
        
        _s = s.strip()
        if _s:
            val = (
                float(_s)
                if len(_s.split(" ")) == 1
                else info.units.getKMS(Quantity(_s))
            )
        else:
            val = cls.v_off_bounds(info)[1]

        return val

    ###

    @classmethod
    def name_converter(cls, s: str) -> str:
        _s = s.strip()
        assert len(_s) > 0
        return _s

    @classmethod
    def linetype_converter(cls, s: str) -> str:
        _s = s.strip().lower()
        assert _s in {"n", "b"}
        return _s

    @classmethod
    def complex_converter(cls, s: str) -> str | None:
        _s = s.strip()
        return _s or None

    @classmethod
    def n_max_converter(cls, s: str) -> int:
        _s = s.strip()
        return int(_s) if _s else 1

    @classmethod
    def wave_converter(cls, info: Info, s: str) -> float:
        _s = s.strip()
        assert len(_s) > 0
        return (
            float(_s)
            if len(_s.split(" ")) == 1
            else info.units.getWavelength(Quantity(_s))
        )

    @classmethod
    def needs_line_converter(cls, s: str) -> str:
        return s.strip()

    @classmethod
    def is_copy_of_converter(cls, s: str) -> str:
        return s.strip()

    @classmethod
    def scale_init_converter(cls, info: Info, s: str) -> float:
        _s = s.strip()
        return float(_s) if _s else info.lines.scale_init

    @classmethod
    def scale_lower_converter(cls, info: Info, s: str) -> float:
        _s = s.strip()
        return float(_s) if _s else info.lines.scale_bounds[0]

    @classmethod
    def scale_upper_converter(cls, info: Info, s: str) -> float:
        _s = s.strip()
        return float(_s) if _s else info.lines.scale_bounds[1]

    @classmethod
    def scale_fixed_converter(cls, info: Info, s: str) -> bool:
        _s = s.strip()
        return bool(_s) if _s else info.lines.scale_fixed

    @classmethod
    def wave_reverser(cls, wave: float, info: Info) -> str:
        return str(info.units.getWavelength(wave))


    @classmethod
    def strength_reverser(cls, strength: float, info: Info) -> str:
        return str(info.units.getStrength(strength))


    @classmethod
    def v_reverser(cls, fwhm_v: float, info: Info) -> str:
        return str(info.units.getKMS(fwhm_v))


class NarrowConverter(Converter):
    @classmethod
    def strength_bounds(cls, info: Info) -> tuple[float, float]:
        return info.lines.strength_bounds_n

    @classmethod
    def fwhm_v_bounds(cls, info: Info) -> tuple[float, float]:
        return info.lines.fwhm_v_bounds_n

    @classmethod
    def v_off_bounds(cls, info: Info) -> tuple[float, float]:
        return info.lines.v_off_bounds_n

class BroadConverter(Converter):
    @classmethod
    def strength_bounds(cls, info: Info) -> tuple[float, float]:
        return info.lines.strength_bounds_b

    @classmethod
    def fwhm_v_bounds(cls, info: Info) -> tuple[float, float]:
        return info.lines.fwhm_v_bounds_b

    @classmethod
    def v_off_bounds(cls, info: Info) -> tuple[float, float]:
        return info.lines.v_off_bounds_b

def df_to_dict(
    *,
    df: DataFrame,
    info: Info,
) -> dict:
    output_dict = {}
    for _, row in df.iterrows():
        name = row["name"]
        output_dict[name] = field = {}

        field["type"] = row["type"]

        if row["complex"]:
            field["complex"] = row["complex"]

        field["n_max"] = row["n_max"]
        field["wave"] = Converter.wave_reverser(row["wave"], info)

        if row["needs_line"]:
            field["needs_line"] = row["needs_line"]

        field["strength_lower"] = Converter.strength_reverser(
            info, row["strength_lower"]
        )
        field["strength_upper"] = Converter.strength_reverser(
            info, row["strength_upper"]
        )

        field["v_off_lower"] = Converter.v_reverser(info, row["v_off_lower"])
        field["v_off_upper"] = Converter.v_reverser(info, row["v_off_upper"])

        field["fwhm_v_lower"] = Converter.v_reverser(info, row["fwhm_v_lower"])
        field["fwhm_v_upper"] = Converter.v_reverser(info, row["fwhm_v_upper"])

        if row["is_copy_of"]:
            field["is_copy_of"] = row["is_copy_of"]

        field["scale_init"] = row["scale_init"]
        field["scale_lower"] = row["scale_lower"]
        field["scale_upper"] = row["scale_upper"]
        field["scale_fixed"] = row["scale_fixed"]

    return output_dict
