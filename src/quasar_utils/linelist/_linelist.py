__all__ = ["DEFAULT_LINE_LIST_PATH", "LineList"]

from pathlib import Path
from typing import ClassVar, Self

from pandas import DataFrame
from pydantic import validate_call
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function
from quasar_typing.pathlib import CSVPath, YAMLPath
from quasar_typing.utils import isinstance_func

from quasar_utils.setup import Info

_this_file: Path = Path(__file__).resolve()
DEFAULT_LINE_LIST_PATH: CSVPath = (
    _this_file.parent / "defaults/line_list.csv"
)


class LineList(DataFrame):
    """
    pandas.DataFrame
    """
    REQUIRED_COLUMNS: ClassVar[list[str]] = [
        "name",
        "linetype",
        "complex",
        "n_max",
        "needs_line",
        "wave",
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

    @classmethod
    @validate_call
    def read_csv(
        cls,
        *,
        path: CSVPath = DEFAULT_LINE_LIST_PATH,
        info: Info = None,
    ) -> Self:
        from .csv_utils import read_csv
        return cls(DataFrame(read_csv(path, info)))

    @classmethod
    @validate_call
    def read_yaml(
        cls,
        *,
        path: YAMLPath,
        info: Info = None,
    ) -> Self:
        from .yaml_utils import read_yaml
        return cls(DataFrame(read_yaml(path, info)))

    @classmethod
    def _validate(cls, value: object) -> Self:
        value = isinstance_func(DataFrame)(value)

        missing_columns = [
            col for col in cls.REQUIRED_COLUMNS if col not in value.columns
        ]
        if len(missing_columns) > 0:
            cols = ", ".join(missing_columns)
            msg = f"Line list DataFrame is missing required columns: {cols}"
            raise PydanticCustomError("validation_error", msg)

        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
