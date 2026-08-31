__all__ = ["DEFAULT_LINE_LIST_PATH", "LineList"]

from pathlib import Path
from typing import ClassVar, Self

from pandas import DataFrame
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function
from quasar_typing.pathlib import AbsoluteCSVPath, AbsoluteYAMLPath

from ..decorators import validate_call
from ..setup import Info
from .csv_utils import read_csv
from .yaml_utils import read_yaml

_this_file: Path = Path(__file__).resolve()
DEFAULT_LINE_LIST_PATH: AbsoluteCSVPath = (
    _this_file.parent / "defaults/line_list.csv"
)


class LineList(DataFrame):
    """
    pandas.DataFrame
    """

    REQUIRED_COLUMNS: ClassVar[list[str]] = [
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

    @classmethod
    @validate_call
    def read_csv(
        cls,
        *,
        path: AbsoluteCSVPath = DEFAULT_LINE_LIST_PATH,
        info: Info = None,
    ) -> Self:
        return cls(read_csv(path, info))

    @classmethod
    @validate_call
    def read_yaml(
        cls,
        *,
        path: AbsoluteYAMLPath,
        info: Info = None,
    ) -> Self:
        return read_yaml(path, info)

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, DataFrame):
            msg = f"Expected a 'pandas.DataFrame', got {type(value).__name__}"
            raise PydanticCustomError("validation_error", msg)

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
