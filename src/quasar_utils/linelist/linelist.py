from collections.abc import Iterator
from functools import wraps
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, model_validator, validate_call
from pydantic_core import PydanticCustomError
from quasar_typing.pathlib import CSVPath

from quasar_utils.setup import Info


class LineList(BaseModel):
    name: list[str]
    linetype: list[Literal["n", "b"]]
    complex: list[str]
    n_max: list[int]
    needs_line: list[str]
    wave: list[float]

    strength_lower: list[float]
    strength_upper: list[float]

    v_off_lower: list[float]
    v_off_upper: list[float]

    fwhm_v_lower: list[float]
    fwhm_v_upper: list[float]

    is_copy_of: list[str]

    scale_init: list[float]
    scale_lower: list[float]
    scale_upper: list[float]
    scale_fixed: list[bool]

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

    @model_validator(mode="after")
    def check_arr_lengths(self) -> Self:
        column_lengths = {
            col_name: len(getattr(self, col_name))
            for col_name in self.REQUIRED_COLUMNS
        }
        bad_columns = {
            col_name: length
            for col_name, length in column_lengths.items()
            if length != column_lengths[self.REQUIRED_COLUMNS[0]]
        }
        if bad_columns:
            raise PydanticCustomError(
                "validation_error",
                "All columns must have the same length ({n}). "
                "Bad columns: {bad_columns}",
                {"n": len(self.name), "bad_columns": bad_columns}
            )
        return self


    def sort_values(self, by: str, inplace: bool = False) -> Self:
        from numpy import argsort
        sorted_indices = argsort(getattr(self, by))

        out = self if inplace else self.model_copy()
        for col_name in self.REQUIRED_COLUMNS:
            setattr(
                out, 
                col_name, 
                [getattr(out, col_name)[i] for i in sorted_indices]
            )
        return out


    def drop(self, index: int, inplace: bool = False) -> Self:
        out = self if inplace else self.model_copy()
        for col_name in self.REQUIRED_COLUMNS:
            getattr(out, col_name).pop(index)
        return out


    def iterrows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        yield from ((i, self[i]) for i in range(len(self.name)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            col_name: getattr(self, col_name)[index] 
            for col_name in self.REQUIRED_COLUMNS
        }

    ### read_csv

    @classmethod
    @validate_call
    def read_csv(
        cls,
        *,
        path: CSVPath,
        info: Info,
    ) -> Self:
        return cls(**cls._read_csv(path=path, info=info))

    @classmethod
    @wraps(read_csv)
    def _read_csv(
        cls,
        *,
        path: CSVPath,
        info: Info,
    ) -> dict[str, list[Any]]:
        from .csv_utils import read_csv
        return read_csv(path, info).to_dict(orient="list")

    ### toDataFrame

    def toDataFrame(self):
        from pandas import DataFrame
        return DataFrame(self.model_dump())


    def to_csv(self, path: CSVPath) -> None:
        self.toDataFrame().to_csv(path, index=False)