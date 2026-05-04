from typing import Self, ClassVar
from pathlib import Path
from json import load as json_load
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import (
    AbsolutePath, AnyAbsolutePath, AbsoluteCSVPath, 
    AbsoluteDirPath, AbsoluteJSONPath,
)
from .defaults import DIRECTORY_TO_DEFAULTS

@dataclass
class BackupFile:
    input_path: AbsolutePath | None = field(default=None, kw_only=True)
    output_path: AnyAbsolutePath | None = field(default=None, kw_only=True)

    fitting_info: AbsoluteJSONPath | None = field(default=None, kw_only=True)
    line_list: AbsoluteCSVPath | None = field(default=None, kw_only=True)
    measure_list: AbsoluteCSVPath | None = field(default=None, kw_only=True)

    directory_to_defaults: ClassVar[AbsoluteDirPath] = DIRECTORY_TO_DEFAULTS

    def __post_init__(self) -> None:
        if self.fitting_info is None:
            self.fitting_info = self.directory_to_defaults / "fitting_info.json"
        if self.line_list is None:
            self.line_list = self.directory_to_defaults / "line_list.csv"
        if self.measure_list is None:
            self.measure_list = self.directory_to_defaults / "measure_list.csv"

    @classmethod
    def load(cls, path: AbsoluteJSONPath | None) -> Self:
        """
        Creates a 'BackupFile' instance from a JSON file.
        """
        if path is None:
            return cls()
        
        with open(path) as f:
            backup_info: dict[str, Path] = json_load(f)

        return cls(
            input_path=backup_info.get("input_path"),
            output_path=backup_info.get("output_path"),
            fitting_info=backup_info.get("fitting_info"),
            line_list=backup_info.get("line_list"),
            measure_list=backup_info.get("measure_list"),
        )