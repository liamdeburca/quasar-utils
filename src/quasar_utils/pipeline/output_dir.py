import os
import shutil
from collections.abc import Iterator
from logging import FileHandler
from typing import Any, Self

from pydantic import BaseModel, validate_call
from quasar_typing.pathlib import (
    AnyDirPath,
    AnyLogPath,
    DirPath,
    FilePath,
    LogPath,
)

from .input_dir import InputDir
from .sub_dir import SubDir


class OutputDir(BaseModel, validate_assignment=True):
    input_dir: InputDir
    path: AnyDirPath
    input_link: FilePath | DirPath
    dangerous: bool = False
    subdirs: set[SubDir]

    @classmethod
    def _create(
        cls,
        input_dir: InputDir,
        path: AnyDirPath | None = None,
        dangerous: bool = False,
        subdirs: set[SubDir] | None = None,
    ) -> dict[str, Any]:
        if path is None:
            path = input_dir.directory
        else:
            path.mkdir(parents=True, exist_ok=True)

        if not subdirs:
            subdirs = {
                cls.createSubdir(in_file, path, dangerous=dangerous)
                for in_file in input_dir.files
            }

        input_link = path / "input"
        if not input_link.exists():
            os.symlink(
                input_dir.path, 
                input_link,
                target_is_directory=True,
            )

        return {
            "input_dir": input_dir,
            "path": path,
            "input_link": input_link,
            "dangerous": dangerous,
            "subdirs": subdirs,
        }


    @classmethod
    @validate_call
    def create(
        cls,
        input_dir: InputDir,
        path: AnyDirPath | None = None,
        dangerous: bool = False,
        subdirs: set[SubDir] | None = None,
    ) -> Self:
        return cls(**cls._create(
            input_dir,
            path=path,
            dangerous=dangerous,
            subdirs=subdirs,
        ))

    @classmethod
    def createSubdir(
        cls,
        in_file: FilePath,
        output_dir_path: DirPath,
        dangerous: bool = False,
    ) -> SubDir:
        out_dir = output_dir_path / f"{in_file.stem}_out"

        msg = f"Output directory @ {out_dir!s} "
        if not out_dir.exists():
            msg += "does not exist."
        else:
            is_empty = next(out_dir.iterdir(), None) is None
            if is_empty:
                msg += "exists and empty -> using directory."
            elif dangerous:
                msg += "exists and is not empty -> deleting contents."
                shutil.rmtree(out_dir, ignore_errors=True)
            else:
                msg += "exists and NOT empty! "
                msg += "Use 'dangerous=True' to automatically delete contents."

        return SubDir(
            in_file=in_file, 
            out_dir=out_dir, 
            current_log=[msg],
        )

        
    @classmethod
    def resume(cls, path: DirPath):
        """
        Re-instantiate an OutputDir object from an existing output directory 
        path.
        """
        input_link = path / "input"
        input_dir = InputDir.create(input_link.resolve())
        return cls.create(input_dir, path=path, dangerous=False)

    def batched(
        self,
        batch_size: int,
        *,
        input_dir: InputDir | None = None,
        path: AnyDirPath | None = None,
    ) -> Iterator[Self]:
        if path is None:
            path = self.path
        else:
            path.mkdir(parents=True, exist_ok=True)

        if input_dir is None:
            input_dir = self.input_dir

        yield from (
            OutputDir.create(input_dir_batch, path=path, dangerous=False)
            for input_dir_batch in input_dir.batched(batch_size)
        )

    ###

    def __len__(self) -> int:
        return len(self.subdirs)

    def __iter__(self) -> Iterator[SubDir]:
        yield from sorted(self.subdirs, key=lambda subdir: subdir.out_dir.stem)

    ###
    
    @property
    def _debug_logs(self) -> set[AnyLogPath]:
        return {subdir._debug_log for subdir in self}

    @property
    def _main_logs(self) -> set[AnyLogPath]:
        return {subdir._main_log for subdir in self}

    @property
    def _flags_logs(self) -> set[AnyLogPath]:
        return {subdir._flags_log for subdir in self}

    ###

    @property
    def debug_logs(self) -> set[LogPath]:
        return {subdir.debug_log for subdir in self}

    @property
    def main_logs(self) -> set[LogPath]:
        return {subdir.main_log for subdir in self}

    @property
    def flags_logs(self) -> set[LogPath]:
        return {subdir.flags_log for subdir in self}

    @property
    def debug_handlers(self) -> list[FileHandler]:
        return [subdir.handlers.get("debug") for subdir in self]

    @property
    def main_handlers(self) -> list[FileHandler]:
        return [subdir.handlers["main"] for subdir in self]

    @property
    def all_handlers(self) -> list[FileHandler]:
        return self.debug_handlers + self.main_handlers