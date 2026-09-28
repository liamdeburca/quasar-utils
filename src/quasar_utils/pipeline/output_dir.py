from collections.abc import Iterator
from dataclasses import field
from logging import FileHandler
from os import symlink
from shutil import rmtree
from typing import Self

from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import (
    AbsoluteDirPath,
    AbsoluteFilePath,
    AbsoluteLogPath,
    AnyAbsoluteDirPath,
    AnyAbsoluteLogPath,
)

from .input_dir import InputDir
from .sub_dir import SubDir


@dataclass
class OutputDir:
    input_dir: InputDir

    path: AnyAbsoluteDirPath | None = None
    input_link: AbsoluteFilePath | AbsoluteDirPath | None = None
    dangerous: bool = False

    subdirs: set[SubDir] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = self.input_dir.directory
        else:
            self.path.mkdir(parents=True, exist_ok=True)

        if not self.subdirs:
            self.create_subdirs()

        self.dangerous = False

        # Create symbolic link to input file/directory
        self.input_link = self.path / "input"
        if not self.input_link.exists():
            symlink(
                self.input_dir.path, 
                self.input_link,
                target_is_directory=self.input_dir.path.is_dir(),
            )

    @classmethod
    def resume(cls, path: AbsoluteDirPath) -> Self:
        """
        Re-instantiate an OutputDir object from an existing output directory 
        path.
        """
        input_link = path / "input"
        return OutputDir(
            InputDir(input_link.resolve()),
            path=path,
            input_link=input_link,
            dangerous=False,
        )

    def batched(
        self,
        batch_size: int,
        *,
        input_dir: InputDir | None = None,
        path: AnyAbsoluteDirPath | None = None,
    ) -> Iterator[Self]:
        if path is None:
            path = self.path
        else:
            path.mkdir(parents=True, exist_ok=True)

        if input_dir is None:
            input_dir = self.input_dir

        for input_dir_batch in input_dir.batched(batch_size):
            yield OutputDir(
                input_dir=input_dir_batch,
                path=path,
                dangerous=False,
            )

    ###

    def __len__(self) -> int:
        return len(self.subdirs)

    def __iter__(self) -> Iterator[SubDir]:
        yield from sorted(self.subdirs, key=lambda subdir: subdir._out_dir.stem)

    def __getstate__(self) -> dict:
        return {
            "input_dir": self.input_dir,
            "path": self.path,
            "input_link": self.input_link,
            "dangerous": False,
            "subdirs": self.subdirs,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(
            input_dir=state["input_dir"],
            path=state["path"],
            input_link=state["input_link"],
            dangerous=False,
            subdirs=state["subdirs"],
        )

    @property
    def _debug_logs(self) -> set[AnyAbsoluteLogPath]:
        """
        Returns the set of debug log paths for all subdirectories.
        """
        return {subdir._debug_log for subdir in self}

    @property
    def debug_logs(self) -> set[AbsoluteLogPath]:
        """
        Returns the set of (existing) debug log paths for all subdirectories.
        """
        return {subdir.debug_log for subdir in self}

    @property
    def _main_logs(self) -> set[AnyAbsoluteLogPath]:
        """
        Returns the set of main log paths for all subdirectories.
        """
        return {subdir._main_log for subdir in self}

    @property
    def main_logs(self) -> set[AbsoluteLogPath]:
        """
        Returns the set of (existing) main log paths for all subdirectories.
        """
        return {subdir.main_log for subdir in self}

    @property
    def debug_handlers(self) -> list[FileHandler]:
        return [subdir.handlers.get("debug") for subdir in self]

    @property
    def main_handlers(self) -> list[FileHandler]:
        return [subdir.handlers["main"] for subdir in self]

    @property
    def all_handlers(self) -> list[FileHandler]:
        return self.debug_handlers + self.main_handlers

    def create_subdir(
        self,
        path: AbsoluteFilePath,
    ) -> SubDir:
        """
        Create a 'SubDir' object for the given input file path.

        If the subdirectory already exists and is not empty, it can be deleted 
        by setting `dangerous` to True.

        Parameters
        ----------
        path : AbsoluteFilePath
            The input file path for which to create the subdirectory.

        Returns
        -------
        SubDir
            The created 'SubDir' object.
        """
        out_dir = self.path / f"{path.stem}_out"

        msg = f"Output directory @ {out_dir!s} "
        if not out_dir.exists():
            msg += "does not exist."
        else:
            is_empty = next(out_dir.iterdir(), None) is None
            if is_empty:
                msg += "exists and empty -> using directory."
            elif self.dangerous:
                msg += "exists and is not empty -> deleting contents."
                rmtree(out_dir, ignore_errors=True)
            else:
                msg += "exists and NOT empty! "
                msg += "Use 'dangerous=True' to automatically delete contents."

        subdir = SubDir(path, out_dir)
        subdir.current_log.append(msg)
        return subdir

    def create_subdirs(self) -> None:
        """
        Creates subdirectories for each input file.
        """
        for in_path in self.input_dir.files:
            self.subdirs.add(self.create_subdir(in_path))