from shutil import rmtree
from typing import Iterator
from logging import FileHandler
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import AnyAbsoluteDirPath, AbsoluteFilePath, AbsoluteLogPath

from .sub_dir import SubDir
from .input_dir import InputDir

@dataclass
class OutputDir:
    input_dir: InputDir

    path: AnyAbsoluteDirPath | None = None
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
            
    def __len__(self) -> int:
        return len(self.subdirs)

    def __iter__(self) -> Iterator[SubDir]:
        return iter(sorted(
            self.subdirs,
            key=lambda subdir: subdir._out_dir.name,
        ))
    
    def __getstate__(self) -> dict:
        return {
            'input_dir': self.input_dir,
            'path': self.path,
            'dangerous': False,
            'subdirs': self.subdirs,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(
            input_dir=state['input_dir'],
            path=state['path'],
            dangerous=False,
            subdirs=state['subdirs'],
        )
    @property
    def debug_logs(self) -> set[AbsoluteLogPath]:
        return {subdir.debug_log for subdir in self}
    
    @property
    def main_logs(self) -> set[AbsoluteLogPath]:
        return {subdir.main_log for subdir in self}
    
    @property
    def debug_handlers(self) -> list[FileHandler]:
        return [subdir.handlers['debug'] for subdir in self]
    
    @property
    def main_handlers(self) -> list[FileHandler]:
        return [subdir.handlers['main'] for subdir in self]
    
    @property
    def all_handlers(self) -> list[FileHandler]:
        return self.debug_handlers + self.main_handlers

    def create_subdir(
        self, 
        path: AbsoluteFilePath, 
        add: bool = True,
    ) -> SubDir:
        current_log: list[str] = [
            "Log for output file/dir initialisation:\n",
        ]
        out_dir = self.path / f"{path.name.split('.')[0]}_out"

        msg = f"Output directory @ {out_dir} "
        if out_dir.exists():
            is_empty = True
            for _ in out_dir.glob("*_out"):
                is_empty = False
                break

            if is_empty:
                msg += "exists but is empty -> using directory."
                rmtree(out_dir, ignore_errors=True)
            elif self.dangerous:
                msg += "exists and is not empty -> deleting contents."
                rmtree(out_dir, ignore_errors=True)
            else:
                msg += "exists and is not empty! "
                msg += "Use 'dangerous=True' to automatically delete contents."
                raise ValueError(msg)
        else:
            msg += "does not exist."
        
        current_log.append(msg)
        subdir = SubDir(path, out_dir, current_log=current_log)
        if add:
            self.subdirs.add(subdir)

        return subdir
    
    def create_subdirs(self) -> None:
        """
        Creates subdirectories for each input file.
        """
        for in_path in self.input_dir.files:
            _ = self.create_subdir(in_path, add=True)