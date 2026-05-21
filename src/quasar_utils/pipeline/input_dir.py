from typing import ClassVar, Iterator, Self
from itertools import batched
from dataclasses import field
from pydantic.dataclasses import dataclass
from quasar_typing.pathlib import AbsoluteFilePath, AbsoluteDirPath

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({'.asc', '.fits'})

@dataclass
class InputDir:
    """
    Data input directory: 

    Either a path to a file containing a spectrum to be fitted OR a path to a
    directory containing one or more spectra to be fitted. Currently, only 
    ASCII and FITS files are supported.
    """
    path: AbsoluteFilePath | AbsoluteDirPath
    
    directory: AbsoluteDirPath | None = None
    files: list[AbsoluteFilePath] = field(default_factory=list)
    
    allowed_extensions: ClassVar[frozenset[str]] = ALLOWED_EXTENSIONS

    def __post_init__(self) -> None:
        if self.files:
            self.directory = self.directory or self.files[0].parent
        else:
            if self.path.is_file() and self.is_valid_extension(self.path):
                self.directory = self.path.parent
                self.files.append(self.path)
            else:
                self.directory = self.path
                self.files.extend(filter(
                    self.is_valid_extension, 
                    self.directory.iterdir()),
                )
                if not self.files:
                    msg = f"InputDir was not able to find any valid data " \
                        f"files in {self.directory}!"
                    raise ValueError(msg)
            
    def __getstate__(self) -> dict:
        return {
            'path': self.path,
            'directory': self.directory,
            'files': self.files,
        }
    
    def __setstate__(self, state: dict) -> None:
        self.__init__(
            state['path'],
            directory=state['directory'],
            files=state['files'],
        )

    @classmethod
    def is_valid_extension(cls, path: AbsoluteFilePath) -> bool:
        return path.is_file() and (path.suffix in cls.allowed_extensions)

    def __iter__(self) -> Iterator[AbsoluteFilePath]:
        return iter(sorted(self.files, key=lambda path: path.name))

    def __len__(self) -> int:
        return len(self.files)
    
    def batched(self, batch_size: int) -> Iterator[Self]:
        for files_batch in batched(self, batch_size):
            yield InputDir(
                self.path,
                directory=self.directory,
                files=list(files_batch),
            )