from typing import ClassVar, Iterator
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
    
    directory: AbsoluteDirPath = field(init=False)
    files: list[AbsoluteFilePath] = field(init=False)
    
    allowed_extensions: ClassVar[frozenset[str]] = ALLOWED_EXTENSIONS

    def __post_init__(self) -> None:
        if self.path.is_file() and self.is_valid_extension(self.path):
            self.directory = self.path.parent
            self.files = [self.path]
        else:
            self.directory = self.path
            self.files = list(filter(
                self.is_valid_extension, 
                self.directory.iterdir()),
            )
            if not self.files:
                msg = f"InputDir was not able to find any valid data files in \
                    {self.directory}!"
                raise ValueError(msg)
            
    def __getstate__(self) -> dict:
        return {'path': self.path}
    
    def __setstate__(self, state: dict) -> None:
        self.__init__(state['path'])

    @classmethod
    def is_valid_extension(cls, path: AbsoluteFilePath) -> bool:
        return path.is_file() and (path.suffix in cls.allowed_extensions)

    def __iter__(self) -> Iterator[AbsoluteFilePath]:
        yield from self.files

    def __len__(self) -> int:
        return len(self.files)