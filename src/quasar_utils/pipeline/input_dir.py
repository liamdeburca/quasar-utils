from collections.abc import Iterator
from itertools import batched
from typing import Any, ClassVar, Self

from pydantic import BaseModel, Field, validate_call
from quasar_typing.pathlib import DirPath, FilePath

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".asc", ".fits"})


class InputDir(BaseModel, validate_assignment=True):
    """
    Data input directory:

    Either a path to a file containing a spectrum to be fitted OR a path to a
    directory containing one or more spectra to be fitted. Currently, only
    ASCII and FITS files are supported.
    """
    path: FilePath | DirPath
    directory: DirPath
    files: list[FilePath] = Field(default_factory=list)

    allowed_extensions: ClassVar[frozenset[str]] = ALLOWED_EXTENSIONS


    @classmethod
    def _create(
        cls,
        path: FilePath | DirPath,
        directory: DirPath | None = None,
        files: list[FilePath] | None = None,
    ) -> dict[str, Any]:
        if files:
            directory = directory or files[0].parent
            path = directory
        else:
            if path.is_file():
                if not cls.is_valid_extension(path):
                    raise ValueError(f"File has an invalid extension: {path}")
                
                directory = path.parent
                files = [path]
            else:
                directory = path
                files = list(filter(
                    cls.is_valid_extension, 
                    directory.iterdir(),
                ))
                if not files:
                    raise ValueError(
                        f"No valid data files found in directory: {directory}"
                    )

        return {
            "path": path,
            "directory": directory,
            "files": files or [],
        }


    @classmethod
    @validate_call
    def create(
        cls,
        path: FilePath | DirPath,
        directory: DirPath | None = None,
        files: list[FilePath] | None = None,
    ) -> Self:
        return cls(**cls._create(path, directory=directory, files=files))

    @classmethod
    def is_valid_extension(cls, path: FilePath) -> bool:
        return path.is_file() and (path.suffix in cls.allowed_extensions)

    def __iter__(self) -> Iterator[FilePath]:
        return iter(sorted(self.files, key=lambda path: path.name))

    def __len__(self) -> int:
        return len(self.files)
    
    def batched(self, batch_size: int) -> Iterator[Self]:
        for files_batch in batched(self, batch_size):
            yield InputDir(
                path=self.path,
                directory=self.directory,
                files=list(files_batch),
            )
