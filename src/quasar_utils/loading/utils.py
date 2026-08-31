__all__ = [
    "LoaderOutput",
]

from typing import TypedDict

from quasar_typing.numpy import FloatVector
from quasar_typing.pathlib import AbsoluteFilePath

from ..setup import Info


class LoaderOutput(TypedDict):
    path: str | AbsoluteFilePath
    title: str
    x: FloatVector
    y: FloatVector
    dy: FloatVector
    dx: FloatVector
    info: Info