__all__ = ["SubDir"]

from typing import Literal
from dataclasses import field
from pathlib import Path
from functools import cached_property
from pydantic.dataclasses import dataclass

from quasar_typing.pathlib import (
    AbsoluteFilePath,
    AnyAbsoluteDirPath, AbsoluteDirPath,
    AnyAbsoluteLogPath, AbsoluteLogPath,
    AnyAbsoluteCSVPath, AbsoluteCSVPath,
)
from quasar_typing.logging import FileHandler_

from .logging.filters import quasar_filter

@dataclass(eq=False)
class SubDir:
    in_file: AbsoluteFilePath
    _out_dir: AnyAbsoluteDirPath

    _debug_log: AnyAbsoluteLogPath | None = None
    _main_log: AnyAbsoluteLogPath | None = None
    _profile: AnyAbsoluteCSVPath | None = None
    
    handlers: dict[Literal['debug', 'main'], FileHandler_] = field(default_factory=dict)
    current_log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self._debug_log is None:
            self._debug_log = self._out_dir / "debug.log"
        if self._main_log is None:
            self._main_log = self._out_dir / "main.log"
        if self._profile is None:
            self._profile = self._out_dir / "profile.csv"
    
    def __hash__(self) -> int:
        return hash((self.in_file, self.out_dir))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubDir):
            return NotImplemented
        return (self.in_file, self.out_dir) == (other.in_file, other.out_dir)
    
    def __getstate__(self) -> dict:
        return {
            'in_file': str(self.in_file),
            '_out_dir': str(self.out_dir),
            '_debug_log': str(self.debug_log),
            '_main_log': str(self.main_log),
            '_profile': str(self.profile),
            'handlers': self.handlers,
            'current_log': self.current_log,
        }
    
    def __setstate__(self, state: dict) -> None:
        for handler in state['handlers'].values():
            handler.addFilter(quasar_filter)

        self.__init__(
            Path(state['in_file']),
            Path(state['_out_dir']),
            _debug_log=Path(state['_debug_log']),
            _main_log=Path(state['_main_log']),
            _profile=Path(state['_profile']),
            handlers=state['handlers'],
            current_log=state['current_log'],
        )

    @cached_property
    def out_dir(self) -> AbsoluteDirPath:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        return self._out_dir
    
    @cached_property
    def debug_log(self) -> AbsoluteLogPath:
        _ = self.out_dir
        self._debug_log.touch(exist_ok=True)
        return self._debug_log
    
    @cached_property
    def main_log(self) -> AbsoluteLogPath:
        _ = self.out_dir
        self._main_log.touch(exist_ok=True)
        return self._main_log
    
    @cached_property
    def profile(self) -> AbsoluteCSVPath:
        _ = self.out_dir
        self._profile.touch(exist_ok=True)
        return self._profile

    @cached_property
    def plots(self) -> AbsoluteDirPath:
        path = self.out_dir / "plots"
        path.mkdir(exist_ok=True)
        return path
    
    @cached_property
    def line_results(self) -> AbsoluteDirPath:
        path = self.out_dir / "line_results"
        path.mkdir(exist_ok=True)
        return path