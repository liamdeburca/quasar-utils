__all__ = ["SubDir"]

from typing import Literal
from dataclasses import field
from pathlib import Path
from logging import FileHandler
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
    
    current_log: list[str] = field(default_factory=list)
    handlers: dict[Literal['debug', 'main'], FileHandler_] = field(default_factory=dict)

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
        state = {
            'in_file': str(self.in_file),
            '_out_dir': str(self._out_dir),
            '_debug_log': str(self._debug_log),
            '_main_log': str(self._main_log),
            '_profile': str(self._profile),
            'current_log': self.current_log,
        }
        handlers = {}
        if 'debug' in self.handlers:
            fh = self.handlers['debug']
            handlers['debug'] = {
                'filename': fh.baseFilename,
                'mode': fh.mode,
                'encoding': fh.encoding,
                'level': fh.level,
            }
        if 'main' in self.handlers:
            fh = self.handlers['main']
            handlers['main'] = {
                'filename': fh.baseFilename,
                'mode': fh.mode,
                'encoding': fh.encoding,
                'level': fh.level,
            }
        state['handlers'] = handlers
        return state
    
    def __setstate__(self, state: dict) -> None:
        _handlers = state.get('handlers', {})
        handlers = {}
        if 'debug' in _handlers:
            debug_info = _handlers['debug']
            handlers['debug'] = fh = FileHandler(
                filename=debug_info['filename'],
                mode=debug_info['mode'],
                encoding=debug_info['encoding'],
            )
            fh.setLevel(debug_info['level'])
            fh.addFilter(quasar_filter)
        if 'main' in _handlers:
            main_info = _handlers['main']
            handlers['main'] = fh = FileHandler(
                filename=main_info['filename'],
                mode=main_info['mode'],
                encoding=main_info['encoding'],
            )
            fh.setLevel(main_info['level'])
            fh.addFilter(quasar_filter)

        self.__init__(
            Path(state['in_file']),
            Path(state['_out_dir']),
            _debug_log=Path(state['_debug_log']),
            _main_log=Path(state['_main_log']),
            _profile=Path(state['_profile']),
            current_log=state['current_log'],
            handlers=handlers,
        )

    @cached_property
    def out_dir(self) -> AbsoluteDirPath:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        return self._out_dir
    
    @cached_property
    def debug_log(self) -> AbsoluteLogPath:
        _ = self.out_dir
        self._debug_log.touch(exist_ok=True)
        with self._debug_log.open('w') as f:
            f.writelines(self.current_log)
        return self._debug_log
    
    @cached_property
    def main_log(self) -> AbsoluteLogPath:
        _ = self.out_dir
        self._main_log.touch(exist_ok=True)
        with self._main_log.open('w') as f:
            f.writelines(self.current_log)
        return self._main_log
    
    @cached_property
    def profile(self) -> AbsoluteCSVPath:
        _ = self.out_dir
        self._profile.touch(exist_ok=True)
        return self._profile

    @cached_property
    def plots(self) -> AbsoluteDirPath:
        path = self.out_dir / "plots"
        if not path.exists():
            path.mkdir()
        return path
    
    @cached_property
    def line_results(self) -> AbsoluteDirPath:
        path = self.out_dir / "line_results"
        if not path.exists():
            path.mkdir()
        return path