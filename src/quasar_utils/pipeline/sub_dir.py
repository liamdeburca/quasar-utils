__all__ = ["SubDir"]

from dataclasses import field
from logging import DEBUG, INFO, getLogger
from typing import Literal, Self

from pydantic.dataclasses import dataclass
from quasar_typing.logging import FileHandler_
from quasar_typing.pathlib import (
    AbsoluteCSVPath,
    AbsoluteDirPath,
    AbsoluteFilePath,
    AbsoluteLogPath,
    AnyAbsoluteCSVPath,
    AnyAbsoluteDirPath,
    AnyAbsoluteLogPath,
)

from ..logging import (
    CONFIG_KWARGS,
    FlagFilter,
    FlagFormatter,
    QuasarFilter,
    QuasarFormatter,
)

logger = getLogger(__name__)
root = logger.root


@dataclass(eq=False)
class SubDir:
    in_file: AbsoluteFilePath
    _out_dir: AnyAbsoluteDirPath

    _debug_log: AnyAbsoluteLogPath | None = None
    _main_log: AnyAbsoluteLogPath | None = None
    _flags_log: AnyAbsoluteLogPath | None = None
    _profile: AnyAbsoluteCSVPath | None = None

    handlers: dict[Literal["debug", "main", "flag"], FileHandler_] = field(
        default_factory=dict,
    )
    current_log: list[str] = field(default_factory=list)

    _prev_handlers: list[FileHandler_] = field(default_factory=list)
    _prev_level: int | None = field(default=None)

    def __post_init__(self) -> None:
        if self._debug_log is None:
            self._debug_log = self._out_dir / "debug.log"
        if self._main_log is None:
            self._main_log = self._out_dir / "main.log"
        if self._flags_log is None:
            self._flags_log = self._out_dir / "flags.log"
        if self._profile is None:
            self._profile = self._out_dir / "profile.csv"

    def create_handlers(
        self,
        use_colors: bool = False,
    ) -> dict[Literal["debug", "main", "flag"], FileHandler_]:
        """Create and return file handlers for this SubDir.

        This method will create the output directory (if not present) and
        instantiate the debug and main FileHandler_ objects, attach a
        non-colored CustomFormatter and the quasar_filter, then store them in
        self.handlers.
        """
        # Ensure output directory exists
        _ = self.out_dir

        handlers: dict[Literal["debug", "main", "flag"], FileHandler_] = {}

        h_debug = FileHandler_(self._debug_log, mode="a", encoding="utf-8")
        h_debug.setLevel(DEBUG)
        h_debug.setFormatter(QuasarFormatter(use_colors=use_colors))
        h_debug.addFilter(QuasarFilter())
        handlers["debug"] = h_debug

        h_main = FileHandler_(self._main_log, mode="a", encoding="utf-8")
        h_main.setLevel(INFO)
        h_main.setFormatter(QuasarFormatter(use_colors=use_colors))
        h_main.addFilter(QuasarFilter())
        handlers["main"] = h_main

        h_flags = FileHandler_(self._flags_log, mode="a", encoding="utf-8")
        h_flags.setLevel(INFO)
        h_flags.setFormatter(FlagFormatter())
        h_flags.addFilter(FlagFilter())
        handlers["flag"] = h_flags

        # Close any existing handlers we own before replacing
        self.close_handlers()
        self.handlers.update(handlers)
        return handlers

    def close_handlers(self) -> None:
        """Close and clear this SubDir's handlers.

        Safe to call multiple times.
        """
        for h in list(self.handlers.values()):
            try:
                h.flush()
            except (OSError, ValueError):
                # Flush may fail if the underlying stream is invalid; ignore
                # these specific errors but allow others to surface.
                pass
            try:
                h.close()
            except (OSError, ValueError):
                # Closing can sometimes raise if the file descriptor is bad;
                # swallow these expected errors to ensure robust cleanup.
                pass
        self.handlers.clear()

    def __enter__(self) -> Self:
        """Context-manager entry: create and attach handlers for this SubDir.

        Handlers are created here so log files are only made when the SubDir is
        actually used in a with-block.
        """
        # create handlers for this subdir
        self.create_handlers()

        # Install our handlers on the root logger, saving previous handlers
        # so they can be restored on exit. We remove previous handlers from
        # the root without closing them so they can be re-attached.
        self._prev_handlers.clear()
        self._prev_handlers.extend(root.handlers)
        _ = [root.removeHandler(h) for h in self._prev_handlers]
        _ = [root.addHandler(h) for h in self.handlers.values()]

        # Apply level from config if present; save previous level to restore
        # on exit.
        self._prev_level = root.level
        level = CONFIG_KWARGS.get("level")
        if level is not None:
            root.setLevel(level)

        # Flush any queued start-up messages into the newly-installed handlers
        _ = [logger.info(line) for line in self.current_log]
        self.current_log.clear()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context-manager exit: restore previous root handlers and close
        this SubDir's handlers.

        We remove our handlers from the root logger, close them, and restore
        the previously-installed handlers (if any).
        """
        # Remove our handlers from the root logger
        _ = [root.removeHandler(h) for h in self.handlers.values()]

        # Close and clear our handlers
        self.close_handlers()

        # Restore prior handlers (if we saved them during __enter__)
        if self._prev_handlers:
            [root.addHandler(h) for h in self._prev_handlers]
            self._prev_handlers.clear()

        # Restore previous logging level if we saved it
        if self._prev_level is not None:
            root.setLevel(self._prev_level)
            self._prev_level = None

    def __tuple__(self) -> tuple[AbsoluteFilePath, AnyAbsoluteDirPath]:
        return (self.in_file, self._out_dir)

    def __hash__(self) -> int:
        # Use the stored path attributes for hashing to avoid creating
        # filesystem side-effects (out_dir cached_property creates dirs).
        return hash(self.__tuple__())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SubDir) \
            and self.__tuple__() == other.__tuple__()

    def __getstate__(self) -> dict:
        # Only serialize persistent metadata; runtime-only fields like
        # _prev_handlers/_prev_level are transient and should not be pickled.
        return {
            "in_file": str(self.in_file),
            "_out_dir": str(self._out_dir),
            "_debug_log": str(self._debug_log),
            "_main_log": str(self._main_log),
            "_flags_log": str(self._flags_log),
            "_profile": str(self._profile),
            "handlers": self.handlers,
            "current_log": self.current_log,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(
            state.pop("in_file"),
            state.pop("_out_dir"),
            _debug_log=state.pop("_debug_log", None),
            _main_log=state.pop("_main_log", None),
            _flags_log=state.pop("_flags_log", None),
            _profile=state.pop("_profile", None),
            handlers=state.pop("handlers", {}),
            current_log=state.pop("current_log", []),
        )

    @property
    def out_dir(self) -> AbsoluteDirPath:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        return self._out_dir

    @property
    def debug_log(self) -> AbsoluteLogPath:
        _ = self.out_dir
        self._debug_log.touch(exist_ok=True)
        return self._debug_log

    @property
    def main_log(self) -> AbsoluteLogPath:
        _ = self.out_dir
        self._main_log.touch(exist_ok=True)
        return self._main_log

    @property
    def flags_log(self) -> AbsoluteLogPath:
        _ = self.out_dir
        self._flags_log.touch(exist_ok=True)
        return self._flags_log

    @property
    def profile(self) -> AbsoluteCSVPath:
        _ = self.out_dir
        self._profile.touch(exist_ok=True)
        return self._profile

    @property
    def plots(self) -> AbsoluteDirPath:
        path = self.out_dir / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def line_results(self) -> AbsoluteDirPath:
        path = self.out_dir / "line_results"
        path.mkdir(parents=True, exist_ok=True)
        return path
