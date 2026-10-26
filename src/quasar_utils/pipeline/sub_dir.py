__all__ = ["SubDir"]

from logging import DEBUG, INFO, FileHandler, getLogger
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, validate_call
from quasar_typing.logging import FileHandlerType
from quasar_typing.pathlib import (
    AnyCSVPath,
    AnyDirPath,
    AnyLogPath,
    CSVPath,
    DirPath,
    FilePath,
    LogPath,
)

from quasar_utils.logging import (
    CONFIG_KWARGS,
    FlagFilter,
    # FlagFormatter,
    QuasarFilter,
    QuasarFormatter,
)

logger = getLogger(__name__)
root = logger.root


class SubDir(BaseModel, validate_assignment=True):
    in_file: FilePath = Field(
        description="Path to the input file",
    )
    out_dir: AnyDirPath = Field(
        description="Path to the output directory."
    )

    handlers: dict[Literal["debug", "main", "flag"], FileHandlerType] = Field(
        default_factory=dict,
        exclude=True,
    )
    current_log: list[str] = Field(
        default_factory=list,
        exclude=True,
    )
    prev_handlers: list[FileHandlerType] = Field(
        default_factory=list,
        exclude=True,
    )
    prev_level: int | None = None


    @classmethod
    def _create(cls, in_file: FilePath, out_dir: AnyDirPath) -> dict[str, Any]:
        return {"in_file": in_file, "out_dir": out_dir}


    @classmethod
    @validate_call
    def create(cls, in_file: FilePath, out_dir: AnyDirPath) -> Self:
        return cls(**cls._create(in_file, out_dir))

    ###

    @property
    def _debug_log(self) -> AnyLogPath:
        return self.out_dir / "debug.log"

    
    @property
    def _main_log(self) -> AnyLogPath:
        return self.out_dir / "main.log"
    
    
    @property
    def _flags_log(self) -> AnyLogPath:
        return self.out_dir / "flags.log"
    
    
    @property
    def _profile(self) -> AnyCSVPath:
        return self.out_dir / "profile.csv"


    @property
    def _plots(self) -> AnyDirPath:
        return self.out_dir / "plots"

    @property
    def _line_results(self) -> AnyDirPath:
        return self.out_dir / "line_results"

    ###

    @property
    def debug_log(self) -> LogPath:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._debug_log.touch(exist_ok=True)
        return self._debug_log

    @property
    def main_log(self) -> LogPath:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._main_log.touch(exist_ok=True)
        return self._main_log
    
    
    @property
    def flags_log(self) -> LogPath:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._flags_log.touch(exist_ok=True)
        return self._flags_log
    
    
    @property
    def profile(self) -> CSVPath:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._profile.touch(exist_ok=True)
        return self._profile


    @property
    def plots(self) -> DirPath:
        self._plots.mkdir(parents=True, exist_ok=True)
        return self._plots

    @property
    def line_results(self) -> DirPath:
        self._line_results.mkdir(parents=True, exist_ok=True)
        return self._line_results

    ###

    def getOutDir(self) -> DirPath:
        """Get the output directory, creating it if necessary."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir

    ###

    def createHandlers(
        self,
        use_colors: bool = False,
    ) -> dict[Literal["debug", "main", "flag"], FileHandlerType]:
        """Create and return file handlers for this SubDir.

        This method will create the output directory (if not present) and
        instantiate the debug and main FileHandler_ objects, attach a
        non-colored CustomFormatter and the quasar_filter, then store them in
        self.handlers.
        """
        # Ensure output directory exists
        _ = self.getOutDir()

        h_debug = FileHandler(self.debug_log, mode="a", encoding="utf-8")
        h_debug.setLevel(DEBUG)
        h_debug.setFormatter(QuasarFormatter(use_colors=use_colors))
        h_debug.addFilter(QuasarFilter())

        h_main = FileHandler(self.main_log, mode="a", encoding="utf-8")
        h_main.setLevel(INFO)
        h_main.setFormatter(QuasarFormatter(use_colors=use_colors))
        h_main.addFilter(QuasarFilter())

        h_flags = FileHandler(self.flags_log, mode="a", encoding="utf-8")
        h_flags.setLevel(INFO)
        # h_flags.setFormatter(FlagFormatter())
        h_flags.addFilter(FlagFilter())

        handlers = {
            "debug": h_debug,
            "main": h_main,
            "flag": h_flags
        }

        # Close any existing handlers we own before replacing
        self.closeHandlers()
        self.handlers = handlers
        return self.handlers

    def closeHandlers(self) -> None:
        """Close and clear this SubDir's handlers.

        Safe to call multiple times.
        """
        for h in list(self.handlers.values()):
            try:
                h.flush()
                h.close()
            except (OSError, ValueError):
                pass
        self.handlers.clear()

    def __enter__(self) -> Self:
        """Context-manager entry: create and attach handlers for this SubDir.

        Handlers are created here so log files are only made when the SubDir is
        actually used in a with-block.
        """
        # create handlers for this subdir
        self.createHandlers()

        # Install our handlers on the root logger, saving previous handlers
        # so they can be restored on exit. We remove previous handlers from
        # the root without closing them so they can be re-attached.
        self.prev_handlers.clear()
        self.prev_handlers.extend(root.handlers)
        _ = [root.removeHandler(h) for h in self.prev_handlers]
        _ = [root.addHandler(h) for h in self.handlers.values()]

        # Apply level from config if present; save previous level to restore
        # on exit.
        self.prev_level = root.level
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
        self.closeHandlers()

        # Restore prior handlers (if we saved them during __enter__)
        if self.prev_handlers:
            [root.addHandler(h) for h in self.prev_handlers]
            self.prev_handlers.clear()

        # Restore previous logging level if we saved it
        if self.prev_level is not None:
            root.setLevel(self.prev_level)
            self.prev_level = None

    def __tuple__(self) -> tuple[FilePath, AnyDirPath]:
        return (self.in_file, self.out_dir)

    def __hash__(self) -> int:
        # Use the stored path attributes for hashing to avoid creating
        # filesystem side-effects (out_dir cached_property creates dirs).
        return hash(self.__tuple__())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SubDir) \
            and self.__tuple__() == other.__tuple__()