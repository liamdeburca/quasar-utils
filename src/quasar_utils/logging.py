__all__ = [
    "DisableLogging",
    "FlagFilter",
    "FlagFormatter",
    "QuasarFilter",
    "QuasarFormatter",
    "flag",
]

import sys
from collections.abc import Iterable
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    Filter,
    Formatter,
    LogRecord,
    getLogger,
)
from typing import Any

from quasar_typing.logging import FileHandler_

FMT: str = "{asctime} | {levelname:<8} | {name:<40}::{funcName:<30} | {message}"
FLAG_FMT: str = "{title}::{value}"
DATEFMT: str = "%Y-%m-%d %H:%M:%S"
STYLE: str = "{"

CONFIG_KWARGS: dict = {
    "format": FMT,
    "datefmt": None,
    "style": STYLE,
    "level": DEBUG,
    "force": True,
}
COLORS: dict[str, str] = {
    'grey': "\x1b[38;21m",
    'blue': "\x1b[38;5;39m",
    'yellow': "\x1b[38;5;226m",
    'red': "\x1b[38;5;196m",
    'bold_red': "\x1b[31;1m",
    'reset': "\x1b[0m",
}
LEVEL_TO_COLOR: dict[int, str] = {
    DEBUG: COLORS["grey"],
    INFO: COLORS["blue"],
    WARNING: COLORS["yellow"],
    ERROR: COLORS["red"],
    CRITICAL: COLORS["bold_red"],
}

FLAG_LOGGER_NAME: str = "_flag_logger"
flag_logger = getLogger(FLAG_LOGGER_NAME)

root_logger = getLogger()

### Standard

class QuasarFilter(Filter):
    """
    Standard filter that allows non-flag records from the 'quasar' namespace.
    """
    def __init__(self) -> None:
        super().__init__("quasar")

    def filter(self, record: LogRecord) -> bool:
        return record.name.startswith(self.name) \
            and record.name != FLAG_LOGGER_NAME
    

class QuasarFormatter(Formatter):
    """Custom formatter that optionally adds color to messages.

    Color output is enabled automatically when stderr is a TTY unless
    overridden via the ``use_colors`` argument. Color codes are never added
    when ``use_colors`` is False (suitable for file handlers).
    """

    def __init__(
        self,
        fmt: str | None = FMT,
        datefmt: str | None = DATEFMT,
        style: str = STYLE,
        use_colors: bool | None = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        if use_colors is None:
            self.use_colors = sys.stderr.isatty()
        else:
            self.use_colors = use_colors

    def format(self, record: LogRecord) -> str:
        msg = super().format(record)
        if self.use_colors:
            col = LEVEL_TO_COLOR.get(record.levelno, COLORS["reset"])
            return f"{col}{msg}{COLORS['reset']}"
        return msg

### Flagging

def flag(title: str, value: Any) -> None:
    """
    Log a flag message with the given title and value.
    """
    flag_logger.info(title, value)


class FlagFilter(Filter):
    """
    Logging filter class that only allows flags.
    """
    def __init__(self) -> None:
        super().__init__(FLAG_LOGGER_NAME)

    def filter(self, record: LogRecord) -> bool:
        return record.name == FLAG_LOGGER_NAME


class FlagFormatter(Formatter):
    """
    Custom formatter for flags.
    """
    def __init__(
        self,
        datefmt: str | None = None,
        style: str = STYLE,
    ) -> None:
        super().__init__(fmt=FLAG_FMT, datefmt=datefmt, style=style)
    
    def format(self, record: LogRecord) -> str:
        if (n_args := len(record.args)) != 1:
            raise ValueError(
                "Must provide exactly one argument when logging a flag, "
                f"got {n_args}"
            )
        return self._fmt.format(title=record.msg, value=record.args[0])

### Utils    

class DisableLogging:
    """
    Context manager for temporarily disabling the root logger by setting all
    levels to 'logging.CRITICAL + 1'.
    """

    def __init__(self, handlers: Iterable[FileHandler_]):
        self._levels = {handler: handler.level for handler in handlers}

    def __enter__(self):
        root_logger.info("Disabling logging temporarily...")
        for handler in self._levels:
            handler.setLevel(CRITICAL + 1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handler, level in self._levels.items():
            handler.setLevel(level)
        root_logger.info("Logging re-enabled.")
