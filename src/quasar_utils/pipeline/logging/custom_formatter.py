"""
Custom formatter for 'logging'.
"""
__all__ = [
    "CustomFormatter",
    "CONFIG_KWARGS",
]

from logging import Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL

FMT: str = "{asctime} | {levelname:<8} | {name:<40}::{funcName:<30} | {message}"
DATEFMT: str = "%Y-%m-%d %H:%M:%S"
STYLE: str = '{'

CONFIG_KWARGS: dict = {
    'format': FMT,
    'datefmt': None,
    'style': STYLE,
    'level': DEBUG,
    'force': True,
}
COLORS: dict[str, str] = dict(
    grey = '\x1b[38;21m',
    blue = '\x1b[38;5;39m',
    yellow = '\x1b[38;5;226m',
    red = '\x1b[38;5;196m',
    bold_red = '\x1b[31;1m',
    reset = '\x1b[0m',
)
LEVEL_TO_COLOR: dict[int, str] = {
    DEBUG: COLORS['grey'],
    INFO: COLORS['blue'],
    WARNING: COLORS['yellow'],
    ERROR: COLORS['red'],
    CRITICAL: COLORS['bold_red'],
}

class CustomFormatter(Formatter):
    """
    Custom formatter for 'logging' module that adds color to log messages based
    on their level.

    Parameters
    ----------
    fmt : str | None, optional
        The log message format string. If None, the default format will be used.
    datefmt : str | None, optional
        The date format string. If None, the default date format will be used.
    style : str, optional
        The style of the format string. Default is '{'.
    validate : bool, optional
        Whether to validate the format string. Default is True.
    defaults : Mapping[str, Any] | None, optional
        A mapping of default values for the format string. If None, no defaults 
        will be used.
    """
    # def __init__(
    #     self, 
    #     fmt: str | None = FMT,
    #     datefmt: str | None = DATEFMT,
    #     style: str = STYLE,
    #     validate: bool = True,
    #     *,
    #     defaults: Mapping[str, Any] | None = None   
    # ):
    #     """
    #     Parameters
    #     ----------
    #     fmt : str | None, optional
    #         The log message format string. If None, the default format will be 
    #         used.
    #     datefmt : str | None, optional
    #         The date format string. If None, the default date format will be 
    #         used.
    #     style : str, optional
    #         The style of the format string. Default is '{'.
    #     validate : bool, optional
    #         Whether to validate the format string. Default is True.
    #     defaults : Mapping[str, Any] | None, optional
    #         A mapping of default values for the format string. If None, no 
    #         defaults will be used.
    #     """
    #     super().__init__(
    #         fmt=fmt, 
    #         datefmt=datefmt, 
    #         style=style, 
    #         validate=validate, 
    #         defaults=defaults,
    #     )
    #     self._formatter_kwargs = {
    #         'fmt': fmt,
    #         'datefmt': datefmt,
    #         'style': style,
    #         'validate': validate,
    #         'defaults': defaults,
    #     }

    # def format(self, record: LogRecord) -> str:
    #     col = LEVEL_TO_COLOR.get(record.levelno, COLORS["reset"])
    #     log_fmt = col + self._fmt + COLORS['reset']
    #     kwargs = self._formatter_kwargs | {'fmt': log_fmt}
    #     return Formatter(**kwargs).format(record)