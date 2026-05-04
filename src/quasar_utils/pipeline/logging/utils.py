"""
Utility functions for managing logs withing a pipeline.
"""
__all__ = [
    'instantiate_logging',
    'switch_filehandlers',
]

from logging import getLogger, FileHandler, basicConfig, DEBUG, INFO
from typing import Iterable

from .custom_formatter import CONFIG_KWARGS
from .filters import quasar_filter

logger = getLogger(__name__)

def instantiate_logging(
    output_dir: Iterable[object],
    config_kwargs: dict = CONFIG_KWARGS,
    reset_logs: bool = True,
) -> None:
    """
    Instantiates all DEBUG-level and INFO-level loggers for the 
    given output directory.

    All loggers are configured together allowing initial messages to be passed
    to all loggers at the same time. 
    """
    for subdir in output_dir:
        if 'debug' not in subdir.handlers or reset_logs:
            handler = FileHandler(subdir.debug_log, mode='a', encoding='utf-8')
            handler.setLevel(DEBUG)
            handler.addFilter(quasar_filter)
            subdir.handlers['debug'] = handler
        if 'main' not in subdir.handlers or reset_logs:
            handler = FileHandler(subdir.main_log, mode='a', encoding='utf-8')
            handler.setLevel(INFO)
            handler.addFilter(quasar_filter)
            subdir.handlers['main'] = handler

        basicConfig(handlers=subdir.handlers.values(), **config_kwargs)
        for line in subdir.current_log:
                logger.info(line)

def switch_filehandlers(
    handlers: Iterable[FileHandler],
    config_kwargs: dict = CONFIG_KWARGS,
) -> None:
    """
    Configures a new set of filehandlers. 
    Removes all existing handlers and adds the new ones.
    """
    basicConfig(handlers=handlers, **config_kwargs)