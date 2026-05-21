"""
Utility functions for managing logs withing a pipeline.
"""
__all__ = [
    'instantiate_logging',
    'switch_filehandlers',
    'switch_subdir',
    'DisableLogging',
]
from logging import getLogger, basicConfig, DEBUG, INFO, CRITICAL
from typing import Iterable
from tqdm import tqdm
from itertools import repeat

from quasar_typing.logging import FileHandler_
from quasar_typing.misc import Pool_

from .custom_formatter import CONFIG_KWARGS
from .filters import quasar_filter

logger = getLogger(__name__)

def get_handlers_for_subdir(args: tuple) -> dict[str, FileHandler_]:
    subdir, reset_logs = args
    handlers = {}
    if 'debug' not in subdir.handlers or reset_logs:
        handler = FileHandler_(subdir.debug_log, mode='a', encoding='utf-8')
        handler.setLevel(DEBUG)
        handler.addFilter(quasar_filter)
        handlers['debug'] = handler
        
    if 'main' not in subdir.handlers or reset_logs:
        handler = FileHandler_(subdir.main_log, mode='a', encoding='utf-8')
        handler.setLevel(INFO)
        handler.addFilter(quasar_filter)
        handlers['main'] = handler

    return handlers

def instantiate_logging(
    output_dir: Iterable[object],
    config_kwargs: dict = CONFIG_KWARGS,
    reset_logs: bool = True,
    tqdm_kwargs: dict = {
        'unit': 'subdir', 
        'disable': False, 
        'leave': False,
    },
    pool: Pool_ | None = None,
) -> None:
    """
    Instantiates all DEBUG-level and INFO-level loggers for the 
    given output directory.

    All loggers are configured together allowing initial messages to be passed
    to all loggers at the same time. 
    """
    args_iter = zip(
        output_dir, 
        repeat(reset_logs, len(output_dir)),
    )
    _map = map if pool is None else pool.imap
    loop = tqdm(
        _map(get_handlers_for_subdir, args_iter),
        total=len(output_dir), desc='Instantiating file handlers', 
        **tqdm_kwargs,
    )

    for handlers, subdir in zip(loop, output_dir):
        subdir.handlers.update(handlers)
    
    # basicConfig(handlers=output_dir.all_handlers, **config_kwargs)

def switch_filehandlers(
    handlers: Iterable[FileHandler_],
    config_kwargs: dict = CONFIG_KWARGS,
) -> None:
    """
    Configures a new set of filehandlers. 
    Removes all existing handlers and adds the new ones.
    """
    basicConfig(handlers=handlers, **config_kwargs)

def switch_subdir(
    subdir: object,
    config_kwargs: dict = CONFIG_KWARGS,
) -> None:
    """
    Switches logging to the given subdir. 
    """
    switch_filehandlers(subdir.handlers.values(), config_kwargs)
    for line in subdir.current_log:
        logger.info(line)
    subdir.current_log.clear()

class DisableLogging:
    """
    Context manager for temporarily disabling the root logger by setting all 
    levels to 'logging.CRITICAL + 1'.
    """
    def __init__(self, handlers: Iterable[FileHandler_]):
        self._levels = {handler: handler.level for handler in handlers}

    def __enter__(self):
        logger.info('Disabling logging temporarily...')
        for handler in self._levels:
            handler.setLevel(CRITICAL + 1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handler, level in self._levels.items():
            handler.setLevel(level)
        logger.info('Logging re-enabled.')