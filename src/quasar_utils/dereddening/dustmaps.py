"""dereddening.dustmaps

Helpers for interacting with the `dustmaps` package and its cached data.

Supported maps:
 - SFD (Schlegel, Finkbeiner & Davis 1998)
 - CSFD (Chiang 2023)

This module exposes a small, validated public API for setup and lookup and
keeps the implementation details in private functions defined below. Low-level
helpers are placed at the end of the file.
"""
__all__ = ["PATH_TO_CACHE", "get_dust_map", "reset_cache", "setup_dustmaps"]

from functools import wraps
from shutil import rmtree
from typing import TYPE_CHECKING

from pydantic import validate_call
from quasar_typing.literals import MapName

from .download_data import PATH_TO_CACHE

### Public functions


@validate_call
def setup_dustmaps() -> None:
    """Configure the dustmaps cache directory and ensure maps are present.

    Public (validated) wrapper. See :func:`_setup_dustmaps` for the
    unvalidated implementation.
    """
    return _setup_dustmaps()


@validate_call
def reset_cache() -> None:
    """Remove cached dustmaps and re-run setup.

    Public (validated) wrapper for :func:`_reset_cache`.
    """
    return _reset_cache()


@validate_call
def get_dust_map(map_name: MapName):
    """Return a query object for a supported dust map.

    Parameters
    ----------
    map_name
        Case-insensitive map identifier, one of ``"sfd"`` or ``"csfd"``.

    Returns
    -------
    dustmaps.Query
        A dustmaps query object for the requested map.
    """
    return _get_dust_map(map_name)


### Private implementations


@wraps(setup_dustmaps)
def _setup_dustmaps() -> None:
    from dustmaps.config import config
    
    # Ensure cache dir exists and configure dustmaps to use it.
    if not PATH_TO_CACHE.exists():
        from .download_data import main
        main(force=True)

    config["data_dir"] = str(PATH_TO_CACHE)


@wraps(reset_cache)
def _reset_cache() -> None:
    # Remove everything under the cache directory then re-setup
    rmtree(PATH_TO_CACHE)
    _setup_dustmaps()


@wraps(get_dust_map)
def _get_dust_map(map_name: MapName):
    match map_name:
        case "sfd":
            from dustmaps.sfd import SFDQuery
            return SFDQuery()
        case "csfd":
            from dustmaps.csfd import CSFDQuery
            return CSFDQuery()


if not TYPE_CHECKING:
    _setup_dustmaps()
    
if __name__ == "__main__":
    _reset_cache()
