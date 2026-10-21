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
from pathlib import Path
from shutil import rmtree

from dustmaps.config import config
from dustmaps.csfd import CSFDQuery
from dustmaps.sfd import SFDQuery
from pydantic import validate_call
from quasar_typing.literals import MapName

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parent / "__cache__"

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
def get_dust_map(map_name: MapName) -> SFDQuery | CSFDQuery:
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
    # Ensure cache dir exists and configure dustmaps to use it.
    if not PATH_TO_CACHE.exists():
        PATH_TO_CACHE.mkdir()

    config["data_dir"] = str(PATH_TO_CACHE)

    # Ensure SFD is available
    if not (PATH_TO_CACHE / "sfd").exists():
        import dustmaps.sfd

        dustmaps.sfd.fetch()

    # Ensure CSFD is available
    if not (PATH_TO_CACHE / "csfd").exists():
        import dustmaps.csfd

        dustmaps.csfd.fetch()


@wraps(reset_cache)
def _reset_cache() -> None:
    # Remove everything under the cache directory then re-setup
    for path in PATH_TO_CACHE.iterdir():
        rmtree(path)

    _setup_dustmaps()


@wraps(get_dust_map)
def _get_dust_map(map_name: MapName) -> SFDQuery | CSFDQuery:
    match map_name:
        case "sfd":
            return SFDQuery()
        case "csfd":
            return CSFDQuery()


if __name__ == "__main__":
    _reset_cache()
