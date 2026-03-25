"""
Package for dereddening quasar spectra using the Python package: `dustmaps`

For simplicity, the following maps are supported:
- SFD (Schlegel, Finkbeiner & Davis 1998)
- CSFD (Chiang 2023)
"""
__all__ = [
    "PATH_TO_CACHE",
    "setup_sfd",
    "setup_csfd",
    "reset_cache",
    "DUST_MAPS",
]

import dustmaps
from dustmaps.sfd import SFDQuery
from dustmaps.csfd import CSFDQuery
from dustmaps.config import config
from pathlib import Path
from shutil import rmtree
from typing import Literal

_this_file: Path = Path(__file__).resolve()

PATH_TO_CACHE: Path = _this_file.parent / "__cache__"

# Set cache location
def setup_dustmaps() -> None:
    """
    Sets up the dustmaps package by configuring the cache location and 
    downloading the necessary dust maps.
    """
    if not PATH_TO_CACHE.exists(): 
        PATH_TO_CACHE.mkdir()

    config["data_dir"] = str(PATH_TO_CACHE)

    # SFD
    if not (PATH_TO_CACHE / "sfd").exists():
        import dustmaps.sfd
        dustmaps.sfd.fetch()

    # CSFD
    if not (PATH_TO_CACHE / "csfd").exists():
        import dustmaps.csfd
        dustmaps.csfd.fetch()

# Reset cache
def reset_cache() -> None:
    for path in PATH_TO_CACHE.iterdir():
        rmtree(path)

    setup_dustmaps()

def get_dust_map(
    map_name: Literal['sfd', 'csfd'],
) -> SFDQuery | CSFDQuery:
    match map_name.strip().lower():
        case 'sfd': return SFDQuery()
        case 'csfd': return CSFDQuery()
        case _: pass

    raise NotImplementedError(
        f"Dust map '{map_name}' is not supported. Supported maps are: \
            'sfd', 'csfd'.",
    )

if __name__ == "__main__":
    reset_cache()