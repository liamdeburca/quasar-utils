import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "dereddening": ["deredden_spectrum", "get_correction"],
    "dustmaps": ["get_dust_map", "reset_cache", "setup_dustmaps"],
    "dust_extinction": ["get_dust_law"],
    "download_data": ["PATH_TO_CACHE"]
}

if TYPE_CHECKING:
    from .dereddening import deredden_spectrum, get_correction  # noqa: F401
    from .download_data import PATH_TO_CACHE  # noqa: F401
    from .dust_extinction import get_dust_law  # noqa: F401
    from .dustmaps import (  # noqa: F401
        get_dust_map,
        reset_cache,
        setup_dustmaps,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )