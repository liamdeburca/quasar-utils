import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "linelist": ["LineList"],
}

if TYPE_CHECKING:
    from .linelist import LineList  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )