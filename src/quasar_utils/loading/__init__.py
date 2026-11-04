import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "paqs_loader": ["PAQSLoader"],
    "sdss_loader": ["SDSSLoader"],
    "metadata": ["SDSSMeta", "PAQSMeta"],
}

if TYPE_CHECKING:
    from .metadata import PAQSMeta, SDSSMeta  # noqa: F401
    from .paqs_loader import PAQSLoader  # noqa: F401
    from .sdss_loader import SDSSLoader  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )