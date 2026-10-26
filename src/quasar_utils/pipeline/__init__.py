import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "input_dir": ["InputDir"],
    "output_dir": ["OutputDir"],
    "sub_dir": ["SubDir"],
}

if TYPE_CHECKING:
    from .input_dir import InputDir  # noqa: F401
    from .output_dir import OutputDir  # noqa: F401
    from .sub_dir import SubDir  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )
