"""
Submodule containing utility functions and classes for running pipelines on
large datasets of (quasar) spectra.
"""

__all__ = [
    "InputDir",
    "OutputDir",
    "SubDir",
]
from .input_dir import InputDir
from .output_dir import OutputDir
from .sub_dir import SubDir
