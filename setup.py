from Cython.Build import cythonize
from numpy import get_include
from setuptools import find_packages, setup

setup(
    name="quasar_utils",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        [
            "src/quasar_utils/binning/alpha_matrix_elements.pyx",
            "src/quasar_utils/interpolation/interp_matrix_elements.pyx",
            "src/quasar_utils/masking/irregular.pyx",
            "src/quasar_utils/masking/linear.pyx",
            "src/quasar_utils/masking/logarithmic.pyx",
        ],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    include_dirs=[get_include()],
)
