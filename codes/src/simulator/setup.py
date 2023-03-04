from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name            = "DM_compute",
    ext_modules     = cythonize("DM_compute.pyx", annotate=True),
    include_dirs    = [numpy.get_include()],
)

# from setuptools import Extension, setup
# from Cython.Build import cythonize

# extensions = [
# 	Extension(
#         "DM_compute", 
#         ["DM_compute.pyx"]
#     )
# ]

# setup(
#     name        = "DM_compute",
# 	ext_modules = cythonize([extensions]), 
#     annotate    = True,
#     include_dirs=[numpy.get_include()],
# )