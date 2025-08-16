from setuptools import setup, Extension
import pybind11
import sys
import os

ext = Extension(
    "dstar_lite",
    sources=[
        os.path.join("src", "DStarLite", "bindings.cpp"),
        os.path.join("src", "DStarLite", "DStarLite.cpp"),
    ],
    include_dirs=[pybind11.get_include(), os.path.join("src", "DStarLite")],
    language="c++",
    extra_compile_args=["/O2"] if sys.platform.startswith("win") else ["-O3"],
)

setup(
    name="dstar_lite",
    version="0.1.0",
    ext_modules=[ext],
)
