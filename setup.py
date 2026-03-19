"""Disentangling Time-series Spectra with Gaussian Processes:
Applications to Radial Velocity Analysis of Spectroscopic Binaries (SB2).

https://github.com/piotrlojko/PSOAP-modernized
"""

from setuptools import setup, find_packages
from setuptools.extension import Extension
from codecs import open
from os import path

import numpy as np
from Cython.Build import cythonize

entry_points = {"console_scripts": [
    "psoap-initialize = psoap.initialize:main",
    "psoap-sample = psoap.sample:main",
    "psoap-sample-parallel = psoap.sample_parallel:main",
    "psoap-plot-samples = psoap.plot_samples:main",
]}

here = path.abspath(path.dirname(__file__))

extensions = [
    Extension(
        "psoap.matrix_functions",
        ["psoap/matrix_functions.pyx"],
        include_dirs=[np.get_include()],
    )
]

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="psoap",
    version="0.2.0",
    description="Gaussian Processes for spectroscopic binary disentangling (SB2/ST3)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/piotrlojko/PSOAP-modernized",
    author="Ian Czekala",
    author_email="iancze@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="science astronomy spectra spectroscopic binary gaussian process",
    python_requires=">=3.12",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "astropy>=6.0",
        "matplotlib>=3.8",
        "pyyaml>=6.0",
        "cython>=3.0",
    ],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["pytest", "coverage"],
        "sklearn": ["scikit-learn>=1.4"],
    },
    package_data={
        "psoap": [
            "data/*.yaml",
        ],
    },
    entry_points=entry_points,
    ext_modules=cythonize(extensions),
)
