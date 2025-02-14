import io
import os
import re

from setuptools import find_packages
from setuptools import setup
import versioneer


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    version="1.1.0",
    cmdclass=versioneer.get_cmdclass(),
    name="lime_tbx",
    url="https://gitlab.npl.co.uk/eco/eo/lime_tbx",
    license="LGPL-3.0-only",
    author="Javier Gatón Herguedas, Pieter De Vis, Stefan Adriaensen, Jacob Fahy, Ramiro González Catón, Carlos Toledano, África Barreto, Agnieszka Bialek, Marc Bouvet",
    author_email="lime_tbx@goa.uva.es",
    maintainer="Javier Gatón Herguedas",
    maintainer_email="gaton@goa.uva.es",
    description="The LIME TBX is a Python package providing a comprehensive toolbox for utilizing the LIME (Lunar Irradiance Model of ESA) model to simulate lunar observations and compare them with remote sensing data of the Moon.",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "comet-maths~=1.0.3",
        "Deprecated~=1.2.18",
        "matheo~=0.1.2",
        "matplotlib~=3.10.0",
        "mplcursors~=0.6",
        "netCDF4~=1.6.5",
        "numpy~=1.26.4",
        "obsarray~=1.0.1",
        "packaging>=24.1",
        "pandas~=2.2.3",
        "punpy~=1.0.2",
        "PySide2~=6.8.2.1",
        "PyYAML~=6.0.2",
        "requests~=2.32",
        "ruamel.yaml==0.17.21",
        "scipy~=1.13.1",
        "setuptools>=75.8.0",
        "spicedmoon==1.0.13",
        "spiceypy~=6.0",
        "xarray>=2022.10.0",
        "xarray_schema>=0.0.3",
    ],
    extras_require={"dev": ["pre-commit", "tox", "sphinx", "sphinx_rtd_theme"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
