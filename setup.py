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
        "certifi>=2022.6.15",
        "cftime~=1.6.1",
        "comet-maths~=1.0.3",
        "cycler~=0.11.0",
        "Deprecated~=1.2.13",
        "fonttools~=4.37.1",
        "joblib~=1.4.2",
        "kiwisolver~=1.4.4",
        "matheo~=0.1.2",
        "matplotlib==3.6.0",
        "mplcursors==0.5.2",
        "netCDF4~=1.6.5",
        "numdifftools~=0.9.41",
        "numpy~=1.26.4",
        "obsarray~=1.0.1",
        "packaging>=24.1",
        "pandas~=2.2.3",
        "Pillow==9.2.0",
        "punpy~=1.0.2",
        "pyparsing==3.0.9",
        "PySide2==5.15.2.1",
        "python-dateutil==2.8.2",
        "pytz==2022.2.1",
        "PyYAML==6.0",
        "requests>=2.28.1",
        "ruamel.yaml==0.17.21",
        "scikit-learn~=1.6.1",
        "scipy~=1.13.1",
        "shiboken2==5.15.2.1",
        "six==1.16.0",
        "spicedmoon==1.0.13",
        "spiceypy~=5.1.2",
        "threadpoolctl==3.1.0",
        "wrapt==1.14.1",
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
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
