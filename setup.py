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
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    name="lime_tbx",
    url="https://gitlab.npl.co.uk/eco/eo/lime_tbx",
    license="None",
    author="Javier Gatón Herguedas",
    author_email="gaton@goa.uva.es",
    description="The LIME TBX isa python package with a toolbox for using the LIME (Lunar Irradiance Model of ESA) model to simulate lunar observations and compare to remote sensing observations of the moon.",
    long_description=read("README.rst"),
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "certifi==2022.6.15",
        "cftime==1.6.1",
        "comet-maths==0.19.4",
        "cycler==0.11.0",
        "Deprecated==1.2.13",
        "fonttools==4.37.1",
        "joblib==1.1.0",
        "kiwisolver==1.4.4",
        "matheo==0.1.0",
        "matplotlib==3.6.0",
        "mplcursors==0.5.2",
        "netCDF4==1.5.8",
        "numdifftools==0.9.39",
        "numpy==1.23.4",
        "obsarray==0.2.2",
        "packaging==21.3",
        "pandas==1.4.4",
        "Pillow==9.2.0",
        "punpy==0.39.5",
        "pyparsing==3.0.9",
        "PySide2==5.15.2.1",
        "python-dateutil==2.8.2",
        "pytz==2022.2.1",
        "PyYAML==6.0",
        "ruamel.yaml==0.17.21",
        "requests==2.28.1",
        "scikit-learn==1.1.2",
        "scipy==1.9.1",
        "shiboken2==5.15.2.1",
        "six==1.16.0",
        "sklearn==0.0",
        "spicedmoon==1.0.7",
        "spiceypy==5.1.2",
        "threadpoolctl==3.1.0",
        "wrapt==1.14.1",
        "xarray==2022.10.0",
    ],
    extras_require={"dev": ["pre-commit", "tox", "sphinx", "sphinx_rtd_theme"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
)
