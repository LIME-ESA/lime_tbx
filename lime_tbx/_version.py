"""
LIME Toolbox Version Handler Module

Handles versioning for the `lime_tbx` package.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lime_tbx")
except PackageNotFoundError:
    __version__ = "0.0.0"
