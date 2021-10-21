"""lime_tbx - The LIME TBX isa python package with a toolbox for using the LIME (Lunar Irradiance Model of ESA) model to simulate lunar observations and compare to remote sensing observations of the moon."""

__author__ = "Pieter De Vis <pieter.de.vis@npl.co.uk>"
__all__ = []

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
