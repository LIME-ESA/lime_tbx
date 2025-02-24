"""
This module contains the functionality that manages the coefficient files
in the local system. Get all the available ones, get the currently
selected one or set a different one as the selected one.
"""

"""___Built-In Modules___"""
from typing import List
import os

"""___NPL Modules___"""
from lime_tbx.common.datatypes import (
    LimeCoefficients,
)
from lime_tbx.persistence.local_storage import programdata
from lime_tbx.application.filedata import coefficients


def get_all_coefficients() -> List[LimeCoefficients]:
    """
    Obtain all available LimeCoefficients

    Returns
    -------
    coeffs: list of LimeCoefficients
        All available LimeCoefficients
    """
    folder = os.path.join(
        programdata.get_programfiles_folder(), "coeff_data", "versions"
    )
    version_files = os.listdir(folder)
    coeffs = []
    for vf in version_files:
        cf = coefficients.read_coeff_nc(os.path.join(folder, vf))
        coeffs.append(cf)
    return sorted(coeffs, key=lambda x: x.version)


def get_previously_selected_version() -> str:
    """
    Get the version name of the previously selected version.

    Returns
    -------
    version_name: str
        Name of the previously selected version
    """
    file = os.path.join(
        programdata.get_programfiles_folder(), "coeff_data", "selected.txt"
    )
    if os.path.exists(file):
        name = None
        with open(file, "r") as fp:
            name = fp.readlines()[0].strip()
        return name
    return None


def set_previusly_selected_version(name: str):
    """
    Set the name of the previously/current selected version.

    Parameters
    ----------
    version_name: str
        Name of the previously selected version
    """
    file = os.path.join(
        programdata.get_programfiles_folder(), "coeff_data", "selected.txt"
    )
    with open(file, "w") as fp:
        fp.write(name)


def get_coefficients_filenames() -> List[str]:
    """Obtain the basenames of LIME TBX's existing coefficient files.

    Returns
    -------
    filenames: list of str
        Filenames (basenames) of LIME TBX's existing coefficient files.
    """
    folder = os.path.join(
        programdata.get_programfiles_folder(), "coeff_data", "versions"
    )
    return os.listdir(folder)
