"""
This module calculates the extra-terrestrial solar irradiance.

It exports the following functions:
    * get_esi_per_nms - Calculates the expected solar extra-terrestrial irradiance
    for a given wavelengths in nanometers. Based on Wehrli 1985 data, passed through
    some filters.
    * get_esi - Calculates the expected solar extra-terrestrial irradiance
    for a specific SRF, based in the TSIS spectrum.
    * get_esi - Calculates the expected uncertainties of the calculation of the
    expected solar extra-terrestrial irradiance for a specific SRF, based in the TSIS spectrum.
"""

"""___Built-In Modules___"""
import pkgutil
import csv
from io import StringIO
import os
from typing import Dict, Tuple

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/03"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_WEHRLI_FILE = "assets/wehrli_asc.csv"
_WEHRLI_CIMEL_FILE = "assets/wehrli_cimel.csv"
_WEHRLI_ASD_FILE = "assets/wehrli_asd.csv"
_TSIS_CIMEL_FILE = "assets/tsis_cimel.csv"
_TSIS_ASD_FILE = "assets/tsis_asd.csv"
_TSIS_GAUSS_1_3_FILE = "assets/tsis_fwhm_3_1_gaussian.csv"
_TSIS_TRIANGULAR_1_1_FILE = "assets/tsis_fwhm_1_1_triangle.csv"

_loaded_data = {}


def _get_wehrli_data() -> Dict[float, Tuple[float, float]]:
    """Returns all wehrli data

    Returns
    -------
    A dict that has the wavelengths as keys (float), and as values it has tuples of the
    (Wm⁻²/nm, Wm⁻²) values.
    """
    global _loaded_data
    if _loaded_data:
        return _loaded_data
    wehrli_bytes = pkgutil.get_data(__name__, _WEHRLI_FILE)
    wehrli_string = wehrli_bytes.decode()
    file = StringIO(wehrli_string)
    csvreader = csv.reader(file)
    next(csvreader)  # Discard the header
    data = {}
    for row in csvreader:
        data[float(row[0])] = (float(row[1]), float(row[2]))
    file.close()
    _loaded_data = data
    return _loaded_data


def get_esi_per_nms(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Gets the expected extraterrestrial solar irradiance at some concrete wavelengths
    Returns the data in Wm⁻²/nm.

    It uses Wehrli 1985 data passed through different filters, the same data used in
    AEMET's RimoApp and others.

    Parameters
    ----------
    wavelengths_nm : np.array of float
        Wavelengths (in nanometers) of which the extraterrestrial solar irradiance will be
        obtained

    Returns
    -------
    np.ndarray of float
        The expected extraterrestrial solar irradiance in Wm⁻²/nm
    """
    wehrli_data = _get_wehrli_data()
    wehrli_x = list(wehrli_data.keys())
    wehrli_y = list(map(lambda x: x[0], wehrli_data.values()))
    return np.interp(wavelengths_nm, wehrli_x, wehrli_y)


def get_esi(srf_type: str) -> np.ndarray:
    """Gets the expected extraterrestrial solar irradiance of a concrete SRF.
    Returns the data in Wm⁻²/nm.

    It uses TSIS data.

    Parameters
    ----------
    srf_type : str
        Name of the srf. Can be 'cimel', 'asd', 'interpolated_gaussian' or 'interpolated_triangle'.
        It can also be 'cimel_wehrli' and 'asd_wehrli', although that option won't be available
        in the toolbox GUI.

    Returns
    -------
    np.ndarray of float
        The expected extraterrestrial solar irradiance in Wm⁻²/nm
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if srf_type == "cimel":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_CIMEL_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
    elif srf_type == "asd":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_ASD_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
    elif srf_type == "interpolated_gaussian":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_GAUSS_1_3_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
    elif srf_type == "interpolated_triangle":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_TRIANGULAR_1_1_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
    elif srf_type == "cimel_wehrli":
        return np.genfromtxt(
            os.path.join(dir_path, _WEHRLI_CIMEL_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
    elif srf_type == "asd_wehrli":
        return np.genfromtxt(
            os.path.join(dir_path, _WEHRLI_ASD_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"srf_type was {srf_type} and must be 'cimel', 'asd', 'interpolated_gaussian' or 'interpolated_triangle'"
        )


def get_u_esi(srf_type: str) -> np.ndarray:
    """Gets the expected extraterrestrial solar irradiance uncertainties of a concrete SRF.
    Returns the data in Wm⁻²/nm.

    It uses TSIS data.

    Parameters
    ----------
    srf_type : str
        Name of the srf. Can be 'cimel', 'asd', 'interpolated_gaussian' or 'interpolated_triangle'.
        It can also be 'cimel_wehrli' and 'asd_wehrli', although that option won't be available
        in the toolbox GUI.

    Returns
    -------
    np.ndarray of float
        The expected extraterrestrial solar irradiance in Wm⁻²/nm
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if srf_type == "cimel":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_CIMEL_FILE),
            delimiter=",",
            usecols=[0, 2],
            dtype=np.float32,
        )
    elif srf_type == "asd":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_ASD_FILE),
            delimiter=",",
            usecols=[0, 2],
            dtype=np.float32,
        )
    elif srf_type == "interpolated_gaussian":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_GAUSS_1_3_FILE),
            delimiter=",",
            usecols=[0, 2],
            dtype=np.float32,
        )
    elif srf_type == "interpolated_triangle":
        return np.genfromtxt(
            os.path.join(dir_path, _TSIS_GAUSS_1_3_FILE),
            delimiter=",",
            usecols=[0, 2],
            dtype=np.float32,
        )
    elif srf_type == "cimel_wehrli":
        vals = np.genfromtxt(
            os.path.join(dir_path, _WEHRLI_CIMEL_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
        vals[:, 1] = vals[:, 1] * 0
        return vals
    elif srf_type == "asd_wehrli":
        vals = np.genfromtxt(
            os.path.join(dir_path, _WEHRLI_ASD_FILE),
            delimiter=",",
            usecols=[0, 1],
            dtype=np.float32,
        )
        vals[:, 1] = vals[:, 1] * 0
        return vals
    else:
        raise ValueError(
            f"srf_type was {srf_type} and must be 'cimel', 'asd', 'interpolated_gaussian' or 'interpolated_triangle'"
        )
