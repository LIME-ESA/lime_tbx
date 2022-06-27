"""
This module calculates the extra-terrestrial solar irradiance.

It exports the following functions:
    * get_esi_per_nms - Calculates the expected solar extra-terrestrial irradiance
    for a given wavelengths in nanometers. Based on Wehrli 1985 data, passed through
    some filters.
"""

"""___Built-In Modules___"""
import pkgutil
import csv
from io import StringIO
from typing import Dict, Tuple, Union, List

"""___Third-Party Modules___"""
import numpy as np

"""___LIME Modules___"""
# import here

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/03"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_WEHRLI_FILE = "assets/wehrli_asc.csv"

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


def get_esi_per_nms(wavelengths_nm: List[float]) -> np.ndarray:
    """Gets the expected extraterrestrial solar irradiance at some concrete wavelengths
    Returns the data in Wm⁻²/nm.

    It uses Wehrli 1985 data passed through different filters, the same data used in
    AEMET's RimoApp and others.

    Parameters
    ----------
    wavelengths_nm : list of float
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
