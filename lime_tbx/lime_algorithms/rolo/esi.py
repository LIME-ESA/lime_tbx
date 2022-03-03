from typing import Dict, Tuple
import pkgutil
from io import StringIO
import csv
import numpy as np

WEHRLI_FILE = "assets/wehrli_asc.csv"

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
    wehrli_bytes = pkgutil.get_data(__name__, WEHRLI_FILE)
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


def get_esi_per_nm(wavelength_nm: float) -> float:
    """Gets the expected extraterrestrial solar irradiance at a concrete wavelength
    Returns the data in Wm⁻²/nm

    Parameters
    ----------
    wavelength_nm : float
        Wavelength (in nanometers) of which the extraterrestrial solar irradiance will be
        obtained

    Returns
    -------
    float
        The expected extraterrestrial solar irradiance in Wm⁻²/nm
    """
    wehrli_data = _get_wehrli_data()
    wehrli_x = list(wehrli_data.keys())
    if wavelength_nm in wehrli_x:
        return wehrli_data[wavelength_nm][0]
    if wavelength_nm < wehrli_x[0]:
        return wehrli_data[wehrli_x[0]][0]
    if wavelength_nm > wehrli_x[-1]:
        return wehrli_data[wehrli_x[-1]][0]
    wehrli_y = list(map(lambda x: x[0], wehrli_data.values()))
    return np.interp(wavelength_nm, wehrli_x, wehrli_y)
