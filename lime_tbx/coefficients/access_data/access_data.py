"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import Dict
import pkgutil
import csv
from io import StringIO

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import IrradianceCoefficients

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class IAccessData(ABC):
    @abstractmethod
    def get_all_coefficients_irradiance() -> list:
        pass

    @abstractmethod
    def get_all_coefficients_polarization() -> list:
        pass


class AccessData(IAccessData):
    def get_all_coefficients_irradiance() -> list:
        pass

    def get_all_coefficients_polarization() -> list:
        pass


_DEFAULT_C_COEFFS = [0.00034115, -0.0013425, 0.00095906, 0.00066229]
_DEFAULT_P_COEFFS = [4.06054, 12.8802, -30.5858, 16.7498]
_DEFAULT_APOLLO_COEFFS = [
    1.0301,
    1.0970,
    0.9325,
    0.9466,
    1.0225,
    1.0157,
    1.0470,
    1.0084,
    1.0100,
    1.0148,
    0.9843,
    1.0134,
    0.9329,
    0.9849,
    0.9994,
    0.9957,
    1.0059,
    0.9618,
    0.9561,
    0.9796,
    0.9568,
    0.9873,
    1.0575,
    1.0108,
    0.9743,
    1.0386,
    1.0338,
    1.0577,
    1.0650,
    1.0815,
    0.8945,
    0.9689,
]


def _get_default_irradiance_coefficients() -> IrradianceCoefficients:
    data = _get_coefficients_data()
    wlens = list(data.keys())
    w_coeffs = list(data.values())
    coeffs = IrradianceCoefficients(
        wlens, w_coeffs, _DEFAULT_C_COEFFS, _DEFAULT_P_COEFFS, _DEFAULT_APOLLO_COEFFS
    )
    return coeffs


def _get_coefficients_data() -> Dict[float, IrradianceCoefficients.CoefficientsWln]:
    """Returns all variable coefficients (a, b and d) for all wavelengths

    Returns
    -------
    A dict that has the wavelengths as keys (float), and as values the _CoefficientsWln associated
    to the wavelength.
    """
    coeff_bytes = pkgutil.get_data(__name__, "assets/coefficients.csv")
    coeff_string = coeff_bytes.decode()
    file = StringIO(coeff_string)
    csvreader = csv.reader(file)
    next(csvreader)  # Discard the header
    data = {}
    for row in csvreader:
        coeffs = []
        for i in range(1, 11):
            coeffs.append(float(row[i]))
        data[float(row[0])] = IrradianceCoefficients.CoefficientsWln(coeffs)
    file.close()
    return data
