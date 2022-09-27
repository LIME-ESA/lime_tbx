"""
This module contains the functionality that access to local coefficients data and other.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import Dict
import pkgutil
import csv
from io import StringIO
import os

"""___Third-Party Modules___"""
import xarray
import obsarray
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    ApolloIrradianceCoefficients,
    PolarizationCoefficients,
    ReflectanceCoefficients,
)
from lime_tbx.datatypes.templates_digital_effects_table import TEMPLATE_CIMEL

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
_POLARIZATION_WLENS = [440, 500, 675, 870, 1020, 1640]
# POLAR COEFFS MIGHT BE WRONG
_DEFAULT_POS_POLAR_COEFFS = [
    (0.003008799098, 0.000177889155, 0.000002581092, 0.000000012553),
    (0.002782607290, 0.000161111675, 0.000002331213, 0.000000011175),
    (0.002467126521, 0.000140139814, 0.000002021823, 0.000000009468),
    (0.002536989960, 0.000150448307, 0.000002233876, 0.000000010661),
    (0.002481149030, 0.000149814043, 0.000002238987, 0.000000010764),
    (0.002135380897, 0.000126059235, 0.000001888331, 0.000000009098),
]
_DEFAULT_NEG_POLAR_COEFFS = [
    (-0.003328093061, 0.000221328429, -0.000003441781, 0.000000018163),
    (-0.002881735316, 0.000186855017, -0.000002860010, 0.000000014778),
    (-0.002659373268, 0.000170314209, -0.000002652223, 0.000000013710),
    (-0.002521475080, 0.000157719602, -0.000002452656, 0.000000012597),
    (-0.002546369943, 0.000158157867, -0.000002469036, 0.000000012675),
    (-0.002726077195, 0.000171190004, -0.000002850707, 0.000000015473),
]


def _get_default_irradiance_coefficients() -> ApolloIrradianceCoefficients:
    data = _get_coefficients_data()
    wlens = list(data.keys())
    w_coeffs = list(data.values())
    coeffs = ApolloIrradianceCoefficients(
        wlens, w_coeffs, _DEFAULT_C_COEFFS, _DEFAULT_P_COEFFS, _DEFAULT_APOLLO_COEFFS
    )
    return coeffs


def _get_default_polarization_coefficients() -> PolarizationCoefficients:
    wlens = _POLARIZATION_WLENS
    pos_coeffs = _DEFAULT_POS_POLAR_COEFFS
    neg_coeffs = _DEFAULT_NEG_POLAR_COEFFS
    coeffs = PolarizationCoefficients(wlens, pos_coeffs, neg_coeffs)
    return coeffs


def _get_coefficients_data() -> Dict[
    float, ApolloIrradianceCoefficients.CoefficientsWln
]:
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
        data[float(row[0])] = ApolloIrradianceCoefficients.CoefficientsWln(coeffs)
    file.close()
    return data


def get_default_cimel_coeffs() -> ReflectanceCoefficients:
    # define dim_size_dict to specify size of arrays
    dim_sizes = {
        "wavelength": 6,
        "i_coeff": 18,
    }
    # create dataset
    ds_cimel: xarray.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)

    # TODO FIX THE EXTRAPOLATION
    ds_cimel = ds_cimel.assign_coords(wavelength=[440, 500, 675, 870, 1020, 1640])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(
        os.path.join(current_dir, "assets/coefficients_cimel.csv"), delimiter=","
    )
    u_data = np.genfromtxt(
        os.path.join(current_dir, "assets/u_coefficients_cimel.csv"), delimiter=","
    )

    ds_cimel.coeff.values = data.T
    ds_cimel.u_coeff.values = u_data.T

    return ReflectanceCoefficients(ds_cimel, "vDemo")
