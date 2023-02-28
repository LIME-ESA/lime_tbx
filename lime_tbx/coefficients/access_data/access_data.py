"""
This module contains the functionality that access to local coefficients data and other.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import List
import os

"""___Third-Party Modules___"""
import xarray
import obsarray
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    LimeCoefficients,
    PolarizationCoefficients,
    ReflectanceCoefficients,
)
from lime_tbx.datatypes.templates import TEMPLATE_CIMEL
from lime_tbx.local_storage import programdata
from lime_tbx.filedata import coefficients

"""___Authorship___"""
__author__ = "Pieter De Vis, Jacob Fahy, Javier Gatón Herguedas, Ramiro González Catón, Carlos Toledano"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class IAccessData(ABC):
    @abstractmethod
    def get_all_coefficients(self) -> List[LimeCoefficients]:
        pass

    @abstractmethod
    def get_previously_selected_version(self) -> str:
        pass

    @abstractmethod
    def set_previusly_selected_version(self, name: str):
        pass


class AccessData(IAccessData):
    def get_all_coefficients(self) -> List[LimeCoefficients]:
        folder = os.path.join(
            programdata.get_programfiles_folder(), "coeff_data", "versions"
        )
        version_files = os.listdir(folder)
        coeffs = []
        for vf in version_files:
            cf = coefficients.read_coeff_nc(os.path.join(folder, vf))
            coeffs.append(cf)
        return coeffs

    def get_previously_selected_version(self) -> str:
        file = os.path.join(
            programdata.get_programfiles_folder(), "coeff_data", "selected.txt"
        )
        if os.path.exists(file):
            name = None
            with open(file, "r") as fp:
                name = fp.readlines()[0].strip()
            return name
        return None

    def set_previusly_selected_version(self, name: str):
        file = os.path.join(
            programdata.get_programfiles_folder(), "coeff_data", "selected.txt"
        )
        with open(file, "w") as fp:
            fp.write(name)


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
_DEFAULT_UNCS = [
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
]


def _get_default_polarization_coefficients() -> PolarizationCoefficients:
    wlens = _POLARIZATION_WLENS
    pos_coeffs = _DEFAULT_POS_POLAR_COEFFS
    neg_coeffs = _DEFAULT_NEG_POLAR_COEFFS
    uncs = _DEFAULT_UNCS
    err_corr_size = len(uncs) * len(uncs[0])
    err_corr = np.zeros((err_corr_size, err_corr_size))
    np.fill_diagonal(err_corr, 1)
    coeffs = PolarizationCoefficients(
        wlens, pos_coeffs, uncs, err_corr, neg_coeffs, uncs, err_corr
    )
    return coeffs


def _read_cimel_coeffs_files(filepath: str, u_filepath: str) -> ReflectanceCoefficients:
    # TODO FIX THE EXTRAPOLATION ?

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # read dataset with error correlation info (the error correlations will not be updated)
    ds_cimel = xarray.open_dataset(
        os.path.join(current_dir, "assets/ds_cimel_coeff.nc")
    )

    # read in updates on cimel coeff and uncs
    data = np.genfromtxt(os.path.join(current_dir, filepath), delimiter=",")
    u_data = np.genfromtxt(os.path.join(current_dir, u_filepath), delimiter=",")

    ds_cimel.coeff.values = data.T
    ds_cimel.u_coeff.values = u_data.T
    ds_cimel.err_corr_coeff.values = np.zeros((18 * 6, 18 * 6))
    np.fill_diagonal(ds_cimel.err_corr_coeff.values, 1)

    return ReflectanceCoefficients(ds_cimel)


def _get_demo_cimel_coeffs() -> ReflectanceCoefficients:
    # Demo coefficients, used for testing only
    return _read_cimel_coeffs_files(
        "assets/coefficients_cimel.csv", "assets/u_coefficients_cimel.csv"
    )


def get_coefficients_filenames() -> List[str]:
    folder = os.path.join(
        programdata.get_programfiles_folder(), "coeff_data", "versions"
    )
    return os.listdir(folder)
