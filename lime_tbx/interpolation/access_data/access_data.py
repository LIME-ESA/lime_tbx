"""describe class"""

"""___Built-In Modules___"""
import os

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    PolarizationCoefficients,
)
from lime_tbx.datatypes.datatypes import SpectralData
from lime_tbx.datatypes.templates_digital_effects_table import TEMPLATE_REFL

"""___Authorship___"""
__author__ = "Pieter De Vis, Jacob Fahy, Javier Gatón Herguedas, Ramiro González Catón, Carlos Toledano"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def get_asd_data(abs_phase_angle: float) -> SpectralData:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(
        os.path.join(current_dir, "assets/SomeMoonReflectances.txt"), delimiter=","
    )
    wavs = np.arange(350, 2501, 1)
    phase_angles = data[:, 3]
    best_id = np.argmin(np.abs(np.abs(phase_angles) - abs_phase_angle))
    refl = data[best_id, 4:]

    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]

    ds_asd = SpectralData.make_reflectance_ds(wavs, refl)
    unc_tot = (
        ds_asd.u_ran_reflectance.values**2 + ds_asd.u_sys_reflectance.values**2
    )

    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)

    return spectral_data


def _get_default_asd_data() -> SpectralData:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(
        os.path.join(current_dir, "assets/SomeMoonReflectances.txt"), delimiter=","
    )
    wavs = np.arange(350, 2501, 1)
    refl = data[5, 4:]

    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]

    ds_asd = SpectralData.make_reflectance_ds(wavs, refl)
    unc_tot = (
        ds_asd.u_ran_reflectance.values**2 + ds_asd.u_sys_reflectance.values**2
    )

    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)

    return spectral_data
