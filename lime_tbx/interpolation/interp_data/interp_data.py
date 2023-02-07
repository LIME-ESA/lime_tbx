"""describe class"""

"""___Built-In Modules___"""
import os
from typing import List

"""___Third-Party Modules___"""
import numpy as np

"""___LIME TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    InterpolationSettings,
)
from lime_tbx.datatypes.datatypes import SpectralData
from lime_tbx.local_storage import programdata
from lime_tbx.datatypes import logger

"""___Authorship___"""
__author__ = "Pieter De Vis, Jacob Fahy, Javier Gatón Herguedas, Ramiro González Catón, Carlos Toledano"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def get_best_asd_data(moon_phase_angle: float) -> SpectralData:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(
        os.path.join(current_dir, "assets/SomeMoonReflectances.txt"), delimiter=","
    )
    wavs = np.arange(350, 2501, 1)
    phase_angles = data[:, 3]
    best_id = np.argmin(np.abs(phase_angles - moon_phase_angle))
    # Default value was best_id = 5
    refl = data[best_id, 4:]

    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]

    ds_asd = SpectralData.make_reflectance_ds(wavs, refl)
    unc_tot = (
        ds_asd.u_ran_reflectance.values**2 + ds_asd.u_sys_reflectance.values**2
    )

    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)

    return spectral_data


def get_apollo16_data() -> SpectralData:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(
        os.path.join(current_dir, "assets/Apollo16.txt"), delimiter=","
    )
    wavs = data[:, 0]
    refl = data[:, 1]
    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]
    ds_asd = SpectralData.make_reflectance_ds(wavs, refl)
    # TODO: Correct uncertainties from data
    unc_tot = (
        ds_asd.u_ran_reflectance.values**2 + ds_asd.u_sys_reflectance.values**2
    )
    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)
    return spectral_data


def get_breccia_data() -> SpectralData:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(os.path.join(current_dir, "assets/Breccia.txt"), delimiter=",")
    wavs = data[:, 0]
    refl = data[:, 1]
    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]
    ds_asd = SpectralData.make_reflectance_ds(wavs, refl)
    unc_tot = (
        ds_asd.u_ran_reflectance.values**2 + ds_asd.u_sys_reflectance.values**2
    )
    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)
    return spectral_data


SPECTRUM_NAME_ASD = "ASD"
SPECTRUM_NAME_APOLLO16 = "Apollo 16"
SPECTRUM_NAME_BRECCIA = "Breccia"
SPECTRUM_NAME_COMPOSITE = "Apollo 16 + Breccia"

_VALID_INTERP_SPECTRA = [
    SPECTRUM_NAME_ASD,
    SPECTRUM_NAME_APOLLO16,
    SPECTRUM_NAME_BRECCIA,
    SPECTRUM_NAME_COMPOSITE,
]


def _get_interp_path() -> str:
    path = os.path.join(
        programdata.get_programfiles_folder(), "coeff_data", "interp_settings.yml"
    )
    return path


def _load_interp_settings() -> InterpolationSettings:
    path = _get_interp_path()
    return InterpolationSettings._load_yaml(path)


def get_available_spectra_names() -> List[str]:
    return _VALID_INTERP_SPECTRA.copy()


def get_interpolation_spectrum_name() -> str:
    setts = _load_interp_settings()
    if setts.interpolation_spectrum in _VALID_INTERP_SPECTRA:
        return setts.interpolation_spectrum
    else:
        logger.get_logger().error(
            f"Unknown interpolation spectrum found: {setts.interpolation_spectrum}"
        )
        return _VALID_INTERP_SPECTRA[0]


def set_interpolation_spectrum_name(spectrum: str):
    setts = _load_interp_settings()
    if spectrum in _VALID_INTERP_SPECTRA:
        path = _get_interp_path()
        setts.interpolation_spectrum = spectrum
        setts._save_disk(path)
    else:
        logger.get_logger().error(
            f"Tried setting unknown interpolation spectrum: {spectrum}"
        )


def can_perform_polarization() -> bool:
    return get_interpolation_spectrum_name() == SPECTRUM_NAME_ASD
