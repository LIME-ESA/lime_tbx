"""Module in charge of retrieving interpolation data (spectra, etc) from storage.

It exports the following functions:
    * get_best_asd_data() - Retrieve the best ASD reflectance spectrum for the given moon phase angle.
    * get_apollo16_data() - Retrieve the Apollo 16 spectrum.
    * get_breccia_data() - Retrieve the Breccia spectrum.
    * get_composite_data() - Retrieve the composite Apollo 16 + Breccia spectrum.
    * get_available_spectra_names() - Obtain the spectra names that the user can use.
    * get_interpolation_spectrum_name() - Obtains the currently chosen interpolation spectrum name.
    * set_interpolation_spectrum_name() - Sets the spectrum name as the currently selected one.
    * can_perform_polarization() - Checks if the currently selected spectrum supports polarization.
"""

"""___Built-In Modules___"""
import os
from typing import List

"""___Third-Party Modules___"""
import numpy as np
import xarray as xr

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    InterpolationSettings,
    LimeException,
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


def _get_default_asd_data(moon_phase_angle: float) -> SpectralData:  # pragma: no cover
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


def get_best_asd_data(moon_phase_angle: float) -> SpectralData:
    """Retrieve the best ASD reflectance spectrum for the given moon phase angle.

    Parameters
    ----------
    moon_phase_angle: float
        Moon phase angle in degrees of which the best ASD reflectance will be retrieved.

    Returns
    -------
    spectral_data: SpectralData
        Reflectance spectrum.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # data = np.genfromtxt(
    #     os.path.join(current_dir, "assets/SomeMoonReflectances.txt"), delimiter=","
    # )

    ds_asd = xr.open_dataset(os.path.join(current_dir, "assets/ds_ASD.nc"))

    wavs = ds_asd.wavelength.values
    phase_angles = ds_asd.phase_angle.values
    best_id = np.argmin(np.abs(np.abs(phase_angles) - moon_phase_angle))

    refl = ds_asd.reflectance.values[:, best_id]
    unc = ds_asd.u_reflectance.values[:, best_id] * refl / 100

    spectral_data = SpectralData(wavs, refl, unc, ds_asd)

    return spectral_data


def get_apollo16_data() -> SpectralData:
    """Retrieve the Apollo 16 spectrum.

    Returns
    -------
    spectral_data: SpectralData
        Apollo 16 reflectance spectrum.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(
        os.path.join(current_dir, "assets/Apollo16.txt"), delimiter=","
    )
    wavs = data[:, 0]
    refl = data[:, 1]
    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]
    # TODO: Correct uncertainties from data
    unc_tot = np.zeros(refl.shape)
    corr = np.zeros((len(refl), len(refl)))
    np.fill_diagonal(corr, 1)
    ds_asd = SpectralData.make_reflectance_ds(wavs, refl, unc_tot, corr)
    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)
    return spectral_data


def get_breccia_data() -> SpectralData:
    """Retrieve the Breccia spectrum.

    Returns
    -------
    spectral_data: SpectralData
        Apollo 16 reflectance spectrum.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(os.path.join(current_dir, "assets/Breccia.txt"), delimiter=",")
    wavs = data[:, 0]
    refl = data[:, 1]
    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]
    # TODO: Correct uncertainties from data
    unc_tot = np.zeros(refl.shape)
    corr = np.zeros((len(refl), len(refl)))
    np.fill_diagonal(corr, 1)
    ds_asd = SpectralData.make_reflectance_ds(wavs, refl, unc_tot, corr)
    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)
    return spectral_data


def get_composite_data() -> SpectralData:
    """Retrieve the composite Apollo 16 + Breccia spectrum.
    95% Apollo 16 and 5% Breccia.

    Returns
    -------
    spectral_data: SpectralData
        Apollo 16 reflectance spectrum.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.genfromtxt(
        os.path.join(current_dir, "assets/Composite.txt"), delimiter=","
    )
    wavs = data[:, 0]
    refl = data[:, 1]
    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]
    valid_wavs = [i for i in range(350, 2501)]
    valid_ids = np.where(np.in1d(wavs, valid_wavs))[0]
    wavs = wavs[valid_ids]
    refl = refl[valid_ids]
    # TODO: Correct uncertainties from data
    unc_tot = np.zeros(refl.shape)
    corr = np.zeros((len(refl), len(refl)))
    np.fill_diagonal(corr, 1)
    ds_asd = SpectralData.make_reflectance_ds(wavs, refl, unc_tot, corr)
    spectral_data = SpectralData(wavs, refl, unc_tot, ds_asd)
    return spectral_data


SPECTRUM_NAME_ASD = "ASD"
SPECTRUM_NAME_APOLLO16 = "Apollo 16"
SPECTRUM_NAME_BRECCIA = "Breccia"
SPECTRUM_NAME_COMPOSITE = "Apollo 16 + Breccia"

_VALID_INTERP_SPECTRA = [
    SPECTRUM_NAME_ASD,
    #    SPECTRUM_NAME_APOLLO16,
    #    SPECTRUM_NAME_BRECCIA,
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
    """Obtain the spectra names that the user can use.

    Returns
    -------
    names: list of str
        Valid interpolation spectra names.
    """
    return _VALID_INTERP_SPECTRA.copy()


def get_interpolation_spectrum_name() -> str:
    """Obtains the currently chosen interpolation spectrum name.

    Returns
    -------
    name: str
        Currently chosen interpolation spectrum name.
    """
    setts = _load_interp_settings()
    if setts.interpolation_spectrum in _VALID_INTERP_SPECTRA:
        return setts.interpolation_spectrum
    logger.get_logger().error(
        f"Unknown interpolation spectrum found: {setts.interpolation_spectrum}"
    )
    return _VALID_INTERP_SPECTRA[0]


def is_show_interpolation_spectrum() -> bool:
    """Checks if the UI should show the spectrum used for interpolation.

    Returns
    -------
    should_show: bool
        True if the spectrum should be shown.
    """
    setts = _load_interp_settings()
    return setts.show_interp_spectrum


def set_interpolation_spectrum_name(spectrum: str):
    """Sets the spectrum name as the currently selected one.

    Parameters
    ----------
    spectrum: str
        Spectrum name to set as chosen.
    """
    setts = _load_interp_settings()
    if spectrum in _VALID_INTERP_SPECTRA:
        path = _get_interp_path()
        setts.interpolation_spectrum = spectrum
        setts._save_disk(path)
    else:
        msg = f"Tried setting unknown interpolation spectrum: {spectrum}"
        logger.get_logger().error(msg)
        raise LimeException(msg)


def set_show_interpolation_spectrum(show: bool):
    """Sets the interpolation spectrum visibility as <show>.

    Parameters
    ----------
    show: bool
        Visibility of the interpolation spectrum.
    """
    setts = _load_interp_settings()
    setts.show_interp_spectrum = show
    path = _get_interp_path()
    setts._save_disk(path)


def can_perform_polarization() -> bool:
    """Checks if the currently selected spectrum supports polarization.

    Returns
    -------
    supports_polarization: bool
        Indicator of support of polarization of the current spectrum.
    """
    return get_interpolation_spectrum_name() == SPECTRUM_NAME_ASD
