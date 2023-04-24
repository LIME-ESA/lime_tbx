"""Module in charge of retrieving interpolation data (spectra, etc) from storage.

It exports the following functions:
    * get_best_asd_data() - Retrieve the best ASD reflectance spectrum for the given moon phase angle.
    * get_apollo16_data() - Retrieve the Apollo 16 spectrum.
    * get_breccia_data() - Retrieve the Breccia spectrum.
    * get_composite_data() - Retrieve the composite Apollo 16 + Breccia spectrum.
    * get_available_spectra_names() - Obtain the spectra names that the user can use.
    * get_interpolation_spectrum_name() - Obtains the currently chosen interpolation spectrum name.
    * set_interpolation_spectrum_name() - Sets the spectrum name as the currently selected one.
"""

"""___Built-In Modules___"""
import os
from typing import List
import shutil

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


def get_best_polar_asd_data(moon_phase_angle: float) -> SpectralData:
    """Retrieve the best ASD polarization spectrum for the given moon phase angle.

    Parameters
    ----------
    moon_phase_angle: float
        Moon phase angle in degrees of which the best ASD polarization will be retrieved.

    Returns
    -------
    spectral_data: SpectralData
        ASD polarization spectrum.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ds_asd = xr.open_dataset(os.path.join(current_dir, "assets/ds_ASD.nc"))

    wavs = ds_asd.wavelength.values
    phase_angles = ds_asd.phase_angle.values
    best_id = np.argmin(np.abs(np.abs(phase_angles) - moon_phase_angle))

    pol = ds_asd.polarization.values[:, best_id]
    unc = ds_asd.u_polarization.values[:, best_id] * pol / 100

    spectral_data = SpectralData(wavs, pol, unc, ds_asd)

    return spectral_data


def get_linear_polar_data() -> SpectralData:
    """Retrieve the linear polarization spectrum.

    Returns
    -------
    spectral_data: SpectralData
        Linear polarization spectrum.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ds_asd = xr.open_dataset(os.path.join(current_dir, "assets/ds_ASD.nc"))

    wavs = ds_asd.wavelength.values
    pol = np.ones(wavs.shape)
    unc = np.ones(wavs.shape) * 0.1 * pol / 100
    corr = np.zeros((len(wavs), len(wavs)))
    np.fill_diagonal(corr, 1)

    ds = SpectralData.make_polarization_ds(wavs, pol, unc, corr)
    spectral_data = SpectralData(wavs, pol, unc, ds)

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

SPECTRUM_NAME_LINEAR = "Linear"

SRF_NAME_ASD = "ASD"
SRF_NAME_GAUSSIAN_1NM_3NM = "Gaussian SRF with 1nm spectral sampling and 3nm resolution"
SRF_NAME_TRIANGULAR_1NM_1NM = (
    "Triangular SRF with 1nm spectral sampling and 1nm resolution"
)
SRF_NAME_GAUSSIAN_P1NM_P3NM = (
    "Gaussian SRF with 0.1nm spectral sampling and 0.3nm resolution"
)
SRF_NAME_GAUSSIAN_P1NM_P3NM = (
    "Triangular SRF with 0.1nm spectral sampling and 0.1nm resolution"
)

_VALID_INTERP_SPECTRA = [
    SPECTRUM_NAME_ASD,
    #    SPECTRUM_NAME_APOLLO16,
    #    SPECTRUM_NAME_BRECCIA,
    SPECTRUM_NAME_COMPOSITE,
]

_VALID_DOLP_INTERP_SPECTRA = [
    SPECTRUM_NAME_ASD,
    SPECTRUM_NAME_LINEAR,
]

_VALID_INTERP_SRFS = [
    SRF_NAME_GAUSSIAN_1NM_3NM,
    SRF_NAME_TRIANGULAR_1NM_1NM,
    # SRF_NAME_GAUSSIAN_P1NM_P3NM,
    # SRF_NAME_GAUSSIAN_P1NM_P3NM,
    SRF_NAME_ASD,
]

SRF_DICT_FWHM_FILES = {
    SRF_NAME_ASD: "asd_fwhm.csv",
    SRF_NAME_GAUSSIAN_1NM_3NM: "interpolated_model_fwhm_3_1_gaussian.csv",
    SRF_NAME_TRIANGULAR_1NM_1NM: "interpolated_model_fwhm_1_1_triangle.csv",
    # SRF_NAME_GAUSSIAN_P1NM_P3NM: "interpolated_model_fwhm_0p3_0p1_gaussian.csv",
    # SRF_NAME_GAUSSIAN_P1NM_P3NM: "interpolated_model_fwhm_0p1_0p1_triangle.csv",
}

SRF_DICT_SOLAR_FILES = {
    SRF_NAME_ASD: "tsis_asd.csv",
    SRF_NAME_GAUSSIAN_1NM_3NM: "tsis_fwhm_3_1_gaussian.csv",
    SRF_NAME_TRIANGULAR_1NM_1NM: "tsis_fwhm_1_1_triangle.csv",
    # SRF_NAME_GAUSSIAN_P1NM_P3NM: "tsis_fwhm_0p3_0p1_gaussian.csv",
    # SRF_NAME_GAUSSIAN_P1NM_P3NM: "tsis_fwhm_0p1_0p1_triangle.csv",
}

SRF_DICT_SOLAR_DIALOG_SRF_TYPE = {
    SRF_NAME_ASD: "asd",
    SRF_NAME_GAUSSIAN_1NM_3NM: "interpolated_gaussian",
    SRF_NAME_TRIANGULAR_1NM_1NM: "interpolated_triangle",
    # SRF_NAME_GAUSSIAN_P1NM_P3NM: "tsis_fwhm_0p3_0p1_gaussian.csv",
    # SRF_NAME_GAUSSIAN_P1NM_P3NM: "tsis_fwhm_0p1_0p1_triangle.csv",
}


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


def get_available_dolp_spectra_names() -> List[str]:
    """Obtain the dolp spectra names that the user can use.

    Returns
    -------
    names: list of str
        Valid interpolation spectra names.
    """
    return _VALID_DOLP_INTERP_SPECTRA.copy()


def get_dolp_interpolation_spectrum_name() -> str:
    """Obtains the currently chosen dolp (polarization) interpolation spectrum name.

    Returns
    -------
    name: str
        Currently chosen dolp interpolation spectrum name.
    """
    setts = _load_interp_settings()
    if setts.interpolation_spectrum_polarization in _VALID_DOLP_INTERP_SPECTRA:
        return setts.interpolation_spectrum_polarization
    logger.get_logger().error(
        f"Unknown interpolation spectrum found: {setts.interpolation_spectrum_polarization}"
    )
    return _VALID_DOLP_INTERP_SPECTRA[0]


def get_available_interp_SRFs() -> List[str]:
    """Obtain the spectra names that the user can use.

    Returns
    -------
    names: list of str
        Valid interpolation spectra names.
    """
    return _VALID_INTERP_SRFS.copy()


def get_interpolation_SRF() -> str:
    """Obtains the currently chosen interpolation spectrum name.

    Returns
    -------
    name: str
        Currently chosen interpolation spectrum name.
    """
    setts = _load_interp_settings()
    if setts.interpolation_SRF in _VALID_INTERP_SRFS:
        return setts.interpolation_SRF
    logger.get_logger().error(
        f"Unknown interpolation spectrum found: {setts.interpolation_SRF}"
    )
    return _VALID_INTERP_SRFS[0]


def get_interpolation_srf_as_srf_type() -> str:
    srf_str = get_interpolation_SRF()
    return SRF_DICT_SOLAR_DIALOG_SRF_TYPE[srf_str]


def is_show_interpolation_spectrum() -> bool:
    """Checks if the UI should show the spectrum used for interpolation.

    Returns
    -------
    should_show: bool
        True if the spectrum should be shown.
    """
    setts = _load_interp_settings()
    return setts.show_interp_spectrum


def is_skip_uncertainties() -> bool:
    """Checks if the user settings are set to skip the uncertainties calculation.

    Returns
    -------
    should_skip: bool
        True if the uncertainties calculation should be skipped."""
    setts = _load_interp_settings()
    return setts.skip_uncertainties


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


def set_dolp_interpolation_spectrum_name(spectrum: str):
    """Sets the dolp spectrum name as the currently selected one.

    Parameters
    ----------
    spectrum: str
        Spectrum name to set as chosen.
    """
    setts = _load_interp_settings()
    if spectrum in _VALID_DOLP_INTERP_SPECTRA:
        path = _get_interp_path()
        setts.interpolation_spectrum_polarization = spectrum
        setts._save_disk(path)
    else:
        msg = f"Tried setting unknown polarisation interpolation spectrum: {spectrum}"
        logger.get_logger().error(msg)
        raise LimeException(msg)


def set_interpolation_SRF(intp_srf: str):
    """Sets the SRF as the currently selected one.

    Parameters
    ----------
    spectrum: str
        SRF name to set as chosen.
    """
    setts = _load_interp_settings()
    if intp_srf in _VALID_INTERP_SRFS:
        path = _get_interp_path()
        setts.interpolation_SRF = intp_srf
        setts._save_disk(path)
    else:
        msg = f"Tried setting unknown interpolation spectrum: {intp_srf}"
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


def set_skip_uncertainties(skip: bool):
    """Sets if the uncertainties should be calculated.

    Parameters
    ----------
    skip: bool
        True if the uncertainties should be skipped.
    """
    setts = _load_interp_settings()
    setts.skip_uncertainties = skip
    path = _get_interp_path()
    setts._save_disk(path)
