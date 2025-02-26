"""
This module calculates the extra-terrestrial lunar reflectance.

It exports the following functions:
    * calculate_elref - Calculates the expected extra-terrestrial lunar reflectance
    for a given wavelength in nanometers. Based on Eq 3 in Roman et al., 2020 for the
    irradiance, then divided by the solar irradiance.
    * calculate_elref_unc - Calculates the uncertainty for the reflectance
    calculations of empirical data points.
"""

"""___Built-In Modules___"""
from typing import List, Union, Tuple

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules"""
import punpy

"""___LIME_TBX Modules___"""
from lime_tbx.common.datatypes import (
    ReflectanceCoefficients,
    MoonData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def _measurement_func_elref(
    coeffs: Union[List[float], np.ndarray],
    phi: float,
    l_phi: float,
    l_theta: float,
    gd_value: float,
) -> Union[float, np.ndarray]:
    """
    Final computation of the calculation of the ln of the reflectance of the Moon's disk, following Eq.2 in
    Roman et al., 2020

    Parameters
    ----------
    coeffs: list of float | np.ndarray of np.ndarray of float
        Coefficients for a wavelength, or all coefficients for all wavelengths (matrix)
    phi: float
        Selenographic longitude of the sun (radians)
    l_phi: float
        Selenographic longitude of the observer (degrees)
    l_theta: float
        Selenographic latitude of the observer (degrees)
    gd_value: float
        Absolute moon phase angle (degrees)

    Returns
    -------
    elrefs: float | np.ndarray of float
        Calculated reflectances.
    """

    # a_coeffs: list of float | np.ndarray of np.ndarray of float
    # A Coefficients for a wavelength, or all the A coefficients for all wavelengths (matrix)
    # b_coeffs: list of float | np.ndarray of np.ndarray of float
    # B Coefficients for a wavelength, or all the B coefficients for all wavelengths (matrix)
    # c_coeffs: list of float | np.ndarray of np.ndarray of float
    # C Coefficients for a wavelength, or all the C coefficients for all wavelengths (matrix)
    # d_coeffs: list of float | np.ndarray of np.ndarray of float
    # D Coefficients for a wavelength, or all the D coefficients for all wavelengths (matrix)
    # p_coeffs: list of float | np.ndarray of np.ndarray of float
    # P Coefficients for a wavelength, or all the P coefficients for all wavelengths (matrix)

    a_coeffs = coeffs[0:4, :]
    b_coeffs = coeffs[4:7, :]
    c_coeffs = coeffs[7:11, :]
    d_coeffs = coeffs[11:14, :]
    p_coeffs = coeffs[14::, :]

    gr_value = np.radians(gd_value)
    d1_value = d_coeffs[0] * np.exp(-gd_value / p_coeffs[0])
    d2_value = d_coeffs[1] * np.exp(-gd_value / p_coeffs[1])
    d3_value = d_coeffs[2] * np.cos((gd_value - p_coeffs[2]) / p_coeffs[3])

    sum_a: float = np.sum(
        [a_coeffs[i] * gr_value**i for i in range(len(a_coeffs))], axis=0
    )
    sum_b: float = np.sum(
        [b_coeffs[j] * phi ** (2 * (j + 1) - 1) for j in range(len(b_coeffs))], axis=0
    )
    result = (
        sum_a
        + sum_b
        + c_coeffs[0] * l_phi
        + c_coeffs[1] * l_theta
        + c_coeffs[2] * phi * l_phi
        + c_coeffs[3] * phi * l_theta
        + d1_value
        + d2_value
        + d3_value
    )
    return np.exp(result)


def calculate_elref(
    refl_coeffs: ReflectanceCoefficients,
    moon_data: MoonData,
) -> np.ndarray:
    """
    The calculation of the reflectance of the Moon's disk, following Eq.2 in Roman et al., 2020,
    without multiplying the value by its apollo coefficient.

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The reflectance is calculated in fractions of unity.

    Parameters
    ----------
    refl_coeffs: ReflectanceCoefficients
        ReflectanceCoefficients with the reflectances coefficients and uncertainties.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance

    Returns
    -------
    np.ndarray of np.ndarray of float
        The extraterrestrial lunar irradiance calculated using the points' coefficients present in the
        reflectance coefficient matrix.
    """
    cfs = refl_coeffs.coeffs
    phi = moon_data.long_sun_radians
    l_theta = moon_data.lat_obs
    l_phi = moon_data.long_obs
    gd_value = moon_data.absolute_mpa_degrees

    result = _measurement_func_elref(
        cfs._coeffs,
        phi,
        l_phi,
        l_theta,
        gd_value,
    )

    return result


def calculate_elref_unc(
    cimel_coef: ReflectanceCoefficients,
    moon_data: MoonData,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the uncertainty for the reflectance calculations of empirical data points.

    This uncertainties is for the empirical CIMEL data.

    Parameters
    ----------
    cimel_coef: ReflectanceCoefficients
        ReflectanceCoefficients with the CIMEL coefficients and uncertainties.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance

    Returns
    -------
    uncs: np.ndarray of float
        The uncertainties calculated
    corr: np.ndarray of float
        The error correlation matrix calculated
    """

    cfs = cimel_coef.coeffs
    ucfs = cimel_coef.unc_coeffs
    corrcfs = cimel_coef.err_corr_coeff

    phi = moon_data.long_sun_radians
    l_theta = moon_data.lat_obs
    l_phi = moon_data.long_obs
    gd_value = moon_data.absolute_mpa_degrees

    prop = punpy.MCPropagation(100, 1, MCdimlast=True)
    unc, corr = prop.propagate_random(
        _measurement_func_elref,
        [
            cfs._coeffs,
            phi,
            l_phi,
            l_theta,
            gd_value,
        ],
        [
            ucfs._coeffs,
            None,
            None,
            None,
            None,
        ],
        corr_x=[
            corrcfs,
            None,
            None,
            None,
            None,
        ],
        return_corr=True,
        allow_some_nans=False,
    )

    if not isinstance(corr, np.ndarray) and not isinstance(corr, list):
        corr = np.array([[corr]])
    corr = np.where(np.isnan(corr), 0, corr)
    np.fill_diagonal(corr, 1)

    return unc, corr
