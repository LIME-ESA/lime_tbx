"""
This module calculates the extra-terrestrial lunar reflectance.

It exports the following functions:
    * calculate_elref - Calculates the expected extra-terrestrial lunar reflectance
    for a given wavelength in nanometers. Based on Eq 3 in Roman et al., 2020 for the
    irradiance, then divided by the solar irradiance.
"""

"""___Built-In Modules___"""
import math
from typing import List

"""___Third-Party Modules___"""
import numpy as np
import punpy

"""___LIME Modules___"""
from ...datatypes.datatypes import (
    CimelCoef,
    MoonData,
    IrradianceCoefficients,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

def band_moon_disk_reflectance(
    cimel_coef: CimelCoef,
    moon_data: MoonData,
) -> np.ndarray:
    """
    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The reflectance is calculated in fractions of unity.

    This simulation is for the empirical CIMEL data.

    Parameters
    ----------
    cimel_coef: CimelCoef
        CimelCoef with the CIMEL coefficients and uncertainties.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance

    Returns
    -------
    np.ndarray of float
        The extraterrestrial lunar irradiance calculated for the uncertainty points
    """
    cfs = cimel_coef.coeffs

    phi = moon_data.long_sun_radians
    l_theta = moon_data.lat_obs
    l_phi = moon_data.long_obs
    gd_value = moon_data.absolute_mpa_degrees

    result = _measurement_func_elref(cfs.a_coeffs, cfs.b_coeffs, cfs.c_coeffs, cfs.d_coeffs,
        cfs.p_coeffs, phi, l_phi, l_theta, gd_value)

    return result

def band_moon_disk_reflectance_unc(
    cimel_coef: CimelCoef,
    moon_data: MoonData,
) -> np.ndarray:
    """
    Calculates the uncertainty for the reflectance calculations of empirical data points.

    This uncertainties is for the empirical CIMEL data.

    Parameters
    ----------
    cimel_coef: CimelCoef
        CimelCoef with the CIMEL coefficients and uncertainties.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance

    Returns
    -------
    np.ndarray of float
        The uncertainties calculated
    """

    cfs = cimel_coef.coeffs
    ucfs = cimel_coef.unc_coeffs

    phi = moon_data.long_sun_radians
    l_theta = moon_data.lat_obs
    l_phi = moon_data.long_obs
    gd_value = moon_data.absolute_mpa_degrees

    prop=punpy.MCPropagation(1000,dtype=np.float64)

    unc, samples_y, samples_x = prop.propagate_random(_measurement_func_elref, [cfs.a_coeffs, cfs.b_coeffs,
        cfs.c_coeffs, cfs.d_coeffs, cfs.p_coeffs, phi, l_phi, l_theta, gd_value],
        [ucfs.a_coeffs, ucfs.b_coeffs, ucfs.c_coeffs, ucfs.d_coeffs, ucfs.p_coeffs, None,
        None, None, None], return_samples=True)

    print("here6", samples_y)

    return unc

def _moon_disk_reflectance(
    wavelength_nm: float,
    moon_data: MoonData,
    coeffs: IrradianceCoefficients,
) -> float:
    """The calculation of the ln of the reflectance of the Moon's disk, following Eq.2 in
    Roman et al., 2020

    If the wavelength has no associated ROLO coefficients, it uses some linearly interpolated
    ones.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers from which one wants to obtain the MDR.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance
    coeffs : IrradianceCoefficients
        Needed coefficients for the simulation.

    Returns
    -------
    float
        The ln of the reflectance of the Moon's disk for the inputed data
    """

    a_coeffs: List[float] = coeffs.get_coefficients_a(wavelength_nm)
    b_coeffs: List[float] = coeffs.get_coefficients_b(wavelength_nm)
    c_coeffs: List[float] = coeffs.get_coefficients_c()
    d_coeffs: List[float] = coeffs.get_coefficients_d(wavelength_nm)
    p_coeffs: List[float] = coeffs.get_coefficients_p()

    phi = moon_data.long_sun_radians
    l_theta = moon_data.lat_obs
    l_phi = moon_data.long_obs
    gd_value = moon_data.absolute_mpa_degrees

    result = _measurement_func_elref(a_coeffs,b_coeffs,c_coeffs,d_coeffs,p_coeffs,phi,l_phi,l_theta,gd_value)

    return result


def _measurement_func_elref(a_coeffs: List[float], b_coeffs: List[float], c_coeffs: List[float],
        d_coeffs: List[float], p_coeffs: List[float], phi: float, l_phi: float, l_theta: float,
        gd_value: float) -> float:
    """
    Final computation of the calculation of the ln of the reflectance of the Moon's disk, following Eq.2 in
    Roman et al., 2020

    Parameters
    ----------
    a_coeffs
    b_coeffs
    """
    if isinstance(gd_value, float):
        gr_value = math.radians(gd_value)
    else:
        gr_value = gd_value
    d1_value = d_coeffs[0]*np.exp(-gd_value/p_coeffs[0])
    d2_value = d_coeffs[1]*np.exp(-gd_value/p_coeffs[1])
    d3_value = d_coeffs[2]*np.cos((gd_value-p_coeffs[2])/p_coeffs[3])

    sum_a: float = np.sum([a_coeffs[i]*gr_value**i for i in range(len(a_coeffs))],axis=0)
    sum_b: float = np.sum([b_coeffs[j]*phi**(2*(j+1)-1) for j in range(len(b_coeffs))],axis=0)
    result = (sum_a+sum_b+c_coeffs[0]*l_phi+c_coeffs[1]*l_theta+c_coeffs[2]*phi*l_phi+
              c_coeffs[3]*phi*l_theta+d1_value+d2_value+d3_value)
    return np.exp(result)

def interpolated_moon_disk_reflectance(
    wavelength_nm: float,
    moon_data: "MoonData",
    coeffs: IrradianceCoefficients,
) -> float:
    """The calculation of the reflectance of the Moon's disk, following Eq.2 in Roman et al., 2020

    If the wavelength is not present in the ROLO coefficients, it calculates the linear
    interpolation between the previous and the next one, or the extrapolation with the two
    nearest ones in case that it's on an extreme.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers from which one wants to obtain the MDR.
    moon_data : 'MoonData'
        Moon data needed to calculate Moon's irradiance
    coeffs : IrradianceCoefficients
        Needed coefficients for the simulation.

    Returns
    -------
    float
        The ln of the reflectance of the Moon's disk for the inputed data
    """
    wvlens = coeffs.get_wavelengths()
    if wavelength_nm < wvlens[0]:
        # The extrapolation done is "nearest"
        return interpolated_moon_disk_reflectance(
            wvlens[0], moon_data, coeffs
        )
    if wavelength_nm > wvlens[-1]:
        # The extrapolation done is "nearest"
        return interpolated_moon_disk_reflectance(
            wvlens[-1], moon_data, coeffs
        )
    apollo_coeffs = coeffs.get_apollo_coefficients()
    if wavelength_nm in wvlens:
        apollo_i = wvlens.index(wavelength_nm)
        return (
                _moon_disk_reflectance(
                    wavelength_nm, moon_data, coeffs
                )
            * apollo_coeffs[apollo_i]
        )
    near_left = -math.inf
    near_right = math.inf
    for wvlen in wvlens:
        if near_left < wvlen < wavelength_nm:
            near_left = wvlen
        elif wavelength_nm < wvlen < near_right:
            near_right = wvlen
    x_values = [near_left, near_right]
    left_index = wvlens.index(x_values[0])
    right_index = wvlens.index(x_values[1])
    y_values = []
    y_values.append(

            _moon_disk_reflectance(
                x_values[0], moon_data, coeffs

        )
        * apollo_coeffs[left_index]
    )
    y_values.append(

            _moon_disk_reflectance(
                x_values[1], moon_data, coeffs

        )
        * apollo_coeffs[right_index]
    )
    return np.interp(wavelength_nm, x_values, y_values)


def calculate_elref(
    wavelength_nm: float, moon_data: MoonData, coefficients: IrradianceCoefficients
) -> float:
    """
    The calculation of the reflectance of the Moon's disk, following Eq.2 in Roman et al., 2020
    and performing interpolation.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength (in nanometers) of which the extraterrestrial lunar irradiance will be
        calculated.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance
    coefficients : IrradianceCoefficients
        Needed coefficients for the simulation.

    Returns
    -------
    float
        The extraterrestrial lunar reflectance calculated, in fraction of unity.
    """
    return interpolated_moon_disk_reflectance(
        wavelength_nm, moon_data, coefficients
    )

