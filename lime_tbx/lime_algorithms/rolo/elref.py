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

"""___LIME Modules___"""
from ...datatypes.datatypes import MoonData, IrradianceCoefficients

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def _summatory_a(
    wavelength_nm: float, gr_value: float, coeffs: IrradianceCoefficients
) -> float:
    """The first summatory of Eq. 2 in Roman et al., 2020

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers from which the moon's disk reflectance is being calculated
    gr_value : float
        Absolute value of MPA in radians
    coeffs : IrradianceCoefficients
        Needed coefficients for the simulation.

    Returns
    -------
    float
        Result of the computation of the first summatory
    """



def _summatory_b(
    wavelength_nm: float, phi: float, coeffs: IrradianceCoefficients
) -> float:
    """The second summatory of Eq. 2 in Roman et al., 2020, without the erratum

    Parameters
    ----------
    wavelength_nm : float
        Wavelength from which the moon's disk reflectance is being calculated
    phi : float
        Selenographic longitude of the Sun (in radians)
    coeffs : IrradianceCoefficients
        Needed coefficients for the simulation.

    Returns
    -------
    float
        Result of the computation of the second summatory
    """
    count: float = 0.0
    b_coeffs: List[float] = coeffs.get_coefficients_b(wavelength_nm)
    for j, b_value in enumerate(b_coeffs):
        count = count + b_value * phi ** (2 * (j + 1) - 1)
    return count


def _ln_moon_disk_reflectance(
    absolute_mpa_degrees: float,
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
    absolute_mpa_degrees : float
        Absolute Moon phase angle (in degrees)
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
    gd_value = absolute_mpa_degrees
    gr_value = math.radians(gd_value)
    phi = moon_data.long_sun_radians
    a_coeffs: List[float] = coeffs.get_coefficients_a(wavelength_nm)
    b_coeffs: List[float] = coeffs.get_coefficients_b(wavelength_nm)
    c_coeffs: List[float] = coeffs.get_coefficients_c()
    d_coeffs: List[float] = coeffs.get_coefficients_d(wavelength_nm)
    p_coeffs: List[float] = coeffs.get_coefficients_p()
    l_theta = moon_data.lat_obs
    l_phi = moon_data.long_obs

    result = measurement_func_elref(a_coeffs,b_coeffs,c_coeffs,d_coeffs,p_coeffs,phi,l_phi,l_theta,gd_value,gr_value)
    return result


def measurement_func_elref(a_coeffs,b_coeffs,c_coeffs,d_coeffs,p_coeffs,phi,l_phi,l_theta,gd_value,gr_value):
    d1_value = d_coeffs[0]*math.exp(-gd_value/p_coeffs[0])
    d2_value = d_coeffs[1]*math.exp(-gd_value/p_coeffs[1])
    d3_value = d_coeffs[2]*math.cos((gd_value-p_coeffs[2])/p_coeffs[3])
    sum_a: float = 0.0
    for i,a_value in enumerate(a_coeffs):
        sum_a = sum_a+a_value*gr_value**i
    sum_b: float = 0.0
    for j,b_value in enumerate(b_coeffs):
        sum_b = sum_b+b_value*phi**(2*(j+1)-1)
    result = (sum_a+sum_b+c_coeffs[0]*l_phi+c_coeffs[1]*l_theta+c_coeffs[2]*phi*l_phi+
              c_coeffs[3]*phi*l_theta+d1_value+d2_value+d3_value)
    return result

def interpolated_moon_disk_reflectance(
    absolute_mpa_degrees: float,
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
    absolute_mpa_degrees : float
        Absolute Moon phase angle (in degrees)
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
            absolute_mpa_degrees, wvlens[0], moon_data, coeffs
        )
    if wavelength_nm > wvlens[-1]:
        # The extrapolation done is "nearest"
        return interpolated_moon_disk_reflectance(
            absolute_mpa_degrees, wvlens[-1], moon_data, coeffs
        )
    apollo_coeffs = coeffs.get_apollo_coefficients()
    if wavelength_nm in wvlens:
        apollo_i = wvlens.index(wavelength_nm)
        return (
            math.exp(
                _ln_moon_disk_reflectance(
                    absolute_mpa_degrees, wavelength_nm, moon_data, coeffs
                )
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
        math.exp(
            _ln_moon_disk_reflectance(
                absolute_mpa_degrees, x_values[0], moon_data, coeffs
            )
        )
        * apollo_coeffs[left_index]
    )
    y_values.append(
        math.exp(
            _ln_moon_disk_reflectance(
                absolute_mpa_degrees, x_values[1], moon_data, coeffs
            )
        )
        * apollo_coeffs[right_index]
    )
    return np.interp(wavelength_nm, x_values, y_values)


def calculate_elref(
    wavelength_nm: float, moon_data: MoonData, coefficients: IrradianceCoefficients
) -> float:
    """
    The calculation of the reflectance of the Moon's disk, following Eq.2 in Roman et al., 2020
    and performin interpolation.

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
        moon_data.absolute_mpa_degrees, wavelength_nm, moon_data, coefficients
    )
