from ...datatypes.datatypes import MoonData, IrradianceCoefficients
import math
from typing import List
import numpy as np
import esi


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
    count: float = 0.0
    a_coeffs: List[float] = coeffs.get_coefficients_a(wavelength_nm)
    for i, a_value in enumerate(a_coeffs):
        count = count + a_value * gr_value**i
    return count


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
    c_coeffs: List[float] = coeffs.get_coefficients_c()
    d_coeffs: List[float] = coeffs.get_coefficients_d(wavelength_nm)
    p_coeffs: List[float] = coeffs.get_coefficients_p()
    l_theta = moon_data.lat_obs
    l_phi = moon_data.long_obs
    sum_a = _summatory_a(wavelength_nm, gr_value)
    sum_b = _summatory_b(wavelength_nm, phi)
    d1_value = d_coeffs[0] * math.exp(-gd_value / p_coeffs[0])
    d2_value = d_coeffs[1] * math.exp(-gd_value / p_coeffs[1])
    d3_value = d_coeffs[2] * math.cos((gd_value - p_coeffs[2]) / p_coeffs[3])
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
    return result


def _interpolated_moon_disk_reflectance(
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
        return _interpolated_moon_disk_reflectance(
            absolute_mpa_degrees, wvlens[0], moon_data, coeffs
        )
    if wavelength_nm > wvlens[-1]:
        # The extrapolation done is "nearest"
        return _interpolated_moon_disk_reflectance(
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
            _ln_moon_disk_reflectance(absolute_mpa_degrees, x_values[0], moon_data)
        )
        * apollo_coeffs[left_index]
    )
    y_values.append(
        math.exp(
            _ln_moon_disk_reflectance(absolute_mpa_degrees, x_values[1], moon_data)
        )
        * apollo_coeffs[right_index]
    )
    return np.interp(wavelength_nm, x_values, y_values)


def calculate_eli(
    wavelength_nm: float, moon_data: MoonData, coefficients: IrradianceCoefficients
) -> float:
    """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

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
        The extraterrestrial lunar irradiance calculated
    """
    a_l = _interpolated_moon_disk_reflectance(
        moon_data.absolute_mpa_degrees, wavelength_nm, moon_data, coefficients
    )

    solid_angle_moon: float = 6.4177e-05
    omega = solid_angle_moon
    esk = esi.get_esi_per_nm(wavelength_nm)
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon
    distance_earth_moon_km: int = 384400

    lunar_irr = (
        ((a_l * omega * esk) / math.pi)
        * ((1 / dsm) ** 2)
        * (distance_earth_moon_km / dom) ** 2
    )
    return lunar_irr
