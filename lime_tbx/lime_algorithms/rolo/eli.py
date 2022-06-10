"""
This module calculates the extra-terrestrial lunar irradiance.

It exports the following functions:
    * calculate_elis - Calculates the expected extra-terrestrial lunar irradiances
    for some given wavelengths in nanometers. Based on Eq 3 in Roman et al., 2020.
"""

"""___Built-In Modules___"""
import math
from typing import List, Union

"""___Third-Party Modules___"""
# import here
import numpy as np
import punpy

"""___LIME Modules___"""
from . import esi, elref
from ...datatypes.datatypes import (
    MoonData,
    IrradianceCoefficients,
    CimelData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def calculate_elis(
    wavelengths_nm: List[float],
    moon_data: MoonData,
    coefficients: IrradianceCoefficients,
) -> np.ndarray:
    """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

    Simulates a lunar observation for some wavelengths for any observer/solar selenographic
    latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

    Parameters
    ----------
    wavelengths_nm : list of float
        Wavelengths (in nanometers) of which the extraterrestrial lunar irradiance will be
        calculated.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance
    coefficients : IrradianceCoefficients
        Needed coefficients for the simulation.

    Returns
    -------
    np.ndarray of float
        The extraterrestrial lunar irradiances calculated
    """
    a_l = np.array(
        [
            elref.interpolated_moon_disk_reflectance(wlen, moon_data, coefficients)
            for wlen in wavelengths_nm
        ]
    )

    esk = list(esi.get_esi_per_nms(wavelengths_nm))
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    lunar_irr = measurement_func_eli(a_l, esk, dsm, dom)

    return lunar_irr


def calculate_eli_band(cimel_data: CimelData, moon_data: MoonData) -> np.ndarray:
    """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020
    for a concrete set of empirical data points.

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

    This simulation is for the empirical CIMEL data.

    Parameters
    ----------
    cimel_data: CimelData
        CimelData with the CIMEL coefficients and uncertainties.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance

    Returns
    -------
    np.ndarray of float
        The extraterrestrial lunar irradiance calculated for the uncertainty points
    """
    a_l = elref.band_moon_disk_reflectance(cimel_data, moon_data)

    esk = list(esi.get_esi_per_nms(cimel_data.wavelengths))
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    lunar_irr = measurement_func_eli(a_l, esk, dsm, dom)
    return lunar_irr


def calculate_eli_band_unc(
    cimel_data: CimelData,
    moon_data: MoonData,
) -> np.ndarray:
    """
    Calculates the uncertainty for the ELI calculations of empirical data points.

    This uncertainties is for the empirical CIMEL data.

    Parameters
    ----------
    cimel_data: CimelData
        CimelData with the CIMEL coefficients and uncertainties.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance

    Returns
    -------
    np.ndarray of float
        The uncertainties calculated
    """
    a_l = elref.band_moon_disk_reflectance(cimel_data, moon_data)
    u_a_l = elref.band_moon_disk_reflectance_unc(cimel_data, moon_data)

    esk = list(esi.get_esi_per_nms(cimel_data.wavelengths))
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    lunar_irr = measurement_func_eli(a_l, esk, dsm, dom)

    # prop = punpy.MCPropagation(1000)
    #
    # unc = prop.propagate_random(measurement_func_eli,
    #                             [a_l,omega,esk,dsm,distance_earth_moon_km,dom],
    #                             [u_a_l,None,None,None,None,None])

    unc = lunar_irr * u_a_l / a_l
    return unc


def measurement_func_eli(
    a_l: np.ndarray, esk: List[float], dsm: float, dom: float
) -> np.ndarray:
    """
    Final computation of the Eq 3 in Roman et al., 2020
    """
    solid_angle_moon: float = 6.4177e-05
    omega = solid_angle_moon
    distance_earth_moon_km: int = 384400
    lunar_irr = (
        ((a_l * omega * esk) / math.pi)
        * ((1 / dsm) ** 2)
        * (distance_earth_moon_km / dom) ** 2
    )
    return lunar_irr
