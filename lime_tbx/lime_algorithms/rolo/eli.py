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
    CimelReflectanceCoeffs,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


SOLID_ANGLE_MOON: float = 6.4177e-05
DIST_EARTH_MOON_KM: int = 384400


def measurement_func_eli(
    a_l: Union[float, np.ndarray],
    omega: float,
    esk: Union[float, List[float]],
    dsm: float,
    distance_earth_moon_km: float,
    dom: float,
) -> Union[float, np.ndarray]:
    """
    Final computation of the Eq 3 in Roman et al., 2020
    """
    lunar_irr = (
        ((a_l * omega * esk) / math.pi)
        * ((1 / dsm) ** 2)
        * (distance_earth_moon_km / dom) ** 2
    )
    return lunar_irr


def calculate_elis(
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
    a_l = elref.calculate_elref(wavelength_nm, moon_data, coefficients)

    esk = esi.get_esi_per_nms(wavelength_nm)
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    lunar_irr = measurement_func_eli(a_l, esk, dsm, dom)

    return lunar_irr


def calculate_eli_band(
    cimel_coef: CimelReflectanceCoeffs, moon_data: MoonData
) -> np.ndarray:
    """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020
    for a concrete set of empirical data points.

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

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
    a_l = elref.band_moon_disk_reflectance(cimel_coef, moon_data)

    esk = list(esi.get_esi_per_nms(cimel_coef.wlens))
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    lunar_irr = measurement_func_eli(
        a_l, SOLID_ANGLE_MOON, esk, dsm, DIST_EARTH_MOON_KM, dom
    )
    return lunar_irr


def calculate_eli_from_elref(
    wavelengths_nm: List[float], moon_data: MoonData, elref: np.ndarray
) -> np.ndarray:
    """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

    Parameters
    ----------
    wavelength_nm : list of float
        Wavelengths (in nanometers) of which the extraterrestrial lunar irradiance will be
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
    esk = list(esi.get_esi_per_nms(wavelengths_nm))
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    lunar_irr = measurement_func_eli(
        elref, SOLID_ANGLE_MOON, esk, dsm, DIST_EARTH_MOON_KM, dom
    )
    return lunar_irr
