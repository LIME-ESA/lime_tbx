"""
This module calculates the extra-terrestrial lunar irradiance.

It exports the following functions:
    * calculate_eli - Calculates the expected extra-terrestrial lunar irradiance
    for a given wavelength in nanometers. Based on Eq 3 in Roman et al., 2020.
"""

"""___Built-In Modules___"""
import math

"""___Third-Party Modules___"""
# import here

"""___LIME Modules___"""
from . import esi, elref
from ...datatypes.datatypes import MoonData, IrradianceCoefficients

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


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
    a_l = elref.interpolated_moon_disk_reflectance(
        wavelength_nm, moon_data, coefficients
    )

    solid_angle_moon: float = 6.4177e-05
    omega = solid_angle_moon
    esk = esi.get_esi_per_nm(wavelength_nm)
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon
    distance_earth_moon_km: int = 384400

    lunar_irr = measurement_func_eli(a_l,omega,esk,dsm,distance_earth_moon_km,dom)
    return lunar_irr

def measurement_func_eli(a_l,omega,esk,dsm,distance_earth_moon_km,dom):

    lunar_irr = (((a_l*omega*esk)/math.pi)*((1/dsm)**2)*(distance_earth_moon_km/dom)**2)
    return lunar_irr