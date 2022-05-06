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
from . import esi, eli
from ...datatypes.datatypes import MoonData, IrradianceCoefficients

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def calculate_elref(
    wavelength_nm: float, moon_data: MoonData, coefficients: IrradianceCoefficients
) -> float:
    """Calculation of Extraterrestrial Lunar Reflectance following Eq 3 in Roman et al., 2020
    and then dividing by the Solar Reflectance using Wehrli 1985.

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The reflectance is returned in

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
    calc_eli = eli.calculate_eli(wavelength_nm, moon_data, coefficients)
    calc_esi = esi.get_esi_per_nm(wavelength_nm)
    return calc_eli / calc_esi
