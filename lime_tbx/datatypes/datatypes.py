"""
This module contains different dataclasses and classes used for the exchange
of data between modules of the package lime-tbx.

It exports the following classes:
    * MoonData - Moon data used in the calculations of the Moon's irradiance.
    * IrradianceCoefficients - Coefficients used in the ROLO algorithm. (ROLO's + Apollo's).
"""

"""___Built-In Modules___"""
from dataclasses import dataclass
from typing import List

"""___Third-Party Modules___"""
# import here

"""___LIME Modules___"""
# import here


@dataclass
class MoonData:
    """
    Moon data needed to calculate Moon's irradiance, probably obtained from NASA's SPICE Toolbox

    Attributes
    ----------
    distance_sun_moon : float
        Distance between the Sun and the Moon (in astronomical units)
    distance_observer_moon : float
        Distance between the Observer and the Moon (in kilometers)
    selen_sun_lon_rad : float
        Selenographic longitude of the Sun (in radians)
    selen_obs_lat : float
        Selenographic latitude of the observer (in degrees)
    selen_obs_lon : float
        Selenographic longitude of the observer (in degrees)
    abs_moon_phase_angle : float
        Absolute Moon phase angle (in degrees)
    """

    distance_sun_moon: float
    distance_observer_moon: float
    long_sun_radians: float
    lat_obs: float
    long_obs: float
    absolute_mpa_degrees: float


class IrradianceCoefficients:
    """
    Coefficients used in the ROLO algorithm. (ROLO's + Apollo's).
    """

    def __init__(self):
        pass

    def get_wavelengths(self) -> List[float]:
        """Gets all wavelengths present in the model, in nanometers

        Returns
        -------
        list of float
            A list of floats that are the wavelengths in nanometers, in order
        """
        pass

    def get_coefficients_a(wavelength_nm: float) -> List[float]:
        """Gets all 'a' coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'a' coefficients for the wavelength
        """
        pass

    def get_coefficients_b(wavelength_nm: float) -> List[float]:
        """Gets all 'b' coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'b' coefficients for the wavelength
        """
        pass

    def get_coefficients_d(wavelength_nm: float) -> List[float]:
        """Gets all 'd' coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'd' coefficients for the wavelength
        """
        pass

    def get_coefficients_c() -> List[float]:
        """Gets all 'c' coefficients

        Returns
        -------
        list of float
            A list containing all 'c' coefficients
        """
        pass

    def get_coefficients_p() -> List[float]:
        """Gets all 'p' coefficients

        Returns
        -------
        list of float
            A list containing all 'p' coefficients
        """
        pass

    def get_apollo_coefficients(self) -> List[float]:
        """Coefficients used for the adjustment of the ROLO model using Apollo spectra.

        Returns
        -------
        list of float
            A list containing all Apollo coefficients
        """
        pass
