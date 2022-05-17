"""
This module contains different dataclasses and classes used for the exchange
of data between modules of the package lime-tbx.

It exports the following classes:
    * MoonData - Moon data used in the calculations of the Moon's irradiance.
    * IrradianceCoefficients - Coefficients used in the ROLO algorithm. (ROLO's + Apollo's).
"""

"""___Built-In Modules___"""
from dataclasses import dataclass
from typing import Dict, List, Union
from datetime import datetime

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


@dataclass
class SpectralResponseFunction:
    """
    Dataclass containing the spectral response function. Consists of a set of pairs wavelength:percentage.

    Attributes
    ----------
    spectral_response : dict of float, float
        Set of pairs wavelength, percentage. 100% = 1.0.
    """

    spectral_response: Dict[float, float]


@dataclass
class SurfacePoint:
    """
    Dataclass representing a point on Earth's surface.

    The needed parameters for the calculation from a surface point.

    Attributes
    ----------
    latitude: float
        Geographic latitude in decimal degrees.
    longitude: float
        Geographic longitude in decimal degrees.
    altitude: float
        Altitude over the sea level in meters.
    dt: datetime | list of datetime
        Time or time series at which the lunar data will be calculated.
    """

    latitude: float
    longitude: float
    altitude: float
    dt: Union[datetime, List[datetime]]


@dataclass
class CustomPoint:
    """
    Dataclass representing a point which custom Moon data.

    The needed parameters for the calculation from a custom point.

    Attributes
    ----------
    distance_sun_moon : float
        Distance between the Sun and the Moon (in astronomical units)
    distance_observer_moon : float
        Distance between the Observer and the Moon (in kilometers)
    selen_obs_lat : float
        Selenographic latitude of the observer (in degrees)
    selen_obs_lon : float
        Selenographic longitude of the observer (in degrees)
    selen_sun_lon : float
        Selenographic longitude of the Sun (in radians)
    abs_moon_phase_angle : float
        Absolute Moon phase angle (in degrees)
    """

    distance_sun_moon: float
    distance_observer_moon: float
    selen_obs_lat: float
    selen_obs_lon: float
    selen_sun_lon: float
    abs_moon_phase_angle: float


class IrradianceCoefficients:
    """
    Coefficients used in the ROLO algorithm. (ROLO's + Apollo's).
    """

    @dataclass
    class CoefficientsWln:
        """
        Coefficients data for a wavelength. It includes only the a, b and d coefficients.

        Attributes
        ----------
        a_coeffs : tuple of 4 floats, corresponding to coefficients a0, a1, a2, and a3
        b_coeffs : tuple of 3 floats, corresponding to coefficients b1, b2, and b3
        d_coeffs : tuple of 3floats, corresponding to coefficients d1, d2, and d3
        """

        __slots__ = ["a_coeffs", "b_coeffs", "d_coeffs"]

        def __init__(self, coeffs: List[float]):
            """
            Parameters
            ----------
            coeffs : list of float
                List of floats consisting of all coefficients. In order: a0, a1, a2, a3, b1, b2, b3,
                d1, d2 and d3.

            Returns
            -------
            CoefficientsWln
                Instance of Coefficients with the correct data
            """
            self.a_coeffs = (coeffs[0], coeffs[1], coeffs[2], coeffs[3])
            self.b_coeffs = (coeffs[4], coeffs[5], coeffs[6])
            self.d_coeffs = (coeffs[7], coeffs[8], coeffs[9])

    def __init__(
        self,
        wavelengths: List[float],
        wlen_coeffs: List[CoefficientsWln],
        c_coeffs: List[float],
        p_coeffs: List[float],
        apollo_coeffs: List[float],
    ):
        self.wavelengths = wavelengths
        self.wlen_coeffs = wlen_coeffs
        self.c_coeffs = c_coeffs
        self.p_coeffs = p_coeffs
        self.apollo_coeffs = apollo_coeffs

    def get_wavelengths(self) -> List[float]:
        """Gets all wavelengths present in the model, in nanometers

        Returns
        -------
        list of float
            A list of floats that are the wavelengths in nanometers, in order
        """
        return self.wavelengths

    def get_coefficients_a(self, wavelength_nm: float) -> List[float]:
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
        index = self.get_wavelengths().index(wavelength_nm)
        return self.wlen_coeffs[index].a_coeffs

    def get_coefficients_b(self, wavelength_nm: float) -> List[float]:
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
        index = self.get_wavelengths().index(wavelength_nm)
        return self.wlen_coeffs[index].b_coeffs

    def get_coefficients_d(self, wavelength_nm: float) -> List[float]:
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
        index = self.get_wavelengths().index(wavelength_nm)
        return self.wlen_coeffs[index].d_coeffs

    def get_coefficients_c(self) -> List[float]:
        """Gets all 'c' coefficients

        Returns
        -------
        list of float
            A list containing all 'c' coefficients
        """
        return self.c_coeffs

    def get_coefficients_p(self) -> List[float]:
        """Gets all 'p' coefficients

        Returns
        -------
        list of float
            A list containing all 'p' coefficients
        """
        return self.p_coeffs

    def get_apollo_coefficients(self) -> List[float]:
        """Coefficients used for the adjustment of the ROLO model using Apollo spectra.

        Returns
        -------
        list of float
            A list containing all Apollo coefficients
        """
        return self.apollo_coeffs
