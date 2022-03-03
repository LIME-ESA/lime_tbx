from dataclasses import dataclass
from typing import List


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
    def __init__(self):
        pass

    def get_wavelengths(self) -> List[float]:
        pass

    def get_coefficients_a(wavelength_nm: float) -> List[float]:
        pass

    def get_coefficients_b(wavelength_nm: float) -> List[float]:
        pass

    def get_coefficients_d(wavelength_nm: float) -> List[float]:
        pass

    def get_coefficients_c() -> List[float]:
        pass

    def get_coefficients_p() -> List[float]:
        pass

    def get_apollo_coefficients(self) -> List[float]:
        pass
