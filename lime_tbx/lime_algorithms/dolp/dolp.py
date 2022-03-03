"""describe class"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

from abc import ABC, abstractmethod


class IDOLP(ABC):
    @abstractmethod
    def get_polarized(
        wavelengths,
        distance_sun_moon,
        distance_observer_moon,
        selen_obs_lat,
        selen_obs_lon,
        selen_sun_lon,
        abs_moon_phase_angle,
    ):
        pass


class DOLP(IDOLP):
    def get_polarized(
        wavelengths,
        distance_sun_moon,
        distance_observer_moon,
        selen_obs_lat,
        selen_obs_lon,
        selen_sun_lon,
        abs_moon_phase_angle,
    ):
        pass
