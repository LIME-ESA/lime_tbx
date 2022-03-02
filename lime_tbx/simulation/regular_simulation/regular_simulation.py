"""describe class"""

"""___Built-In Modules___"""
#import here

"""___Third-Party Modules___"""
#import here

"""___NPL Modules___"""
#import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

from abc import ABC, abstractmethod

class IRegularSimulation(ABC):
    @abstractmethod
    def get_eli_from_surface(srf, latitude, longitude, altitude, datetime) -> list:
        pass
    @abstractmethod
    def get_polarized_from_surface(srf, latitude, longitude, altitude, datetime) -> list:
        pass
    @abstractmethod
    def get_eli_from_custom(srf, distance_sun_moon, distance_observer_moon, selen_obs_lat,
        selen_obs_lon, selen_sun_lon, abs_moon_phase_angle) -> list:
        pass
    @abstractmethod
    def get_polarized_from_custom(srf, selen_obs_lat, selen_obs_lon, selen_sun_lon,
        abs_moon_phase_angle) -> list:
        pass

class RegularSimulation(IRegularSimulation):
    def get_eli_from_surface(srf, latitude, longitude, altitude, datetime) -> list:
        pass

    def get_polarized_from_surface(srf, latitude, longitude, altitude, datetime) -> list:
        pass

    def get_eli_from_custom(srf, distance_sun_moon, distance_observer_moon, selen_obs_lat,
        selen_obs_lon, selen_sun_lon, abs_moon_phase_angle) -> list:
        pass

    def get_polarized_from_custom(srf, selen_obs_lat, selen_obs_lon, selen_sun_lon,
        abs_moon_phase_angle) -> list:
        pass
