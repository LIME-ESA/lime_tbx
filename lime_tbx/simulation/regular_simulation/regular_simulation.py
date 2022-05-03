"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime
from typing import List

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...spice_adapter.spice_adapter import SPICEAdapter
from ...datatypes.datatypes import (
    IrradianceCoefficients,
    MoonData,
    SpectralResponseFunction,
)
from ...lime_algorithms.rolo import rolo

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

from abc import ABC, abstractmethod


class IRegularSimulation(ABC):
    @staticmethod
    @abstractmethod
    def get_eli_from_surface(
        srf: SpectralResponseFunction,
        latitude: float,
        longitude: float,
        altitude: float,
        dt: datetime,
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_surface(srf, latitude, longitude, altitude, datetime) -> list:
        pass

    @staticmethod
    @abstractmethod
    def get_polarized_from_surface(
        srf, latitude, longitude, altitude, datetime
    ) -> list:
        pass

    @staticmethod
    @abstractmethod
    def get_eli_from_custom(
        srf: SpectralResponseFunction,
        distance_sun_moon: float,
        distance_observer_moon: float,
        selen_obs_lat: float,
        selen_obs_lon: float,
        selen_sun_lon: float,
        abs_moon_phase_angle: float,
        coefficients,
    ) -> List[float]:
        pass

    @staticmethod
    @abstractmethod
    def get_elref_from_custom(
        srf,
        distance_sun_moon,
        distance_observer_moon,
        selen_obs_lat,
        selen_obs_lon,
        selen_sun_lon,
        abs_moon_phase_angle,
    ) -> list:
        pass

    @staticmethod
    @abstractmethod
    def get_polarized_from_custom(
        srf, selen_obs_lat, selen_obs_lon, selen_sun_lon, abs_moon_phase_angle
    ) -> list:
        pass


class RegularSimulation(IRegularSimulation):
    @staticmethod
    def get_eli_from_surface(
        srf: SpectralResponseFunction,
        latitude: float,
        longitude: float,
        altitude: float,
        dt: datetime,
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        rl = rolo.ROLO()
        wlens = list(srf.spectral_response.keys())
        md = SPICEAdapter().get_moon_data_from_earth(latitude, longitude, altitude, dt)
        irradiances = rl.get_eli(wlens, md, coefficients)
        for i, w in enumerate(wlens):
            irradiances[i] = irradiances[i] * srf[w]
        return irradiances

    @staticmethod
    def get_elref_from_surface(srf, latitude, longitude, altitude, datetime) -> list:
        pass

    @staticmethod
    def get_polarized_from_surface(
        srf, latitude, longitude, altitude, datetime
    ) -> list:
        pass

    @staticmethod
    def get_eli_from_custom(
        srf: SpectralResponseFunction,
        distance_sun_moon: float,
        distance_observer_moon: float,
        selen_obs_lat: float,
        selen_obs_lon: float,
        selen_sun_lon: float,
        abs_moon_phase_angle: float,
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        rl = rolo.ROLO()
        wlens = list(srf.spectral_response.keys())
        md = MoonData(
            distance_sun_moon,
            distance_observer_moon,
            selen_sun_lon,
            selen_obs_lat,
            selen_obs_lon,
            abs_moon_phase_angle,
        )
        irradiances = rl.get_eli(wlens, md, coefficients)
        for i, w in enumerate(wlens):
            irradiances[i] = irradiances[i] * srf[w]
        return irradiances

    @staticmethod
    def get_elref_from_custom(
        srf,
        distance_sun_moon,
        distance_observer_moon,
        selen_obs_lat,
        selen_obs_lon,
        selen_sun_lon,
        abs_moon_phase_angle,
    ) -> list:
        pass

    @staticmethod
    def get_polarized_from_custom(
        srf, selen_obs_lat, selen_obs_lon, selen_sun_lon, abs_moon_phase_angle
    ) -> list:
        pass
