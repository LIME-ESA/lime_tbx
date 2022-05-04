"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime
from typing import List, Union

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
        dt: Union[datetime, List[datetime]],
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Union[List[float], List[List[float]]]:
        """
        Simulate the extraterrestrial lunar irradiance for a geographic point.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        latitude: float
            Geographic latitude in decimal degrees.
        longitude: float
            Geographic longitude in decimal degrees.
        altitude: float
            Altitude over the sea level in meters.
        dt: datetime | list of datetime
            Time or time series at which the lunar data will be calculated.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.
        kernels_path: str
            Path where the needed SPICE kernels are located.
            The user must have write access to that directory.

        Returns
        -------
        elis: list of float | list of list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
            It will be a list of lists of float if the parameter dt is a list. Otherwise it
            will only be a list of float.
        """
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
        coefficients: IrradianceCoefficients,
    ) -> List[float]:
        """
        Simulate the extraterrestrial lunar irradiance for custom lunar parameters.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
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
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        elis: list of float
            Extraterrestrial lunar irradiances for the given srf and the specified parameters.
        """
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
    def _get_eli_from_md(
        srf: SpectralResponseFunction,
        md: Union[MoonData, List[MoonData]],
        coefficients: IrradianceCoefficients,
    ) -> Union[List[float], List[List[float]]]:
        """
        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral Response Function that the simulation will be computed for.
        md: MoonData | list of MoonData
            MoonData for one observation, or for multiple observations in case that
            it's a list.
        coefficients: IrradianceCoefficients
            Values of the chosen coefficients for the ROLO algorithm.

        Returns
        -------
        irradiances: list of float | list of list of float
            Extraterrestrial lunar irradiances for the given srf at the specified point.
            It will be a list of lists of float if the parameter md is a list. Otherwise it
            will only be a list of float.
        """
        rl = rolo.ROLO()
        wlens = list(srf.spectral_response.keys())
        if not isinstance(md, list):
            irradiances = rl.get_eli(wlens, md, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i] * srf[w]
            return irradiances
        times_irr = []
        for m in md:
            irradiances = rl.get_eli(wlens, m, coefficients)
            for i, w in enumerate(wlens):
                irradiances[i] = irradiances[i] * srf[w]
            times_irr.append(irradiances)
        return times_irr

    @staticmethod
    def get_eli_from_surface(
        srf: SpectralResponseFunction,
        latitude: float,
        longitude: float,
        altitude: float,
        dt: Union[datetime, List[datetime]],
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Union[List[float], List[List[float]]]:
        md = SPICEAdapter().get_moon_data_from_earth(
            latitude, longitude, altitude, dt, kernels_path
        )
        return RegularSimulation._get_eli_from_md(srf, md, coefficients)

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
        md = MoonData(
            distance_sun_moon,
            distance_observer_moon,
            selen_sun_lon,
            selen_obs_lat,
            selen_obs_lon,
            abs_moon_phase_angle,
        )
        return RegularSimulation._get_eli_from_md(srf, md, coefficients)

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
