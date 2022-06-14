"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
import math
from typing import List, Tuple
from datetime import datetime

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
from ...datatypes.datatypes import (
    ApolloIrradianceCoefficients,
    LunarObservation,
    SpectralResponseFunction,
    SurfacePoint,
)
from lime_tbx.simulation.lime_simulation import LimeSimulation
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/03/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def to_llh(x: float, y: float, z: float):
    """
    Changes from coordinates to latitude longitude and height

    Returns
    -------
    lat: float
        Latitude
    lon: float
        Longitude
    h: float
        Height (in meters)
    """
    a = 6378137.0  # in meters
    b = 6356752.314245  # in meters

    f = (a - b) / a

    e_sq = f * (2 - f)
    eps = e_sq / (1.0 - e_sq)

    p = math.sqrt(x * x + y * y)
    q = math.atan2((z * a), (p * b))

    sin_q = math.sin(q)
    cos_q = math.cos(q)

    sin_q_3 = sin_q * sin_q * sin_q
    cos_q_3 = cos_q * cos_q * cos_q

    phi = math.atan2((z + eps * b * sin_q_3), (p - e_sq * a * cos_q_3))
    lam = math.atan2(y, x)

    v = a / math.sqrt(1.0 - e_sq * math.sin(phi) * math.sin(phi))
    h = (p / math.cos(phi)) - v

    lat = math.degrees(phi)
    lon = math.degrees(lam)

    return lat, lon, h


class IComparison(ABC):
    @abstractmethod
    def get_simulations(
        self,
        observations: LunarObservation,
        srf: SpectralResponseFunction,
        coefficients: ApolloIrradianceCoefficients,
        kernels_path: str,
    ) -> Tuple[List[List[float]], List[List[datetime]], List[List[SurfacePoint]]]:
        """
        Simulate the moon irradiance for the given scenarios.

        Parameters
        ----------
        observations: MoonObservation
            MoonObservationn read from a GLOD datafile.
        srf: SpectralResponseFunction
            SpectralResponseFunction that corresponds to the observations file
        coefficients: IrradianceCoefficients
            Irradiance Coefficients to be used
        kernels_path: str
            Path where the needed SPICE kernels are located.

        Returns
        -------
        irrs: list of list of float
            List containing one list per SRF channel, containing all the simulated measures
            that have a counterpart in the observations data object.
        dts: list of list of datetime
            List containing one list per SRF channel, containing the corresponding datetimes
            for every irradiance measure.
        sps: list of list of SurfacePoint
            List containing one list per SRF channel, containing the corresponding SurfacePoint
            for every irradiance measure.
        """
        pass


class Comparison(IComparison):
    def get_simulations(
        self,
        observations: List[LunarObservation],
        srf: SpectralResponseFunction,
        coefficients: ApolloIrradianceCoefficients,
        lime_simulation: LimeSimulation,
    ) -> Tuple[List[List[float]], List[List[datetime]], List[List[SurfacePoint]]]:
        ch_names = srf.get_channels_names()
        irrs = [[] for _ in ch_names]
        ch_dates = [[] for _ in ch_names]
        sps = [[] for _ in ch_names]
        for obs in observations:
            sat_pos = obs.sat_pos
            dt = obs.dt
            lat, lon, h = to_llh(sat_pos.x * 1000, sat_pos.y * 1000, sat_pos.z * 1000)
            sp = SurfacePoint(lat, lon, h, dt)
            lime_simulation.update_irradiance(sp, coefficients)
            elis = lime_simulation.elis
            integrated_irrs = SpectralIntegration.integrate_elis(srf, elis)
            for j, ch in enumerate(ch_names):
                if obs.has_ch_value(ch):
                    ch_dates[j].append(dt)
                    irrs[j].append(integrated_irrs[j])
                    sps[j].append(sp)
        return irrs, ch_dates, sps
