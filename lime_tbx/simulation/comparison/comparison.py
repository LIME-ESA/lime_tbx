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
    IrradianceCoefficients,
    LunarObservation,
    SpectralResponseFunction,
    SurfacePoint,
)
from ...simulation.regular_simulation.regular_simulation import RegularSimulation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def to_llh(x: float, y: float, z: float):
    a = 6378137.0  # in meters
    b = 6356752.314245  # in meters

    f = (a - b) / a
    f_inv = 1.0 / f

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
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Tuple[List[List[float]], List[List[datetime]]]:
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
        """
        pass


class Comparison(IComparison):
    def get_simulations(
        self,
        observations: List[LunarObservation],
        srf: SpectralResponseFunction,
        coefficients: IrradianceCoefficients,
        kernels_path: str,
    ) -> Tuple[List[List[float]], List[List[datetime]]]:
        ch_names = srf.get_channels_names()
        irrs = [[] for _ in ch_names]
        ch_dates = [[] for _ in ch_names]
        for obs in observations:
            sat_pos = obs.sat_pos
            dt = obs.dt
            lat, lon, h = to_llh(sat_pos.x * 1000, sat_pos.y * 1000, sat_pos.z * 1000)
            sp = SurfacePoint(lat, lon, h, dt)
            elis = RegularSimulation.get_eli_from_surface(
                srf, sp, coefficients, kernels_path
            )
            integrated_irrs = RegularSimulation.integrate_elis(srf, elis)
            for j, ch in enumerate(ch_names):
                if obs.has_ch_value(ch):
                    ch_dates[j].append(dt)
                    irrs[j].append(integrated_irrs[j])
        return irrs, ch_dates
