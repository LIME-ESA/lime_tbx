"""
This module abstracts and encapsulates use-cases related to simulations from esa satellites,
and performs the actions and calculations that are related to each of them.

It exports the following classes:
    * IESASatellites - Interface that contains the methods of this module.
    * ESASatellites - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
import math
from typing import List, Callable

from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter

"""___Third-Party Modules___"""
import numpy as np
import pyproj

"""___NPL Modules___"""
from ...datatypes.datatypes import (
    ComparisonData,
    KernelsPath,
    LunarObservation,
    ReflectanceCoefficients,
    SpectralData,
    SpectralResponseFunction,
    SurfacePoint,
    SRFChannel,
)
from lime_tbx.simulation.lime_simulation import ILimeSimulation
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration
from ...datatypes import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/03/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def to_llh(x: float, y: float, z: float):
    """
    Changes from coordinates to latitude longitude and height
    UNUSED

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


def to_xyz(latitude, longitude, altitude):
    # (lat, lon) in WSG-84 degrees
    # altitude in meters
    # unused
    R = 6378137.0
    f_inv = 298.257224
    f = 1.0 / f_inv
    e2 = 1 - (1 - f) * (1 - f)
    cosLat = math.cos(latitude * math.pi / 180)
    sinLat = math.sin(latitude * math.pi / 180)

    cosLong = math.cos(longitude * math.pi / 180)
    sinLong = math.sin(longitude * math.pi / 180)

    c = 1 / math.sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat)
    s = (1 - f) * (1 - f) * c

    x = (R * c + altitude) * cosLat * cosLong
    y = (R * c + altitude) * cosLat * sinLong
    z = (R * s + altitude) * sinLat

    return x, y, z


class IComparison(ABC):
    @abstractmethod
    def get_simulations(
        self,
        observations: List[LunarObservation],
        def_srf: SpectralResponseFunction,
        srf: SpectralResponseFunction,
        coefficients: ReflectanceCoefficients,
        lime_simulation: ILimeSimulation,
    ) -> List[ComparisonData]:
        """
        Simulate the moon irradiance for the given scenarios.

        Parameters
        ----------
        observations: list of MoonObservation
            MoonObservations read from a GLOD datafile.
        def_srf: SpectralResponseFunction
            SpectralResponseFunction that corresponds to the default spectrum.
        srf: SpectralResponseFunction
            SpectralResponseFunction that corresponds to the observations file
        coefficients: ReflectanceCoefficients
            Coefficients to be used
        lime_simulation: ILimeSimulation
            Lime simulation instance, storing the current state of the simulation.

        Returns
        -------
        comparisons: list of ComparisonData
            List containing all comparisons of all channels
        """
        pass


class Comparison(IComparison):
    def _get_full_srf(self) -> SpectralResponseFunction:
        spectral_response = {
            i: 1.0 for i in np.arange(constants.MIN_WLEN, constants.MAX_WLEN)
        }
        ch = SRFChannel(
            (constants.MAX_WLEN - constants.MIN_WLEN) / 2, "Full", spectral_response
        )
        srf = SpectralResponseFunction("Full", [ch])
        return srf

    def __init__(self, kernels_path: KernelsPath):
        self.kernels_path = kernels_path

    def get_simulations(
        self,
        observations: List[LunarObservation],
        srf: SpectralResponseFunction,
        coefficients: ReflectanceCoefficients,
        lime_simulation: ILimeSimulation,
        callback_observation: Callable = None,
    ) -> List[ComparisonData]:
        """
        Parameters
        ----------
        observations: list of LunarObservations
        srf: SpectralResponseFunction
        coefficients: ReflectanceCoefficients
        lime_simulation: ILimeSimulation
        callback_observation: Callable
            Function that will be called once for every observation when simulated.
        """
        ch_names = srf.get_channels_names()
        comparisons = []
        sigs = [[] for _ in ch_names]
        ch_dates = [[] for _ in ch_names]
        sps = [[] for _ in ch_names]
        obs_irrs = [[] for _ in ch_names]
        for obs in observations:
            sat_pos = obs.sat_pos
            dt = obs.dt
            lat, lon, h = SPICEAdapter.to_planetographic(
                sat_pos.x * 1000,
                sat_pos.y * 1000,
                sat_pos.z * 1000,
                "EARTH",
                self.kernels_path.main_kernels_path,
            )
            sp = SurfacePoint(lat, lon, h, dt)
            lime_simulation.set_simulation_changed()
            lime_simulation.update_irradiance(srf, sp, coefficients)
            signals = lime_simulation.get_signals()
            for j, ch in enumerate(ch_names):
                if obs.has_ch_value(ch):
                    ch_dates[j].append(dt)
                    sigs[j].append((signals.data[j][0], signals.uncertainties[j][0]))
                    # [0] because obs.dt is one datetime, only one dt
                    sps[j].append(sp)
                    obs_irrs[j].append(obs.ch_irrs[ch])
            if callback_observation:
                callback_observation()
        for i, ch in enumerate(ch_names):
            if len(ch_dates[i]) > 0:
                irrs = np.array([s[0] for s in sigs[i]])
                uncs = np.array([s[1] for s in sigs[i]])
                # Observed and Simulated
                specs = (
                    SpectralData(
                        ch_dates[i], obs_irrs[i], np.zeros(len(ch_dates[i])), None
                    ),
                    SpectralData(ch_dates[i], irrs, uncs, None),
                )
                # Relative Differences
                rel_diffs = []
                uncs_r = []
                tot_rel_diff = 0
                num_samples = len(specs[0].wlens)
                for j in range(num_samples):
                    sim = specs[1].data[j]
                    ref = specs[0].data[j]
                    rel_dif = (sim - ref) / ref
                    tot_rel_diff += rel_dif
                    rel_diffs.append(rel_dif)
                    unc_sim = specs[1].uncertainties[j]
                    unc_ref = specs[0].uncertainties[j]
                    uncs_r.append(
                        (unc_sim + unc_ref) + unc_ref
                    )  # i dont know if this is the correct way of propagating the uncertainty
                mean_rel_diff = tot_rel_diff / num_samples
                std = np.std(rel_diffs)
                ratio_spec = SpectralData(
                    specs[0].wlens, np.array(rel_diffs), np.array(uncs_r), None
                )
                cp = ComparisonData(
                    specs[0],
                    specs[1],
                    ratio_spec,
                    mean_rel_diff,
                    std,
                    0,  # TODO add real valid correct value
                    num_samples,
                    ch_dates[i],
                    sps[i],
                )
                comparisons.append(cp)
            else:
                comparisons.append(
                    ComparisonData(None, None, None, None, None, None, None, [], [])
                )
        return comparisons
