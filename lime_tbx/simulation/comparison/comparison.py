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
from deprecated import deprecated

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    ComparisonData,
    KernelsPath,
    LunarObservation,
    ReflectanceCoefficients,
    SpectralData,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
    SRFChannel,
    LimeException,
)
from lime_tbx.simulation.lime_simulation import ILimeSimulation, is_ampa_valid_range
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.spectral_integration.spectral_integration import get_default_srf

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/03/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


@deprecated
def _to_llh(x: float, y: float, z: float):
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


@deprecated
def _to_xyz(latitude, longitude, altitude):
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
    """Interface that contains the methods of this module.

    It exports the following functions:
        * get_simulations - Simulate the moon irradiance for the given scenarios.
        * sort_by_mpa - Returns a copy of the given list of ComparisonData but sorted by
            moon phase angle.
    """

    @abstractmethod
    def get_simulations(
        self,
        observations: List[LunarObservation],
        srf: SpectralResponseFunction,
        coefficients: ReflectanceCoefficients,
        lime_simulation: ILimeSimulation,
        callback_observation: Callable = None,
    ) -> List[ComparisonData]:
        """
        Obtain a list of comparison data for the given observation scenarios,
        each element corresponding to the comparisons of one channel.

        Parameters
        ----------
        observations: list of LunarObservations
            LunarObservations read from a GLOD datafile.
        srf: SpectralResponseFunction
            SpectralResponseFunction that is used.
        coefficients: ReflectanceCoefficients
            Coefficients to be used.
        lime_simulation: ILimeSimulation
            Lime simulation instance, storing the current state of the simulation.
        callback_observation: Callable
            Function that will be called once for every observation when simulated.

        Returns
        -------
        comparisons: list of ComparisonData
            List of ComparisonData, each element corresponding to the comparisons of one channel.
        """
        pass

    @abstractmethod
    def sort_by_mpa(self, comparisons: List[ComparisonData]) -> List[ComparisonData]:
        """Returns a copy of the given list of ComparisonData but sorted by moon phase angle.

        Parameters
        ----------
        comparisons: list of ComparisonData
            List of ComparisonData that will be sorted by mpa.

        Returns
        -------
        sorted_comparisons: list of ComparisonData
            List with the same ComparisonData instances but sorted by the moon phase angle.
        """
        pass


class Comparison(IComparison):
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
        ch_names = srf.get_channels_names()
        comparisons = []
        sigs = [[] for _ in ch_names]
        ch_dates = [[] for _ in ch_names]
        sps = [[] for _ in ch_names]
        mpas = [[] for _ in ch_names]
        obs_irrs = [[] for _ in ch_names]
        mpa_calcs = []
        sp_calcs = []
        #
        dts = [o.dt for o in observations]
        xyzs = [
            (
                o.sat_pos.x,
                o.sat_pos.y,
                o.sat_pos.z,
            )
            for o in observations
        ]
        if observations and observations[0].sat_pos_ref in (
            "IAU_MOON",
            "MOON_ME",
            "MOON",
        ):
            llhs = SPICEAdapter.to_planetographic_multiple(
                xyzs,
                "MOON",
                self.kernels_path.main_kernels_path,
            )
            mdas = SPICEAdapter.get_moon_datas_from_rectangular_multiple(
                xyzs,
                dts,
                self.kernels_path,
                "MOON_ME",
            )
            sp_calcs = [
                CustomPoint(
                    mdam.distance_sun_moon,
                    mdam.distance_observer_moon,
                    llh[0],
                    llh[1],
                    mdam.long_sun_radians,
                    mdam.absolute_mpa_degrees,
                    mdam.mpa_degrees,
                )
                for mdam, llh in zip(mdas, llhs)
            ]
        else:
            llhs = SPICEAdapter.to_planetographic_multiple(
                xyzs,
                "EARTH",
                self.kernels_path.main_kernels_path,
            )
            sp_calcs = [
                SurfacePoint(llh[0], llh[1], llh[2], dt) for llh, dt in zip(llhs, dts)
            ]
            mdas = SPICEAdapter.get_moon_datas_from_rectangular_multiple(
                xyzs, dts, self.kernels_path
            )
        mpa_calcs = [md.mpa_degrees for md in mdas]
        #
        for obs, mpa, sp, mda, dt in zip(observations, mpa_calcs, sp_calcs, mdas, dts):
            if callback_observation:
                callback_observation()
            lime_simulation.set_simulation_changed()
            def_srf = get_default_srf()
            lime_simulation.update_irradiance(
                def_srf, srf, sp, coefficients, mda_precalculated=mda
            )
            signals = lime_simulation.get_signals()
            for j, ch in enumerate(ch_names):
                if obs.has_ch_value(ch):
                    ch_dates[j].append(dt)
                    sigs[j].append((signals.data[j][0], signals.uncertainties[j][0]))
                    # [0] because obs.dt is one datetime, only one dt
                    sps[j].append(sp)
                    mpas[j].append(mpa)
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
                tot_rel_diff = tot_abs_rel_diff = tot_perc_diff = 0
                num_samples = len(specs[0].wlens)
                perc_diffs = []
                uncs_p = []
                for j in range(num_samples):
                    sim = specs[1].data[j]
                    ref = specs[0].data[j]
                    rel_dif = 100 * (sim - ref) / ref
                    tot_rel_diff += rel_dif
                    tot_abs_rel_diff += abs(rel_dif)
                    rel_diffs.append(rel_dif)
                    perc_diff = 100 * abs(sim - ref) / ((sim + ref) / 2)
                    tot_perc_diff += perc_diff
                    perc_diffs.append(perc_diff)
                    unc_r = unc_p = 0
                    if not lime_simulation.is_skipping_uncs():
                        unc_sim = specs[1].uncertainties[j]
                        unc_ref = specs[0].uncertainties[j]
                        if sim > ref:
                            rel_dif1 = (
                                100 * (sim + unc_sim - ref - unc_ref) / (ref - unc_ref)
                            )
                            perc_dif1 = (
                                100
                                * abs(sim + unc_sim - ref - unc_ref)
                                / ((sim + unc_sim + ref - unc_ref) / 2)
                            )
                        else:
                            rel_dif1 = (
                                100 * (sim - unc_sim - ref + unc_ref) / (ref + unc_ref)
                            )
                            perc_dif1 = (
                                100
                                * abs(sim - unc_sim - ref + unc_ref)
                                / ((sim - unc_sim + ref + unc_ref) / 2)
                            )
                        unc_r = abs(rel_dif - rel_dif1)
                        unc_p = abs(perc_diff - perc_dif1)
                    uncs_r.append(unc_r)
                    uncs_p.append(unc_p)
                    # i dont know if this is the correct way of propagating the uncertainty
                mean_rel_diff = tot_rel_diff / num_samples
                mean_abs_rel_diff = tot_abs_rel_diff / num_samples
                mean_perc_diff = tot_perc_diff / num_samples
                std = np.std(rel_diffs)
                ratio_spec = SpectralData(
                    specs[0].wlens, np.array(rel_diffs), np.array(uncs_r), None
                )
                perc_spec = SpectralData(
                    specs[0].wlens, np.array(perc_diffs), np.array(uncs_p), None
                )
                ampa_valid_range = [is_ampa_valid_range(abs(mpa)) for mpa in mpas[i]]
                cp = ComparisonData(
                    specs[0],
                    specs[1],
                    ratio_spec,
                    mean_rel_diff,
                    mean_abs_rel_diff,
                    std,
                    num_samples,
                    ch_dates[i],
                    sps[i],
                    mpas[i],
                    ampa_valid_range,
                    perc_spec,
                    mean_perc_diff,
                )
                comparisons.append(cp)
            else:
                comparisons.append(
                    ComparisonData(
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        [],
                        [],
                        [],
                        [],
                        None,
                        None,
                    )
                )
        return comparisons

    def sort_by_mpa(self, comparisons: List[ComparisonData]) -> List[ComparisonData]:
        new_comparisons = []
        for c in comparisons:
            if c.observed_signal == None:
                new_comparisons.append(c)
                continue
            spectrals = [
                c.observed_signal,
                c.simulated_signal,
                c.diffs_signal,
                c.perc_diffs,
            ]
            sp_vals = []
            for spectr in spectrals:
                sp_vals.append(spectr.wlens)
                sp_vals.append(spectr.data)
                sp_vals.append(spectr.uncertainties)
            vals = list(zip(*sp_vals, c.dts, c.points, c.mpas))
            vals.sort(key=lambda v: v[-1])
            mpas = [v[-1] for v in vals]
            ampa_valid_range = [is_ampa_valid_range(abs(mpa)) for mpa in mpas]
            new_spectrals = []
            index = 0
            for i, spectr in enumerate(spectrals):
                index = i * 3
                wlens = np.array(mpas)  # [v[index] for v in vals]
                data = np.array([v[index + 1] for v in vals])
                uncertainties = np.array([v[index + 2] for v in vals])
                new_spectrals.append(SpectralData(wlens, data, uncertainties, None))
            mrd = c.mean_relative_difference
            mard = c.mean_absolute_relative_difference
            mpd = c.mean_perc_difference
            std = c.standard_deviation_mrd
            nsamp = c.number_samples
            dts = [v[-3] for v in vals]
            points = [v[-2] for v in vals]
            nc = ComparisonData(
                new_spectrals[0],
                new_spectrals[1],
                new_spectrals[2],
                mrd,
                mard,
                std,
                nsamp,
                dts,
                points,
                mpas,
                ampa_valid_range,
                new_spectrals[3],
                mpd,
            )
            new_comparisons.append(nc)
        return new_comparisons
