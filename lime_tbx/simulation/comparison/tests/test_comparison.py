"""Tests for comparison module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import random
from typing import Tuple

"""___Third-Party Modules___"""
import unittest
import numpy as np
import pytest

"""___LIME_TBX Modules___"""
from .. import comparison
from ....datatypes.datatypes import (
    ComparisonData,
    LunarObservation,
    SatellitePosition,
    SpectralResponseFunction,
    SRFChannel,
    ReflectanceCoefficients,
    SatellitePoint,
    KernelsPath,
    SpectralData,
)
from ....coefficients.access_data.access_data import _get_demo_cimel_coeffs
from ...lime_simulation import LimeSimulation, ILimeSimulation
from lime_tbx.interpolation.interp_data import interp_data
from lime_tbx.gui.settings import SettingsManager

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "25/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Production"

KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = "./eocfi_data"

CH_WLENS = np.array([350, 400, 450, 500])
CH_SRF = np.array([0.2, 0.2, 0.3, 0.3])
CH_ELIS = np.array([0.005, 0.0002, 0.3, 0.0001])
SP = SatellitePosition(3753240, -196698.975, 5138362)
DT1 = datetime(2022, 1, 17, 2, tzinfo=timezone.utc)
DT2 = datetime(2022, 2, 16, 2, tzinfo=timezone.utc)
OBS1 = LunarObservation(
    ["default"], "ITRF93", {"Default": 0.000001}, DT1, SP, "Test", None
)
OBS2 = LunarObservation(
    ["first", "second"],
    "ITRF93",
    {"First": 0.000001, "Second": 0.000002},
    DT1,
    SP,
    "Test",
    None,
)
OBS3 = LunarObservation(
    ["first", "second"],
    "ITRF93",
    {"First": 0.0000012, "Second": 0.0000025},
    DT2,
    SP,
    "Test",
    None,
)
SATELLITE_POINT = SatellitePoint("BIOMASS", DT1)


def get_comparison() -> comparison.IComparison:
    return comparison.Comparison(KERNELS_PATH)


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_WLENS[i]: CH_SRF[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "Default", spectral_response)
    return SpectralResponseFunction("default", [ch])


def get_second_srf() -> SpectralResponseFunction:
    half = len(CH_SRF) // 2
    spectral_response1 = {CH_WLENS[i]: CH_SRF[i] for i in range(half)}
    ch1 = SRFChannel((CH_WLENS[half] - CH_WLENS[0]) / 2, "First", spectral_response1)
    spectral_response2 = {CH_WLENS[i]: CH_SRF[i] for i in range(half, len(CH_SRF))}
    ch2 = SRFChannel((CH_WLENS[-1] - CH_WLENS[half]) / 2, "Second", spectral_response2)
    return SpectralResponseFunction("default", [ch1, ch2])


def get_cimel_coeffs() -> ReflectanceCoefficients:
    return _get_demo_cimel_coeffs()


def get_lime_simulation() -> ILimeSimulation:
    interp_data.set_interpolation_spectrum_name("ASD")
    return LimeSimulation(EOCFI_PATH, KERNELS_PATH, SettingsManager(), verbose=False)


def get_random_spectral_data_dts(
    n_elems: int = 30,
) -> Tuple[SpectralData, SpectralData]:
    dts = list(
        map(datetime.fromtimestamp, sorted(random.sample(range(1700000000), n_elems)))
    )
    data = [random.random() for _ in range(n_elems)]
    uncerts = [random.random() / 10 for _ in range(n_elems)]
    data2 = [random.random() for _ in range(n_elems)]
    uncerts2 = [random.random() / 10 for _ in range(n_elems)]
    s0 = SpectralData(dts, data, uncerts, None)
    s1 = SpectralData(dts, data2, uncerts2, None)
    return s0, s1


class TestComparison(unittest.TestCase):
    # Function to_llh
    def test_to_llh_ok(self):
        with self.assertWarns(DeprecationWarning):
            lat, lon, h = comparison._to_llh(3196669.145, 3196669.145, 4490530.389)
        self.assertAlmostEqual(lat, 45)
        self.assertAlmostEqual(lon, 45)
        self.assertAlmostEqual(h, 4500, delta=4)

    def test_to_llh_diff_vals_ok(self):
        with self.assertWarns(DeprecationWarning):
            lat, lon, h = comparison._to_llh(3753240, -196698.975, 5138362)
        self.assertAlmostEqual(lat, 54)
        self.assertAlmostEqual(lon, -3)
        self.assertAlmostEqual(h, 2000, delta=4)

    # Function to_xyz
    def test_to_xyz_ok(self):
        with self.assertWarns(DeprecationWarning):
            x, y, z = comparison._to_xyz(45, 45, 4500)
        self.assertAlmostEqual(x, 3196669.145, delta=4)
        self.assertAlmostEqual(y, 3196669.145, delta=4)
        self.assertAlmostEqual(z, 4490530.389, delta=4)

    def test_to_xyz_diff_vals_ok(self):
        with self.assertWarns(DeprecationWarning):
            x, y, z = comparison._to_xyz(54, -3, 2000)
        self.assertAlmostEqual(x, 3753240, delta=4)
        self.assertAlmostEqual(y, -196698.975, delta=4)
        self.assertAlmostEqual(z, 5138362, delta=4)

    # Function get_simulations
    @pytest.mark.slow
    def test_get_simulations_ok(self):
        co = get_comparison()
        lime = get_lime_simulation()
        srf = get_srf()
        coeffs = get_cimel_coeffs()
        comps = co.get_simulations([OBS1], srf, coeffs, lime)
        self.assertEqual(len(comps), 1)
        comp: ComparisonData = comps[0]
        self.assertEqual(len(comp.dts), 1)
        self.assertEqual(comp.dts[0], OBS1.dt)
        self.assertEqual(comp.number_samples, 1)
        self.assertEqual(comp.observed_signal.data[0], OBS1.ch_irrs["Default"])
        self.assertAlmostEqual(comp.simulated_signal.data[0], 2.59881401749616e-06)
        self.assertAlmostEqual(
            comp.simulated_signal.uncertainties[0], 1.5069602623384797e-08
        )

    @pytest.mark.slow
    def test_get_simulations_multiple_obs_and_channels(self):
        co = get_comparison()
        lime = get_lime_simulation()
        srf = get_second_srf()
        coeffs = get_cimel_coeffs()
        comps = co.get_simulations([OBS2, OBS3], srf, coeffs, lime)
        self.assertEqual(len(comps), 2)
        comp: ComparisonData = comps[0]
        self.assertEqual(len(comp.dts), 2)
        self.assertEqual(comp.dts[0], OBS2.dt)
        self.assertEqual(comp.number_samples, 2)
        self.assertEqual(comp.observed_signal.data[0], OBS2.ch_irrs["First"])
        self.assertEqual(comp.observed_signal.data[1], OBS3.ch_irrs["First"])
        np.testing.assert_array_almost_equal(
            comp.simulated_signal.data, np.array([1.398473e-06, 1.547618e-06])
        )
        np.testing.assert_array_almost_equal(
            comp.simulated_signal.uncertainties, np.array([1.016344e-08, 1.118738e-08])
        )
        self.assertEqual(comps[1].observed_signal.data[0], OBS2.ch_irrs["Second"])
        self.assertEqual(comps[1].observed_signal.data[1], OBS3.ch_irrs["Second"])
        np.testing.assert_array_almost_equal(
            comps[1].simulated_signal.data, np.array([3.30473368e-06, 3.66841620e-06])
        )
        np.testing.assert_array_almost_equal(
            comps[1].simulated_signal.uncertainties,
            np.array([2.290940e-08, 2.594562e-08]),
        )

    @pytest.mark.slow
    def test_get_simulations_other_srf(self):
        # Wrong srf in comparisons output empty comparison data instances.
        co = get_comparison()
        lime = get_lime_simulation()
        srf = get_second_srf()
        coeffs = get_cimel_coeffs()
        comps = co.get_simulations([OBS1], srf, coeffs, lime)
        self.assertEqual(len(comps), 2)
        for comp in comps:
            self.assertEqual(len(comp.ampa_valid_range), 0)
            self.assertIsNone(comp.diffs_signal)
            self.assertEqual(len(comp.dts), 0)
            self.assertIsNone(comp.mean_relative_difference)
            self.assertEqual(len(comp.mdas), 0)
            self.assertIsNone(comp.number_samples)
            self.assertIsNone(comp.observed_signal)
            self.assertEqual(len(comp.points), 0)
            self.assertIsNone(comp.simulated_signal)
            self.assertIsNone(comp.standard_deviation_mrd)

    def test_get_simulations_empty(self):
        # No observations input outputs empty comparison data instances.
        co = get_comparison()
        lime = get_lime_simulation()
        srf = get_srf()
        coeffs = get_cimel_coeffs()
        comps = co.get_simulations([], srf, coeffs, lime)
        self.assertEqual(len(comps), 1)
        for comp in comps:
            self.assertEqual(len(comp.ampa_valid_range), 0)
            self.assertIsNone(comp.diffs_signal)
            self.assertEqual(len(comp.dts), 0)
            self.assertIsNone(comp.mean_relative_difference)
            self.assertEqual(len(comp.mdas), 0)
            self.assertIsNone(comp.number_samples)
            self.assertIsNone(comp.observed_signal)
            self.assertEqual(len(comp.points), 0)
            self.assertIsNone(comp.simulated_signal)
            self.assertIsNone(comp.standard_deviation_mrd)
