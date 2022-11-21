"""Tests for comparison module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import unittest
import numpy as np

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
)
from ....coefficients.access_data.access_data import _get_default_cimel_coeffs
from ...lime_simulation import LimeSimulation, ILimeSimulation

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "25/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = "./eocfi_data"

CH_WLENS = np.array([350, 400, 450, 500])
CH_SRF = np.array([0.2, 0.2, 0.3, 0.3])
CH_ELIS = np.array([0.005, 0.0002, 0.3, 0.0001])
SP = SatellitePosition(3753240, -196698.975, 5138362)
DT1 = datetime(2022, 1, 17, 2, tzinfo=timezone.utc)
OBS1 = LunarObservation(["default"], "ITRF93", {"Default": 0.000001}, DT1, SP)
SATELLITE_POINT = SatellitePoint("BIOMASS", DT1)


def get_comparison() -> comparison.IComparison:
    return comparison.Comparison(KERNELS_PATH)


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_WLENS[i]: CH_SRF[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "Default", spectral_response)
    return SpectralResponseFunction("default", [ch])


def get_cimel_coeffs() -> ReflectanceCoefficients:
    return _get_default_cimel_coeffs()


def get_lime_simulation() -> ILimeSimulation:
    return LimeSimulation(EOCFI_PATH, KERNELS_PATH, verbose=False)


class TestComparison(unittest.TestCase):
    # Function to_llh
    def test_to_llh_ok(self):
        lat, lon, h = comparison.to_llh(3196669.145, 3196669.145, 4490530.389)
        self.assertAlmostEqual(lat, 45)
        self.assertAlmostEqual(lon, 45)
        self.assertAlmostEqual(h, 4500, delta=4)

    def test_to_llh_diff_vals_ok(self):
        lat, lon, h = comparison.to_llh(3753240, -196698.975, 5138362)
        self.assertAlmostEqual(lat, 54)
        self.assertAlmostEqual(lon, -3)
        self.assertAlmostEqual(h, 2000, delta=4)

    # Function to_xyz
    def test_to_xyz_ok(self):
        x, y, z = comparison.to_xyz(45, 45, 4500)
        self.assertAlmostEqual(x, 3196669.145, delta=4)
        self.assertAlmostEqual(y, 3196669.145, delta=4)
        self.assertAlmostEqual(z, 4490530.389, delta=4)

    def test_to_xyz_diff_vals_ok(self):
        x, y, z = comparison.to_xyz(54, -3, 2000)
        self.assertAlmostEqual(x, 3753240, delta=4)
        self.assertAlmostEqual(y, -196698.975, delta=4)
        self.assertAlmostEqual(z, 5138362, delta=4)

    # Function get_simulations
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


if __name__ == "__main__":
    unittest.main()
