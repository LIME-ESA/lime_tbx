"""Tests for lime_simulation module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import sys
import io

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from ..lime_simulation import ILimeSimulation, LimeSimulation
from ...datatypes.datatypes import (
    PolarizationCoefficients,
    ReflectanceCoefficients,
    SpectralData,
    SpectralResponseFunction,
    SRFChannel,
    SurfacePoint,
    KernelsPath,
    LGLODData,
)
from ...coefficients.access_data.access_data import (
    _get_default_polarization_coefficients,
    _get_demo_cimel_coeffs,
)
from ...filedata import moon, srf as srflib

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

LAT = 21
LON = 21
ALT = 2400
DT1 = datetime(2022, 1, 17, 2, tzinfo=timezone.utc)
DT2 = datetime(2022, 2, 16, 2, tzinfo=timezone.utc)

SURFACE_POINT = SurfacePoint(LAT, LON, ALT, [DT1, DT2])


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_WLENS[i]: CH_SRF[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "Default", spectral_response)
    return SpectralResponseFunction("default", [ch])


def get_cimel_coeffs() -> ReflectanceCoefficients:
    return _get_demo_cimel_coeffs()


def get_polar_coeffs() -> PolarizationCoefficients:
    return _get_default_polarization_coefficients()


def get_lime_simulation() -> ILimeSimulation:
    return LimeSimulation(EOCFI_PATH, KERNELS_PATH, verbose=False)


class TestLimeSimulation(unittest.TestCase):

    # Function set_simulation_changed
    def test_set_simulation_changed_ok(self):
        ls = get_lime_simulation()
        ls.set_simulation_changed()

    def test_set_simulation_changed_multiple_times(self):
        ls = get_lime_simulation()
        for _ in range(6):
            ls.set_simulation_changed()

    # Function update_reflectance
    def test_update_reflectance(self):
        ls = get_lime_simulation()
        ls.update_reflectance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)

    # Function update_irradiance
    def test_update_irradiance(self):
        ls = get_lime_simulation()
        ls.update_irradiance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        elis = ls.get_elis()
        self.assertIsNotNone(elis)
        self.assertIsInstance(elis, list)
        signals = ls.get_signals()
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, SpectralData)

    # Function update_polarization
    def test_update_polarization(self):
        ls = get_lime_simulation()
        ls.update_polarization(get_srf(), SURFACE_POINT, get_polar_coeffs())
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)

    def test_update_irr_polar_verbose(self):
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr

        ls = LimeSimulation(EOCFI_PATH, KERNELS_PATH, verbose=True)
        ls.update_irradiance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        ls.update_polarization(get_srf(), SURFACE_POINT, get_polar_coeffs())
        elis = ls.get_elis()
        self.assertIsNotNone(elis)
        self.assertIsInstance(elis, list)
        signals = ls.get_signals()
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, SpectralData)
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def test_update_get_point(self):
        ls = get_lime_simulation()
        ls.update_reflectance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        pt = ls.get_point()
        self.assertEqual(pt, SURFACE_POINT)

    def test_load_lglod(self):
        ls = get_lime_simulation()
        lglod: LGLODData = moon.read_lglod_file(
            "test_files/moon/simulation.nc", KERNELS_PATH
        )
        srf = srflib.read_srf(
            "test_files/srf/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc"
        )
        ls.set_observations(lglod, srf)
        ls.get_elis()
        np.testing.assert_array_equal(
            lglod.observations[0].irrs.data, ls.get_elis()[0].data
        )


if __name__ == "__main__":
    unittest.main()
