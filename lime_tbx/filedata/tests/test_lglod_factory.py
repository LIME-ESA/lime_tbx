"Tests for the lglod_factory module"

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from .. import lglod_factory
from lime_tbx.simulation.lime_simulation import LimeSimulation
from lime_tbx.datatypes.datatypes import (
    KernelsPath,
    SurfacePoint,
    SatellitePoint,
    CustomPoint,
)
from lime_tbx.spectral_integration.spectral_integration import get_default_srf
from lime_tbx.lime_algorithms.lime.tests.test_elref import get_coeffs
from lime_tbx.lime_algorithms.dolp.tests.test_dolp import POL_COEFFS
from lime_tbx.gui.settings import SettingsManager

EOCFI_PATH = "eocfi_data"
KERNELS_PATH = KernelsPath("kernels", "kernels")


def get_lime_simulation() -> LimeSimulation:
    return LimeSimulation(EOCFI_PATH, KERNELS_PATH, SettingsManager())


class TestLGLODFactory(unittest.TestCase):
    def test_surface_point_ok(self):
        ls = get_lime_simulation()
        srf = get_default_srf()
        pt = SurfacePoint(
            40, 40, 400, datetime(2000, 1, 1, 1, 1, 1, tzinfo=timezone.utc)
        )
        ls.update_irradiance(srf, srf, pt, get_coeffs())
        ls.update_polarisation(srf, pt, POL_COEFFS)
        mdas = ls.get_moon_datas()
        lglod = lglod_factory.create_lglod_data(
            pt,
            srf,
            ls,
            KERNELS_PATH,
            "test",
            "testp",
            "test",
            mdas,
        )
        self.assertEqual(lglod.spectrum_name, "test")
        self.assertEqual(lglod.dolp_spectrum_name, "testp")
        np.testing.assert_array_equal(
            lglod.elis_cimel[0].data, ls.get_elis_cimel().data
        )

    def test_satellite_point_ok(self):
        ls = get_lime_simulation()
        srf = get_default_srf()
        pt = SatellitePoint(
            "ENVISAT", datetime(2005, 10, 10, 10, 10, 10, tzinfo=timezone.utc)
        )
        ls.update_irradiance(srf, srf, pt, get_coeffs())
        ls.update_polarisation(srf, pt, POL_COEFFS)
        mdas = ls.get_moon_datas()
        lglod = lglod_factory.create_lglod_data(
            pt,
            srf,
            ls,
            KERNELS_PATH,
            "test",
            "testp",
            "test",
            mdas,
        )
        self.assertEqual(lglod.spectrum_name, "test")
        self.assertEqual(lglod.dolp_spectrum_name, "testp")
        np.testing.assert_array_equal(
            lglod.elis_cimel[0].data, ls.get_elis_cimel().data
        )

    def test_custom_point_ok(self):
        ls = get_lime_simulation()
        srf = get_default_srf()
        pt = CustomPoint(1, 400000, 30, 30, 1, 40, -40)
        ls.update_irradiance(srf, srf, pt, get_coeffs())
        ls.update_polarisation(srf, pt, POL_COEFFS)
        mdas = ls.get_moon_datas()
        lglod = lglod_factory.create_lglod_data(
            pt,
            srf,
            ls,
            KERNELS_PATH,
            "test",
            "testp",
            "test",
            mdas,
        )
        self.assertEqual(lglod.spectrum_name, "test")
        self.assertEqual(lglod.dolp_spectrum_name, "testp")
        np.testing.assert_array_equal(
            lglod.elis_cimel[0].data, ls.get_elis_cimel().data
        )


if __name__ == "__main__":
    unittest.main()
