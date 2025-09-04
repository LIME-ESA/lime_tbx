"Tests for the lglod_factory module"

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from .. import lglod_factory, lglod as lglodlib, srf as srflib
from lime_tbx.application.simulation.lime_simulation import LimeSimulation
from lime_tbx.common.datatypes import (
    KernelsPath,
    SurfacePoint,
    SatellitePoint,
    CustomPoint,
    MultipleCustomPoint,
    EocfiPath,
    LGLODData,
)
from lime_tbx.business.spectral_integration.spectral_integration import get_default_srf
from lime_tbx.business.lime_algorithms.lime.tests.test_elref import get_coeffs
from lime_tbx.business.lime_algorithms.polar.tests.test_dolp import POL_COEFFS
from lime_tbx.presentation.gui.settings import SettingsManager

EOCFI_PATH = EocfiPath("eocfi_data", "eocfi_data2")
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
            "testa",
            "test",
            mdas,
        )
        self.assertEqual(lglod.spectrum_name, "test")
        self.assertEqual(lglod.dolp_spectrum_name, "testp")
        self.assertEqual(lglod.aolp_spectrum_name, "testa")
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
            "testa",
            "test",
            mdas,
        )
        self.assertEqual(lglod.spectrum_name, "test")
        self.assertEqual(lglod.dolp_spectrum_name, "testp")
        self.assertEqual(lglod.aolp_spectrum_name, "testa")
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
            "testa",
            "test",
            mdas,
        )
        self.assertEqual(lglod.spectrum_name, "test")
        self.assertEqual(lglod.dolp_spectrum_name, "testp")
        self.assertEqual(lglod.aolp_spectrum_name, "testa")
        np.testing.assert_array_equal(
            lglod.elis_cimel[0].data, ls.get_elis_cimel().data
        )

    def test_multiple_custom_point_ok(self):
        ls = get_lime_simulation()
        srf = get_default_srf()
        pts = [
            CustomPoint(1, 400000, 30, 30, 1, 40, -40),
            CustomPoint(1.00002, 400001, 32, 31, 2, 41, -41),
        ]
        pt = MultipleCustomPoint(pts)
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
            "testa",
            "test",
            mdas,
        )
        self.assertEqual(lglod.spectrum_name, "test")
        self.assertEqual(lglod.dolp_spectrum_name, "testp")
        self.assertEqual(lglod.aolp_spectrum_name, "testa")
        np.testing.assert_array_equal(
            lglod.elis_cimel[0].data, ls.get_elis_cimel()[0].data
        )
        np.testing.assert_array_equal(
            lglod.elis_cimel[1].data, ls.get_elis_cimel()[1].data
        )


class TestLGLODFactoryLGLODIntegration(unittest.TestCase):
    def assert_eq_lglod(self, a: LGLODData, b: LGLODData):
        self.assertEqual(len(a.elis_cimel), len(b.elis_cimel))
        for i, (aec, bec) in enumerate(zip(a.elis_cimel, b.elis_cimel)):
            np.testing.assert_array_equal(
                aec.data,
                bec.data,
                f"Mismatch in elis_cimel.data at {i}.\n{aec.data}\n{bec.data}",
            )
            np.testing.assert_array_equal(
                aec.wlens,
                bec.wlens,
                f"Mismatch in elis_cimel.wlens at {i}\n{aec.wlens}\n{bec.wlens}",
            )
        np.testing.assert_array_equal(a.signals.wlens, b.signals.wlens)
        np.testing.assert_array_equal(a.signals.data, b.signals.data)

    def test_createlglod_write_read_write_multiseleno(self):
        ls = get_lime_simulation()
        srf = get_default_srf()
        pts = [
            CustomPoint(1, 400000, 30, 30, 1, 40, -40),
            CustomPoint(1.00002, 400001, 32, 31, 2, 41, -41),
        ]
        pt = MultipleCustomPoint(pts)
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
            "testa",
            "test",
            mdas,
        )
        path = "./test_files/moon/sim_multiseleno.e2e.test.nc"
        now = datetime.now(timezone.utc)
        inside_mpa_range = [True for _ in range(len(lglod.elis_cimel))]
        lglodlib.write_obs(lglod, path, now, inside_mpa_range)
        rlglod = lglodlib.read_lglod_file(path, KERNELS_PATH)
        self.assert_eq_lglod(lglod, rlglod)
        path = "./test_files/moon/sim_multiseleno2.e2e.test.nc"
        lglodlib.write_obs(rlglod, path, now, inside_mpa_range)
        lglod = lglodlib.read_lglod_file(path, KERNELS_PATH)
        self.assert_eq_lglod(rlglod, lglod)

    def test_createlglod_write_read_write_multiseleno_cimel(self):
        ls = get_lime_simulation()
        srf = get_default_srf()
        sigsrf = srflib.read_srf("./test_files/moon/cimel_srf.nc")
        pts = [
            CustomPoint(1, 400000, 30, 30, 1, 40, -40),
            CustomPoint(1.00002, 400001, 32, 31, 2, 41, -41),
        ]
        pt = MultipleCustomPoint(pts)
        ls.update_irradiance(srf, sigsrf, pt, get_coeffs())
        ls.update_polarisation(srf, pt, POL_COEFFS)
        mdas = ls.get_moon_datas()
        lglod = lglod_factory.create_lglod_data(
            pt,
            sigsrf,
            ls,
            KERNELS_PATH,
            "test",
            "testp",
            "testa",
            "test",
            mdas,
        )
        path = "./test_files/moon/sim_multiseleno_cimel.e2e.test.nc"
        now = datetime.now(timezone.utc)
        inside_mpa_range = [True for _ in range(len(lglod.elis_cimel))]
        lglodlib.write_obs(lglod, path, now, inside_mpa_range)
        rlglod = lglodlib.read_lglod_file(path, KERNELS_PATH)
        self.assert_eq_lglod(lglod, rlglod)
        path = "./test_files/moon/sim_multiseleno_cimel2.e2e.test.nc"
        lglodlib.write_obs(rlglod, path, now, inside_mpa_range)
        lglod = lglodlib.read_lglod_file(path, KERNELS_PATH)
        self.assert_eq_lglod(rlglod, lglod)


if __name__ == "__main__":
    unittest.main()
