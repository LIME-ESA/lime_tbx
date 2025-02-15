"""Tests for moon module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    KernelsPath,
    SatellitePosition,
    EocfiPath,
)
from .. import moon
from ..srf import read_srf

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "31/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

PATH = "./test_files/moon/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140318140112_01.nc"

KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = EocfiPath("./eocfi_data", "./eocfi_data2")


class TestMoon(unittest.TestCase):
    def test_read_moon_obs_ok(self):
        lo = moon.read_moon_obs(PATH, KERNELS_PATH, EOCFI_PATH)
        ch_names = ["VIS006", "VIS008", "NIR016", "HRVIS"]
        ch_irrs = {
            "VIS006": 1.9233498386870263e-06,
            "VIS008": 1.6566640151377671e-06,
            "NIR016": 5.949228451947656e-07,
        }
        dt = datetime(2014, 3, 18, 14, 1, 12, 25, tzinfo=timezone.utc)
        sat_pos = SatellitePosition(
            42164810.38833844, -75054.8191222299, 66493.62502083843
        )
        for i, name in enumerate(lo.ch_names):
            self.assertEqual(name, ch_names[i])
        for irr in lo.ch_irrs:
            self.assertEqual(lo.ch_irrs[irr], ch_irrs[irr])
        self.assertEqual(lo.dt, dt)
        self.assertEqual(lo.sat_pos, sat_pos)
        self.assertEqual(lo.sat_pos_ref, "ITRF93")

    def test_read_moon_obs_check_srf(self):
        lo = moon.read_moon_obs(PATH, KERNELS_PATH, EOCFI_PATH)
        srf = read_srf(
            "./test_files/srf/W_XX-EUMETSAT-Darmstadt_VIS+IR+SRF_MSG3+SEVIRI_C_EUMG.nc"
        )
        self.assertTrue(lo.check_valid_srf(srf))


if __name__ == "__main__":
    unittest.main()
