"""Tests for the esi module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from .. import esi

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

WLENS0 = np.array([350, 370, 390, 410, 430, 450])
WLENS1 = np.array([i for i in range(350, 2501)])
WLENS2 = np.array([i for i in range(0, 1000)])


class TestESI(unittest.TestCase):
    def test_esi_equal(self):
        esi0 = esi.get_esi_per_nms(WLENS0)
        esi1 = esi.get_esi_per_nms(WLENS0)
        np.testing.assert_array_equal(esi0, esi1)

    def test_esi_equal_diff_arrays(self):
        esi0 = esi.get_esi_per_nms(WLENS0)
        esi1 = esi.get_esi_per_nms(WLENS1)
        self.assertEqual(esi0[0], esi1[0])
        self.assertEqual(esi0[1], esi1[20])
        self.assertEqual(esi0[2], esi1[40])
        self.assertEqual(esi0[3], esi1[60])
        self.assertEqual(esi0[4], esi1[80])
        self.assertEqual(esi0[5], esi1[100])

    def test_esi_outrange_ok(self):
        esi0 = esi.get_esi_per_nms(WLENS2)
        self.assertGreater(esi0[0], 0)
        self.assertEqual(esi0[0], esi0[1])

    def test_esi_get_wehrli_data_ok(self):
        wehrli_first_vals = {
            330.5: (1.006, 332.483),
            331.5: (0.9676, 320.7594),
            332.5: (0.9207, 306.13275),
            333.5: (0.9047, 301.71745),
            334.5: (0.9397, 314.32965),
            335.5: (0.9816, 329.3268),
            336.5: (0.7649, 257.38885),
            337.5: (0.8658, 292.2075),
            338.5: (0.9157, 309.96445),
            339.5: (0.9367, 318.00964999999997),
            340.5: (0.9916, 337.63980000000004),
        }
        wehrli = esi._get_wehrli_data()
        for wlen in wehrli_first_vals:
            self.assertEqual(wehrli_first_vals[wlen], wehrli[wlen])


if __name__ == "__main__":
    unittest.main()
