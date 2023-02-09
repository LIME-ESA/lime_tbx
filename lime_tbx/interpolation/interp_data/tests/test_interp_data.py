"""Tests for classname module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME TBX Modules___"""
from .. import interp_data as ipd

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "26/01/2023"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class TestAccessData(unittest.TestCase):
    def test_get_asd_data_ok(self):
        asd_data = ipd.get_best_asd_data(19)
        self.assertIsNotNone(asd_data)

    def test_get_asd_data_not_mpa_absolute(self):
        asd_data_neg = ipd.get_best_asd_data(-80)
        asd_data_pos = ipd.get_best_asd_data(80)
        self.assertTrue(np.not_equal(asd_data_neg.data, asd_data_pos.data).any())

    def test_get_asd_data_consistent(self):
        asd_data0 = ipd.get_best_asd_data(7.2)
        asd_data1 = ipd.get_best_asd_data(7.2)
        self.assertIsNotNone(asd_data0)
        np.testing.assert_array_equal(asd_data0.data, asd_data1.data)

    def test_get_apollo_data(self):
        apdata = ipd.get_apollo16_data()
        self.assertEqual(apdata.data[0], 0.07254)  # First
        self.assertEqual(apdata.data[24], 0.10901)  # 420
        self.assertEqual(apdata.data[40], 0.13049)  # 500
        self.assertEqual(apdata.data[-1], 0.35048)  # Last

    def test_get_breccia_data(self):
        brdata = ipd.get_breccia_data()
        self.assertEqual(brdata.data[0], 0.314064)  # First
        self.assertEqual(brdata.data[12], 0.41268)  # 410.589
        self.assertEqual(brdata.data[18], 0.449469)  # 503.017
        self.assertEqual(brdata.data[-1], 0.626718)  # Last

    def test_get_composite_data(self):
        codata = ipd.get_composite_data()
        self.assertEqual(codata.data[0], 0.09966)  # First
        self.assertEqual(codata.data[700], 0.12451)  # 420.00
        self.assertEqual(codata.data[1500], 0.14640)  # 500.00
        self.assertEqual(codata.data[-1], 0.36427)  # Last

    def test_change_spectrum_persistency(self):
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_ASD)
        self.assertEqual(ipd.get_interpolation_spectrum_name(), ipd.SPECTRUM_NAME_ASD)
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_APOLLO16)
        self.assertEqual(
            ipd.get_interpolation_spectrum_name(), ipd.SPECTRUM_NAME_APOLLO16
        )
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_BRECCIA)
        self.assertEqual(
            ipd.get_interpolation_spectrum_name(), ipd.SPECTRUM_NAME_BRECCIA
        )
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_COMPOSITE)
        self.assertEqual(
            ipd.get_interpolation_spectrum_name(), ipd.SPECTRUM_NAME_COMPOSITE
        )
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_ASD)
        self.assertEqual(ipd.get_interpolation_spectrum_name(), ipd.SPECTRUM_NAME_ASD)

    def test_can_perform_polarization_only_asd(self):
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_APOLLO16)
        self.assertFalse(ipd.can_perform_polarization())
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_BRECCIA)
        self.assertFalse(ipd.can_perform_polarization())
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_COMPOSITE)
        self.assertFalse(ipd.can_perform_polarization())
        ipd.set_interpolation_spectrum_name(ipd.SPECTRUM_NAME_ASD)
        self.assertTrue(ipd.can_perform_polarization())

    def test_get_available_spectra_names(self):
        names = ipd.get_available_spectra_names()
        t_names = ["ASD", "Apollo 16", "Breccia", "Apollo 16 + Breccia"]
        self.assertEqual(names, t_names)


if __name__ == "__main__":
    unittest.main()
