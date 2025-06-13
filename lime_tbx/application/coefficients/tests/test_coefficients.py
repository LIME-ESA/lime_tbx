"""Tests for coefficients integration"""

"""___Built-In Modules___"""
import os

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.application.filedata import coefficients
from lime_tbx.business.lime_algorithms.lime import lime
from lime_tbx.business.lime_algorithms.polar import dolp
from lime_tbx.common.datatypes import MoonData

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "04/12/2024"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_6WL_COEFFS = os.path.join(_CURRENT_DIR, "assets/LIME_MODEL_COEFS_20231120_V01.nc")
PATH_7WL_COEFFS_2130 = os.path.join(_CURRENT_DIR, "assets/SEVENWLENS_COEFFS.nc")
MD1 = MoonData(0.999998, 388499, 1.1, 15, -16, 11, -11)


class TestCoefficients(unittest.TestCase):
    # If something fails check that the lime toolbox is NOT installed in the machine

    def test_run_lime_6wlens_standard(self):
        lc = coefficients.read_coeff_nc(PATH_6WL_COEFFS)
        elref = lime.LIME().get_elrefs(lc.reflectance, MD1)
        self.assertEqual(elref.wlens.shape, (6,))
        self.assertEqual(elref.data.shape, (6,))
        self.assertEqual(elref.uncertainties.shape, (6,))
        self.assertTrue(np.all(elref.uncertainties != 0))
        self.assertEqual(elref.err_corr.shape, (6, 6))
        eli = lime.LIME().get_elis_from_elrefs(elref, MD1, "cimel")
        self.assertEqual(eli.wlens.shape, (6,))
        self.assertEqual(eli.data.shape, (6,))
        self.assertEqual(eli.uncertainties.shape, (6,))
        self.assertTrue(np.all(eli.uncertainties != 0))
        self.assertIsNone(eli.err_corr)
        pol = dolp.DOLP().get_polarized(MD1.mpa_degrees, lc.polarisation)
        self.assertEqual(pol.wlens.shape, (6,))
        self.assertEqual(pol.data.shape, (6,))
        self.assertEqual(pol.uncertainties.shape, (6,))
        self.assertTrue(np.all(pol.uncertainties != 0))
        self.assertEqual(pol.err_corr.shape, (6, 6))

    def test_run_lime_6wlens_standard_skip_uncs(self):
        lc = coefficients.read_coeff_nc(PATH_6WL_COEFFS)
        elref = lime.LIME().get_elrefs(lc.reflectance, MD1, True)
        self.assertEqual(elref.wlens.shape, (6,))
        self.assertEqual(elref.data.shape, (6,))
        self.assertEqual(elref.uncertainties.shape, (6,))
        self.assertTrue(np.all(elref.uncertainties == 0))
        self.assertIsNone(elref.err_corr)
        eli = lime.LIME().get_elis_from_elrefs(elref, MD1, "cimel", True)
        self.assertEqual(eli.wlens.shape, (6,))
        self.assertEqual(eli.data.shape, (6,))
        self.assertEqual(eli.uncertainties.shape, (6,))
        self.assertTrue(np.all(eli.uncertainties == 0))
        self.assertIsNone(eli.err_corr)
        pol = dolp.DOLP().get_polarized(MD1.mpa_degrees, lc.polarisation, True)
        self.assertEqual(pol.wlens.shape, (6,))
        self.assertEqual(pol.data.shape, (6,))
        self.assertEqual(pol.uncertainties.shape, (6,))
        self.assertTrue(np.all(pol.uncertainties == 0))
        self.assertIsNone(pol.err_corr)

    def test_run_lime_7wlens_2130(self):
        lc = coefficients.read_coeff_nc(PATH_7WL_COEFFS_2130)
        elref = lime.LIME().get_elrefs(lc.reflectance, MD1)
        self.assertEqual(elref.wlens.shape, (7,))
        self.assertEqual(elref.data.shape, (7,))
        self.assertEqual(elref.uncertainties.shape, (7,))
        self.assertTrue(np.all(elref.uncertainties != 0))
        self.assertEqual(elref.err_corr.shape, (7, 7))
        eli = lime.LIME().get_elis_from_elrefs(elref, MD1, "cimel")
        self.assertEqual(eli.wlens.shape, (7,))
        self.assertEqual(eli.data.shape, (7,))
        self.assertEqual(eli.uncertainties.shape, (7,))
        self.assertTrue(np.all(eli.uncertainties != 0))
        self.assertIsNone(eli.err_corr)
        pol = dolp.DOLP().get_polarized(MD1.mpa_degrees, lc.polarisation)
        self.assertEqual(pol.wlens.shape, (7,))
        self.assertEqual(pol.data.shape, (7,))
        self.assertEqual(pol.uncertainties.shape, (7,))
        self.assertTrue(np.all(pol.uncertainties != 0))
        self.assertEqual(pol.err_corr.shape, (7, 7))

    def test_run_lime_7wlens_2130_skip_uncs(self):
        lc = coefficients.read_coeff_nc(PATH_7WL_COEFFS_2130)
        elref = lime.LIME().get_elrefs(lc.reflectance, MD1, True)
        self.assertEqual(elref.wlens.shape, (7,))
        self.assertEqual(elref.data.shape, (7,))
        self.assertEqual(elref.uncertainties.shape, (7,))
        self.assertTrue(np.all(elref.uncertainties == 0))
        self.assertIsNone(elref.err_corr)
        eli = lime.LIME().get_elis_from_elrefs(elref, MD1, "cimel", True)
        self.assertEqual(eli.wlens.shape, (7,))
        self.assertEqual(eli.data.shape, (7,))
        self.assertEqual(eli.uncertainties.shape, (7,))
        self.assertTrue(np.all(eli.uncertainties == 0))
        self.assertIsNone(eli.err_corr)
        pol = dolp.DOLP().get_polarized(MD1.mpa_degrees, lc.polarisation, True)
        self.assertEqual(pol.wlens.shape, (7,))
        self.assertEqual(pol.data.shape, (7,))
        self.assertEqual(pol.uncertainties.shape, (7,))
        self.assertTrue(np.all(pol.uncertainties == 0))
        self.assertIsNone(pol.err_corr)
