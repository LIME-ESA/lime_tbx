"""Tests for templates module"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import unittest
import xarray
import obsarray

"""___LIME_TBX Modules___"""
from ..templates_digital_effects_table import (
    TEMPLATE_POL,
    TEMPLATE_CIMEL,
    TEMPLATE_IRR,
    TEMPLATE_REFL,
    TEMPLATE_SIGNALS,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "29/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class TestTemplates(unittest.TestCase):
    def test_template_cimel(self):
        dim_sizes = {
            "wavelength": 6,
            "i_coeff": 18,
            "i_coeff.wavelength": 18 * 6,
        }
        ds: xarray.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)
        self.assertIsInstance(ds, xarray.Dataset)

    def test_template_refl(self):
        dim_sizes = {
            "wavelength": 6,
        }
        ds: xarray.Dataset = obsarray.create_ds(TEMPLATE_REFL, dim_sizes)
        self.assertIsInstance(ds, xarray.Dataset)

    def test_template_irr(self):
        dim_sizes = {
            "wavelength": 6,
        }
        ds: xarray.Dataset = obsarray.create_ds(TEMPLATE_IRR, dim_sizes)
        self.assertIsInstance(ds, xarray.Dataset)

    def test_template_signals(self):
        dim_sizes = {
            "channels": 6,
            "dts": 3,
        }
        ds: xarray.Dataset = obsarray.create_ds(TEMPLATE_SIGNALS, dim_sizes)
        self.assertIsInstance(ds, xarray.Dataset)

    def test_template_pol(self):
        dim_sizes = {
            "wavelength": 6,
        }
        ds: xarray.Dataset = obsarray.create_ds(TEMPLATE_POL, dim_sizes)
        self.assertIsInstance(ds, xarray.Dataset)


if __name__ == "__main__":
    unittest.main()
