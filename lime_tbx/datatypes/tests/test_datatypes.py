"""Tests for datatypes module"""

"""___Built-In Modules___"""
from datetime import datetime, timedelta

from lime_tbx.datatypes.templates_digital_effects_table import TEMPLATE_CIMEL

"""___Third-Party Modules___"""
import unittest
import numpy as np
import xarray
import obsarray

"""___LIME_TBX Modules___"""
from ..datatypes import (
    LunarObservation,
    OrbitFile,
    PolarizationCoefficients,
    ReflectanceCoefficients,
    SRFChannel,
    Satellite,
    SatellitePosition,
    SpectralData,
    SpectralResponseFunction,
    SpectralValidity,
)
from .. import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "29/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


MIN_WLEN = constants.MIN_WLEN
MAX_WLEN = constants.MAX_WLEN
SRF_CENTER = 450
SRF_ID = "SRFChannel"


class TestSRFChannel(unittest.TestCase):
    def test_creation_valid(self):
        srfc = SRFChannel(
            SRF_CENTER,
            SRF_ID,
            {MIN_WLEN + 1: 0.5, MAX_WLEN - 1: 0.5},
            MIN_WLEN,
            MAX_WLEN,
        )
        self.assertEqual(srfc.valid_spectre, SpectralValidity.VALID)

    def test_creation_low_equal_min(self):
        srfc = SRFChannel(
            SRF_CENTER, SRF_ID, {MIN_WLEN: 0.5, MAX_WLEN - 1: 0.5}, MIN_WLEN, MAX_WLEN
        )
        self.assertEqual(srfc.valid_spectre, SpectralValidity.VALID)

    def test_creation_high_equal_max(self):
        srfc = SRFChannel(
            SRF_CENTER, SRF_ID, {MIN_WLEN + 1: 0.5, MAX_WLEN: 0.5}, MIN_WLEN, MAX_WLEN
        )
        self.assertEqual(srfc.valid_spectre, SpectralValidity.VALID)

    def test_creation_partly(self):
        srfc = SRFChannel(
            SRF_CENTER,
            SRF_ID,
            {MIN_WLEN - 1: 0.5, MAX_WLEN - 1: 0.5},
            MIN_WLEN,
            MAX_WLEN,
        )
        self.assertEqual(srfc.valid_spectre, SpectralValidity.PARTLY_OUT)

    def test_creation_out(self):
        srfc = SRFChannel(
            SRF_CENTER,
            SRF_ID,
            {MIN_WLEN - 2: 0.5, MIN_WLEN - 1: 0.5},
            MIN_WLEN,
            MAX_WLEN,
        )
        self.assertEqual(srfc.valid_spectre, SpectralValidity.OUT)


class TestSpectralResponseFunction(unittest.TestCase):
    def test_all_functions(self):
        srfc = [
            SRFChannel(
                SRF_CENTER,
                SRF_ID,
                {MIN_WLEN: 0.5, MAX_WLEN - 1: 0.5},
                MIN_WLEN,
                MAX_WLEN,
            )
        ]
        srf = SpectralResponseFunction("test", srfc)
        self.assertEqual(srf.get_channel_from_name(SRF_ID), srfc[0])
        self.assertIsNone(srf.get_channel_from_name("Nonexistent"))
        ch_names = srf.get_channels_names()
        self.assertEqual(len(ch_names), 1)
        self.assertEqual(ch_names[0], SRF_ID)
        ch_vals = srf.get_values()
        self.assertEqual(len(ch_vals), 2)
        self.assertEqual(ch_vals[0], 0.5)
        ch_wlens = srf.get_wavelengths()
        self.assertEqual(len(ch_wlens), 2)
        self.assertEqual(ch_wlens[0], MIN_WLEN)


COEF_WLENS = [350, 500, 650]
POS_COEFFS = [(1, 2, 3, 4), (10, 2, 3, 4), (0.1, 2, 3, 4)]
NEG_COEFFS = [(-1, -2, -3, -4), (-1, -2, -3, -4), (-1, -2, -3, -4)]


class TestPolarizationCoefficients(unittest.TestCase):
    def test_func(self):
        coeffs = PolarizationCoefficients(COEF_WLENS, POS_COEFFS, NEG_COEFFS)
        self.assertEqual(coeffs.get_wavelengths(), COEF_WLENS)
        self.assertEqual(coeffs.get_coefficients_positive(COEF_WLENS[0]), POS_COEFFS[0])
        self.assertEqual(coeffs.get_coefficients_negative(COEF_WLENS[0]), NEG_COEFFS[0])


class TestApolloIrradianceCoefficients(unittest.TestCase):
    # This class is not deleted for legacy purposes, but it's not used. If it were to be used in the future, these tests should be developed.
    pass


DT1 = datetime(2000, 1, 16, 2)
DT2 = datetime(2022, 1, 17, 3)
DT3 = datetime(2032, 1, 17, 3)


class TestSatellite(unittest.TestCase):
    def test_satellite_functions_ok(self):
        obfs = [OrbitFile("a", DT1, DT2), OrbitFile("e", DT2, DT3)]
        sat = Satellite("SAT", 0, obfs)
        self.assertEqual(sat.get_best_orbit_file(DT2 - timedelta(5)), obfs[0])
        self.assertEqual(sat.get_best_orbit_file(DT2 + timedelta(5)), obfs[1])
        self.assertIsNone(sat.get_best_orbit_file(DT3 + timedelta(5)))
        self.assertEqual(sat.get_datetime_range(), (DT1, DT3))


CH_WLENS = np.array([350, 400, 450, 500])
CH_SRF = np.array([0.2, 0.2, 0.3, 0.3])
CH_ELIS = np.array([0.005, 0.0002, 0.3, 0.0001])


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_SRF[i]: CH_WLENS[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "default", spectral_response)
    return SpectralResponseFunction("Default", [ch])


class TestLunarObservation(unittest.TestCase):
    def test_lunar_functions_ok(self):
        stp = SatellitePosition(10000, 10000, 10000)
        ch_names = ["default"]
        ch_irrs = {ch_names[0]: 0.003}
        lo = LunarObservation(ch_names, "ITRF93", ch_irrs, DT2, stp)
        self.assertEqual(lo.get_ch_irradiance(ch_names[0]), ch_irrs[ch_names[0]])
        self.assertTrue(lo.has_ch_value(ch_names[0]))
        self.assertFalse(lo.has_ch_value(""))
        self.assertFalse(lo.has_ch_value("23asd"))
        srf = get_srf()
        self.assertTrue(lo.check_valid_srf(srf))
        srf.channels[0].id = "a"
        self.assertFalse(lo.check_valid_srf(srf))


COEFF_LINE = np.array(
    [
        -2.263172432,
        -1.953409783,
        0.691585146,
        -0.301894577,
        0.052456211,
        0.008714468,
        -0.004148856,
        0.001216634,
        -0.000358287,
        0.001610105,
        0.000732067,
        -0.092938476,
        2.000625563,
        -0.005710425,
        1.354459689,
        1.314673623,
        9.324088764,
        9.596769204,
    ]
)
U_COEFF_LINE = np.array(
    [
        9.3686983712e-03,
        1.2017105711e-08,
        1.4446094233e-08,
        5.0688276442e-09,
        5.4710503849e-11,
        1.0713699068e-10,
        4.2666117090e-11,
        9.4897416668e-14,
        3.9586194983e-13,
        1.3962709717e-12,
        2.1961497567e-12,
        1.4488250162e-05,
        1.4521860048e-05,
        4.1449084817e-10,
        5.3081256301e-10,
        4.0658906933e-07,
        1.7771204867e-08,
        3.2858296080e-09,
    ]
)


class TestReflectanceCoefficients(unittest.TestCase):
    def test_creation_ok(self):
        dim_sizes = {
            "wavelength": 6,
            "i_coeff": 18,
        }
        ds_cimel: xarray.Dataset = obsarray.create_ds(TEMPLATE_CIMEL, dim_sizes)
        ds_cimel = ds_cimel.assign_coords(wavelength=[440, 500, 675, 870, 1020, 1640])
        data = np.array([i * COEFF_LINE for i in range(1, 7)])
        u_data = np.array([i * U_COEFF_LINE for i in range(1, 7)])
        ds_cimel.coeff.values = data.T
        ds_cimel.u_coeff.values = u_data.T
        coeffs = ReflectanceCoefficients(ds_cimel)
        self.assertEqual(len(coeffs.coeffs.a_coeffs), 4)
        for i in range(4):  # a coeffs are 4 coefficients
            a_coeffs = coeffs.coeffs.a_coeffs[i]
            a_coeffs_check = [d[i] for d in data]
            for j, a in enumerate(a_coeffs):
                self.assertEqual(a, a_coeffs_check[j])


SPD_WAVS = np.array([350, 380, 400, 430, 450, 500, 600, 750, 800])
SPD_VALS = np.array([0.002, 0.0048, 0.0043, 0.004, 0.0007, 0.0067, 0.0001, 0.7, 0.08])
CH_IDS = np.array(["a", "b", "c", "d", "e"])
SIGNALS_DATA = np.array(
    [[0, 0.01], [0.01, 0.01], [0.01, 0.02], [0.04, 0.004], [0.06, 0.01]]
)


class TestSpectralData(unittest.TestCase):
    def test_make_reflectance_ds_ok(self):
        ds = SpectralData.make_reflectance_ds(SPD_WAVS, SPD_VALS)
        for i, val in enumerate(ds.reflectance.values):
            self.assertEqual(val, SPD_VALS[i])
        self.assertIsNotNone(ds.u_ran_reflectance.values)
        self.assertIsNotNone(ds.u_sys_reflectance.values)

    def test_make_reflectance_ds_list_err(self):
        self.assertRaises(
            TypeError, SpectralData.make_reflectance_ds, list(SPD_WAVS), list(SPD_VALS)
        )

    def test_make_reflectance_ds_Spectral_Data(self):
        ds = SpectralData.make_reflectance_ds(SPD_WAVS, SPD_VALS)
        uncs = ds.u_ran_reflectance.values**2 + ds.u_sys_reflectance.values**2
        _ = SpectralData(SPD_WAVS, SPD_VALS, uncs, ds)

    def test_make_irradiance_ds_ok(self):
        ds = SpectralData.make_irradiance_ds(SPD_WAVS, SPD_VALS)
        for i, val in enumerate(ds.irradiance.values):
            self.assertEqual(val, SPD_VALS[i])
        self.assertIsNotNone(ds.u_ran_irradiance.values)
        self.assertIsNotNone(ds.u_sys_irradiance.values)

    def test_make_polarization_ds_ok(self):
        ds = SpectralData.make_polarization_ds(SPD_WAVS, SPD_VALS)
        for i, val in enumerate(ds.polarization.values):
            self.assertEqual(val, SPD_VALS[i])
        self.assertIsNotNone(ds.u_ran_polarization.values)
        self.assertIsNotNone(ds.u_sys_polarization.values)

    def test_make_signals_ds_ok(self):
        ds = SpectralData.make_signals_ds(CH_IDS, SIGNALS_DATA)
        for i, arr in enumerate(ds.signals.values):
            for j, val in enumerate(arr):
                self.assertEqual(val, SIGNALS_DATA[i][j])
        self.assertIsNotNone(ds.u_ran_signals.values)
        self.assertIsNotNone(ds.u_sys_signals.values)

    def test_make_signals_ds_invalid_signals_transposed(self):
        self.assertRaises(
            ValueError, SpectralData.make_signals_ds, CH_IDS, SIGNALS_DATA.T
        )

    def test_make_signals_ds_invalid_signals_unnested(self):
        self.assertRaises(
            TypeError, SpectralData.make_signals_ds, CH_IDS, SIGNALS_DATA.flatten()
        )


if __name__ == "__main__":
    unittest.main()