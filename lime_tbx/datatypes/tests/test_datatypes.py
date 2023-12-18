"""Tests for datatypes module"""

"""___Built-In Modules___"""
from datetime import datetime, timedelta, timezone

"""___Third-Party Modules___"""
import unittest
import numpy as np
import xarray

"""___NPL Modules___"""
import obsarray

"""___LIME_TBX Modules___"""
from ..datatypes import (
    LunarObservation,
    OrbitFile,
    PolarisationCoefficients,
    ReflectanceCoefficients,
    SRFChannel,
    Satellite,
    SatellitePosition,
    SpectralData,
    SpectralResponseFunction,
    SpectralValidity,
    LunarObservationWrite,
)
from .. import constants
from lime_tbx.datatypes.templates import TEMPLATE_CIMEL

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

    def test_creation_both_equal_max(self):
        srfc = SRFChannel(
            SRF_CENTER, SRF_ID, {MIN_WLEN: 0.5, MAX_WLEN: 0.5}, MIN_WLEN, MAX_WLEN
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
                {MIN_WLEN: 0.5, MAX_WLEN: 0.5},
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


DT1 = datetime(2000, 1, 16, 2, tzinfo=timezone.utc)
DT2 = datetime(2022, 1, 17, 3, tzinfo=timezone.utc)
DT3 = datetime(2032, 1, 17, 3, tzinfo=timezone.utc)


class TestSatellite(unittest.TestCase):
    def test_satellite_functions_ok(self):
        obfs = [OrbitFile("a", DT1, DT2), OrbitFile("e", DT2, DT3)]
        sat = Satellite("SAT", 0, obfs, None, None, None)
        self.assertEqual(sat.get_best_orbit_file(DT2 - timedelta(5)), obfs[0])
        self.assertEqual(sat.get_best_orbit_file(DT2 + timedelta(5)), obfs[1])
        self.assertIsNone(sat.get_best_orbit_file(DT3 + timedelta(5)))
        self.assertEqual(sat.get_datetime_range(), (DT1, DT3))


CH_WLENS = np.array([350, 400, 450, 500])
CH_SRF = np.array([0.2, 0.2, 0.3, 0.3])
CH_ELIS = np.array([0.005, 0.0002, 0.3, 0.0001])


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_WLENS[i]: CH_SRF[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "default", spectral_response)
    return SpectralResponseFunction("Default", [ch])


class TestLunarObservation(unittest.TestCase):
    def test_lunar_functions_ok(self):
        stp = SatellitePosition(10000, 10000, 10000)
        ch_names = ["default"]
        ch_irrs = {ch_names[0]: 0.003}
        lo = LunarObservation(ch_names, "ITRF93", ch_irrs, DT2, stp, "Test")
        self.assertEqual(lo.get_ch_irradiance(ch_names[0]), ch_irrs[ch_names[0]])
        self.assertTrue(lo.has_ch_value(ch_names[0]))
        self.assertFalse(lo.has_ch_value(""))
        self.assertFalse(lo.has_ch_value("23asd"))
        srf = get_srf()
        self.assertTrue(lo.check_valid_srf(srf))
        srf.channels[0].id = "a"
        self.assertFalse(lo.check_valid_srf(srf))

    def test_lunar_get_ch_irradiance_error(self):
        stp = SatellitePosition(10000, 10000, 10000)
        ch_names = ["default"]
        ch_irrs = {ch_names[0]: 0.003}
        lo = LunarObservation(ch_names, "ITRF93", ch_irrs, DT2, stp, "Test")
        self.assertRaises(ValueError, lo.get_ch_irradiance, "")


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
            "i_coeff.wavelength": 18 * 6,
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


COEF_WLENS = [350, 500, 650]
POS_COEFFS = [(1, 2, 3, 4), (10, 2, 3, 4), (0.1, 2, 3, 4)]
NEG_COEFFS = [(-1, -2, -3, -4), (-1, -2, -3, -4), (-1, -2, -3, -4)]
UNCERTAINTIES = np.array([(1, 1, 2, 1), (2, 2, 3, 2), (3, 3, 4, 3)])
_nounc_size = len(UNCERTAINTIES) * len(UNCERTAINTIES[0])
ERR_CORR = np.random.random_sample((_nounc_size, _nounc_size))


class TestPolarisationCoefficients(unittest.TestCase):
    def test_polcoeffs_ok(self):
        coeffs = PolarisationCoefficients(
            COEF_WLENS,
            POS_COEFFS,
            UNCERTAINTIES,
            ERR_CORR,
            NEG_COEFFS,
            UNCERTAINTIES * -1,
            ERR_CORR * -1,
        )
        self.assertEqual(coeffs.get_wavelengths(), COEF_WLENS)
        self.assertEqual(coeffs.get_coefficients_positive(COEF_WLENS[0]), POS_COEFFS[0])
        self.assertEqual(coeffs.get_coefficients_negative(COEF_WLENS[0]), NEG_COEFFS[0])
        np.testing.assert_array_equal(
            coeffs.get_uncertainties_positive(COEF_WLENS[0]), UNCERTAINTIES[0]
        )
        np.testing.assert_array_equal(
            coeffs.get_uncertainties_negative(COEF_WLENS[0]), -1 * UNCERTAINTIES[0]
        )
        self.assertTrue(coeffs.is_calculable())

    def test_polcoeffs_not_calculable(self):
        pos_coeffs = POS_COEFFS.copy()
        pos_coeffs[2] = tuple(np.nan for _ in range(4))
        coeffs = PolarisationCoefficients(
            COEF_WLENS,
            pos_coeffs,
            UNCERTAINTIES,
            ERR_CORR,
            NEG_COEFFS,
            UNCERTAINTIES * -1,
            ERR_CORR * -1,
        )
        self.assertFalse(coeffs.is_calculable())
        coeffs = PolarisationCoefficients(
            COEF_WLENS,
            POS_COEFFS,
            UNCERTAINTIES,
            ERR_CORR,
            pos_coeffs,
            UNCERTAINTIES * -1,
            ERR_CORR * -1,
        )
        self.assertFalse(coeffs.is_calculable())


LOW_WLENS = [350, 400, 500, 700]


class TestLunarObservationWrite(unittest.TestCase):
    def test_low_ok(self):
        stp = SatellitePosition(10000, 10000, 10000)
        ch_names = ["default"]
        irrs = SpectralData(LOW_WLENS, [1, 2, 3, 4], [0, 1, 1, 2], None)
        refls = SpectralData(LOW_WLENS, [2, 3, 4, 5], [-1, 0, 1, 1], None)
        polars = SpectralData(LOW_WLENS, [3, 4, 5, 6], [-2, -1, 0, 1], None)
        low = LunarObservationWrite(
            ch_names,
            "ITRF93",
            DT1,
            stp,
            irrs,
            refls,
            polars,
            None,
            None,
            "Test",
        )
        self.assertTrue(low.has_ch_value("default"))
        self.assertFalse(low.has_ch_value("3w"))

        srfc = [
            SRFChannel(
                500,
                "default",
                {LOW_WLENS[0]: 0.5, LOW_WLENS[-1]: 0.5},
                LOW_WLENS[0],
                LOW_WLENS[-1],
            )
        ]
        srf0 = SpectralResponseFunction("a", srfc)
        srfc = [
            SRFChannel(
                500,
                "default2",
                {LOW_WLENS[0]: 0.5, LOW_WLENS[-1]: 0.5},
                LOW_WLENS[0],
                LOW_WLENS[-1],
            )
        ]
        srf1 = SpectralResponseFunction("a", srfc)
        self.assertTrue(low.check_valid_srf(srf0))
        self.assertFalse(low.check_valid_srf(srf1))


SPD_WAVS = np.array([350, 380, 400, 430, 450, 500, 600, 750, 800])
SPD_VALS = np.array([0.002, 0.0048, 0.0043, 0.004, 0.0007, 0.0067, 0.0001, 0.7, 0.08])
SPD_CORR = np.zeros((len(SPD_WAVS), len(SPD_WAVS)))
np.fill_diagonal(SPD_CORR, 1)
CH_IDS = np.array(["a", "b", "c", "d", "e"])
SIGNALS_DATA = np.array(
    [[0, 0.01], [0.01, 0.01], [0.01, 0.02], [0.04, 0.004], [0.06, 0.01]]
)


class TestSpectralData(unittest.TestCase):
    def test_make_reflectance_ds_ok(self):
        ds = SpectralData.make_reflectance_ds(SPD_WAVS, SPD_VALS)
        for i, val in enumerate(ds.reflectance.values):
            self.assertEqual(val, SPD_VALS[i])
        self.assertIsNotNone(ds.u_reflectance.values)
        self.assertIsNotNone(ds.err_corr_reflectance.values)

    def test_make_reflectance_ds_Spectral_Data(self):
        ds = SpectralData.make_reflectance_ds(
            SPD_WAVS, SPD_VALS, SPD_VALS * 0.05, SPD_CORR
        )
        uncs = ds.u_reflectance.values**2
        _ = SpectralData(SPD_WAVS, SPD_VALS, uncs, ds)

    def test_make_irradiance_ds_ok(self):
        ds = SpectralData.make_irradiance_ds(SPD_WAVS, SPD_VALS)
        for i, val in enumerate(ds.irradiance.values):
            self.assertEqual(val, SPD_VALS[i])
        self.assertIsNotNone(ds.u_irradiance.values)
        self.assertIsNotNone(ds.err_corr_irradiance.values)

    def test_make_polarisation_ds_ok(self):
        ds = SpectralData.make_polarisation_ds(SPD_WAVS, SPD_VALS)
        for i, val in enumerate(ds.polarisation.values):
            self.assertEqual(val, SPD_VALS[i])
        self.assertIsNotNone(ds.u_polarisation.values)
        self.assertIsNotNone(ds.err_corr_polarisation.values)

    def test_make_signals_ds_ok(self):
        ds = SpectralData.make_signals_ds(CH_IDS, SIGNALS_DATA)
        for i, arr in enumerate(ds.signals.values):
            for j, val in enumerate(arr):
                self.assertEqual(val, SIGNALS_DATA[i][j])
        self.assertIsNotNone(ds.u_signals.values)
        self.assertIsNotNone(ds.err_corr_signals_channels.values)
        self.assertIsNotNone(ds.err_corr_signals_dts.values)

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
