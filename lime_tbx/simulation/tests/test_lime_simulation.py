"""Tests for lime_simulation module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import sys
import io

"""___Third-Party Modules___"""
import unittest
import numpy as np
import pytest

"""___LIME_TBX Modules___"""
from ..lime_simulation import ILimeSimulation, LimeSimulation, is_ampa_valid_range
from ...datatypes.datatypes import (
    PolarisationCoefficients,
    ReflectanceCoefficients,
    SpectralData,
    SpectralResponseFunction,
    SRFChannel,
    SurfacePoint,
    SatellitePoint,
    KernelsPath,
    LGLODData,
    EocfiPath,
)
from ...coefficients.access_data.access_data import (
    _get_default_polarisation_coefficients,
    _get_demo_cimel_coeffs,
)
from ...filedata import srf as srflib, lglod as lglodlib
from lime_tbx.interpolation.interp_data import interp_data
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.eocfi_adapter.eocfi_adapter import EOCFIConverter
from lime_tbx.spectral_integration.spectral_integration import get_default_srf
from lime_tbx.gui.settings import SettingsManager


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "25/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = EocfiPath("./eocfi_data", "./eocfi_data2")

CH_WLENS = np.array([350, 400, 450, 500])
CH_DEF_INDICES = np.where(
    np.in1d(list(get_default_srf().channels[0].spectral_response.keys()), CH_WLENS)
)
CH_SRF = np.array([0.2, 0.2, 0.3, 0.3])
CH_ELIS = np.array([0.005, 0.0002, 0.3, 0.0001])

LAT = 21
LON = 21
ALT = 2400
DT1 = datetime(2022, 1, 17, 2, tzinfo=timezone.utc)
DT2 = datetime(2022, 2, 16, 2, tzinfo=timezone.utc)

SURFACE_POINT = SurfacePoint(LAT, LON, ALT, [DT1, DT2])
SATELLITE_POINT = SatellitePoint("BIOMASS", DT1)
SATELLITE_POINT_2 = SatellitePoint("BIOMASS", [DT1, DT2])


def get_srf() -> SpectralResponseFunction:
    spectral_response = {CH_WLENS[i]: CH_SRF[i] for i in range(len(CH_SRF))}
    ch = SRFChannel((CH_WLENS[-1] - CH_WLENS[0]) / 2, "Default", spectral_response)
    return SpectralResponseFunction("default", [ch])


def get_cimel_coeffs() -> ReflectanceCoefficients:
    return _get_demo_cimel_coeffs()


def get_polar_coeffs() -> PolarisationCoefficients:
    return _get_default_polarisation_coefficients()


def get_lime_simulation() -> ILimeSimulation:
    interp_data.set_interpolation_spectrum_name("ASD")
    return LimeSimulation(EOCFI_PATH, KERNELS_PATH, SettingsManager(), verbose=False)


class TestLimeSimulation(unittest.TestCase):
    # Function set_simulation_changed
    def test_set_simulation_changed_ok(self):
        ls = get_lime_simulation()
        ls.set_simulation_changed()

    def test_set_simulation_changed_multiple_times(self):
        ls = get_lime_simulation()
        for _ in range(6):
            ls.set_simulation_changed()

    @pytest.mark.slow
    def test_set_simulation_changed_diff_output_objects(self):
        ls = get_lime_simulation()
        srf = get_srf()
        sp = SURFACE_POINT
        cf = get_cimel_coeffs()
        ls.update_reflectance(srf, sp, cf)
        elrefs_0 = ls.get_elrefs()
        ls.update_reflectance(srf, sp, cf)
        elrefs_1 = ls.get_elrefs()
        ls.set_simulation_changed()
        ls.update_reflectance(srf, sp, cf)
        elrefs_2 = ls.get_elrefs()
        self.assertIs(elrefs_0, elrefs_1)
        self.assertIsNot(elrefs_0, elrefs_2)

    # Function update_reflectance
    @pytest.mark.slow
    def test_update_reflectance(self):
        ls = get_lime_simulation()
        ls._skip_uncs = False
        ls.update_reflectance(get_default_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        cimels = ls.get_elrefs_cimel()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        elrefs_refs = np.array(
            [
                [0.055, 0.065, 0.074, 0.083],
                [0.059, 0.069, 0.079, 0.088],
            ]
        )
        uncs_refs = np.array(
            [
                [1.14e-04, 1.1e-04, 1.02e-04, 1.7e-05],
                [1.2e-04, 1.17e-04, 1.13e-04, 1.5e-05],
            ]
        )
        cimel_refs = np.array(
            [
                [
                    0.07238157588156337,
                    0.08303552269318298,
                    0.10680491219671362,
                    0.12074845803729924,
                    0.13089216637453846,
                    0.1819960991552666,
                ],
                [
                    0.07703189717011595,
                    0.08820668632490204,
                    0.11297783058324386,
                    0.12723529600040906,
                    0.13759411628920393,
                    0.18992325498435048,
                ],
            ]
        )
        cimel_unc_refs = np.array(
            [
                [
                    1.59259961e-05,
                    1.61469037e-05,
                    1.64896686e-05,
                    2.02907432e-05,
                    1.58269992e-05,
                    1.90806538e-05,
                ],
                [
                    1.70530811e-05,
                    1.72827426e-05,
                    1.84880585e-05,
                    2.03905382e-05,
                    1.61238620e-05,
                    2.15202391e-05,
                ],
            ]
        )
        for elref, elref_ref, unc in zip(elrefs, elrefs_refs, uncs_refs):
            np.testing.assert_array_almost_equal(
                elref.data[CH_DEF_INDICES], elref_ref, 3
            )
            np.testing.assert_allclose(elref.uncertainties[CH_DEF_INDICES], unc, 1)
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_almost_equal(cimel.data, data)
            np.testing.assert_array_almost_equal(cimel.uncertainties, unc, 4)

    def test_update_reflectance_skip_uncs(self):
        ls = get_lime_simulation()
        ls._skip_uncs = True
        ls.update_reflectance(get_default_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        cimels = ls.get_elrefs_cimel()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        elrefs_refs = np.array(
            [
                [0.055, 0.065, 0.074, 0.083],
                [0.059, 0.069, 0.079, 0.088],
            ]
        )
        uncs_refs = np.zeros(elrefs_refs.shape)
        cimel_refs = np.array(
            [
                [
                    0.07238157588156337,
                    0.08303552269318298,
                    0.10680491219671362,
                    0.12074845803729924,
                    0.13089216637453846,
                    0.1819960991552666,
                ],
                [
                    0.07703189717011595,
                    0.08820668632490204,
                    0.11297783058324386,
                    0.12723529600040906,
                    0.13759411628920393,
                    0.18992325498435048,
                ],
            ]
        )
        cimel_unc_refs = np.zeros(cimel_refs.shape)
        for elref, elref_ref, unc in zip(elrefs, elrefs_refs, uncs_refs):
            np.testing.assert_array_almost_equal(
                elref.data[CH_DEF_INDICES], elref_ref, 3
            )
            np.testing.assert_array_equal(elref.uncertainties[CH_DEF_INDICES], unc)
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_almost_equal(cimel.data, data)
            np.testing.assert_array_equal(cimel.uncertainties, unc)

    def _check_irr_output(self, elis, signals, cimels):
        self.assertIsNotNone(elis)
        self.assertIsInstance(elis, list)
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, SpectralData)
        elis_refs = np.array(
            [
                [7.508914e-07, 1.981703e-06, 3.090270e-06, 3.096121e-06],
                [8.308273e-07, 2.192665e-06, 3.418179e-06, 3.419373e-06],
            ]
        )
        uncs_refs = np.array(
            [
                [1.22816289e-08, 1.66544863e-08, 1.39227268e-08, 8.35593656e-09],
                [1.36277807e-08, 1.84579950e-08, 1.54306939e-08, 9.22275540e-09],
            ]
        )
        for eli, eli_ref, unc in zip(elis, elis_refs, uncs_refs):
            np.testing.assert_array_almost_equal(eli.data[CH_DEF_INDICES], eli_ref)
            np.testing.assert_array_almost_equal(
                eli.uncertainties[CH_DEF_INDICES], unc, 8
            )
        np.testing.assert_array_almost_equal(
            signals.data, np.array([[3.16668016e-06, 3.75979585e-06]])
        )
        np.testing.assert_array_almost_equal(signals.uncertainties, np.array([[0, 0]]))
        cimel_refs = np.array(
            [
                [
                    2.6237802025548067e-06,
                    3.1685931571619723e-06,
                    3.1508629556748644e-06,
                    2.1881525065050035e-06,
                    1.787553267951884e-06,
                    8.068699993533158e-07,
                ],
                [
                    2.9030938832113895e-06,
                    3.4994114313499266e-06,
                    3.4651529237610166e-06,
                    2.3971455172115902e-06,
                    1.953601260023857e-06,
                    8.754072576766165e-07,
                ],
            ]
        )
        cimel_unc_refs = np.array(
            [
                [2e-08, 1e-08, 1e-08, 7e-09, 6e-09, 3e-09],
                [2e-08, 1e-08, 1e-08, 1e-09, 6e-09, 3e-09],
            ]
        )
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_almost_equal(cimel.data, data)
            np.testing.assert_array_almost_equal(cimel.uncertainties, unc, 8)

    @pytest.mark.slow
    def test_update_irradiance_and_reflectance(self):
        ls: LimeSimulation = get_lime_simulation()
        ls._skip_uncs = False
        ls._update_irradiance_and_reflectance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        elis = ls.get_elis()
        signals = ls.get_signals()
        cimels = ls.get_elis_cimel()
        self._check_irr_output(elis, signals, cimels)

    # Function update_irradiance
    @pytest.mark.slow
    def test_update_irradiance(self):
        ls = get_lime_simulation()
        ls._skip_uncs = False
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        elis = ls.get_elis()
        signals = ls.get_signals()
        cimels = ls.get_elis_cimel()
        self._check_irr_output(elis, signals, cimels)

    def test_update_irradiance_skip_uncs(self):
        ls = get_lime_simulation()
        ls._skip_uncs = True
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        elis = ls.get_elis()
        self.assertIsNotNone(elis)
        self.assertIsInstance(elis, list)
        signals = ls.get_signals()
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, SpectralData)
        elis_refs = np.array(
            [
                [7.508914e-07, 1.981703e-06, 3.090270e-06, 3.096121e-06],
                [8.308273e-07, 2.192665e-06, 3.418179e-06, 3.419373e-06],
            ]
        )
        uncs_refs = np.zeros(elis_refs.shape)
        for eli, eli_ref, unc in zip(elis, elis_refs, uncs_refs):
            np.testing.assert_array_almost_equal(eli.data[CH_DEF_INDICES], eli_ref)
            np.testing.assert_array_equal(eli.uncertainties[CH_DEF_INDICES], unc)
        np.testing.assert_array_almost_equal(
            signals.data, np.array([[3.16668016e-06, 3.75979585e-06]])
        )
        np.testing.assert_array_almost_equal(signals.uncertainties, np.array([[0, 0]]))
        cimels = ls.get_elis_cimel()
        cimel_refs = np.array(
            [
                [
                    2.6237802025548067e-06,
                    3.1685931571619723e-06,
                    3.1508629556748644e-06,
                    2.1881525065050035e-06,
                    1.787553267951884e-06,
                    8.068699993533158e-07,
                ],
                [
                    2.9030938832113895e-06,
                    3.4994114313499266e-06,
                    3.4651529237610166e-06,
                    2.3971455172115902e-06,
                    1.953601260023857e-06,
                    8.754072576766165e-07,
                ],
            ]
        )
        cimel_unc_refs = np.zeros(cimel_refs.shape)
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_almost_equal(cimel.data, data)
            np.testing.assert_array_equal(cimel.uncertainties, unc)

    # Function update_polarisation
    def test_update_polarisation(self):
        ls = get_lime_simulation()
        ls._skip_uncs = False
        ls.update_polarisation(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)
        polars_refs = np.array(
            [
                [0.07129, 0.07129, 0.06957, 0.06093],
                [0.05147, 0.05147, 0.05024, 0.04407],
            ]
        )
        uncs_refs = np.array(
            [
                [0, 0, 0, 0],  # [8.91e-10, 1.06e-9, 4.74e-05, 1e-16],
                [0, 0, 0, 0],  # [4.82e-10, 5.28e-10, 2.7e-06, 7.25e-17],
            ]
        )
        for polar, pref, uref in zip(polars, polars_refs, uncs_refs):
            np.testing.assert_array_almost_equal(polar.data[CH_DEF_INDICES], pref, 5)
            np.testing.assert_allclose(
                polar.uncertainties[CH_DEF_INDICES], uref, rtol=0.5
            )
        cimels = ls.get_polars_cimel()
        cimel_refs = np.array(
            [
                [
                    0.07129322780306957,
                    0.06092845598091574,
                    0.055975108631756476,
                    0.052494153098236906,
                    0.052857023414860475,
                    0.05715870106352176,
                ],
                [
                    0.05147248258427663,
                    0.04407339069207665,
                    0.04050642065393072,
                    0.038043429661297735,
                    0.03831977551866521,
                    0.04134994535789782,
                ],
            ]
        )
        cimel_unc_refs = np.array(
            [
                [
                    0.00158481,
                    0.00175289,
                    0.00166579,
                    0.00164374,
                    0.00169278,
                    0.00192381,
                ],
                [0.00077383, 0.00072996, 0.00068953, 0.00057121, 0.00072269, 0.0007257],
            ]
        )
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_equal(cimel.data, data)
            np.testing.assert_allclose(cimel.uncertainties, unc, rtol=0.4)

    def test_update_polarisation_skip_uncs(self):
        ls = get_lime_simulation()
        ls._skip_uncs = True
        ls.update_polarisation(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)
        polars_refs = np.array(
            [
                [0.07129, 0.07129, 0.06957, 0.06093],
                [0.05147, 0.05147, 0.05024, 0.04407],
            ]
        )
        uncs_refs = np.zeros(polars_refs.shape)
        for polar, pref, uref in zip(polars, polars_refs, uncs_refs):
            np.testing.assert_array_almost_equal(polar.data[CH_DEF_INDICES], pref, 5)
            np.testing.assert_array_equal(polar.uncertainties[CH_DEF_INDICES], uref)
        cimels = ls.get_polars_cimel()
        cimel_refs = np.array(
            [
                [
                    0.07129322780306957,
                    0.06092845598091574,
                    0.055975108631756476,
                    0.052494153098236906,
                    0.052857023414860475,
                    0.05715870106352176,
                ],
                [
                    0.05147248258427663,
                    0.04407339069207665,
                    0.04050642065393072,
                    0.038043429661297735,
                    0.03831977551866521,
                    0.04134994535789782,
                ],
            ]
        )
        cimel_unc_refs = np.zeros(cimel_refs.shape)
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_equal(cimel.data, data)
            np.testing.assert_array_equal(cimel.uncertainties, unc)

    @pytest.mark.slow
    def test_update_irr_polar_verbose(self):
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr

        ls = LimeSimulation(EOCFI_PATH, KERNELS_PATH, SettingsManager(), verbose=True)
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        ls.update_polarisation(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
        elis = ls.get_elis()
        self.assertIsNotNone(elis)
        self.assertIsInstance(elis, list)
        signals = ls.get_signals()
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, SpectralData)
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)
        verbose_output_l = [
            "starting reflectance update",
            "reflectance update done",
            "starting irradiance update",
            "auxiliar irradiance update done",
            "irradiance & signals update done",
            "starting polarisation update",
            "polarisation update done\n",
        ]
        verbose_output = "\n".join(verbose_output_l)
        self.assertEqual(self.capturedOutput.getvalue(), verbose_output)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # get_surfacepoints
    @pytest.mark.slow
    def test_get_surfacepoints_ok(self):
        ls = get_lime_simulation()
        sp = SATELLITE_POINT
        ls.update_irradiance(get_default_srf(), get_srf(), sp, get_cimel_coeffs())
        eocfi = EOCFIConverter(EOCFI_PATH, KERNELS_PATH)
        dts = [sp.dt]
        llhs = eocfi.get_satellite_position(sp.name, dts)
        srps = []
        for i, llh in enumerate(llhs):
            srp = SurfacePoint(llh[0], llh[1], llh[2], dts[i])
            srps.append(srp)
        pt = ls.get_surfacepoints()
        self.assertIsInstance(pt, SurfacePoint)
        self.assertEqual(pt, srps[0])

    @pytest.mark.slow
    def test_get_surfacepoints_ok_multiple(self):
        ls = get_lime_simulation()
        sp = SATELLITE_POINT_2
        ls.update_irradiance(get_default_srf(), get_srf(), sp, get_cimel_coeffs())
        eocfi = EOCFIConverter(EOCFI_PATH, KERNELS_PATH)
        dts = sp.dt
        llhs = eocfi.get_satellite_position(sp.name, dts)
        srps = []
        for i, llh in enumerate(llhs):
            srp = SurfacePoint(llh[0], llh[1], llh[2], dts[i])
            srps.append(srp)
        pts = ls.get_surfacepoints()
        self.assertIsInstance(pts, list)
        for pt, srp in zip(pts, srps):
            self.assertEqual(pt, srp)

    @pytest.mark.slow
    def test_get_surfacepoints_not_sat_point(self):
        ls = get_lime_simulation()
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        pts = ls.get_surfacepoints()
        self.assertIsNone(pts)

    # get point
    @pytest.mark.slow
    def test_update_get_point(self):
        ls = get_lime_simulation()
        ls.update_reflectance(get_default_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        pt = ls.get_point()
        self.assertEqual(pt, SURFACE_POINT)

    @pytest.mark.slow
    def test_update_get_point_satellite(self):
        ls = get_lime_simulation()
        ls.update_reflectance(get_default_srf(), SATELLITE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(
            elrefs, SpectralData
        )  # It's not list if the point.dt is datetime instead of list of dt
        pt = ls.get_point()
        self.assertEqual(pt, SATELLITE_POINT)

    # get_moon_datas
    @pytest.mark.slow
    def test_get_moon_datas_ok(self):
        ls = get_lime_simulation()
        sp = SURFACE_POINT
        ls.update_reflectance(get_default_srf(), sp, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        mds = ls.get_moon_datas()
        mds_ref = SPICEAdapter().get_moon_data_from_earth(
            sp.latitude,
            sp.longitude,
            sp.altitude,
            sp.dt,
            KERNELS_PATH,
        )
        for md, mdr in zip(mds, mds_ref):
            md = np.array(list(md.__dict__.values()))
            mdr = np.array(list(mdr.__dict__.values()))
            np.testing.assert_equal(md, mdr)

    @pytest.mark.slow
    def test_get_moon_datas_ok_sat(self):
        ls = get_lime_simulation()
        sp = SATELLITE_POINT_2
        ls.update_reflectance(get_default_srf(), sp, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        mds = ls.get_moon_datas()
        eocfi = EOCFIConverter(EOCFI_PATH, KERNELS_PATH)
        llhs = eocfi.get_satellite_position(sp.name, sp.dt)
        mds_ref = []
        for i, llh in enumerate(llhs):
            srp = SurfacePoint(llh[0], llh[1], llh[2], sp.dt[i])
            md_ref = SPICEAdapter().get_moon_data_from_earth(
                srp.latitude,
                srp.longitude,
                srp.altitude,
                sp.dt[i],
                KERNELS_PATH,
            )
            mds_ref.append(md_ref)
        for md, mdr in zip(mds, mds_ref):
            md = np.array(list(md.__dict__.values())).astype(float)
            mdr = np.array(list(mdr.__dict__.values())).astype(float)
            np.testing.assert_almost_equal(md, mdr, 8)

    # is_ampa_valid_range
    def test_is_ampa_valid_range_vals(self):
        ampas = [0, 1, 2, 3, 5, 30, 89, 90, 91, 180]
        vals = [False, False, True, True, True, True, True, True, False, False]
        for ampa, val in zip(ampas, vals):
            valcalc = is_ampa_valid_range(ampa)
            self.assertEqual(valcalc, val)

    # are_mpas_inside_mpa_range
    @pytest.mark.slow
    def test_are_mpas_inside_mpa_range(self):
        ls = get_lime_simulation()
        sp = SURFACE_POINT
        ls.update_reflectance(get_default_srf(), sp, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        mds = ls.get_moon_datas()
        valids = ls.are_mpas_inside_mpa_range()
        for md, valid in zip(mds, valids):
            self.assertEqual(valid, is_ampa_valid_range(md.absolute_mpa_degrees))

    # is_polarisation_updated
    @pytest.mark.slow
    def test_is_polarisation_updated(self):
        ls = get_lime_simulation()
        self.assertFalse(ls.is_polarisation_updated())
        ls.update_reflectance(get_default_srf(), SURFACE_POINT, get_cimel_coeffs())
        self.assertFalse(ls.is_polarisation_updated())
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        self.assertFalse(ls.is_polarisation_updated())
        ls.update_polarisation(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
        self.assertTrue(ls.is_polarisation_updated())
        ls.set_simulation_changed()
        self.assertFalse(ls.is_polarisation_updated())

    # set_observations
    def test_load_lglod(self):
        ls = get_lime_simulation()
        lglod: LGLODData = lglodlib.read_lglod_file(
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

    def test_set_observations_unrelated_lglod_srf(self):
        # loading does not check that they are related (at this level)
        ls = get_lime_simulation()
        lglod: LGLODData = lglodlib.read_lglod_file(
            "test_files/moon/simulation.nc", KERNELS_PATH
        )
        srf = get_srf()
        ls.set_observations(lglod, srf)
        ls.get_elis()
        np.testing.assert_array_equal(
            lglod.observations[0].irrs.data, ls.get_elis()[0].data
        )


if __name__ == "__main__":
    unittest.main()
