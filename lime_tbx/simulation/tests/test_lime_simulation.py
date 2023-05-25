"""Tests for lime_simulation module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import sys
import io

"""___Third-Party Modules___"""
import unittest
import numpy as np

"""___LIME_TBX Modules___"""
from ..lime_simulation import ILimeSimulation, LimeSimulation, is_ampa_valid_range
from ...datatypes.datatypes import (
    PolarizationCoefficients,
    ReflectanceCoefficients,
    SpectralData,
    SpectralResponseFunction,
    SRFChannel,
    SurfacePoint,
    SatellitePoint,
    KernelsPath,
    LGLODData,
)
from ...coefficients.access_data.access_data import (
    _get_default_polarization_coefficients,
    _get_demo_cimel_coeffs,
)
from ...filedata import moon, srf as srflib
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
EOCFI_PATH = "./eocfi_data"

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


def get_polar_coeffs() -> PolarizationCoefficients:
    return _get_default_polarization_coefficients()


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
                [0.033, 0.059, 0.076, 0.083],
                [0.036, 0.063, 0.081, 0.088],
            ]
        )
        uncs_refs = np.array(
            [
                [4.5e-04, 6.7e-04, 8.1e-04, 1.5e-05],
                [5.2e-04, 7.1e-04, 8.1e-04, 1.5e-05],
            ]
        )
        cimel_refs = np.array(
            [
                [
                    0.07238056082331103,
                    0.08303440568286098,
                    0.10680356018219804,
                    0.12074701286894197,
                    0.1308906606347773,
                    0.1819942973421718,
                ],
                [
                    0.0770311205388483,
                    0.08820582157140683,
                    0.11297679824393579,
                    0.12723419553845355,
                    0.13759296700829238,
                    0.18992187815923675,
                ],
            ]
        )
        cimel_unc_refs = np.array(
            [
                [
                    1.4840e-05,
                    1.6218e-05,
                    1.6754e-05,
                    1.9120e-05,
                    1.6623e-05,
                    1.9480e-05,
                ],
                [
                    1.6008e-05,
                    1.7083e-05,
                    1.7156e-05,
                    1.9426e-05,
                    1.7426e-05,
                    1.9985e-05,
                ],
            ]
        )
        for elref, elref_ref, unc in zip(elrefs, elrefs_refs, uncs_refs):
            np.testing.assert_array_almost_equal(
                elref.data[CH_DEF_INDICES], elref_ref, 3
            )
            np.testing.assert_array_almost_equal(
                elref.uncertainties[CH_DEF_INDICES], unc, 4
            )
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
                [0.033, 0.059, 0.076, 0.083],
                [0.036, 0.063, 0.081, 0.088],
            ]
        )
        uncs_refs = np.zeros(elrefs_refs.shape)
        cimel_refs = np.array(
            [
                [
                    0.07238056082331103,
                    0.08303440568286098,
                    0.10680356018219804,
                    0.12074701286894197,
                    0.1308906606347773,
                    0.1819942973421718,
                ],
                [
                    0.0770311205388483,
                    0.08820582157140683,
                    0.11297679824393579,
                    0.12723419553845355,
                    0.13759296700829238,
                    0.18992187815923675,
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

    # Function update_irradiance
    def test_update_irradiance(self):
        ls = get_lime_simulation()
        ls._skip_uncs = False
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
        uncs_refs = np.array(
            [
                [1.2e-08, 2.2e-08, 3.30652426e-08, 1e-08],
                [1.3e-08, 2.5e-08, 4e-08, 1e-08],
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
        cimel_unc_refs = np.array(
            [
                [2e-08, 1e-08, 1e-08, 7e-09, 6e-09, 3e-09],
                [2e-08, 1e-08, 1e-08, 1e-09, 6e-09, 3e-09],
            ]
        )
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_almost_equal(cimel.data, data)
            np.testing.assert_array_almost_equal(cimel.uncertainties, unc, 8)

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

    # Function update_polarization
    def test_update_polarization(self):
        ls = get_lime_simulation()
        ls._skip_uncs = False
        ls.update_polarization(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)
        polars_refs = np.array(
            [
                [0.08249, 0.08154, 0.06838, 0.06093],
                [0.05846, 0.05787, 0.0495, 0.04407],
            ]
        )
        uncs_refs = np.array(
            [
                [8.91e-10, 1.06e-9, 4.74e-05, 1e-16],
                [4.82e-10, 5.28e-10, 2.7e-06, 7.25e-17],
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
                    0.07129743958463086,
                    0.060932033847697825,
                    0.05597839201872012,
                    0.05249721820731961,
                    0.05286010644891919,
                    0.05716205906906083,
                ],
                [
                    0.05147480895004593,
                    0.044075371275099785,
                    0.04050823849767443,
                    0.03804512926660234,
                    0.03832148558139749,
                    0.041351801850708816,
                ],
            ]
        )
        cimel_unc_refs = np.array(
            [
                [1.94e-16, 8e-17, 5.5e-17, 4e-17, 1.4e-17, 7.5e-17],
                [4.85e-17, 4.85e-17, 3.5e-17, 5.5e-17, 7e-17, 6.2e-17],
            ]
        )
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_equal(cimel.data, data)
            np.testing.assert_allclose(cimel.uncertainties, unc, rtol=0.4)

    def test_update_polarization_skip_uncs(self):
        ls = get_lime_simulation()
        ls._skip_uncs = True
        ls.update_polarization(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)
        polars_refs = np.array(
            [
                [0.08249, 0.08154, 0.06838, 0.06093],
                [0.05846, 0.05787, 0.0495, 0.04407],
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
                    0.07129743958463086,
                    0.060932033847697825,
                    0.05597839201872012,
                    0.05249721820731961,
                    0.05286010644891919,
                    0.05716205906906083,
                ],
                [
                    0.05147480895004593,
                    0.044075371275099785,
                    0.04050823849767443,
                    0.03804512926660234,
                    0.03832148558139749,
                    0.041351801850708816,
                ],
            ]
        )
        cimel_unc_refs = np.zeros(cimel_refs.shape)
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_equal(cimel.data, data)
            np.testing.assert_array_equal(cimel.uncertainties, unc)

    def test_update_irr_polar_verbose(self):
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr

        ls = LimeSimulation(EOCFI_PATH, KERNELS_PATH, SettingsManager(), verbose=True)
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        ls.update_polarization(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
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
            "irradiance update done",
            "signals update done",
            "starting polarisation update",
            "polarisation update done\n",
        ]
        verbose_output = "\n".join(verbose_output_l)
        self.assertEqual(self.capturedOutput.getvalue(), verbose_output)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # get_surfacepoints
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

    def test_get_surfacepoints_not_sat_point(self):
        ls = get_lime_simulation()
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        pts = ls.get_surfacepoints()
        self.assertIsNone(pts)

    # get point
    def test_update_get_point(self):
        ls = get_lime_simulation()
        ls.update_reflectance(get_default_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        pt = ls.get_point()
        self.assertEqual(pt, SURFACE_POINT)

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
            self.assertEqual(md, mdr)

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
            self.assertEqual(md, mdr)

    # is_ampa_valid_range
    def test_is_ampa_valid_range_vals(self):
        ampas = [0, 1, 2, 3, 5, 30, 89, 90, 91, 180]
        vals = [False, False, True, True, True, True, True, True, False, False]
        for ampa, val in zip(ampas, vals):
            valcalc = is_ampa_valid_range(ampa)
            self.assertEqual(valcalc, val)

    # are_mpas_inside_mpa_range
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

    # is_polarization_updated
    def test_is_polarization_updated(self):
        ls = get_lime_simulation()
        self.assertFalse(ls.is_polarization_updated())
        ls.update_reflectance(get_default_srf(), SURFACE_POINT, get_cimel_coeffs())
        self.assertFalse(ls.is_polarization_updated())
        ls.update_irradiance(
            get_default_srf(), get_srf(), SURFACE_POINT, get_cimel_coeffs()
        )
        self.assertFalse(ls.is_polarization_updated())
        ls.update_polarization(get_default_srf(), SURFACE_POINT, get_polar_coeffs())
        self.assertTrue(ls.is_polarization_updated())
        ls.set_simulation_changed()
        self.assertFalse(ls.is_polarization_updated())

    # set_observations
    def test_load_lglod(self):
        ls = get_lime_simulation()
        lglod: LGLODData = moon.read_lglod_file(
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
        lglod: LGLODData = moon.read_lglod_file(
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
