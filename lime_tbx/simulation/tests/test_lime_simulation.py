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


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "25/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = "./eocfi_data"

CH_WLENS = np.array([350, 400, 450, 500])
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
    return LimeSimulation(EOCFI_PATH, KERNELS_PATH, verbose=False)


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
        ls.update_reflectance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        cimels = ls.get_elrefs_cimel()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        elrefs_refs = np.array(
            [
                [0.24393677, 0.0669242, 0.08324007, 0.08303441],
                [0.34973767, 0.07214307, 0.08856552, 0.08820582],
            ]
        )
        uncs_refs = np.array(
            [
                [0.00243937, 0.00066924, 0.00082782, 0.00083034],
                [0.00349738, 0.00072143, 0.00088566, 0.00088206],
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
                [0.00066694, 0.00078409, 0.00088831, 0.0010316, 0.000949, 0.00129523],
                [
                    0.00071119,
                    0.00084021,
                    0.00094079,
                    0.00106406,
                    0.00098252,
                    0.00134816,
                ],
            ]
        )
        for elref, elref_ref, unc in zip(elrefs, elrefs_refs, uncs_refs):
            np.testing.assert_array_almost_equal(elref.data, elref_ref, 3)
            np.testing.assert_array_almost_equal(elref.uncertainties, unc, 5)
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_equal(cimel.data, data)
            np.testing.assert_array_almost_equal(cimel.uncertainties, unc, 4)

    # Function update_irradiance
    def test_update_irradiance(self):
        ls = get_lime_simulation()
        ls.update_irradiance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        elis = ls.get_elis()
        self.assertIsNotNone(elis)
        self.assertIsInstance(elis, list)
        signals = ls.get_signals()
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, SpectralData)
        elis_refs = np.array(
            [
                [4.71002101e-06, 2.15214190e-06, 3.36387167e-06, 3.09612094e-06],
                [7.02065021e-06, 2.41196890e-06, 3.74160725e-06, 3.41937272e-06],
            ]
        )
        uncs_refs = np.array(
            [
                [4.70780879e-08, 2.16128372e-08, 3.30652426e-08, 3.13265180e-08],
                [7.08220248e-08, 2.37747582e-08, 3.81527548e-08, 3.42065865e-08],
            ]
        )
        for eli, eli_ref, unc in zip(elis, elis_refs, uncs_refs):
            np.testing.assert_array_almost_equal(eli.data, eli_ref)
            np.testing.assert_array_almost_equal(eli.uncertainties, unc)
        np.testing.assert_array_almost_equal(
            signals.data, np.array([[3.16668016e-06, 3.75979585e-06]])
        )
        np.testing.assert_array_almost_equal(signals.uncertainties, np.array([[0, 0]]))
        cimels = ls.get_elis_cimel()
        cimel_refs = np.array(
            [
                [
                    2.4952736664368175e-06,
                    3.096120936905104e-06,
                    3.1435125546491005e-06,
                    2.294878010642595e-06,
                    1.814630922972356e-06,
                    8.344850619622514e-07,
                ],
                [
                    2.760907224971663e-06,
                    3.419372719075061e-06,
                    3.4570693403225917e-06,
                    2.514064499346069e-06,
                    1.983194191275128e-06,
                    9.053680025902102e-07,
                ],
            ]
        )
        cimel_unc_refs = np.array(
            [
                [
                    2.27518170e-08,
                    2.80105042e-08,
                    2.47471487e-08,
                    2.02626150e-08,
                    1.22957246e-08,
                    5.92643584e-09,
                ],
                [
                    2.52788677e-08,
                    3.26865432e-08,
                    2.75493808e-08,
                    2.25832274e-08,
                    1.36390032e-08,
                    6.30731765e-09,
                ],
            ]
        )
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_equal(cimel.data, data)
            np.testing.assert_array_almost_equal(cimel.uncertainties, unc, 4)

    # Function update_polarization
    def test_update_polarization(self):
        ls = get_lime_simulation()
        ls.update_polarization(get_srf(), SURFACE_POINT, get_polar_coeffs())
        polars = ls.get_polars()
        self.assertIsNotNone(polars)
        self.assertIsInstance(polars, list)
        polars_refs = np.array(
            [
                [-0.01480269, -0.01480269, -0.01466682, -0.01405207],
                [-0.01451446, -0.01451446, -0.0143594, -0.01366697],
            ]
        )
        uncs_refs = np.array(
            [
                [-0.00014803, -0.00014803, -0.00014667, -0.00014052],
                [-0.00014514, -0.00014514, -0.00014359, -0.00013667],
            ]
        )
        for polar, pref, uref in zip(polars, polars_refs, uncs_refs):
            np.testing.assert_array_almost_equal(polar.data, pref, 5)
            np.testing.assert_array_almost_equal(polar.uncertainties, uref)
        cimels = ls.get_polars_cimel()
        cimel_refs = np.array(
            [
                [
                    -0.014802688966332765,
                    -0.014052070025738387,
                    -0.012747300769599328,
                    -0.012505471715559003,
                    -0.011960469879127763,
                    -0.010608879424041475,
                ],
                [
                    -0.01451445683680234,
                    -0.013666972260325032,
                    -0.012311244374498992,
                    -0.012243784592411986,
                    -0.011790552874105948,
                    -0.010358460669534122,
                ],
            ]
        )
        cimel_unc_refs = np.array(
            [
                [0.00014803, 0.00014052, 0.00012747, 0.00012505, 0.0001196, 0.00010609],
                [
                    0.00014514,
                    0.00013667,
                    0.00012311,
                    0.00012244,
                    0.00011791,
                    0.00010358,
                ],
            ]
        )
        for cimel, data, unc in zip(cimels, cimel_refs, cimel_unc_refs):
            np.testing.assert_array_equal(cimel.data, data)
            np.testing.assert_array_almost_equal(cimel.uncertainties, unc, 4)

    def test_update_irr_polar_verbose(self):
        self.capturedOutput = io.StringIO()
        self.capturedErr = io.StringIO()
        sys.stdout = self.capturedOutput
        sys.stderr = self.capturedErr

        ls = LimeSimulation(EOCFI_PATH, KERNELS_PATH, verbose=True)
        ls.update_irradiance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        ls.update_polarization(get_srf(), SURFACE_POINT, get_polar_coeffs())
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
        ls.update_irradiance(get_srf(), sp, get_cimel_coeffs())
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
        ls.update_irradiance(get_srf(), sp, get_cimel_coeffs())
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
        ls.update_irradiance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        pts = ls.get_surfacepoints()
        self.assertIsNone(pts)

    # get point
    def test_update_get_point(self):
        ls = get_lime_simulation()
        ls.update_reflectance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        elrefs = ls.get_elrefs()
        self.assertIsNotNone(elrefs)
        self.assertIsInstance(elrefs, list)
        pt = ls.get_point()
        self.assertEqual(pt, SURFACE_POINT)

    def test_update_get_point_satellite(self):
        ls = get_lime_simulation()
        ls.update_reflectance(get_srf(), SATELLITE_POINT, get_cimel_coeffs())
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
        ls.update_reflectance(get_srf(), sp, get_cimel_coeffs())
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
        ls.update_reflectance(get_srf(), sp, get_cimel_coeffs())
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
        ls.update_reflectance(get_srf(), sp, get_cimel_coeffs())
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
        ls.update_reflectance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        self.assertFalse(ls.is_polarization_updated())
        ls.update_irradiance(get_srf(), SURFACE_POINT, get_cimel_coeffs())
        self.assertFalse(ls.is_polarization_updated())
        ls.update_polarization(get_srf(), SURFACE_POINT, get_polar_coeffs())
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
