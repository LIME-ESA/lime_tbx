"""Tests for eocfi_adapter module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
import os

"""___Third-Party Modules___"""
import unittest
import numpy as np
import spiceypy as spice

"""___LIME_TBX Modules___"""
from ..eocfi_adapter import EOCFIConverter, _get_file_datetimes
from lime_tbx.common.datatypes import KernelsPath, LimeException, EocfiPath
from lime_tbx.business.spice_adapter.spice_adapter import SPICEAdapter


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "30/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

EOCFI_PATH = EocfiPath("./eocfi_data", "./eocfi_data2")
KERNELS_PATH = KernelsPath("./kernels", "./kernels")
MANDATORY_SATS = [
    "ENVISAT",
    "PROBA-V",
    "SENTINEL-2A",
    "SENTINEL-2B",
    "SENTINEL-3A",
    "SENTINEL-3B",
    "FLEX",
]
_BASIC_KERNELS = [
    "pck00010.tpc",
    "naif0011.tls",
    "earth_assoc_itrf93.tf",
    "de421.bsp",
    "earth_latest_high_prec.bpc",
    "earth_070425_370426_predict.bpc",
]
_MOON_KERNELS = [
    "moon_pa_de421_1900-2050.bpc",
    "moon_080317.tf",
]


DT1 = datetime(2016, 1, 1, 15, 0, 2, tzinfo=timezone.utc)


def get_eocfi_converter() -> EOCFIConverter:
    return EOCFIConverter(EOCFI_PATH, KERNELS_PATH)


class TestEOCFIConverter(unittest.TestCase):
    def test__get_file_datetimes(self):
        dts = _get_file_datetimes(
            "SENTINEL2A/OSF/S2A_OPER_MPL_ORBSCT_20150625T073255_99999999T999999_0008.EOF"
        )
        self.assertEqual(dts[0], datetime(2015, 6, 25, 7, 32, 55, tzinfo=timezone.utc))
        self.assertEqual(
            dts[1], datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        )
        dts = _get_file_datetimes(
            "SMOS/OSF/SM_OPER_MPL_ORBSCT_20091102T031142_20500101T000000_4540031.EEF"
        )
        self.assertEqual(dts[0], datetime(2009, 11, 2, 3, 11, 42, tzinfo=timezone.utc))
        self.assertEqual(dts[1], datetime(2050, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

    def test_get_sat_names_mandatory(self):
        eo = get_eocfi_converter()
        names = eo.get_sat_names()
        for name in MANDATORY_SATS:
            self.assertIn(
                name,
                names,
                "The available satellites list doesn't include all mandatory satellites.",
            )

    def test_get_sat_list_ok(self):
        eo = get_eocfi_converter()
        sat_list = eo.get_sat_list()
        self.assertEqual(len(sat_list), len(eo.get_sat_names()))

    def test_get_satellite_position_ok(self):
        eo = get_eocfi_converter()
        supposed_lat = 10.361596356893193
        supposed_h = 791027.2075147488
        lat, lon, h = eo.get_satellite_position("SENTINEL-2A", [DT1])[0]
        self.assertEqual(lon, -65.83522069800632)
        self.assertAlmostEqual(lat, supposed_lat)
        self.assertAlmostEqual(h, supposed_h)
        if lat != supposed_lat:
            print(
                "\nWARNING: In test_get_satellite_position_ok, lat != supposed_lat. Unexplained Windows behaviour.\n {} != {}\n".format(
                    lat, supposed_lat
                )
            )
        if h != supposed_h:
            print(
                "\nWARNING: In test_get_satellite_position_ok, h != supposed_h. Unexplained Windows behaviour.\n {} != {}\n".format(
                    h, supposed_h
                )
            )

    def test_get_satellite_position_naive_datetime(self):
        eo = get_eocfi_converter()
        self.assertRaises(
            TypeError,
            eo.get_satellite_position,
            "SENTINEL-2A",
            [datetime(2020, 1, 1, 1, 1, 1)],
        )

    def test_get_satellite_position_outrange(self):
        eo = get_eocfi_converter()
        self.assertRaises(
            LimeException,
            eo.get_satellite_position,
            "SENTINEL-2A",
            [datetime(2000, 1, 1, 1, 1, 1, tzinfo=timezone.utc)],
        )

    def test_get_satellite_position_non_existing_exception(self):
        eo = get_eocfi_converter()
        self.assertRaises(Exception, eo.get_satellite_position, "MISSINGSAT", [DT1])

    def test_get_satellite_position_true_data(self):
        # data obtained with OSV data calc (https://eop-cfi.esa.int/index.php/applications/tools/command-line-tools-osvdata-calc)
        eo = get_eocfi_converter()
        dts = [datetime(2022, 1, 2, 0, 0, 0, tzinfo=timezone.utc)]
        lat, lon, hhh = eo.get_satellite_position("SENTINEL-5P", dts)[0]
        lat2, lon2, hh2 = SPICEAdapter.to_planetographic_multiple(
            [
                (
                    -3496004.37772468,
                    -404044.1700419,
                    6279426.45463165,
                )
            ],
            "EARTH",
            KERNELS_PATH.main_kernels_path,
            dts,
            "ITRF93",
        )[0]
        self.assertAlmostEqual(lat, lat2)
        self.assertAlmostEqual(lon, lon2)
        self.assertAlmostEqual(hhh, hh2)

    def test_get_satellite_position_envisat_ok(self):
        # data obtained from .EOF file supplied by ESA
        """
        <OSV>
            <TAI>TAI=2006-07-20T02:39:02.989786</TAI>
            <UTC>UTC=2006-07-20T02:38:29.989786</UTC>
            <UT1>UT1=2006-07-20T02:38:29.989786</UT1>
            <Absolute_Orbit>+22933</Absolute_Orbit>
            <X unit="m">+2500149.266</X>
            <Y unit="m">-6714929.064</Y>
            <Z unit="m">-0000000.000</Z>
            <VX unit="m/s">-1530.874437</VX>
            <VY unit="m/s">-0560.873432</VY>
            <VZ unit="m/s">+7377.493385</VZ>
            <Quality>0000000000000</Quality>
        </OSV>
        """
        eo = get_eocfi_converter()
        dts = [datetime(2006, 7, 20, 2, 38, 29, 989786, tzinfo=timezone.utc)]
        lat, lon, hhh = eo.get_satellite_position(
            "ENVISAT",
            dts,
        )[0]
        lat2, lon2, hh2 = SPICEAdapter.to_planetographic_multiple(
            [(2500149.266, -6714929.064, -0)],
            "EARTH",
            KERNELS_PATH.main_kernels_path,
            dts,
            "ITRF93",
        )[0]
        self.assertAlmostEqual(lat, lat2)
        self.assertAlmostEqual(lon, lon2, 2)
        self.assertAlmostEqual(hhh, hh2, 3)

    def test_get_satellite_position_probav_ok(self):
        # data obtained from .EOF file supplied by ESA
        """
        <OSV>
            <TAI>TAI=2018-05-03T08:48:12.370522</TAI>
            <UTC>UTC=2018-05-03T08:47:35.370522</UTC>
            <UT1>UT1=2018-05-03T08:47:35.370522</UT1>
            <Absolute_Orbit>+25914</Absolute_Orbit>
            <X unit="m">-6616761.405</X>
            <Y unit="m">-2836593.220</Y>
            <Z unit="m">+0000000.000</Z>
            <VX unit="m/s">-0634.280972</VX>
            <VY unit="m/s">+1503.649992</VY>
            <VZ unit="m/s">+7358.394000</VZ>
            <Quality>0000000000000</Quality>
        </OSV>
        """
        eo = get_eocfi_converter()
        dts = [datetime(2018, 5, 3, 8, 47, 35, 370522, tzinfo=timezone.utc)]
        xyz_esa = np.array((-6616761.405, -2836593.220, 0))
        xyz = np.array(eo.get_satellite_position_rectangular("PROBA-V", dts)[0])
        np.testing.assert_array_almost_equal(xyz, xyz_esa, 3)
        lat, lon, hhh = eo.get_satellite_position("PROBA-V", dts)[0]
        lat2, lon2, hh2 = SPICEAdapter.to_planetographic_multiple(
            [xyz_esa],
            "EARTH",
            KERNELS_PATH.main_kernels_path,
            dts,
            "ITRF93",
        )[0]
        self.assertAlmostEqual(lat, lat2)
        self.assertAlmostEqual(lon, lon2)
        self.assertAlmostEqual(hhh, hh2, 3)

    def test_get_against_sentinel3a(self):
        # The coordinates are taken from SW2 GLOD datafiles
        coords0 = np.array(
            [
                [-1367.947271, -6186.552006, -3393.027741],
                [724.439474, 6548.852351, 2852.697976],
                [-5980.021454, 3961.94646, -396.730665],
                [-5459.598289, -3033.917013, -3556.042138],
                [-270.294847, -5953.919906, -4017.467633],
            ]
        )
        sat_pos_ref = "J2000"
        dates = [
            datetime(2020, 7, 4, 18, 13, 5, 413824, tzinfo=timezone.utc),
            datetime(2022, 1, 18, 9, 6, 39, 651946, tzinfo=timezone.utc),
            datetime(2022, 3, 18, 16, 54, 3, 527799, tzinfo=timezone.utc),
            datetime(2022, 5, 16, 17, 16, 38, 629877, tzinfo=timezone.utc),
            datetime(2022, 7, 14, 1, 36, 35, 327542, tzinfo=timezone.utc),
        ]
        eo = get_eocfi_converter()
        coords1 = (
            np.array(eo.get_satellite_position_rectangular("SENTINEL-3A", dates)) / 1000
        )
        SPICEAdapter._load_kernels(KERNELS_PATH.main_kernels_path)
        coords1 = np.array(
            [
                SPICEAdapter._change_frames(
                    coord, "ITRF93", sat_pos_ref, spice.datetime2et(date)
                )
                for coord, date in zip(coords1, dates)
            ]
        )
        SPICEAdapter._clear_kernels()
        distances = np.linalg.norm(coords0 - coords1, axis=1)
        max_dist = 14000
        for c0, c1, dist in zip(coords0, coords1, distances):
            msg = f"Distance too big. {dist} > {max_dist}. Coordinates: {c0} and {c1}"
            self.assertLessEqual(dist, max_dist, msg)
        np.testing.assert_allclose(coords0, coords1, 2, 3000)

    def test_get_against_sentinel3b(self):
        # The coordinates are taken from SW2 GLOD datafiles
        coords0 = np.array(
            [
                [956.428977, -6474.181784, -2969.738931],
                [-2917.455755, 6359.521681, 1621.118477],
                [-6811.023693, 235.044077, -2279.050897],
                [-3009.816054, -5029.841617, -4159.906603],
            ]
        )
        sat_pos_ref = "J2000"
        dates = [
            datetime(2018, 7, 27, 7, 22, 43, 462684, tzinfo=timezone.utc),
            datetime(2022, 2, 17, 0, 21, 44, 30567, tzinfo=timezone.utc),
            datetime(2022, 4, 17, 7, 26, 50, 352449, tzinfo=timezone.utc),
            datetime(2022, 6, 15, 0, 8, 20, 171066, tzinfo=timezone.utc),
        ]
        eo = get_eocfi_converter()
        coords1 = (
            np.array(eo.get_satellite_position_rectangular("SENTINEL-3B", dates)) / 1000
        )
        SPICEAdapter._load_kernels(KERNELS_PATH.main_kernels_path)
        coords1 = np.array(
            [
                SPICEAdapter._change_frames(
                    coord, "ITRF93", sat_pos_ref, spice.datetime2et(date)
                )
                for coord, date in zip(coords1, dates)
            ]
        )
        SPICEAdapter._clear_kernels()
        distances = np.linalg.norm(coords0 - coords1, axis=1)
        max_dist = 14000
        for c0, c1, dist in zip(coords0, coords1, distances):
            msg = f"Distance too big. {dist} > {max_dist}. Coordinates: {c0} and {c1}"
            self.assertLessEqual(dist, max_dist, msg)
        np.testing.assert_allclose(coords0, coords1, 2, 3000)

    def test_get_against_probav(self):
        # The coordinates are taken from SW2 GLOD datafiles
        coords0 = np.array(
            [
                [-673.04352, 1786.783284, -6945.481248],
                [-799.559962, 678.006872, 7109.660942],
                [-4554.017121, -5322.674454, 1627.142592],
            ]
        )
        sat_pos_ref = "J2000"
        dates = [
            datetime(2017, 9, 6, 22, 0, 31, 939209, tzinfo=timezone.utc),
            datetime(2023, 8, 5, 19, 20, 53, 203194, tzinfo=timezone.utc),
            datetime(2023, 7, 31, 2, 18, 18, 489596, tzinfo=timezone.utc),
        ]
        eo = get_eocfi_converter()
        coords1 = (
            np.array(eo.get_satellite_position_rectangular("PROBA-V", dates)) / 1000
        )
        SPICEAdapter._load_kernels(KERNELS_PATH.main_kernels_path)
        coords1 = np.array(
            [
                SPICEAdapter._change_frames(
                    coord, "ITRF93", sat_pos_ref, spice.datetime2et(date)
                )
                for coord, date in zip(coords1, dates)
            ]
        )
        SPICEAdapter._clear_kernels()
        distances = np.linalg.norm(coords0 - coords1, axis=1)
        max_dist = 8000
        for c0, c1, dist in zip(coords0, coords1, distances):
            msg = f"Distance too big. {dist} > {max_dist}. Coordinates: {c0} and {c1}"
            self.assertLessEqual(dist, max_dist, msg)
        np.testing.assert_allclose(coords0, coords1, 1, 5000)

    def test_get_against_pleiades1b(self):
        # The coordinates are taken from SW2 GLOD datafiles
        coords0 = np.array(
            [
                [376047.094026, -17931.27136, 19077.743743],
                [403764.134505, 25331.937873, -1241.291837],
            ]
        )
        sat_pos_ref = "IAU_MOON"
        dates = [
            datetime(2013, 3, 1, 7, 9, 54, 999991, tzinfo=timezone.utc),
            datetime(2016, 4, 18, 14, 30, 24, 4, tzinfo=timezone.utc),
        ]
        eo = get_eocfi_converter()
        coords1 = (
            np.array(eo.get_satellite_position_rectangular("PLEIADES 1B", dates)) / 1000
        )
        SPICEAdapter._load_kernels(KERNELS_PATH.main_kernels_path)
        coords1 = np.array(
            [
                SPICEAdapter._change_frames(
                    coord, "ITRF93", sat_pos_ref, spice.datetime2et(date)
                )
                for coord, date in zip(coords1, dates)
            ]
        )
        SPICEAdapter._clear_kernels()
        distances = np.linalg.norm(coords0 - coords1, axis=1)
        max_dist = 11000
        for c0, c1, dist in zip(coords0, coords1, distances):
            msg = f"Distance too big. {dist} > {max_dist}. Coordinates: {c0} and {c1}"
            self.assertLessEqual(dist, max_dist, msg)
        np.testing.assert_allclose(coords0, coords1, 1, 5000)


if __name__ == "__main__":
    unittest.main()
