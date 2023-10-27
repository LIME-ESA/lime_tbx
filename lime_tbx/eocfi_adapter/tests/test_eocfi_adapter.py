"""Tests for eocfi_adapter module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from ..eocfi_adapter import EOCFIConverter, IEOCFIConverter, _get_file_datetimes
from lime_tbx.datatypes.datatypes import KernelsPath, LimeException
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "30/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

EOCFI_PATH = "./eocfi_data"
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


DT1 = datetime(2016, 1, 1, 15, 0, 2, tzinfo=timezone.utc)


def get_eocfi_converter() -> IEOCFIConverter:
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
        supposed_lat = 10.381308335388914
        supposed_h = 791026.6206381248
        lat, lon, h = eo.get_satellite_position("SENTINEL-2A", [DT1])[0]
        self.assertEqual(lon, -65.8307798442806)
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
        lat, lon, hhh = eo.get_satellite_position("PROBA-V", dts)[0]
        lat2, lon2, hh2 = SPICEAdapter.to_planetographic_multiple(
            [(-6616761.405, -2836593.220, 0)],
            "EARTH",
            KERNELS_PATH.main_kernels_path,
            dts,
            "ITRF93",
        )[0]
        self.assertAlmostEqual(lat, lat2)
        self.assertAlmostEqual(lon, lon2)
        self.assertAlmostEqual(hhh, hh2, 3)


if __name__ == "__main__":
    unittest.main()
