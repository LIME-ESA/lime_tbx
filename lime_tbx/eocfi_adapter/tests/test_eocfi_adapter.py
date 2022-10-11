"""Tests for eocfi_adapter module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import unittest
import pyproj

"""___LIME_TBX Modules___"""
from ..eocfi_adapter import EOCFIConverter, IEOCFIConverter, _get_file_datetimes


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "30/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

EOCFI_PATH = "./eocfi_data"
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
    return EOCFIConverter(EOCFI_PATH)


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
        lat, lon, h = eo.get_satellite_position("SENTINEL-2A", DT1)[0]
        self.assertEqual(lat, -65.90847446723077)
        self.assertEqual(lon, 10.38388866324515)
        self.assertEqual(h, 791026.5592273567)

    def test_get_satellite_position_true_data(self):
        # data obtained with OSV data calc (https://eop-cfi.esa.int/index.php/applications/tools/command-line-tools-osvdata-calc)
        eo = get_eocfi_converter()
        lat, lon, hhh = eo.get_satellite_position(
            "SENTINEL-5P", [datetime(2022, 1, 2, 0, 0, 0, tzinfo=timezone.utc)]
        )[0]
        transformer = pyproj.Transformer.from_crs(
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        )
        lat2, lon2, hh2 = transformer.transform(
            -3496004.37772468, -404044.1700419, 6279426.45463165, radians=False
        )
        self.assertAlmostEqual(lat, lat2)
        self.assertAlmostEqual(lon, lon2)
        self.assertAlmostEqual(hhh, hh2)


if __name__ == "__main__":
    unittest.main()
