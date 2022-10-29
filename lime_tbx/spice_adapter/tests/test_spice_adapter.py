"""Tests for spice_adapter module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
from typing import List

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from ..spice_adapter import SPICEAdapter, ISPICEAdapter
from ...datatypes.datatypes import KernelsPath, MoonData


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "24/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

LAT = 21
LON = 21
ALT = 2400
KERNELS_PATH = KernelsPath("./kernels", "./kernels")
DT1 = datetime(2022, 1, 17, 2, tzinfo=timezone.utc)
DT2 = datetime(2022, 2, 16, 2, tzinfo=timezone.utc)
MD1 = MoonData(
    0.9863676197729848,
    399227.54900652857,
    0.1343504656066533,
    -4.658809009228347,
    -3.139429310609046,
    11.317038213996295,
    -11.317038213996295,
)
MD2 = MoonData(
    0.9904106311343145,
    389941.3911970312,
    0.049851687502014026,
    -6.15147366076081,
    -5.052383178500747,
    9.11726084520294,
    -9.11726084520294,
)


def get_spice_adapter() -> ISPICEAdapter:
    return SPICEAdapter()


# spicedmoon is a GOA library, which has been tested and used, so we are certain
# that is quite robust, so extremely intense testing is not needed.


class TestSPICEAdapter(unittest.TestCase):

    #### Function: get_moon_data_from_earth

    # One dt
    def test_gmdfe_ok(self):
        sp = get_spice_adapter()
        md: MoonData = sp.get_moon_data_from_earth(LAT, LON, ALT, DT1, KERNELS_PATH)
        self.assertIsInstance(md, MoonData)
        self.assertEqual(md.absolute_mpa_degrees, MD1.absolute_mpa_degrees)
        self.assertEqual(md.distance_observer_moon, MD1.distance_observer_moon)
        self.assertEqual(md.distance_sun_moon, MD1.distance_sun_moon)
        self.assertEqual(md.lat_obs, MD1.lat_obs)
        self.assertEqual(md.long_obs, MD1.long_obs)
        self.assertEqual(md.long_sun_radians, MD1.long_sun_radians)
        self.assertEqual(md.mpa_degrees, MD1.mpa_degrees)

    # List dt
    def test_gmdfe_ok_list_dt(self):
        sp = get_spice_adapter()
        mds: List[MoonData] = sp.get_moon_data_from_earth(
            LAT, LON, ALT, [DT1, DT2], KERNELS_PATH
        )
        self.assertIsInstance(mds, list)
        md1 = mds[0]
        self.assertIsInstance(md1, MoonData)
        self.assertEqual(md1.absolute_mpa_degrees, MD1.absolute_mpa_degrees)
        self.assertEqual(md1.distance_observer_moon, MD1.distance_observer_moon)
        self.assertEqual(md1.distance_sun_moon, MD1.distance_sun_moon)
        self.assertEqual(md1.lat_obs, MD1.lat_obs)
        self.assertEqual(md1.long_obs, MD1.long_obs)
        self.assertEqual(md1.long_sun_radians, MD1.long_sun_radians)
        self.assertEqual(md1.mpa_degrees, MD1.mpa_degrees)
        md2 = mds[1]
        self.assertIsInstance(md2, MoonData)
        self.assertEqual(md2.absolute_mpa_degrees, MD2.absolute_mpa_degrees)
        self.assertEqual(md2.distance_observer_moon, MD2.distance_observer_moon)
        self.assertEqual(md2.distance_sun_moon, MD2.distance_sun_moon)
        self.assertEqual(md2.lat_obs, MD2.lat_obs)
        self.assertEqual(md2.long_obs, MD2.long_obs)
        self.assertEqual(md2.long_sun_radians, MD2.long_sun_radians)
        self.assertEqual(md2.mpa_degrees, MD2.mpa_degrees)

    # Empty list dt
    def test_gmdfe_empty_list_dt(self):
        sp = get_spice_adapter()
        mds = sp.get_moon_data_from_earth(LAT, LON, ALT, [], KERNELS_PATH)
        self.assertIsInstance(mds, list)
        self.assertEqual(len(mds), 0)

    def test_to_rectangular(self):
        x, y, z = SPICEAdapter.to_rectangular(
            LAT, LON, ALT, "EARTH", KERNELS_PATH.main_kernels_path
        )
        # Compared data obtained using library pyproj
        """
        import pyproj
        transformer = pyproj.Transformer.from_crs(
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        )
        x2, y2, z2 = transformer.transform(LON, LAT, ALT/1000, radians=False)
        print("PYPROJ XYZ: ", x2, y2, z2)
        """
        x2, y2, z2 = (5561401.547028502, 2134822.038294565, 2271395.8792928336)
        self.assertAlmostEqual(x, x2, -4)
        self.assertAlmostEqual(y, y2, -4)
        self.assertAlmostEqual(z, z2, -4)

    def test_to_planetographic(self):
        x, y, z = (5563490.882007386, 2135624.0588501, 2272254.9494123333)
        lat, lon, hhh = SPICEAdapter.to_planetographic(
            x, y, z, "EARTH", KERNELS_PATH.main_kernels_path
        )
        # Compared data obtained using library pyproj
        """
        import pyproj
        transformer = pyproj.Transformer.from_crs(
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        )
        lon2, lat2, hhh2 = transformer.transform(x, y, z, radians=False)
        print("PYPROJ DATA: ", lon2, lat2, hhh2)
        """
        lon2, lat2, hhh2 = (21.000000000000004, 20.999999914165222, 2399.598176131025)
        self.assertAlmostEqual(lat, lat2, 6)
        self.assertAlmostEqual(lon, lon2)
        self.assertAlmostEqual(hhh, hhh2, 0)


if __name__ == "__main__":
    unittest.main()
