"""Tests for spice_adapter module"""

"""___Built-In Modules___"""
from datetime import datetime
from typing import List

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from ..spice_adapter import SPICEAdapter, ISPICEAdapter
from ...datatypes.datatypes import MoonData


"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "24/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

LAT = 21
LON = 21
ALT = 2400
KERNELS_PATH = "./kernels"
DT1 = datetime(2022, 1, 17, 2)
DT2 = datetime(2022, 2, 16, 2)
MD1 = MoonData(
    0.9863616457381059,
    398239.6861064414,
    0.14317166573418066,
    -4.714231814650785,
    -2.971616952867521,
    11.655526370063395,
    -11.655526370063395,
)
MD2 = MoonData(
    0.9904004369336903,
    389056.3599474183,
    0.05867768649163748,
    -6.246903062051175,
    -4.90385616224728,
    9.473277579252443,
    -9.473277579252443,
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


if __name__ == "__main__":
    unittest.main()
