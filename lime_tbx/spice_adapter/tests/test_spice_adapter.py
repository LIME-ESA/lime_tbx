"""Tests for spice_adapter module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
from typing import List
import random

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
LAT2 = 41
LON = 21
LON2 = -40
ALT = 2400
KERNELS_PATH = KernelsPath("./kernels", "./kernels")
DT1 = datetime(2022, 1, 17, 2, tzinfo=timezone.utc)
DT2 = datetime(2022, 2, 16, 2, tzinfo=timezone.utc)
MD1 = MoonData(
    0.9863676197729848,
    399227.54900652857,
    0.1343504656066533,
    -4.6700417589443175,
    -3.139429310609046,
    11.317038213996295,
    -11.317038213996295,
)
MD2 = MoonData(
    0.9904106311343145,
    389941.3911970312,
    0.049851687502014026,
    -6.1662564380208424,
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
        x2, y2, z2 = SPICEAdapter.to_rectangular(
            LAT2, LON2, ALT, "EARTH", KERNELS_PATH.main_kernels_path
        )
        # Compared data obtained using library pyproj
        """
        import pyproj
        transformer = pyproj.Transformer.from_crs(
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
        )
        xp, yp, zp = transformer.transform(LON, LAT, ALT, radians=False)
        print("PYPROJ XYZ: ", xp, yp, zp)
        """
        xp, yp, zp = (5563491.229045285, 2135624.1920654676, 2272255.1022886634)
        self.assertAlmostEqual(x, xp, 0)
        self.assertAlmostEqual(y, yp, 0)
        self.assertAlmostEqual(z, zp, 0)
        x2p, y2p, z2p = (3694174.4986328664, -3099780.459307351, 4163997.7423554775)
        self.assertAlmostEqual(x2, x2p, 0)
        self.assertAlmostEqual(y2, y2p, 0)
        self.assertAlmostEqual(z2, z2p, 0)

    def test_to_planetographic(self):
        x, y, z = (5563491.229045285, 2135624.1920654676, 2272255.1022886634)
        x2, y2, z2 = (3694174.4986328664, -3099780.459307351, 4163997.7423554775)
        lat, lon, hhh = SPICEAdapter.to_planetographic(
            x, y, z, "EARTH", KERNELS_PATH.main_kernels_path
        )
        lat2, lon2, hhh2 = SPICEAdapter.to_planetographic(
            x2, y2, z2, "EARTH", KERNELS_PATH.main_kernels_path
        )
        # Compared data obtained using library pyproj
        """
        import pyproj
        transformer = pyproj.Transformer.from_crs(
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        )
        lonp, latp, hhhp = transformer.transform(x, y, z, radians=False)
        print("PYPROJ DATA: ", lonp, latp, hhhp)
        """
        self.assertAlmostEqual(lat, LAT, 6)
        self.assertAlmostEqual(lon, LON)
        self.assertAlmostEqual(hhh, ALT, 0)
        self.assertAlmostEqual(lat2, LAT2, 6)
        self.assertAlmostEqual(lon2, LON2)
        self.assertAlmostEqual(hhh2, ALT, 0)

    def test_to_planetographic_multiple(self):
        xyz = [
            (5563491.229045285, 2135624.1920654676, 2272255.1022886634),
            (3694174.4986328664, -3099780.459307351, 4163997.7423554775),
        ]
        llhs = SPICEAdapter.to_planetographic_multiple(
            xyz, "EARTH", KERNELS_PATH.main_kernels_path
        )
        llhps = [(LAT, LON, ALT), (LAT2, LON2, ALT)]
        for llh, llhp in zip(llhs, llhps):
            self.assertAlmostEqual(llh[0], llhp[0], 6)
            self.assertAlmostEqual(llh[1], llhp[1])
            self.assertAlmostEqual(llh[2], llhp[2], 0)

    def test_to_planetographic_same_multiple(self):
        xyzs = [
            tuple(random.randint(-1000000, 1000000) for _ in range(3)) for _ in range(5)
        ]
        llhs = SPICEAdapter.to_planetographic_multiple(
            xyzs, "EARTH", KERNELS_PATH.main_kernels_path
        )
        llhsmanual = [
            SPICEAdapter.to_planetographic(
                xyz[0], xyz[1], xyz[2], "EARTH", KERNELS_PATH.main_kernels_path
            )
            for xyz in xyzs
        ]
        for llh, llhm in zip(llhs, llhsmanual):
            self.assertEqual(
                llh,
                llhm,
                f"Seed used for generation: {random.seed}. Lists: {llh} and {llhm}.",
            )

    def test_to_planetographic_to_rectangular_same(self):
        xyzs = [
            tuple(random.randint(-1000000, 1000000) for _ in range(3)) for _ in range(5)
        ]
        llhs = [
            SPICEAdapter.to_planetographic(
                xyz[0], xyz[1], xyz[2], "EARTH", KERNELS_PATH.main_kernels_path
            )
            for xyz in xyzs
        ]
        xyzs_back = [
            SPICEAdapter.to_rectangular(
                llh[0], llh[1], llh[2], "EARTH", KERNELS_PATH.main_kernels_path
            )
            for llh in llhs
        ]
        for xyz, xyzb in zip(xyzs, xyzs_back):
            err_msg = (
                f"Seed used for generation: {random.seed}. Lists: {xyz} and {xyzb}."
            )
            self.assertAlmostEqual(xyz[0], xyzb[0], msg=err_msg)
            self.assertAlmostEqual(xyz[1], xyzb[1], msg=err_msg)
            self.assertAlmostEqual(xyz[2], xyzb[2], msg=err_msg)


if __name__ == "__main__":
    unittest.main()
