"""Tests for moon_data_factory module"""

"""___Built-In Modules___"""
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from ..moon_data_factory import MoonDataFactory
from ...datatypes.datatypes import (
    CustomPoint,
    MoonData,
    SatellitePoint,
    SurfacePoint,
    KernelsPath,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "25/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = "./eocfi_data"

LAT = 21
LON = 21
ALT = 2400
DT1 = datetime(2022, 1, 17, 2, tzinfo=timezone.utc)
DT2 = datetime(2022, 2, 16, 2, tzinfo=timezone.utc)

SURFACE_POINT = SurfacePoint(LAT, LON, ALT, DT1)
SURFACE_POINT_DTS = SurfacePoint(LAT, LON, ALT, [DT1, DT2])
CUSTOM_POINT = CustomPoint(
    0.9863676197729848,
    401782.2206941528,
    -4.713616769942523,
    -1.5456745748151752,
    0.1343504656066533,
    9.824985380107215,
    -9.824985380107215,
)
SATELLITE_POINT = SatellitePoint("BIOMASS", DT1)


class TestMoonDataFactory(unittest.TestCase):
    # Function get_md
    def test_get_md_spoint(self):
        md = MoonDataFactory.get_md(SURFACE_POINT, EOCFI_PATH, KERNELS_PATH)
        self.assertIsInstance(md, MoonData)

    def test_get_md_spoint_dts(self):
        mds = MoonDataFactory.get_md(SURFACE_POINT_DTS, EOCFI_PATH, KERNELS_PATH)
        self.assertIsInstance(mds, list)
        self.assertIsInstance(mds[0], MoonData)

    def test_get_md_cpoint(self):
        md = MoonDataFactory.get_md(CUSTOM_POINT, EOCFI_PATH, KERNELS_PATH)
        self.assertIsInstance(md, MoonData)

    def test_get_md_satpoint(self):
        md = MoonDataFactory.get_md(SATELLITE_POINT, EOCFI_PATH, KERNELS_PATH)
        self.assertIsInstance(md, MoonData)

    # Function get_md_from_surface
    def test_get_md_from_surface_ok(self):
        md = MoonDataFactory.get_md_from_surface(SURFACE_POINT, KERNELS_PATH)
        cp = CUSTOM_POINT
        self.assertEqual(md.distance_observer_moon, cp.distance_observer_moon)
        self.assertEqual(md.distance_sun_moon, cp.distance_sun_moon)
        self.assertEqual(md.lat_obs, cp.selen_obs_lat)
        self.assertEqual(md.long_obs, cp.selen_obs_lon)
        self.assertEqual(md.long_sun_radians, cp.selen_sun_lon)
        self.assertEqual(md.mpa_degrees, cp.moon_phase_angle)
        self.assertEqual(md.absolute_mpa_degrees, cp.abs_moon_phase_angle)

    def test_get_md_from_surface_list_ok(self):
        mds = MoonDataFactory.get_md_from_surface(SURFACE_POINT_DTS, KERNELS_PATH)
        cp = CUSTOM_POINT
        md = mds[0]
        self.assertEqual(md.distance_observer_moon, cp.distance_observer_moon)
        self.assertEqual(md.distance_sun_moon, cp.distance_sun_moon)
        self.assertEqual(md.lat_obs, cp.selen_obs_lat)
        self.assertEqual(md.long_obs, cp.selen_obs_lon)
        self.assertEqual(md.long_sun_radians, cp.selen_sun_lon)
        self.assertEqual(md.mpa_degrees, cp.moon_phase_angle)
        self.assertEqual(md.absolute_mpa_degrees, cp.abs_moon_phase_angle)

    # Function get_md_from_custom
    def test_get_md_from_custom_ok(self):
        md = MoonDataFactory.get_md_from_custom(CUSTOM_POINT)
        cp = CUSTOM_POINT
        self.assertEqual(md.distance_observer_moon, cp.distance_observer_moon)
        self.assertEqual(md.distance_sun_moon, cp.distance_sun_moon)
        self.assertEqual(md.lat_obs, cp.selen_obs_lat)
        self.assertEqual(md.long_obs, cp.selen_obs_lon)
        self.assertEqual(md.long_sun_radians, cp.selen_sun_lon)
        self.assertEqual(md.mpa_degrees, cp.moon_phase_angle)
        self.assertEqual(md.absolute_mpa_degrees, cp.abs_moon_phase_angle)

    # Function get_md_from_satellite
    def test_get_md_from_satellite_ok(self):
        md = MoonDataFactory.get_md_from_satellite(
            SATELLITE_POINT, EOCFI_PATH, KERNELS_PATH
        )
        self.assertEqual(md.distance_sun_moon, 0.9863676197729848)
        self.assertEqual(md.distance_observer_moon, 406214.0769307725)
        self.assertEqual(md.long_sun_radians, 0.1343504656066533)
        self.assertEqual(md.lat_obs, -5.234187076763817)
        self.assertEqual(md.long_obs, -3.2374416337423626)
        self.assertEqual(md.absolute_mpa_degrees, 11.58747830181146)
        self.assertEqual(md.mpa_degrees, -11.58747830181146)


if __name__ == "__main__":
    unittest.main()
