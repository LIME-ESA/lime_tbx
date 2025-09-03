"""Tests for lglod module"""

"""___Built-In Modules___"""
import unittest
from datetime import datetime, timezone

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""

from lime_tbx.common.datatypes import (
    KernelsPath,
    LGLODComparisonData,
    LGLODData,
    EocfiPath,
)
from .. import lglod
from ..srf import read_srf

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "15/02/2025"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

PATH = "./test_files/moon/W_XX-EUMETSAT-Darmstadt,VISNIR+SUBSET+MOON,MSG3+SEVIRI_C_EUMG_20140318140112_01.nc"

KERNELS_PATH = KernelsPath("./kernels", "./kernels")
EOCFI_PATH = EocfiPath("./eocfi_data", "./eocfi_data2")


class TestMoon(unittest.TestCase):
    def test_read_lglod_sim(self):
        path = "./test_files/moon/simulation.nc"
        data = lglod.read_lglod_file(path, KERNELS_PATH)
        self.assertFalse(data.not_default_srf)
        self.assertIsInstance(data, LGLODData)

    def test_read_lglod_comp(self):
        path = "./test_files/moon/comparison.nc"
        srf = read_srf("./test_files/moon/cimel_srf.nc")
        cdata = lglod.read_lglod_file(path, KERNELS_PATH)
        self.assertIsInstance(cdata, LGLODComparisonData)
        for ch_name in cdata.ch_names:
            self.assertIn(ch_name, srf.get_channels_names())

    def test_read_lglod_sim_multiseleno(self):
        path = "./test_files/moon/sim_multiseleno.nc"
        data = lglod.read_lglod_file(path, KERNELS_PATH)
        self.assertFalse(data.not_default_srf)
        self.assertIsInstance(data, LGLODData)
        self.assertEqual(len(data.elis_cimel), 2)
        self.assertEqual(data.signals.data.shape, (1, 2))

    def test_read_lglod_sim_multiseleno_cimelsrf(self):
        path = "./test_files/moon/sim_multiseleno_cimel.nc"
        data = lglod.read_lglod_file(path, KERNELS_PATH)
        self.assertTrue(data.not_default_srf)
        self.assertIsInstance(data, LGLODData)
        self.assertEqual(len(data.elis_cimel), 2)
        self.assertEqual(data.signals.data.shape, (6, 2))

    def test_read_write_lglod_multiseleno(self):
        path = "./test_files/moon/sim_multiseleno.nc"
        data = lglod.read_lglod_file(path, KERNELS_PATH)
        tpath = "./test_files/moon/sim_multiseleno.test.nc"
        now = datetime.now(timezone.utc)
        inside_mpa_range = [True for _ in range(len(data.elis_cimel))]
        lglod.write_obs(data, tpath, now, inside_mpa_range)
        tdata = lglod.read_lglod_file(tpath, KERNELS_PATH)
        np.testing.assert_array_equal(data.elis_cimel[0].data, tdata.elis_cimel[0].data)
        np.testing.assert_array_equal(
            data.elis_cimel[0].wlens, tdata.elis_cimel[0].wlens
        )
        np.testing.assert_array_equal(data.elis_cimel[1].data, tdata.elis_cimel[1].data)
        np.testing.assert_array_equal(
            data.elis_cimel[1].wlens, tdata.elis_cimel[1].wlens
        )
        np.testing.assert_array_equal(data.signals.wlens, tdata.signals.wlens)
        np.testing.assert_array_equal(data.signals.data, tdata.signals.data)

    def test_read_write_lglod_multiseleno_cimel(self):
        path = "./test_files/moon/sim_multiseleno_cimel.nc"
        data = lglod.read_lglod_file(path, KERNELS_PATH)
        tpath = "./test_files/moon/sim_multiseleno_cimel.test.nc"
        now = datetime.now(timezone.utc)
        inside_mpa_range = [True for _ in range(len(data.elis_cimel))]
        lglod.write_obs(data, tpath, now, inside_mpa_range)
        tdata = lglod.read_lglod_file(tpath, KERNELS_PATH)
        np.testing.assert_array_equal(data.elis_cimel[0].data, tdata.elis_cimel[0].data)
        np.testing.assert_array_equal(
            data.elis_cimel[0].wlens, tdata.elis_cimel[0].wlens
        )
        np.testing.assert_array_equal(data.elis_cimel[1].data, tdata.elis_cimel[1].data)
        np.testing.assert_array_equal(
            data.elis_cimel[1].wlens, tdata.elis_cimel[1].wlens
        )
        np.testing.assert_array_equal(data.signals.wlens, tdata.signals.wlens)
        np.testing.assert_array_equal(data.signals.data, tdata.signals.data)


if __name__ == "__main__":
    unittest.main()
