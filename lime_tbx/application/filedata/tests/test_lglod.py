"""Tests for lglod module"""

"""___Built-In Modules___"""
pass

"""___Third-Party Modules___"""
import unittest

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


if __name__ == "__main__":
    unittest.main()
