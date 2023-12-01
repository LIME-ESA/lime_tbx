"""Tests for access_data module"""

"""___Built-In Modules___"""
import os

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
from ..access_data import AccessData, IAccessData
from lime_tbx.filedata import coefficients

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "29/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def get_access_data() -> IAccessData:
    return AccessData()


def in_development_repo() -> bool:
    return os.path.exists(os.path.join(".", "coeff_data", "versions"))


not_dev_str = "Development coefficients folder not found."


class TestAccessData(unittest.TestCase):
    # If something fails check that the lime toolbox is NOT installed in the machine

    @unittest.skipIf(not in_development_repo, not_dev_str)
    def test_get_all_coefficients(self):
        ad = get_access_data()
        cfs = ad.get_all_coefficients()
        self.assertGreater(len(cfs), 0)
        folder = os.path.join(".", "coeff_data", "versions")
        version_files = sorted(os.listdir(folder))
        cfs_check = []
        for vf in version_files:
            cf = coefficients.read_coeff_nc(os.path.join(folder, vf))
            cfs_check.append(cf)
        for cf, cfc in zip(cfs, cfs_check):
            self.assertEqual(cf.version, cfc.version)

    @unittest.skipIf(not in_development_repo, not_dev_str)
    def test_get_previously_selected_version(self):
        ad = get_access_data()
        prev = ad.get_previously_selected_version()
        self.assertIsNone(prev)  # By default, in development repo, it should be None

    @unittest.skipIf(not in_development_repo, not_dev_str)
    def test_set_selected_version(self):
        ad = get_access_data()
        vers_sel_test = "23012023"
        ad.set_previusly_selected_version(vers_sel_test)
        prev = ad.get_previously_selected_version()
        self.assertEqual(prev, vers_sel_test)
        os.remove(os.path.join(".", "coeff_data", "selected.txt"))


if __name__ == "__main__":
    unittest.main()
