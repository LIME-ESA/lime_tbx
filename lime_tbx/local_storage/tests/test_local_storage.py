"""Tests for classname module"""

"""___Built-In Modules___"""
import logging
import sys

"""___Third-Party Modules___"""
import unittest

"""___LIME TBX Modules___"""
from .. import appdata, programdata

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def get_logger() -> logging.Logger:
    test_logger = logging.getLogger("test_logger")
    handler = logging.StreamHandler(None)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    test_logger.addHandler(handler)
    return test_logger


class TestLocalStorage(unittest.TestCase):
    def test_is_valid_appdata_repo_ok(self):
        self.assertTrue(appdata._is_valid_appdata(".", get_logger()))

    def test_is_valid_programfiles_repo_ok(self):
        self.assertTrue(programdata._is_valid_programfiles("."))

    def test_is_not_valid_programfiles_repo_coeff_data(self):
        self.assertFalse(programdata._is_valid_programfiles("./coeff_data"))


if __name__ == "__main__":
    unittest.main()
