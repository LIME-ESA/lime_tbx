"""Tests for classname module"""

"""___Built-In Modules___"""
import logging
import os
import pathlib

"""___Third-Party Modules___"""
import unittest

"""___LIME_TBX Modules___"""
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
    for hdl in test_logger.handlers:
        test_logger.removeHandler(hdl)
    handler.setFormatter(formatter)
    test_logger.addHandler(handler)
    test_logger.disabled = True
    return test_logger


class TestAppdata(unittest.TestCase):
    def test_is_valid_appdata_repo_ok(self):
        self.assertTrue(appdata._is_valid_appdata(".", get_logger()))

    def test_is_valid_appdata_repo_not(self):
        self.assertFalse(
            appdata._is_valid_appdata("./test_files/non_editable/file", get_logger())
        )

    def test_platforms_get_appdata_folder(self):
        linappdata = appdata._get_appdata_folder(get_logger(), "linux")
        self.assertEqual(
            linappdata, os.path.expanduser(os.path.join("~", "." + appdata.APPNAME))
        )
        winappdata = appdata._get_appdata_folder(get_logger(), "win32")
        self.assertEqual(
            winappdata, os.path.join(os.getcwd(), "appdata", appdata.APPNAME)
        )
        macappdata = appdata._get_appdata_folder(get_logger(), "darwin")
        self.assertEqual(
            macappdata,
            str(pathlib.Path.home() / "Library/Application Support" / appdata.APPNAME),
        )

    def test_get_appdata_folder(self):
        self.assertEqual(
            appdata.get_appdata_folder(get_logger()),
            os.path.expanduser(os.path.join("~", "." + appdata.APPNAME)),
        )


class TestProgramdata(unittest.TestCase):
    # get_programfiles_folder can't be tested for other platforms in linux as the function
    # has dependencies only available in the platform.

    def test_get_appdata_folder_programdata(self):
        self.assertEqual(
            programdata.get_appdata_folder(),
            os.path.expanduser(os.path.join("~", "." + appdata.APPNAME)),
        )
        self.assertEqual(
            programdata.get_appdata_folder(), appdata.get_appdata_folder(get_logger())
        )

    def test_is_valid_programfiles_repo_ok(self):
        self.assertTrue(programdata._is_valid_programfiles("."))

    def test_is_not_valid_programfiles_repo_coeff_data(self):
        self.assertFalse(programdata._is_valid_programfiles("./coeff_data"))

    def test_get_programfiles(self):
        # This will fail if lime_tbx is installed
        self.assertEqual(programdata.get_programfiles_folder(), ".")


if __name__ == "__main__":
    unittest.main()
