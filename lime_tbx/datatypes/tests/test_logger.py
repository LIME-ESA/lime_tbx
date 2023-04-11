"""Tests for the logger module"""

"""___Built-In Modules___"""
import sys
import io
import logging
import os

"""___Third-Party Modules___"""
import unittest
import numpy as np
import xarray

"""___NPL Modules___"""
import obsarray

"""___LIME_TBX Modules___"""
from .. import logger, constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "14/02/2023"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class TestLogger(unittest.TestCase):
    def test_printout_logger_print_warning(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        plg = logger._get_printout_logger()
        plg.warning("Test string hey")
        capturedval = capturedOutput.getvalue()
        self.assertTrue(capturedval.startswith("WARNING: [test_logger.py:"))
        self.assertTrue(
            capturedval.endswith(
                " - test_printout_logger_print_warning() ] Test string hey\n"
            )
        )
        capturedOutput.close()
        sys.stdout = sys.__stdout__

    def test_printout_logger_values(self):
        plg = logger._get_printout_logger()
        self.assertEqual(plg.level, logging.WARNING)
        self.assertEqual(plg.name, "printout_logger")

    def test_get_logger_equal(self):
        lg = logger.get_logger()
        lg2 = logger.get_logger()
        self.assertEqual(lg, lg2)

    def test_get_logger_name(self):
        lg = logger.get_logger()
        self.assertEqual(lg.name, "lime_tbx.datatypes.logger")

    def test_get_logger_level(self):
        debug_value = "INFO"
        if constants.DEBUG_ENV_NAME in os.environ:
            debug_value = os.environ[constants.DEBUG_ENV_NAME]
        os.environ[constants.DEBUG_ENV_NAME] = "INFO"
        logger._logger = None
        lg = logger.get_logger()
        self.assertEqual(lg.level, logging.INFO)
        os.environ[constants.DEBUG_ENV_NAME] = "DEBUG"
        logger._logger = None
        lg = logger.get_logger()
        self.assertEqual(lg.level, logging.DEBUG)
        os.environ[constants.DEBUG_ENV_NAME] = debug_value
        logger._logger = None
