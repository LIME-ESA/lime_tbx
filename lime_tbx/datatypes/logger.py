"""This module is in charge of logging the output data."""

"""___Built-In Modules___"""
import logging
from datetime import datetime
import os

"""___Third-Party Modules___"""
# import here


"""___LIME Modules___"""
from lime_tbx.coefficients.access_data.appdata import get_appdata_folder

# SINGLETON
_logger: logging.Logger = None

_FORMAT = "%(levelname)s: [%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s"
_DATEFORMAT = "%H:%M:%S"


def get_logger() -> logging.Logger:
    global _logger
    if _logger == None:
        dtnow = datetime.utcnow()
        logname = dtnow.strftime("%Y%m%d")
        logname = "log_{}.txt".format(logname)
        logname = os.path.join(get_appdata_folder(), logname)
        logging.basicConfig(
            filename=logname, filemode="a", format=_FORMAT, datefmt=_DATEFORMAT
        )
        _logger = logging.getLogger(__name__)
        _logger.setLevel(logging.DEBUG)
    return _logger
