"""Module in charge of obtaining the path for the appdata folder. It does not import the Lime logger module.
The appdata folder is a user or system lime_tbx folder which contents can be modified without superuser permissions.
Useful for the SPICE custom kernel or the log files.

It exports the following functions:
    * get_appdata_folder - Get the path of the appdata folder as a string.
"""

"""___Built-In Modules___"""
import sys
import pathlib
from os import path
import os
from logging import Logger

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
# import here

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

APPNAME = "LimeTBX"


def _is_valid_appdata(appdata: str, logger: Logger) -> bool:
    """Checks if a given appdata path is valid, and tries to modify it if not so it is.

    Parameters
    ----------
    appdata: str
        Appdata folder absolute path
    logger: logging.Logger
        Logger that will log the messages emitted during the process

    Returns
    -------
    valid: bool
        Validity of the appdata path.
    """
    kpath = path.join(appdata, "kernels")
    if not path.exists(kpath):
        try:
            os.makedirs(kpath)
        except Exception as e:
            logger.critical(e)
            return False
    cpath = path.join(appdata, "coeff_data")
    if not path.exists(cpath):
        try:
            os.makedirs(cpath)
        except Exception as e:
            logger.critical(e)
            return False
    return True


def _get_appdata_folder(logger: Logger, platform: str) -> str:
    """Find the theoretical path of the appdata folder for a given platform. Function useful for testing.

    Parameters
    ----------
    logger: logging.Logger
        Logger that will log the messages emitted during the process
    platform: str
        System OS platform (darwin, win32 or linux)

    Returns
    -------
    appdata_path: str
        Appdata folder absolute path for that platform
    """
    if platform == "darwin":
        home = pathlib.Path.home()
        appdata = str(home / "Library/Application Support" / APPNAME)
    elif platform == "win32":
        appdata = path.join(
            os.environ.get("APPDATA", path.join(os.getcwd(), "appdata")), APPNAME
        )
    else:
        appdata = path.expanduser(path.join("~", "." + APPNAME))
    logger.debug(f"Appdata folder: {appdata}")
    return appdata


def get_appdata_folder(logger: Logger) -> str:
    """Find the path of the appdata folder

    Parameters
    ----------
    logger: logging.Logger
        Logger that will log the messages emitted during the process

    Returns
    -------
    appdata_path: str
        Appdata folder absolute path
    """
    platf = sys.platform
    appdata = _get_appdata_folder(logger, platf)
    if not _is_valid_appdata(appdata, logger):
        logger.warning("Appdata folder not valid, using '.'")
        appdata = "."
    return appdata
