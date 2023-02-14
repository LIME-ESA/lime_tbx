"""Module in charge of obtaining the path for the appdata folder. It does not import the Lime logger module."""

"""___Built-In Modules___"""
import sys
import pathlib
from os import path, environ
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


def get_appdata_folder(logger: Logger) -> str:
    if sys.platform == "darwin":
        home = pathlib.Path.home()
        appdata = str(home / "Library/Application Support" / APPNAME)
    elif sys.platform == "win32":
        appdata = path.join(environ["APPDATA"], APPNAME)
    else:
        appdata = path.expanduser(path.join("~", "." + APPNAME))
    logger.info(f"Appdata folder: {appdata}")
    if not _is_valid_appdata(appdata, logger):
        logger.warning("Appdata folder not valid, using '.'")
        appdata = "."
    return appdata
