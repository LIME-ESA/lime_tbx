"""describe class"""

"""___Built-In Modules___"""
import sys
from os import path, environ

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

APPNAME = "LimeTBX"


def _is_valid_appdata(appdata: str) -> bool:
    if path.exists(path.join(appdata, "kernels")) and path.exists(
        path.join(appdata, "eocfi_data")
    ):
        return True
    return False


def get_appdata_folder() -> str:
    if sys.platform == "darwin":
        from AppKit import NSSearchPathForDirectoriesInDomains

        appdata = path.join(
            NSSearchPathForDirectoriesInDomains(14, 1, True)[0], APPNAME
        )
    elif sys.platform == "win32":
        appdata = path.join(environ["APPDATA"], APPNAME)
    else:
        appdata = path.join("/opt/esa", APPNAME)
        if not _is_valid_appdata(appdata):
            appdata = get_local_appdata_folder()
    if not _is_valid_appdata(appdata):
        appdata = "."
    return appdata


def get_local_appdata_folder() -> str:
    if sys.platform == "linux":
        appdata = path.expanduser(path.join("~", "." + APPNAME))
    else:
        appdata = get_appdata_folder()
    if not _is_valid_appdata(appdata):
        appdata = "."
    return appdata
