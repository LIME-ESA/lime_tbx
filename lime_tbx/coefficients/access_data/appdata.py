"""describe class"""

"""___Built-In Modules___"""
import sys
import pathlib
from os import path, environ
import os

"""___Third-Party Modules___"""
# import here

"""___LIME Modules___"""
from ...datatypes import logger

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

APPNAME = "LimeTBX"


def _is_valid_appdata(appdata: str) -> bool:
    kpath = path.join(appdata, "kernels")
    if not path.exists(kpath):
        try:
            os.makedirs(kpath)
        except Exception as e:
            logger.get_logger().critical(e)
            return False
    cpath = path.join(appdata, "coeff_data")
    if not path.exists(cpath):
        try:
            os.makedirs(cpath)
        except Exception as e:
            logger.get_logger().critical(e)
            return False
    return True


def _is_valid_programfiles(programdata: str) -> bool:
    if path.exists(path.join(programdata, "kernels")) and path.exists(
        path.join(programdata, "eocfi_data")
        and path.exists(path.join(programdata, "coeff_data"))
    ):
        return True
    return False


def get_programfiles_folder() -> str:
    log = logger.get_logger()
    if sys.platform == "darwin":
        _stream = os.popen('mdfind "kMDItemCFBundleIdentifier = int.esa.LimeTBX"')
        output = _stream.read()
        _stream.close()
        possible_bundles = output.splitlines()
        programfiles = ""
        for bundle in possible_bundles:
            bundle = path.join(bundle, "Contents/Resources")
            if _is_valid_programfiles(bundle):
                programfiles = bundle
                break
        if programfiles == "":
            programfiles = "/Applications/{}.app/Contents/Resources".format(APPNAME)
    elif sys.platform == "win32":
        import winreg

        programfiles = ""
        a_reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        valid = True
        try:
            sub_key_install_folder = "SOFTWARE\\ESA\\LimeTBX\\Settings"
            a_key = winreg.OpenKey(a_reg, sub_key_install_folder)
            programfiles = winreg.QueryValueEx(a_key, "InstallPath")[0]
        except:
            valid = False
        if not valid:  # Search for admin uninstall key
            sub_key_admin = (
                "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\LimeTBX_is1"
            )
            try:
                a_key = winreg.OpenKey(a_reg, sub_key_admin)
            except:
                valid = False
            if not valid:  # Search for user uninstall key
                sub_key_user = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\LimeTBX_is1"
                a_reg = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                valid = True
                try:
                    a_key = winreg.OpenKey(a_reg, sub_key_user)
                except:
                    valid = False
            if valid:
                try:
                    programfiles = winreg.QueryValueEx(a_key, "Inno Setup: App Path")[0]
                except:
                    valid = False
        if not valid:
            log.warning("Did not find LimeTBX key in winreg registry.")
            programfiles = path.join(environ["PROGRAMFILES"], APPNAME)
    else:
        programfiles = path.join("/opt/esa", APPNAME)
    log.info("Programfiles: {}".format(programfiles))
    if not _is_valid_programfiles(programfiles):
        log.warning("Programfiles directory not valid. Using current dir.")
        programfiles = "."
    return programfiles


def get_appdata_folder() -> str:
    if sys.platform == "darwin":
        home = pathlib.Path.home()
        appdata = str(home / "Library/Application Support" / APPNAME)
    elif sys.platform == "win32":
        appdata = path.join(environ["APPDATA"], APPNAME)
    else:
        appdata = path.expanduser(path.join("~", "." + APPNAME))
    if not _is_valid_appdata(appdata):
        appdata = "."
    return appdata
