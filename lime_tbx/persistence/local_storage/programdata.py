"""Module in charge of obtaining the path for the programfiles folder and the appdata folder.

It exports the following functions:
    * get_programfiles_folder() - Get the path of the programfiles folder as a string.
    * get_appdata_folder() - Get the path of the appdata folder as a string.
"""

import sys
from os import path
import os

from lime_tbx.common import logger
from lime_tbx.persistence.local_storage import appdata
import lime_tbx.persistence.local_storage.config_paths as config_paths
from lime_tbx import __version__

APPNAME = "LimeTBX"


def _is_valid_programfiles(programdata: str) -> bool:
    if (
        path.exists(path.join(programdata, "kernels"))
        and path.exists(path.join(programdata, "eocfi_data"))
        and path.exists(path.join(programdata, "coeff_data"))
    ):
        return True
    return False


def get_programfiles_folder() -> str:
    log = logger.get_logger()
    if config_paths.PROGRAMFILES_OVERRIDE is not None:
        programfiles = config_paths.PROGRAMFILES_OVERRIDE
    elif sys.platform == "darwin":
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
        fullappname = f"{APPNAME} {__version__}"
        a_reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        valid = True
        try:
            sub_key_install_folder = f"SOFTWARE\\ESA\\{fullappname}\\Settings"
            a_key = winreg.OpenKey(a_reg, sub_key_install_folder)
            programfiles = winreg.QueryValueEx(a_key, "InstallPath")[0]
        except:
            valid = False
        if not valid:  # Search for admin uninstall key
            sub_key_admin = f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{fullappname}_is1"
            try:
                a_key = winreg.OpenKey(a_reg, sub_key_admin)
            except:
                valid = False
            if not valid:  # Search for user uninstall key
                sub_key_user = f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{fullappname}_is1"
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
            log.warning(f"Did not find {fullappname} key in winreg registry.")
            programfiles = path.join(os.environ["PROGRAMFILES"], fullappname)
    else:
        programfiles = path.join("/opt/esa", APPNAME)
    log.debug("Programfiles: {}".format(programfiles))
    if not _is_valid_programfiles(programfiles):
        log.info("Programfiles directory not valid. Using current dir.")
        programfiles = "."  # os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    return programfiles


def get_appdata_folder() -> str:
    return appdata.get_appdata_folder(logger.get_logger())
