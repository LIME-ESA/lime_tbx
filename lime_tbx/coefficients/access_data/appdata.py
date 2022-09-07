"""describe class"""

"""___Built-In Modules___"""
import sys
from os import path, environ, makedirs

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
    kpath = path.join(appdata, "kernels")
    if not path.exists(kpath):
        try:
            makedirs(kpath)
        except Exception as e:
            print(e)
            return False
    return True

def _is_valid_programfiles(programdata: str) -> bool:
    if path.exists(path.join(programdata, "kernels")) and path.exists(
        path.join(programdata, "eocfi_data")
    ):
        return True
    return False

def get_programfiles_folder() -> str:
    if sys.platform == "darwin":
        programfiles = get_appdata_folder()
    elif sys.platform == "win32":
        import winreg
        a_reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        sub_key_admin = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\LimeTBX_is1"
        sub_key_user = "HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\LimeTBX_is1"
        valid = True
        try:
            a_key = winreg.OpenKey(a_reg, sub_key_admin)
        except:
            valid = False
        if not valid:
            valid = True
            try:
                a_key =  winreg.OpenKey(a_reg, sub_key_user)
            except:
                valid = False
        if valid:
            programfiles = winreg.QueryValueEx(a_key, "Inno Setup: App Path")[0]
        else:
            programfiles = path.join(environ["PROGRAMFILES"], APPNAME)
    else:
        programfiles = path.join("/opt/esa", APPNAME)
    if not _is_valid_programfiles(programfiles):
        programfiles = "."
    return programfiles

def get_appdata_folder() -> str:
    if sys.platform == "darwin":
        from AppKit import NSSearchPathForDirectoriesInDomains
        appdata = path.join(
            NSSearchPathForDirectoriesInDomains(14, 1, True)[0], APPNAME
        )
    elif sys.platform == "win32":
        appdata = path.join(environ["APPDATA"], APPNAME)
    else:
        appdata = path.expanduser(path.join("~", "." + APPNAME))
    if not _is_valid_appdata(appdata):
        appdata = "."
    return appdata
