"""___Built-In Modules___"""
import os
import getopt
import sys

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.cli.cli import (
    CLI,
    OPTIONS,
    LONG_OPTIONS,
    print_help,
)
from lime_tbx.coefficients.access_data.appdata import (
    get_appdata_folder,
    get_programfiles_folder,
)
from lime_tbx.datatypes.datatypes import KernelsPath

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def main():
    programfiles = get_programfiles_folder()
    appdata = get_appdata_folder()
    kernels_path = KernelsPath(
        os.path.join(programfiles, "kernels"), os.path.join(appdata, "kernels")
    )
    eocfi_path = os.path.join(programfiles, "eocfi_data")
    args = sys.argv[1:]
    options = OPTIONS
    long_options = LONG_OPTIONS
    try:
        opts, args = getopt.getopt(args, options, long_options)
    except getopt.GetoptError:
        print("Error parsing input parameters.")
        print_help()
        sys.exit(2)
    if len(opts) == 0:
        from lime_tbx.gui.gui import GUI

        if sys.platform.lower().startswith("win"):
            import ctypes

            whnd = ctypes.windll.kernel32.GetConsoleWindow()
            if whnd != 0:
                ctypes.windll.user32.ShowWindow(whnd, 0)
        gui = GUI(kernels_path, eocfi_path)
    else:
        cli = CLI(kernels_path, eocfi_path)
        status = cli.handle_input(opts)
        sys.exit(status)


if __name__ == "__main__":
    main()
