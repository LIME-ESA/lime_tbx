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
from lime_tbx.local_storage.programdata import (
    get_appdata_folder,
    get_programfiles_folder,
)
from lime_tbx.datatypes.datatypes import KernelsPath
from lime_tbx.datatypes import logger
from lime_tbx.coefficients.access_data.access_data import AccessData

"""___Authorship___"""
__author__ = "Pieter De Vis, Jacob Fahy, Javier Gatón Herguedas, Ramiro González Catón, Carlos Toledano"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def main():
    logger.get_logger().info("ToolBox started")
    programfiles = get_programfiles_folder()
    appdata = get_appdata_folder()
    kernels_path = KernelsPath(
        os.path.join(programfiles, "kernels"), os.path.join(appdata, "kernels")
    )
    eocfi_path = os.path.join(programfiles, "eocfi_data")
    args = sys.argv[1:]
    options = OPTIONS
    long_options = LONG_OPTIONS
    selected_version = AccessData().get_previously_selected_version()
    try:
        opts, args = getopt.getopt(args, options, long_options)
    except getopt.GetoptError as e:
        print("Error parsing input parameters: " + str(e))
        print_help()
        sys.exit(2)
    if len(opts) == 0:
        from lime_tbx.gui.gui import GUI

        if sys.platform.lower().startswith("win"):
            import ctypes

            whnd = ctypes.windll.kernel32.GetConsoleWindow()
            if whnd != 0:
                ctypes.windll.user32.ShowWindow(whnd, 0)
        gui = GUI(kernels_path, eocfi_path, selected_version)
    else:
        cli = CLI(kernels_path, eocfi_path, selected_version)
        status = cli.handle_input(opts)
        sys.exit(status)


if __name__ == "__main__":
    main()
