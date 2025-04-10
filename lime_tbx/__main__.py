"""
LIME Toolbox Entry Point.

This module serves as the entry point for the LIME Toolbox when executed as a standalone 
script (`python3 -m lime_tbx`). It processes command-line arguments, invokes the CLI module 
if arguments are provided, and ensures proper error handling. If no arguments are given, 
it launches the graphical user interface (GUI) by default.

Main Responsibilities
----------------------
- Parses command-line arguments.
- Initializes necessary paths and settings.
- Calls the appropriate CLI functions to perform simulations or comparisons.
- Launches the GUI if no command-line arguments are provided.
- Suppresses specific warnings to improve user experience.
- Handles errors and logs execution details.

Usage
-----
To run the LIME Toolbox via the command line:
    $ python3 -m lime_tbx -e lat,lon,height,datetime -o csv,output.csv

To launch the GUI:
    $ python3 -m lime_tbx

For help, run:
    $ python3 -m lime_tbx -h

Error Handling
--------------
Any errors encountered during execution are logged, and shown to the user
in a window or printed to standard error (stderr).
If a critical error occurs in the CLI, the program exits with a non-zero status code.

References
----------
- `lime_tbx.presentation.cli.CLI`: Handles command-line execution.
- `lime_tbx.presentation.gui.GUI`: Manages the graphical user interface.
- `lime_tbx.logger`: Handles logging.
"""


import os
import getopt
import sys
import warnings
from traceback import format_exc, format_exception
import faulthandler

from lime_tbx.presentation.cli.cli import (
    CLI,
    OPTIONS,
    LONG_OPTIONS,
    print_help,
)
from lime_tbx.persistence.local_storage.programdata import (
    get_appdata_folder,
    get_programfiles_folder,
)
from lime_tbx.common.datatypes import KernelsPath, EocfiPath
from lime_tbx.common import logger
from lime_tbx.application.coefficients import access_data


def excepthook(exc_type, exc_value, exc_traceback):
    error_msg = "".join(format_exception(exc_type, exc_value, exc_traceback))
    logger.get_logger().error("Uncaught exception:\n%s", error_msg)
    print(f"Uncaught exception logged: {str(error_msg)}")


def main():
    """
    Launches the LIME Toolbox.

    This function:
    - Initializes logging.
    - Filters out unneeded warnings to improve the user experience.
    - Retrieves necessary paths for kernels and EO-CFI data.
    - Parses command-line arguments to determine execution mode:
      - If arguments are provided, it runs the **command-line interface (CLI)**.
      - If no arguments are provided, it launches the **graphical user interface (GUI)**.

    On Windows, it also hides the console window when launching the GUI.

    If errors occur during CLI argument parsing or execution, they are logged,
    and the program exits with an appropriate error code.

    Returns
    -------
    None
        The function does not return but either starts the GUI or CLI
        and exits with a status code.
    """
    logger.get_logger().info("ToolBox started")
    sys.excepthook = excepthook
    warnings.filterwarnings("ignore", ".*Gtk-WARNING.*")
    warnings.filterwarnings("ignore", ".*Fontconfig warning.*")
    warnings.filterwarnings(
        "ignore",
        ".*elementwise comparison failed; returning scalar instead.*",
        FutureWarning,
        "punpy",
    )
    programfiles = get_programfiles_folder()
    appdata = get_appdata_folder()
    faulthandler.enable(open(os.path.join(appdata, "crash.log"), "a"))
    logger.get_logger().info(
        f"Appdata folder: {appdata}. Programfiles folder: {programfiles}."
    )
    kernels_path = KernelsPath(
        os.path.join(programfiles, "kernels"), os.path.join(appdata, "kernels")
    )
    eocfi_path = EocfiPath(
        os.path.join(programfiles, "eocfi_data"), os.path.join(appdata, "eocfi_data")
    )
    sysargs = sys.argv[1:]
    options = OPTIONS
    long_options = LONG_OPTIONS
    selected_version = access_data.get_previously_selected_version()
    try:
        opts, args = getopt.gnu_getopt(sysargs, options, long_options)
    except getopt.GetoptError as e:
        print("Error parsing input parameters: " + str(e))
        print_help()
        sys.exit(2)
    try:
        if len(opts) == 0:
            from lime_tbx.presentation.gui.gui import GUI

            if sys.platform.lower().startswith("win"):
                import ctypes

                whnd = ctypes.windll.kernel32.GetConsoleWindow()
                if whnd != 0:
                    ctypes.windll.user32.ShowWindow(whnd, 0)
            gui = GUI(kernels_path, eocfi_path, selected_version)
        else:
            cli = CLI(kernels_path, eocfi_path, selected_version)
            status = cli.check_sys_args(sysargs)
            if status == 0:
                status = cli.handle_input(opts, args)
            sys.exit(status)
    except Exception as e:
        trace = format_exc()
        logger.get_logger().critical(e)
        logger.get_logger().critical(trace)
        print(e, trace)


if __name__ == "__main__":
    main()
