"""___Built-In Modules___"""
import os

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.coefficients.access_data.appdata import (
    get_appdata_folder,
    get_local_appdata_folder,
)
from lime_tbx.gui.gui import GUI
from lime_tbx.datatypes.datatypes import KernelsPath

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def main():
    appdata = get_appdata_folder()
    local_appdata = get_local_appdata_folder()
    kernels_path = KernelsPath(
        os.path.join(appdata, "kernels"), os.path.join(local_appdata, "kernels")
    )
    gui = GUI(kernels_path, os.path.join(appdata, "eocfi_data"))


if __name__ == "__main__":
    main()
