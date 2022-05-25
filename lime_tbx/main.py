"""___Built-In Modules___"""
from lime_tbx.lime_algorithms.rolo import rolo
from lime_tbx.gui.gui import GUI

"""___Third-Party Modules___"""
# import here

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


def main():
    gui = GUI("kernels", "eocfi_data")


if __name__ == "__main__":
    main()
