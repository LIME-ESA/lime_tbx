"""describe class"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import input
from ..simulation.regular_simulation import regular_simulation

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ELISurfaceWidget(QtWidgets.QWidget):
    def __init__(self, kernels_path):
        super().__init__()
        self.kernels_path = kernels_path
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QWidget(self)
        self.surface_widget = input.SurfaceWidget("Calculate ELI", self.calculate)
        pass

    def calculate(self):
        rs = regular_simulation.RegularSimulation()
        latitude = self.surface_widget.get_latitude()
        longitude = self.surface_widget.get_longitude()
        altitude = self.surface_widget.get_altitude()
        datetime = self.surface_widget.get_datetime()
        rs.get_eli_from_surface(
            srf, latitude, longitude, altitude, datetime, coeffs, self.kernels_path
        )
