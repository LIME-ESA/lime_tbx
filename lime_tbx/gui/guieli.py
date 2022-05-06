"""describe class"""

"""___Built-In Modules___"""
from typing import List

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import input, settings, output
from ..simulation.regular_simulation import regular_simulation

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "03/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class ELISurfaceWidget(QtWidgets.QWidget):
    def __init__(self, kernels_path: str, settings_manager: settings.ISettingsManager):
        super().__init__()
        self.kernels_path = kernels_path
        self.settings_manager = settings_manager
        self.graph = output.GraphWindow()
        self._build_layout()

    def propagate_close_event(self):
        self.graph.close()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.surface_widget = input.SurfaceWidget("Calculate ELI", self.calculate)
        self.main_layout.addWidget(self.surface_widget)

    def calculate(self):
        rs = regular_simulation.RegularSimulation()
        latitude = self.surface_widget.get_latitude()
        longitude = self.surface_widget.get_longitude()
        altitude = self.surface_widget.get_altitude()
        datetime = self.surface_widget.get_datetime()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        elis: List[float] = rs.get_eli_from_surface(
            srf, latitude, longitude, altitude, datetime, coeffs, self.kernels_path
        )
        self.graph.show()
        self.graph.update_plot(list(srf.spectral_response.keys()), elis)
        self.graph.update_labels(
            "Extraterrestrial Lunar Irradiances",
            "wavelengths (nm)",
            "irradiances (Wm⁻²/nm)",
        )
