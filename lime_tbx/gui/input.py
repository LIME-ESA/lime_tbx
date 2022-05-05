"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime

"""___Third-Party Modules___"""
from typing import Callable
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "03/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class SurfaceInputWidget(QtWidgets.QWidget):
    """
    Input widget that contains the GUI elements for the input of the needed parameters for
    the simulation of lunar values for a geographic position at a concrete time.
    """

    def __init__(self):
        super().__init__()
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QFormLayout(self)
        self.latitude_label = QtWidgets.QLabel("Latitude:")
        self.longitude_label = QtWidgets.QLabel("Longitude:")
        self.altitude_label = QtWidgets.QLabel("Altitude (m):")
        self.datetime_label = QtWidgets.QLabel("UTC DateTime:")
        self.latitude_spinbox = QtWidgets.QDoubleSpinBox()
        self.longitude_spinbox = QtWidgets.QDoubleSpinBox()
        self.altitude_spinbox = QtWidgets.QDoubleSpinBox()
        self.datetime_edit = QtWidgets.QDateTimeEdit()
        self.latitude_spinbox.setMinimum(-90)
        self.latitude_spinbox.setMaximum(90)
        self.longitude_spinbox.setMinimum(-180)
        self.longitude_spinbox.setMaximum(180)
        self.altitude_spinbox.setMinimum(0)
        self.altitude_spinbox.setMaximum(10000000)
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.main_layout.addRow(self.latitude_label, self.latitude_spinbox)
        self.main_layout.addRow(self.longitude_label, self.longitude_spinbox)
        self.main_layout.addRow(self.altitude_label, self.altitude_spinbox)
        self.main_layout.addRow(self.datetime_label, self.datetime_edit)

    def get_latitude(self) -> float:
        return self.latitude_spinbox.value()

    def get_longitude(self) -> float:
        return self.longitude_spinbox.value()

    def get_altitude(self) -> float:
        return self.altitude_spinbox.value()

    def get_datetime(self) -> datetime:
        return self.datetime_edit.dateTime().toPython()


class SurfaceWidget(QtWidgets.QWidget):
    def __init__(self, title: str, callback: Callable):
        super().__init__()
        self.title = title
        self.callback = callback
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.title_label = QtWidgets.QLabel(self.title, alignment=QtCore.Qt.AlignCenter)
        self.surface_input = SurfaceInputWidget()
        self.action_button = QtWidgets.QPushButton("Calculate")
        self.action_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.surface_input)
        self.main_layout.addWidget(self.action_button)

    @QtCore.Slot()
    def push_action_button(self):
        self.callback()

    def get_latitude(self) -> float:
        return self.surface_input.get_latitude()

    def get_longitude(self) -> float:
        return self.surface_input.get_longitude()

    def get_altitude(self) -> float:
        return self.surface_input.get_altitude()

    def get_datetime(self) -> datetime:
        return self.surface_input.get_datetime()
