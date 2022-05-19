"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime
from typing import Union, Tuple

"""___Third-Party Modules___"""
from typing import Callable
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    SurfacePoint,
    CustomPoint,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "03/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class CustomInputWidget(QtWidgets.QWidget):
    """
    Input widget that contains the GUI elements for the input of the needed parameters for
    the simulation of lunar values for a custom point with custom lunar data.
    """

    def __init__(self):
        super().__init__()
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QFormLayout(self)
        self.dist_sun_moon_label = QtWidgets.QLabel("Distance Sun-Moon (AU):")
        self.dist_obs_moon_label = QtWidgets.QLabel("Distance Observer-Moon (km):")
        self.selen_obs_lat_label = QtWidgets.QLabel(
            "Selenographic latitude of the observer (°):"
        )
        self.selen_obs_lon_label = QtWidgets.QLabel(
            "Selenographic longitude of the observer (°):"
        )
        self.selen_sun_lon_label = QtWidgets.QLabel(
            "Selenographic longitude of the Sun (RAD):"
        )
        self.abs_moon_phase_angle_label = QtWidgets.QLabel(
            "Absolute Moon phase angle (°):"
        )
        self.dist_sun_moon_spinbox = QtWidgets.QDoubleSpinBox()
        self.dist_obs_moon_spinbox = QtWidgets.QDoubleSpinBox()
        self.selen_obs_lat_spinbox = QtWidgets.QDoubleSpinBox()
        self.selen_obs_lon_spinbox = QtWidgets.QDoubleSpinBox()
        self.selen_sun_lon_spinbox = QtWidgets.QDoubleSpinBox()
        self.abs_moon_phase_angle_spinbox = QtWidgets.QDoubleSpinBox()
        self.dist_sun_moon_spinbox.setMinimum(0.00001)
        self.dist_sun_moon_spinbox.setMaximum(1.5)
        self.dist_sun_moon_spinbox.setDecimals(6)
        self.dist_sun_moon_spinbox.setSingleStep(0.001)
        self.dist_sun_moon_spinbox.setValue(0.8)
        self.dist_obs_moon_spinbox.setMinimum(1)
        self.dist_obs_moon_spinbox.setMaximum(500000)
        self.dist_obs_moon_spinbox.setDecimals(4)
        self.dist_obs_moon_spinbox.setValue(300000)
        self.selen_obs_lat_spinbox.setMinimum(-90)
        self.selen_obs_lat_spinbox.setMaximum(90)
        self.selen_obs_lat_spinbox.setDecimals(4)
        self.selen_obs_lon_spinbox.setMinimum(-180)
        self.selen_obs_lon_spinbox.setMaximum(180)
        self.selen_obs_lon_spinbox.setMaximum(4)
        self.selen_sun_lon_spinbox.setMinimum(-180)
        self.selen_sun_lon_spinbox.setMaximum(180)
        self.selen_sun_lon_spinbox.setDecimals(4)
        self.abs_moon_phase_angle_spinbox.setMinimum(-180)
        self.abs_moon_phase_angle_spinbox.setMaximum(180)
        self.abs_moon_phase_angle_spinbox.setDecimals(5)
        self.main_layout.addRow(self.dist_sun_moon_label, self.dist_sun_moon_spinbox)
        self.main_layout.addRow(self.dist_obs_moon_label, self.dist_obs_moon_spinbox)
        self.main_layout.addRow(self.selen_obs_lat_label, self.selen_obs_lat_spinbox)
        self.main_layout.addRow(self.selen_obs_lon_label, self.selen_obs_lon_spinbox)
        self.main_layout.addRow(self.selen_sun_lon_label, self.selen_sun_lon_spinbox)
        self.main_layout.addRow(
            self.abs_moon_phase_angle_label, self.abs_moon_phase_angle_spinbox
        )

    def get_dist_sun_moon(self) -> float:
        return self.dist_sun_moon_spinbox.value()

    def get_dist_obs_moon(self) -> float:
        return self.dist_obs_moon_spinbox.value()

    def get_selen_obs_lat(self) -> float:
        return self.selen_obs_lat_spinbox.value()

    def get_selen_obs_lon(self) -> float:
        return self.selen_obs_lon_spinbox.value()

    def get_selen_sun_lon(self) -> float:
        return self.selen_sun_lon_spinbox.value()

    def get_abs_moon_phase_angle(self) -> float:
        return self.abs_moon_phase_angle_spinbox.value()


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
        self.datetime_edit.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
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


class InputWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        self.surface = SurfaceInputWidget()
        self.tabs.addTab(self.surface, "Surface")
        self.custom = CustomInputWidget()
        self.tabs.addTab(self.custom, "Custom")
        self.satellite = QtWidgets.QLabel("Satellite not yet implemented")
        self.tabs.addTab(self.satellite, "Satellite")
        self.main_layout.addWidget(self.tabs)

    def get_point(self) -> Union[SurfacePoint, CustomPoint]:
        tab = self.tabs.currentWidget()
        if isinstance(tab, SurfaceInputWidget):
            lat = self.surface.get_latitude()
            lon = self.surface.get_longitude()
            alt = self.surface.get_altitude()
            dt = self.surface.get_datetime()
            return SurfacePoint(lat, lon, alt, dt)
        elif isinstance(tab, CustomInputWidget):
            dsm = self.custom.get_dist_sun_moon()
            dom = self.custom.get_dist_obs_moon()
            olat = self.custom.get_selen_obs_lat()
            olon = self.custom.get_selen_obs_lon()
            slon = self.custom.get_selen_sun_lon()
            mpa = self.custom.get_abs_moon_phase_angle()
            return CustomPoint(dsm, dom, olat, olon, slon, mpa)
        else:
            pass
