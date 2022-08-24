"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime
from typing import Union, Tuple

from lime_tbx.filedata import moon

"""___Third-Party Modules___"""
from typing import Callable, List, Optional
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    LunarObservation,
    Point,
    Satellite,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
    SatellitePoint,
)
from ..filedata import csv, srf

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "03/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

MAX_PATH_LEN = 35


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
        self.moon_phase_angle_label = QtWidgets.QLabel("Moon phase angle (°):")
        self.dist_sun_moon_spinbox = QtWidgets.QDoubleSpinBox()
        self.dist_obs_moon_spinbox = QtWidgets.QDoubleSpinBox()
        self.selen_obs_lat_spinbox = QtWidgets.QDoubleSpinBox()
        self.selen_obs_lon_spinbox = QtWidgets.QDoubleSpinBox()
        self.selen_sun_lon_spinbox = QtWidgets.QDoubleSpinBox()
        self.moon_phase_angle_spinbox = QtWidgets.QDoubleSpinBox()
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
        self.selen_obs_lon_spinbox.setDecimals(4)
        self.selen_sun_lon_spinbox.setMinimum(-180)
        self.selen_sun_lon_spinbox.setMaximum(180)
        self.selen_sun_lon_spinbox.setDecimals(4)
        self.moon_phase_angle_spinbox.setMinimum(-180)
        self.moon_phase_angle_spinbox.setMaximum(180)
        self.moon_phase_angle_spinbox.setDecimals(5)
        self.main_layout.addRow(self.dist_sun_moon_label, self.dist_sun_moon_spinbox)
        self.main_layout.addRow(self.dist_obs_moon_label, self.dist_obs_moon_spinbox)
        self.main_layout.addRow(self.selen_obs_lat_label, self.selen_obs_lat_spinbox)
        self.main_layout.addRow(self.selen_obs_lon_label, self.selen_obs_lon_spinbox)
        self.main_layout.addRow(self.selen_sun_lon_label, self.selen_sun_lon_spinbox)
        self.main_layout.addRow(
            self.moon_phase_angle_label, self.moon_phase_angle_spinbox
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

    def get_moon_phase_angle(self) -> float:
        return self.moon_phase_angle_spinbox.value()


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
        self.latitude_spinbox = QtWidgets.QDoubleSpinBox()
        self.longitude_spinbox = QtWidgets.QDoubleSpinBox()
        self.altitude_spinbox = QtWidgets.QDoubleSpinBox()
        self.latitude_spinbox.setMinimum(-90)
        self.latitude_spinbox.setMaximum(90)
        self.longitude_spinbox.setMinimum(-180)
        self.longitude_spinbox.setMaximum(180)
        self.altitude_spinbox.setMinimum(0)
        self.altitude_spinbox.setMaximum(10000000)
        self.main_layout.addRow(self.latitude_label, self.latitude_spinbox)
        self.main_layout.addRow(self.longitude_label, self.longitude_spinbox)
        self.main_layout.addRow(self.altitude_label, self.altitude_spinbox)
        self._build_layout_single_datetime()

    def _build_layout_single_datetime(self):
        self.single_datetime = True
        self.loaded_datetimes = []
        self.datetime_label = QtWidgets.QLabel("UTC DateTime:")
        self.datetime_edit = QtWidgets.QDateTimeEdit()
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.datetime_edit.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
        self.datetime_switch = QtWidgets.QPushButton(" Load time-series file ")
        self.datetime_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.switch_layout = QtWidgets.QHBoxLayout()
        self.switch_layout.addWidget(QtWidgets.QLabel(), 1)
        self.switch_layout.addWidget(self.datetime_switch)
        self.main_layout.addRow(self.datetime_label, self.datetime_edit)
        self.main_layout.addRow(QtWidgets.QLabel(), self.switch_layout)
        self.datetime_switch.clicked.connect(self.change_multiple_datetime)

    def _build_layout_multiple_datetime(self):
        self.single_datetime = False
        self.datetime_label = QtWidgets.QLabel("Time-series file:")
        self.datetimes_layout = QtWidgets.QHBoxLayout()
        self.load_datetimes_button = QtWidgets.QPushButton("Load file")
        self.load_datetimes_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.load_datetimes_button.clicked.connect(self.load_datetimes)
        self.loaded_datetimes_label = QtWidgets.QLabel("")
        self.datetimes_layout.addWidget(self.load_datetimes_button)
        self.datetimes_layout.addWidget(self.loaded_datetimes_label, 1)
        self.datetime_switch = QtWidgets.QPushButton(" Input single datetime ")
        self.datetime_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.switch_layout = QtWidgets.QHBoxLayout()
        self.switch_layout.addWidget(QtWidgets.QLabel(), 1)
        self.switch_layout.addWidget(self.datetime_switch)
        self.main_layout.addRow(self.datetime_label, self.datetimes_layout)
        self.main_layout.addRow(QtWidgets.QLabel(), self.switch_layout)
        self.datetime_switch.clicked.connect(self.change_single_datetime)

    def _clear_form_rows(self):
        self.main_layout.removeRow(4)
        self.main_layout.removeRow(3)

    @QtCore.Slot()
    def change_single_datetime(self):
        self._clear_form_rows()
        self._build_layout_single_datetime()

    @QtCore.Slot()
    def change_multiple_datetime(self):
        self._clear_form_rows()
        self._build_layout_multiple_datetime()

    @QtCore.Slot()
    def load_datetimes(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        self.loaded_datetimes = csv.read_datetimes(path)
        self.loaded_datetimes_label.setText(path)

    def get_latitude(self) -> float:
        return self.latitude_spinbox.value()

    def get_longitude(self) -> float:
        return self.longitude_spinbox.value()

    def get_altitude(self) -> float:
        return self.altitude_spinbox.value()

    def get_datetimes(self) -> Union[datetime, List[datetime]]:
        if self.single_datetime:
            return self.datetime_edit.dateTime().toPython()
        else:
            return self.loaded_datetimes


class SatelliteInputWidget(QtWidgets.QWidget):
    def __init__(self, satellites: List[Satellite]) -> None:
        super().__init__()
        self.satellites = satellites
        self.sat_names = [s.name for s in self.satellites]
        self._build_layout()
        self.update_from_combobox(0)

    def _build_layout(self):
        self.main_layout = QtWidgets.QFormLayout(self)
        # satellite
        self.satellite_label = QtWidgets.QLabel("Satellite:")
        self.combo_sats = QtWidgets.QComboBox()
        self.combo_sats.addItems(self.sat_names)
        self.combo_sats.currentIndexChanged.connect(self.update_from_combobox)
        # finish layout
        self.main_layout.addRow(self.satellite_label, self.combo_sats)
        self._build_layout_single_datetime()

    def _build_layout_single_datetime(self):
        self.single_datetime = True
        self.loaded_datetimes = []
        # datetime
        self.datetime_label = QtWidgets.QLabel("UTC DateTime:")
        self.datetime_edit = QtWidgets.QDateTimeEdit()
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.datetime_edit.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
        self.datetime_switch = QtWidgets.QPushButton(" Load time-series file ")
        self.datetime_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.switch_layout = QtWidgets.QHBoxLayout()
        self.switch_layout.addWidget(QtWidgets.QLabel(), 1)
        self.switch_layout.addWidget(self.datetime_switch)
        #
        self.main_layout.addRow(self.datetime_label, self.datetime_edit)
        self.main_layout.addRow(QtWidgets.QLabel(), self.switch_layout)
        self.datetime_switch.clicked.connect(self.change_multiple_datetime)

    def _build_layout_multiple_datetime(self):
        self.single_datetime = False
        self.datetime_label = QtWidgets.QLabel("Time-series file:")
        self.datetimes_layout = QtWidgets.QHBoxLayout()
        self.load_datetimes_button = QtWidgets.QPushButton("Load file")
        self.load_datetimes_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.load_datetimes_button.clicked.connect(self.load_datetimes)
        self.loaded_datetimes_label = QtWidgets.QLabel("")
        self.datetimes_layout.addWidget(self.load_datetimes_button)
        self.datetimes_layout.addWidget(self.loaded_datetimes_label, 1)
        self.datetime_switch = QtWidgets.QPushButton(" Input single datetime ")
        self.datetime_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.switch_layout = QtWidgets.QHBoxLayout()
        self.switch_layout.addWidget(QtWidgets.QLabel(), 1)
        self.switch_layout.addWidget(self.datetime_switch)
        self.main_layout.addRow(self.datetime_label, self.datetimes_layout)
        self.main_layout.addRow(QtWidgets.QLabel(), self.switch_layout)
        self.datetime_switch.clicked.connect(self.change_single_datetime)

    def _clear_form_rows(self):
        self.main_layout.removeRow(2)
        self.main_layout.removeRow(1)

    @QtCore.Slot()
    def change_single_datetime(self):
        self._clear_form_rows()
        self._build_layout_single_datetime()

    @QtCore.Slot()
    def change_multiple_datetime(self):
        self._clear_form_rows()
        self._build_layout_multiple_datetime()

    @QtCore.Slot()
    def load_datetimes(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        self.loaded_datetimes = csv.read_datetimes(path)
        shown_path = path
        if len(shown_path) > MAX_PATH_LEN:
            shown_path = "..." + shown_path[-(MAX_PATH_LEN - 3) : -1]
        self.loaded_datetimes_label.setText(path)

    def get_satellite(self) -> str:
        return self.sat_names[self.combo_sats.currentIndex()]

    def get_datetimes(self) -> Union[datetime, List[datetime]]:
        if self.single_datetime:
            return self.datetime_edit.dateTime().toPython()
        else:
            return self.loaded_datetimes

    @QtCore.Slot()
    def update_from_combobox(self, i: int):
        sat = self.satellites[i]
        d0, df = sat.get_datetime_range()
        dt0 = QtCore.QDateTime(d0.year, d0.month, d0.day, d0.hour, d0.minute, d0.second)
        dtf = QtCore.QDateTime(df.year, df.month, df.day, df.hour, df.minute, df.second)
        self.datetime_edit.setMinimumDateTime(dt0)
        self.datetime_edit.setMaximumDateTime(dtf)


class InputWidget(QtWidgets.QWidget):
    def __init__(self, satellites: List[Satellite], change_callback: Callable):
        super().__init__()
        self.satellites = satellites
        self.change_callback = change_callback
        self.last_point: Point = None
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        self.surface = SurfaceInputWidget()
        self.tabs.addTab(self.surface, "Geographic")
        self.custom = CustomInputWidget()
        self.tabs.addTab(self.custom, "Selenographic")
        self.satellite = SatelliteInputWidget(self.satellites)
        self.tabs.addTab(self.satellite, "Satellite")
        self.main_layout.addWidget(self.tabs)

    @staticmethod
    def _are_different_points(point_a: Point, point_b: Point) -> bool:
        if type(point_a) != type(point_b):
            return True
        for att in dir(point_a):
            if len(att) < 2 or att[0:2] != "__":
                a0 = getattr(point_a, att)
                a1 = getattr(point_b, att)
                if a0 != a1:
                    return True
        return False

    def _check_last_point(self, point: Point):
        if InputWidget._are_different_points(point, self.last_point):
            self.change_callback()
        self.last_point = point

    def get_point(self) -> Point:
        tab = self.tabs.currentWidget()
        if isinstance(tab, SurfaceInputWidget):
            lat = self.surface.get_latitude()
            lon = self.surface.get_longitude()
            alt = self.surface.get_altitude()
            dts = self.surface.get_datetimes()
            point = SurfacePoint(lat, lon, alt, dts)
        elif isinstance(tab, CustomInputWidget):
            dsm = self.custom.get_dist_sun_moon()
            dom = self.custom.get_dist_obs_moon()
            olat = self.custom.get_selen_obs_lat()
            olon = self.custom.get_selen_obs_lon()
            slon = self.custom.get_selen_sun_lon()
            mpa = self.custom.get_moon_phase_angle()
            ampa = abs(mpa)
            point = CustomPoint(dsm, dom, olat, olon, slon, ampa, mpa)
        else:
            sat = self.satellite.get_satellite()
            dts = self.satellite.get_datetimes()
            point = SatellitePoint(sat, dts)
        self._check_last_point(point)
        return point


class ComparisonInput(QtWidgets.QWidget):
    def __init__(self, callback_change: Callable):
        super().__init__()
        self.callback_change = callback_change
        self.loaded_srf: SpectralResponseFunction = None
        self.loaded_moons: List[LunarObservation] = []
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # MOON Observation data filepath
        self.moon_obs_layout = QtWidgets.QFormLayout()
        self.moon_obs_label = QtWidgets.QLabel("Lunar Observation files:")
        self.moon_obs_feedback = QtWidgets.QLabel("No files loaded")
        self.moon_obs_feedback.setWordWrap(True)
        self.clear_moon_obs_button = QtWidgets.QPushButton(" Unload files ")
        self.clear_moon_obs_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.clear_moon_obs_button.clicked.connect(self.clear_obs_files)
        self.moon_obs_button = QtWidgets.QPushButton(" Load files ")
        self.moon_obs_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.moon_obs_button.clicked.connect(self.load_obs_files)
        self.moon_obs_layout.addRow(self.moon_obs_label, self.moon_obs_feedback)
        obs_but_layout = QtWidgets.QHBoxLayout()
        obs_but_layout.addWidget(QtWidgets.QLabel(), 1)
        obs_but_layout.addWidget(self.moon_obs_button)
        obs_but_layout.addWidget(self.clear_moon_obs_button)
        self.moon_obs_layout.addRow(QtWidgets.QLabel(), obs_but_layout)
        # SRF filepath
        self.srf_layout = QtWidgets.QFormLayout()
        self.srf_label = QtWidgets.QLabel("SRF file:")
        self.srf_feedback = QtWidgets.QLabel("No file loaded")
        self.srf_feedback.setWordWrap(True)
        self.srf_button = QtWidgets.QPushButton(" Load file ")
        self.srf_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.srf_button.clicked.connect(self.load_srf_file)
        self.srf_layout.addRow(self.srf_label, self.srf_feedback)
        srf_but_layout = QtWidgets.QHBoxLayout()
        srf_but_layout.addWidget(QtWidgets.QLabel(), 1)
        srf_but_layout.addWidget(self.srf_button)
        self.srf_layout.addRow(QtWidgets.QLabel(), srf_but_layout)
        # Finish main layout
        self.main_layout.addLayout(self.moon_obs_layout)
        self.main_layout.addLayout(self.srf_layout)

    @QtCore.Slot()
    def load_srf_file(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        self.loaded_srf = srf.read_srf(path)
        shown_path = path
        if len(shown_path) > MAX_PATH_LEN:
            shown_path = "..." + shown_path[-(MAX_PATH_LEN - 3) : -1]
        self.srf_feedback.setText(shown_path)
        self.callback_change()

    @QtCore.Slot()
    def load_obs_files(self):
        paths = QtWidgets.QFileDialog().getOpenFileNames(self)[0]
        for path in paths:
            self._add_observation(moon.read_moon_obs(path))
        shown_path = "Loaded {} files".format(len(self.loaded_moons))
        self.moon_obs_feedback.setText(shown_path)
        self.callback_change()

    def _add_observation(self, obs: LunarObservation):
        for i, pob in enumerate(self.loaded_moons):
            if obs.dt < pob.dt:
                self.loaded_moons.insert(i, obs)
                return
        self.loaded_moons.append(obs)

    @QtCore.Slot()
    def clear_obs_files(self):
        self.loaded_moons = []
        self.moon_obs_feedback.setText("No files loaded")
        self.callback_change()

    def get_srf(self) -> Union[SpectralResponseFunction, None]:
        return self.loaded_srf

    def get_moon_obs(self) -> List[LunarObservation]:
        return self.loaded_moons
