"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime, timezone
from typing import Union, Tuple, Iterable

import PySide2.QtWidgets

"""___Third-Party Modules___"""
from typing import Callable, List
from PySide2 import QtWidgets, QtCore, QtGui
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    LunarObservation,
    Point,
    Satellite,
    SpectralResponseFunction,
    SurfacePoint,
    CustomPoint,
    SatellitePoint,
    KernelsPath,
)
from lime_tbx.datatypes import constants
from lime_tbx.gui.util import (
    CallbackWorker,
    start_thread as _start_thread,
)
from lime_tbx.filedata import csv, srf
from lime_tbx.filedata import moon

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "03/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

MAX_PATH_LEN = 35

_A_LOT_DATETIMES_MSG = f"Usually, calculated reflectance values are used to obtain irradiances. \
However, due to the large size of the error correlation matrices, these matrices are not \
stored in memory for more than {constants.MAX_LIMIT_REFL_ERR_CORR_ARE_STORED} datetimes. \
This requires recalculating reflectances whenever irradiances are computed. Therefore, \
if you want both values in this case, it is advisable to choose to calculate irradiances first."


def _callback_read_obs_files(
    paths: List[str], kernels_path: KernelsPath, eocfi_path: str
) -> List[LunarObservation]:
    obss = []
    for path in paths:
        obss.append(moon.read_moon_obs(path, kernels_path, eocfi_path))
    return [obss]


def _callback_read_srf(path: str) -> Tuple[SpectralResponseFunction, str]:
    read_srf = srf.read_srf(path)
    return (read_srf, path)


class _LimeDoubleInput(QtWidgets.QDoubleSpinBox):
    def __init__(self) -> None:
        super().__init__()

    def textFromValue(self, val: float) -> str:
        return f"{val}"


class CustomInputWidget(QtWidgets.QWidget):
    """
    Input widget that contains the GUI elements for the input of the needed parameters for
    the simulation of lunar values for a custom point with custom lunar data.
    """

    def __init__(self):
        super().__init__()
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.dist_sun_moon_label = QtWidgets.QLabel("Dist. Sun-Moon (AU):")
        self.dist_obs_moon_label = QtWidgets.QLabel("Dist. Observer-Moon (km):")
        self.moon_phase_angle_label = QtWidgets.QLabel("Moon phase angle (°):")
        self.selen_obs_lat_label = QtWidgets.QLabel("Observer sel. latitude (°):")
        self.selen_obs_lon_label = QtWidgets.QLabel("Observer sel. longitude (°):")
        self.selen_sun_lon_label = QtWidgets.QLabel("Sun sel. longitude (RAD):")
        self.dist_sun_moon_label.setToolTip(
            "Distance between the Sun and the Moon in Astronomical Units"
        )
        self.dist_obs_moon_label.setToolTip(
            "Distance between the observer and the Moon in kilometers"
        )
        self.moon_phase_angle_label.setToolTip("Moon phase angles in degrees")
        self.selen_obs_lat_label.setToolTip(
            "Selenographic latitude of the observer in degrees"
        )
        self.selen_obs_lon_label.setToolTip(
            "Selenographic longitude of the observer in degrees"
        )
        self.selen_sun_lon_label.setToolTip(
            "Selenographic longitude of the Sun in radians"
        )
        self.dist_sun_moon_spinbox = _LimeDoubleInput()
        self.dist_obs_moon_spinbox = _LimeDoubleInput()
        self.selen_obs_lat_spinbox = _LimeDoubleInput()
        self.selen_obs_lon_spinbox = _LimeDoubleInput()
        self.selen_sun_lon_spinbox = _LimeDoubleInput()
        self.moon_phase_angle_spinbox = _LimeDoubleInput()
        self.dist_sun_moon_spinbox.setMinimum(0.00001)
        self.dist_sun_moon_spinbox.setMaximum(1.5)
        self.dist_sun_moon_spinbox.setDecimals(20)
        self.dist_sun_moon_spinbox.setSingleStep(0.001)
        self.dist_sun_moon_spinbox.setValue(1)
        self.dist_obs_moon_spinbox.setMinimum(1)
        self.dist_obs_moon_spinbox.setMaximum(1000000)
        self.dist_obs_moon_spinbox.setDecimals(20)
        self.dist_obs_moon_spinbox.setValue(400000)
        self.selen_obs_lat_spinbox.setMinimum(-90)
        self.selen_obs_lat_spinbox.setMaximum(90)
        self.selen_obs_lat_spinbox.setDecimals(20)
        self.selen_obs_lon_spinbox.setMinimum(-180)
        self.selen_obs_lon_spinbox.setMaximum(180)
        self.selen_obs_lon_spinbox.setDecimals(20)
        self.selen_sun_lon_spinbox.setMinimum(-3.141592653589793)
        self.selen_sun_lon_spinbox.setMaximum(3.141592653589793)
        self.selen_sun_lon_spinbox.setDecimals(20)
        self.selen_sun_lon_spinbox.setSingleStep(0.1)
        self.moon_phase_angle_spinbox.setMinimum(-180)
        self.moon_phase_angle_spinbox.setMaximum(180)
        self.moon_phase_angle_spinbox.setDecimals(20)
        self.moon_phase_angle_spinbox.setValue(30)
        self.dists_layout = QtWidgets.QFormLayout()
        self.sel_coords_layout = QtWidgets.QFormLayout()
        self.dists_layout.addRow(self.dist_sun_moon_label, self.dist_sun_moon_spinbox)
        self.dists_layout.addRow(self.dist_obs_moon_label, self.dist_obs_moon_spinbox)
        self.dists_layout.addRow(
            self.moon_phase_angle_label, self.moon_phase_angle_spinbox
        )
        self.main_layout.addLayout(self.dists_layout)
        self.sel_coords_layout.addRow(
            self.selen_obs_lat_label, self.selen_obs_lat_spinbox
        )
        self.sel_coords_layout.addRow(
            self.selen_obs_lon_label, self.selen_obs_lon_spinbox
        )
        self.sel_coords_layout.addRow(
            self.selen_sun_lon_label, self.selen_sun_lon_spinbox
        )
        self.main_layout.addLayout(self.sel_coords_layout)

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

    def set_moon_phase_angle(self, mpa: float) -> float:
        self.moon_phase_angle_spinbox.setValue(mpa)

    def set_dist_sun_moon(self, dist_sun_moon: float) -> float:
        return self.dist_sun_moon_spinbox.setValue(dist_sun_moon)

    def set_dist_obs_moon(self, dist_obs_moon: float) -> float:
        return self.dist_obs_moon_spinbox.setValue(dist_obs_moon)

    def set_selen_obs_lat(self, selen_obs_lat: float) -> float:
        return self.selen_obs_lat_spinbox.setValue(selen_obs_lat)

    def set_selen_obs_lon(self, selen_obs_lon: float) -> float:
        return self.selen_obs_lon_spinbox.setValue(selen_obs_lon)

    def set_selen_sun_lon(self, selen_sun_lon: float) -> float:
        return self.selen_sun_lon_spinbox.setValue(selen_sun_lon)


class ShowDatetimeWidget(QtWidgets.QWidget):
    def __init__(self, datetimes: List[datetime]):
        super().__init__()
        self.dts = datetimes
        self._build_layout()
        self._fill_table()

    def _build_layout(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.groupbox = QtWidgets.QGroupBox()
        self.container_layout = QtWidgets.QVBoxLayout()
        self.data_layout = QtWidgets.QFormLayout()
        self.groupbox.setLayout(self.container_layout)
        # table
        self.table = QtWidgets.QTableWidget()

        self.container_layout.addLayout(self.data_layout, 1)
        self.data_layout.addWidget(self.table)
        self.container_layout.addStretch()

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.groupbox)
        self.main_layout.addWidget(self.scroll_area)

    def _fill_table(self):
        self.table.setRowCount(1 + len(self.dts))
        self.table.setColumnCount(6)
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("Year"))
        self.table.setItem(0, 1, QtWidgets.QTableWidgetItem("Month"))
        self.table.setItem(0, 2, QtWidgets.QTableWidgetItem("Day"))
        self.table.setItem(0, 3, QtWidgets.QTableWidgetItem("Hour"))
        self.table.setItem(0, 4, QtWidgets.QTableWidgetItem("Minute"))
        self.table.setItem(0, 5, QtWidgets.QTableWidgetItem("Second"))
        for i, dt in enumerate(self.dts):
            self.table.setItem(1 + i, 0, QtWidgets.QTableWidgetItem(str(dt.year)))
            self.table.setItem(1 + i, 1, QtWidgets.QTableWidgetItem(str(dt.month)))
            self.table.setItem(1 + i, 2, QtWidgets.QTableWidgetItem(str(dt.day)))
            self.table.setItem(1 + i, 3, QtWidgets.QTableWidgetItem(str(dt.hour)))
            self.table.setItem(1 + i, 4, QtWidgets.QTableWidgetItem(str(dt.minute)))
            self.table.setItem(
                1 + i,
                5,
                QtWidgets.QTableWidgetItem(str(dt.second + dt.microsecond / 1000)),
            )


class FlexibleDateTimeInput(QtWidgets.QWidget):
    def __init__(self, callback_check_calculable: Callable):
        super().__init__()
        self.callback_check_calculable = callback_check_calculable
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        # Single DT input
        self.single_dt_layout = QtWidgets.QHBoxLayout()
        ## DT form
        self.datetime_label = QtWidgets.QLabel("UTC DateTime:")
        self.datetime_edit = QtWidgets.QDateTimeEdit()
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd hh:mm:ss.zzz")
        self.datetime_edit.setDateTime(QtCore.QDateTime.currentDateTimeUtc())
        self.single_dt_layout.addWidget(self.datetime_label)
        self.single_dt_layout.addWidget(self.datetime_edit, 1)
        ## DT switch
        self.datetime_switch = QtWidgets.QPushButton(" Load time-series ")
        self.datetime_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.datetime_switch.clicked.connect(self.change_multiple_datetime)
        self.single_dt_layout.addWidget(self.datetime_switch)
        ## Add Layout
        self.single_dt_frame = QtWidgets.QWidget()
        self.single_dt_layout.setContentsMargins(0, 0, 0, 0)
        self.single_dt_frame.setLayout(self.single_dt_layout)
        self.main_layout.addWidget(self.single_dt_frame)
        self.single_dt_frame.setHidden(True)
        # Multiple DT
        self.multiple_dt_layout = QtWidgets.QHBoxLayout()
        ## DTs form
        self.datetimes_label = QtWidgets.QLabel("Time-series file:")
        self.load_datetimes_button = QtWidgets.QPushButton("Load file")
        self.load_datetimes_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.load_datetimes_button.clicked.connect(self.load_datetimes)
        self.loaded_datetimes_label = QtWidgets.QLabel("")
        self.datetimes_input_layout = QtWidgets.QHBoxLayout()
        self.datetimes_input_layout.addWidget(self.load_datetimes_button)
        self.datetimes_input_layout.addWidget(self.loaded_datetimes_label, 1)
        self.multiple_dt_layout.addWidget(self.datetimes_label)
        self.multiple_dt_layout.addLayout(self.datetimes_input_layout)
        ## Aux buttons
        self.dts_buttons = QtWidgets.QHBoxLayout()
        ### Show DTs
        self.show_datetimes_button = QtWidgets.QPushButton(" See Times ")
        self.show_datetimes_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.show_datetimes_button.clicked.connect(self.show_datetimes)
        self.dts_buttons.addWidget(self.show_datetimes_button)
        self.dts_buttons.addWidget(QtWidgets.QLabel(), 1)
        ### Info 'a lot dts'
        self.info_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MessageBoxInformation
        )
        self.info_pixmap = self.info_icon.pixmap(32)
        self.info_hidden_datetimes_a_lot = QtWidgets.QLabel(" ")
        self.info_hidden_datetimes_a_lot.setPixmap(self.info_pixmap)
        self.info_hidden_datetimes_a_lot.setWordWrap(True)
        self.info_hidden_datetimes_a_lot.hide()
        self.dts_buttons.addWidget(self.info_hidden_datetimes_a_lot)
        self.dts_buttons.addWidget(QtWidgets.QLabel(), 1)
        ### Datetimes switch
        self.datetimes_switch = QtWidgets.QPushButton(" Input single time ")
        self.datetimes_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.datetimes_switch.clicked.connect(self.change_single_datetime)
        self.dts_buttons.addWidget(self.datetimes_switch)
        self.dts_buttons.setContentsMargins(0, 0, 0, 0)
        self.multiple_dt_layout.addLayout(self.dts_buttons)
        ## Add Layout
        self.multiple_dt_frame = QtWidgets.QWidget()
        self.multiple_dt_layout.setContentsMargins(0, 0, 0, 0)
        self.multiple_dt_frame.setLayout(self.multiple_dt_layout)
        self.main_layout.addWidget(self.multiple_dt_frame)
        self.multiple_dt_frame.setHidden(True)
        # Build single dt
        self._build_layout_single_datetime()

    def _build_layout_single_datetime(self):
        self.single_datetime = True
        self.loaded_datetimes = []
        self.single_dt_frame.setHidden(False)

    def _build_layout_multiple_datetime(self):
        self.single_datetime = False
        self.multiple_dt_frame.setHidden(False)
        self.check_if_a_lot_dts_and_update_msg()

    def _clear_layout(self):
        self.single_dt_frame.setHidden(True)
        self.multiple_dt_frame.setHidden(True)

    def change_single_datetime(self):
        self._clear_layout()
        self._build_layout_single_datetime()
        self.callback_check_calculable()

    def change_multiple_datetime(self):
        self._clear_layout()
        self._build_layout_multiple_datetime()
        self.callback_check_calculable()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    def get_datetimes(self) -> Union[datetime, List[datetime]]:
        if self.single_datetime:
            return self.datetime_edit.dateTime().toPython().replace(tzinfo=timezone.utc)
        else:
            return self.loaded_datetimes

    @QtCore.Slot()
    def load_datetimes(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        if path != "":
            try:
                self.loaded_datetimes = csv.read_datetimes(path)
            except Exception as e:
                self.show_error(e)
            else:
                shown_path = path
                if len(shown_path) > MAX_PATH_LEN:
                    shown_path = "..." + shown_path[-(MAX_PATH_LEN - 3) : -1]
                self.loaded_datetimes_label.setText(shown_path)
                self.callback_check_calculable()
        self.check_if_a_lot_dts_and_update_msg()

    @QtCore.Slot()
    def show_datetimes(self):
        self.datetimes_window = QtWidgets.QMainWindow(self)
        self.datetimes_widget = ShowDatetimeWidget(self.loaded_datetimes)
        self.datetimes_window.setCentralWidget(self.datetimes_widget)
        self.datetimes_window.show()
        self.datetimes_window.resize(660, 230)

    def set_datetimes(self, dt: Union[List[datetime], datetime]):
        if isinstance(dt, list) and len(dt) == 1:
            dt = dt[0]
        if (
            isinstance(dt, list)
            or isinstance(dt, np.ndarray)
            or isinstance(dt, Iterable)
        ):
            if self.single_datetime:
                self.change_multiple_datetime()
            self.loaded_datetimes = dt
            self.loaded_datetimes_label.setText("Loaded from LGLOD file.")
            self.callback_check_calculable()
        else:
            if not self.single_datetime:
                self.change_single_datetime()
            self.datetime_edit.setDateTime(
                QtCore.QDateTime(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
                )
            )
        self.check_if_a_lot_dts_and_update_msg()

    def check_if_a_lot_dts_and_update_msg(self):
        if self.single_datetime:
            return
        max_dts = constants.MAX_LIMIT_REFL_ERR_CORR_ARE_STORED
        if len(self.loaded_datetimes) > max_dts and not self._skip_uncs:
            self.info_hidden_datetimes_a_lot.show()
            self.info_hidden_datetimes_a_lot.setToolTip(_A_LOT_DATETIMES_MSG)
        else:
            self.info_hidden_datetimes_a_lot.hide()
            self.info_hidden_datetimes_a_lot.setToolTip("")

    def is_calculable(self) -> bool:
        if self.single_datetime:
            return True
        else:
            return len(self.loaded_datetimes) > 0


class SurfaceInputWidget(QtWidgets.QWidget):
    """
    Input widget that contains the GUI elements for the input of the needed parameters for
    the simulation of lunar values for a geographic position at a concrete time.
    """

    def __init__(self, callback_check_calculable: Callable, skip_uncs: bool):
        super().__init__()
        self.callback_check_calculable = callback_check_calculable
        self._skip_uncs = skip_uncs
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.latitude_label = QtWidgets.QLabel("Latitude (°):")
        self.longitude_label = QtWidgets.QLabel("Longitude (°):")
        self.altitude_label = QtWidgets.QLabel("Altitude (km):")
        self.latitude_spinbox = _LimeDoubleInput()
        self.longitude_spinbox = _LimeDoubleInput()
        self.altitude_spinbox = _LimeDoubleInput()
        self.latitude_spinbox.setMinimum(-90)
        self.latitude_spinbox.setMaximum(90)
        self.latitude_spinbox.setDecimals(20)
        self.longitude_spinbox.setMinimum(-180)
        self.longitude_spinbox.setMaximum(180)
        self.longitude_spinbox.setDecimals(20)
        self.altitude_spinbox.setMinimum(-1)
        self.altitude_spinbox.setMaximum(1000000)
        self.altitude_spinbox.setDecimals(20)
        self.coordinates_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.coordinates_layout)
        self.coord_forms_layouts = [QtWidgets.QFormLayout() for _ in range(3)]
        self.coord_forms_layouts[0].addRow(self.latitude_label, self.latitude_spinbox)
        self.coord_forms_layouts[1].addRow(self.longitude_label, self.longitude_spinbox)
        self.coord_forms_layouts[2].addRow(self.altitude_label, self.altitude_spinbox)
        for lay in self.coord_forms_layouts:
            self.coordinates_layout.addLayout(lay)
        self.flexdt_wg = FlexibleDateTimeInput(self.callback_check_calculable)
        self.main_layout.addWidget(self.flexdt_wg)
        self.main_layout.addStretch()

    @QtCore.Slot()
    def change_single_datetime(self):
        self.flexdt_wg.change_single_datetime()

    @QtCore.Slot()
    def change_multiple_datetime(self):
        self.flexdt_wg.change_multiple_datetime()

    def get_latitude(self) -> float:
        return self.latitude_spinbox.value()

    def get_longitude(self) -> float:
        return self.longitude_spinbox.value()

    def get_altitude(self) -> float:
        # Get altitude in meters
        return self.altitude_spinbox.value() * 1000

    def get_datetimes(self) -> Union[datetime, List[datetime]]:
        return self.flexdt_wg.get_datetimes()

    def set_latitude(self, lat: float):
        self.latitude_spinbox.setValue(lat)

    def set_longitude(self, lon: float):
        self.longitude_spinbox.setValue(lon)

    def set_altitude(self, alt: float):
        self.altitude_spinbox.setValue(alt)

    def set_datetimes(self, dt: Union[List[datetime], datetime]):
        self.flexdt_wg.set_datetimes(dt)

    def is_calculable(self) -> bool:
        return self.flexdt_wg.is_calculable()

    def set_is_skipping_uncs(self, skip_uncs: bool):
        self._skip_uncs = skip_uncs
        self.flexdt_wg.check_if_a_lot_dts_and_update_msg()


_DEF_MAX_DATE = datetime(2037, 7, 16, 23, 59, 55, tzinfo=timezone.utc)


class SatelliteInputWidget(QtWidgets.QWidget):
    def __init__(
        self,
        satellites: List[Satellite],
        callback_check_calculable: Callable,
        skip_uncs: bool,
    ) -> None:
        super().__init__()
        self.satellites = satellites
        self.sat_names = [s.name for s in self.satellites]
        self._build_layout()
        self.callback_check_calculable = callback_check_calculable
        self.current_min_date = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.current_max_date = _DEF_MAX_DATE
        self._skip_uncs = skip_uncs
        self.all_loaded_datetimes = []
        self.update_from_combobox(0)

    def _build_layout(self):
        self.main_layout = QtWidgets.QFormLayout(self)
        # satellite
        self.satellite_label = QtWidgets.QLabel("Satellite:")
        self.combo_sats = QtWidgets.QComboBox()
        self.combo_sats.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.combo_sats.addItems(self.sat_names)
        for i, sat in enumerate(self.satellites):
            if not sat.orbit_files:
                self.combo_sats.model().item(i).setEnabled(False)
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
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd hh:mm:ss.zzz")
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
        self.loaded_datetimes_label.setWordWrap(True)
        self.datetimes_layout.addWidget(self.load_datetimes_button)
        self.datetimes_layout.addWidget(self.loaded_datetimes_label, 1)
        self.datetime_switch = QtWidgets.QPushButton(" Input single datetime ")
        self.datetime_switch.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_datetimes_button = QtWidgets.QPushButton(" See datetimes ")
        self.show_datetimes_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        # info
        self.info_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MessageBoxInformation
        )
        self.info_pixmap = self.info_icon.pixmap(32)
        self.info_hidden_datetimes_a_lot = QtWidgets.QLabel(" ")
        self.info_hidden_datetimes_a_lot.setPixmap(self.info_pixmap)
        self.info_hidden_datetimes_a_lot.setWordWrap(True)
        self.info_hidden_datetimes_a_lot.hide()
        # warning
        self.warning_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MessageBoxWarning
        )  # QtGui.QIcon.fromTheme('dialog-warning')
        self.warning_pixmap = self.warning_icon.pixmap(32)
        self.warn_hidden_datetimes_invalid = QtWidgets.QLabel(" ")
        self.warn_hidden_datetimes_invalid.setPixmap(self.warning_pixmap)
        self.warn_hidden_datetimes_invalid.setWordWrap(True)
        self.warn_hidden_datetimes_invalid.hide()
        self.switch_layout = QtWidgets.QHBoxLayout()
        self.switch_layout.addWidget(self.show_datetimes_button)
        self.switch_layout.addWidget(self.info_hidden_datetimes_a_lot)
        self.switch_layout.addWidget(self.warn_hidden_datetimes_invalid)
        self.switch_layout.addWidget(QtWidgets.QLabel(), 1)
        self.switch_layout.addWidget(self.datetime_switch)
        self.main_layout.addRow(self.datetime_label, self.datetimes_layout)
        self.main_layout.addRow(QtWidgets.QLabel(), self.switch_layout)
        self.datetime_switch.clicked.connect(self.change_single_datetime)
        self.show_datetimes_button.clicked.connect(self.show_datetimes)
        self._check_if_a_lot_dts_and_update_msg()

    def _clear_form_rows(self):
        self.main_layout.removeRow(2)
        self.main_layout.removeRow(1)

    @QtCore.Slot()
    def change_single_datetime(self):
        self._clear_form_rows()
        self._build_layout_single_datetime()
        self.callback_check_calculable()

    @QtCore.Slot()
    def change_multiple_datetime(self):
        self._clear_form_rows()
        self._build_layout_multiple_datetime()
        self.callback_check_calculable()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    @QtCore.Slot()
    def load_datetimes(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        if path != "":
            try:
                self.loaded_datetimes = csv.read_datetimes(path)
                self.all_loaded_datetimes = self.loaded_datetimes
                self.update_dates_with_limits()
            except Exception as e:
                self.show_error(e)
            else:
                shown_path = path
                if len(shown_path) > MAX_PATH_LEN:
                    shown_path = "..." + shown_path[-(MAX_PATH_LEN - 3) : -1]
                self.loaded_datetimes_label.setText(path)
                self.callback_check_calculable()
        self._check_if_a_lot_dts_and_update_msg()

    @QtCore.Slot()
    def show_datetimes(self):
        self.datetimes_window = QtWidgets.QMainWindow(self)
        self.datetimes_widget = ShowDatetimeWidget(self.loaded_datetimes)
        self.datetimes_window.setCentralWidget(self.datetimes_widget)
        self.datetimes_window.show()
        self.datetimes_window.resize(660, 230)

    def get_satellite(self) -> str:
        return self.sat_names[self.combo_sats.currentIndex()]

    def get_datetimes(self) -> Union[datetime, List[datetime]]:
        if self.single_datetime:
            return self.datetime_edit.dateTime().toPython().replace(tzinfo=timezone.utc)
        else:
            return self.loaded_datetimes

    def set_satellite(self, name: str):
        self.combo_sats.setCurrentIndex(self.sat_names.index(name))

    def set_datetimes(self, dt: Union[datetime, List[datetime]]):
        if isinstance(dt, list) and len(dt) == 1:
            dt = dt[0]
        if isinstance(dt, list):
            if self.single_datetime:
                self.change_multiple_datetime()
            self.all_loaded_datetimes = dt
            self.loaded_datetimes = dt
            self.loaded_datetimes_label.setText("Loaded from LGLOD file.")
            self.update_dates_with_limits()
            self.callback_check_calculable()
        else:
            if not self.single_datetime:
                self.change_single_datetime()
            self.datetime_edit.setDateTime(
                QtCore.QDateTime(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
                )
            )
        self._check_if_a_lot_dts_and_update_msg()

    def update_dates_with_limits(self):
        self.loaded_datetimes = [
            dt
            for dt in self.all_loaded_datetimes
            if self.current_min_date < dt < self.current_max_date
        ]
        if len(self.loaded_datetimes) != len(self.all_loaded_datetimes):
            missing_dts = [
                dt
                for dt in self.all_loaded_datetimes
                if dt not in self.loaded_datetimes
            ]
            missing_dts_msg = ",".join(
                [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in missing_dts]
            )
            self.warn_hidden_datetimes_invalid.show()
            self.warn_hidden_datetimes_invalid.setToolTip(
                f"The following datetimes are not available for the selected satellite: {missing_dts_msg}"
            )
        else:
            self.warn_hidden_datetimes_invalid.hide()
            self.warn_hidden_datetimes_invalid.setToolTip("")
        self._check_if_a_lot_dts_and_update_msg()

    @QtCore.Slot()
    def update_from_combobox(self, i: int):
        sat = self.satellites[i]
        d0, df = sat.get_datetime_range()
        if d0 == None:
            d0 = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.current_min_date = d0
        if df == None:
            df = _DEF_MAX_DATE
        df = min(df, _DEF_MAX_DATE)
        self.current_max_date = df
        if self.single_datetime:
            dt0 = QtCore.QDateTime(
                d0.year, d0.month, d0.day, d0.hour, d0.minute, d0.second
            )
            self.datetime_edit.setMinimumDateTime(dt0)
            dtf = QtCore.QDateTime(
                df.year, df.month, df.day, df.hour, df.minute, df.second
            )
            self.datetime_edit.setMaximumDateTime(dtf)
        else:
            self.update_dates_with_limits()
            self.callback_check_calculable()

    def _check_if_a_lot_dts_and_update_msg(self):
        if self.single_datetime:
            return
        max_dts = constants.MAX_LIMIT_REFL_ERR_CORR_ARE_STORED
        if len(self.loaded_datetimes) > max_dts and not self._skip_uncs:
            self.info_hidden_datetimes_a_lot.show()
            self.info_hidden_datetimes_a_lot.setToolTip(_A_LOT_DATETIMES_MSG)
        else:
            self.info_hidden_datetimes_a_lot.hide()
            self.info_hidden_datetimes_a_lot.setToolTip("")

    def get_current_min_max_dates(self) -> Tuple[datetime, datetime]:
        return self.current_min_date, self.current_max_date

    def is_calculable(self) -> bool:
        if self.single_datetime:
            return True
        else:
            return len(self.loaded_datetimes) > 0

    def set_is_skipping_uncs(self, skip_uncs: bool):
        self._skip_uncs = skip_uncs
        self._check_if_a_lot_dts_and_update_msg()


class InputWidget(QtWidgets.QWidget):
    def __init__(
        self,
        satellites: List[Satellite],
        change_callback: Callable,
        callback_check_calculable: Callable,
        skip_uncs: bool,
    ):
        super().__init__()
        self.satellites = satellites
        self.change_callback = change_callback
        self.last_point: Point = None
        self.callback_check_calculable = callback_check_calculable
        self.skip_uncs = skip_uncs
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        self.surface = SurfaceInputWidget(
            self.callback_check_calculable, self.skip_uncs
        )
        self.tabs.addTab(self.surface, "Geographic")
        self.custom = CustomInputWidget()
        self.tabs.addTab(self.custom, "Selenographic")
        self.satellite = SatelliteInputWidget(
            self.satellites, self.callback_check_calculable, self.skip_uncs
        )
        self.tabs.addTab(self.satellite, "Satellite")
        self.tabs.currentChanged.connect(self.callback_check_calculable)
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

    def set_last_point_to_point(self):
        self.last_point = self._get_point()

    def set_point(self, point: Point):
        if isinstance(point, SurfacePoint):
            self.tabs.setCurrentIndex(0)
            self.surface.set_latitude(point.latitude)
            self.surface.set_longitude(point.longitude)
            self.surface.set_altitude(point.altitude)
            self.surface.set_datetimes(point.dt)
        elif isinstance(point, CustomPoint):
            self.tabs.setCurrentIndex(1)
            self.custom.set_dist_sun_moon(point.distance_sun_moon)
            self.custom.set_dist_obs_moon(point.distance_observer_moon)
            self.custom.set_selen_obs_lat(point.selen_obs_lat)
            self.custom.set_selen_obs_lon(point.selen_obs_lon)
            self.custom.set_selen_sun_lon(point.selen_sun_lon)
            self.custom.set_moon_phase_angle(point.moon_phase_angle)
        elif isinstance(point, SatellitePoint):
            self.tabs.setCurrentIndex(2)
            self.satellite.set_satellite(point.name)
            self.satellite.set_datetimes(point.dt)

    def _get_point(self) -> Point:
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
        return point

    def is_calculable(self) -> bool:
        tab = self.tabs.currentWidget()
        if isinstance(tab, SurfaceInputWidget):
            return self.surface.is_calculable()
        elif isinstance(tab, SatelliteInputWidget):
            return self.satellite.is_calculable()
        return True

    def get_point(self) -> Point:
        point = self._get_point()
        self._check_last_point(point)
        return point

    def set_is_skipping_uncs(self, skip_uncs: bool):
        self.skip_uncs = skip_uncs
        self.surface.set_is_skipping_uncs(skip_uncs)
        self.satellite.set_is_skipping_uncs(skip_uncs)


class ComparisonInput(QtWidgets.QWidget):
    def __init__(
        self,
        callback_change: Callable,
        callback_compare_but_enable: Callable,
        kernels_path: KernelsPath,
        eocfi_path: str,
    ):
        super().__init__()
        self.callback_change = callback_change
        self.callback_compare_but_enable = callback_compare_but_enable
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
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

    def _load_srf_file_finished(self, data):
        self.loaded_srf = data[0]
        path = data[1]
        shown_path = path
        if len(shown_path) > MAX_PATH_LEN:
            shown_path = "..." + shown_path[-(MAX_PATH_LEN - 3) : -1]
        self.srf_feedback.setText(shown_path)
        self.callback_change()
        self._set_enabled_gui_input(True)

    def _load_srf_file_error(self, e):
        self.show_error(e)
        self.clear_srf()

    @QtCore.Slot()
    def load_srf_file(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        if path != "":
            self.callback_compare_but_enable(False)
            self._set_enabled_gui_input(False)
            self.srf_feedback.setText("Loading...")
            self.worker = CallbackWorker(
                _callback_read_srf,
                [path],
            )
            self._start_thread(self._load_srf_file_finished, self._load_srf_file_error)

    def show_error(self, error: Exception):
        self._set_enabled_gui_input(True)
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    def _start_thread(self, finished: Callable, error: Callable, info: Callable = None):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error, info)

    def _set_enabled_gui_input(self, enabled: bool):
        self.setEnabled(enabled)

    @QtCore.Slot()
    def load_obs_files(self):
        paths = QtWidgets.QFileDialog().getOpenFileNames(self)[0]
        if len(paths) == 0:
            return
        self.callback_compare_but_enable(False)
        self._set_enabled_gui_input(False)
        self.moon_obs_feedback.setText("Loading...")
        self.worker = CallbackWorker(
            _callback_read_obs_files,
            [paths, self.kernels_path, self.eocfi_path],
        )
        self._start_thread(
            self.loading_obs_files_finished, self.loading_obs_files_error
        )

    def loading_obs_files_finished(self, data):
        obs = data[0]
        for ob in obs:
            self._add_observation(ob)
        if len(self.loaded_moons) > 0:
            shown_path = "Loaded {} files".format(len(self.loaded_moons))
            self.moon_obs_feedback.setText(shown_path)
        self._set_enabled_gui_input(True)
        self.callback_change()

    def loading_obs_files_error(self, e):
        self.show_error(e)
        self.clear_obs_files()

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

    def clear_srf(self):
        self.loaded_srf = None
        self.srf_feedback.setText("No file loaded")
        self.callback_change()

    def get_srf(self) -> Union[SpectralResponseFunction, None]:
        return self.loaded_srf

    def get_moon_obs(self) -> List[LunarObservation]:
        return self.loaded_moons

    def clear_input(self) -> None:
        self.clear_srf()
        self.clear_obs_files()
