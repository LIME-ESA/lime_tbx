"""describe class"""

"""___Built-In Modules___"""
from typing import List, Callable, Union, Tuple
from datetime import datetime
import time

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import settings, output, input, srf, help
from ..simulation.regular_simulation import regular_simulation
from ..datatypes.datatypes import (
    PolarizationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    IrradianceCoefficients,
    SurfacePoint,
    CustomPoint,
)
from ..eocfi_adapter import eocfi_adapter

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "02/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class CallbackWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    exception = QtCore.Signal(Exception)

    def __init__(self, callback: Callable, args: list):
        super().__init__()
        self.callback = callback
        self.args = args

    def run(self):
        try:
            res = self.callback(*self.args)
            self.finished.emit(list(res))
        except Exception as e:
            self.exception.emit(e)


def eli_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: IrradianceCoefficients,
    kernels_path: str,
    eocfi_path: str,
) -> Tuple[List[float], List[float]]:
    rs = regular_simulation.RegularSimulation()
    time.sleep(0.01)  # For some reason without this the GUI doesn't get disabled.
    if isinstance(point, SurfacePoint):
        elis: List[float] = rs.get_eli_from_surface(srf, point, coeffs, kernels_path)
    elif isinstance(point, CustomPoint):
        elis: List[float] = rs.get_eli_from_custom(srf, point, coeffs)
    else:
        elis: List[float] = rs.get_eli_from_satellite(
            srf, point, coeffs, kernels_path, eocfi_path
        )
    wlens = srf.get_wavelengths()
    return wlens, elis, point


def elref_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: IrradianceCoefficients,
    kernels_path: str,
    eocfi_path: str,
) -> Tuple[List[float], List[float]]:
    rs = regular_simulation.RegularSimulation()
    if isinstance(point, SurfacePoint):
        elrefs: List[float] = rs.get_elref_from_surface(
            srf, point, coeffs, kernels_path
        )
    elif isinstance(point, CustomPoint):
        elrefs: List[float] = rs.get_elref_from_custom(srf, point, coeffs)
    else:
        elrefs: List[float] = rs.get_elref_from_satellite(
            srf, point, coeffs, kernels_path, eocfi_path
        )
    wlens = srf.get_wavelengths()
    return wlens, elrefs, point


def polar_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: PolarizationCoefficients,
    kernels_path: str,
    eocfi_path: str,
) -> Tuple[List[float], List[float]]:
    rs = regular_simulation.RegularSimulation()
    if isinstance(point, SurfacePoint):
        polars: List[float] = rs.get_polarized_from_surface(
            srf, point, coeffs, kernels_path
        )
    elif isinstance(point, CustomPoint):
        polars: List[float] = rs.get_polarized_from_custom(srf, point, coeffs)
    else:
        polars: List[float] = rs.get_polarized_from_satellite(
            srf, point, coeffs, kernels_path, eocfi_path
        )
    wlens = srf.get_wavelengths()
    return wlens, polars, point


class MainSimulationsWidget(QtWidgets.QWidget):
    """
    Widget containing the landing gui, which lets the user calculate the eli, elref and polar
    of a surface point, custom input or satellite input at one moment.
    """

    def __init__(
        self,
        kernels_path: str,
        eocfi_path: str,
        settings_manager: settings.ISettingsManager,
    ):
        super().__init__()
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.settings_manager = settings_manager
        self.eocfi = eocfi_adapter.EOCFIConverter(self.eocfi_path)
        self.satellites = self.eocfi.get_sat_list()
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # input
        self.input_widget = input.InputWidget(self.satellites)
        # srf
        # self.srf_widget = srf.CurrentSRFWidget(self.settings_manager)
        # buttons
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.eli_button = QtWidgets.QPushButton("Irradiance")
        self.eli_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.eli_button.clicked.connect(self.show_eli)
        self.elref_button = QtWidgets.QPushButton("Reflectance")
        self.elref_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.elref_button.clicked.connect(self.show_elref)
        self.polar_button = QtWidgets.QPushButton("Polarization")
        self.polar_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.polar_button.clicked.connect(self.show_polar)
        self.buttons_layout.addWidget(self.eli_button)
        self.buttons_layout.addWidget(self.elref_button)
        self.buttons_layout.addWidget(self.polar_button)
        # Lower tab
        self.lower_tabs = QtWidgets.QTabWidget()
        self.lower_tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        # graph
        self.graph = output.GraphWidget(
            "Simulation output", "Wavelengths (nm)", "Units"
        )
        # srf widget
        self.srf_widget = srf.SRFEditWidget(self.settings_manager)
        # finish tab
        self.lower_tabs.addTab(self.graph, "Result")
        self.lower_tabs.addTab(self.srf_widget, "SRF")
        # finish main layout
        self.main_layout.addWidget(self.input_widget)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addWidget(self.lower_tabs, 1)

    def _unblock_gui(self):
        self.eli_button.setDisabled(False)
        self.elref_button.setDisabled(False)
        self.polar_button.setDisabled(False)
        self.input_widget.setDisabled(False)
        self.graph.setDisabled(False)
        self.srf_widget.setDisabled(False)
        self.lower_tabs.setDisabled(False)

    def _block_gui_loading(self):
        self.eli_button.setDisabled(True)
        self.elref_button.setDisabled(True)
        self.polar_button.setDisabled(True)
        self.input_widget.setDisabled(True)
        self.graph.setDisabled(True)
        self.srf_widget.setDisabled(True)
        self.lower_tabs.setDisabled(True)

    def _start_thread(self, finished: Callable, error: Callable):
        self.worker_th = QtCore.QThread()
        self.worker.moveToThread(self.worker_th)
        self.worker_th.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_th.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(finished)
        self.worker.exception.connect(self.worker_th.quit)
        self.worker.exception.connect(self.worker.deleteLater)
        self.worker.exception.connect(error)
        self.worker_th.finished.connect(self.worker_th.deleteLater)
        self.worker_th.start()

    @QtCore.Slot()
    def show_eli(self):
        """
        Calculate and show extraterrestrial lunar irradiances for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        self.worker = CallbackWorker(
            eli_callback, [srf, point, coeffs, self.kernels_path, self.eocfi_path]
        )
        self._start_thread(self.eli_finished, self.eli_error)

    def eli_finished(
        self,
        data: Tuple[
            List[float], List[float], Union[SurfacePoint, CustomPoint, SatellitePoint]
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[0], data[1], data[2])
        self.graph.update_labels(
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²/nm)",
        )

    def eli_error(self, error: Exception):
        self._unblock_gui()
        raise error

    @QtCore.Slot()
    def show_elref(self):
        """
        Calculate and show extraterrestrial lunar reflectances for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        self.worker = CallbackWorker(
            elref_callback, [srf, point, coeffs, self.kernels_path, self.eocfi_path]
        )
        self._start_thread(self.elref_finished, self.elref_error)

    def elref_finished(
        self,
        data: Tuple[
            List[float], List[float], Union[SurfacePoint, CustomPoint, SatellitePoint]
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[0], data[1], data[2])
        self.graph.update_labels(
            "Extraterrestrial Lunar Reflectances",
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
        )

    def elref_error(self, error: Exception):
        self._unblock_gui()
        raise error

    @QtCore.Slot()
    def show_polar(self):
        """
        Calculate and show extraterrestrial lunar polarization for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_polar_coeffs()
        self.worker = CallbackWorker(
            polar_callback, [srf, point, coeffs, self.kernels_path, self.eocfi_path]
        )
        self._start_thread(self.polar_finished, self.polar_error)

    def polar_finished(
        self,
        data: Tuple[
            List[float], List[float], Union[SurfacePoint, CustomPoint, SatellitePoint]
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[0], data[1], data[2])
        self.graph.update_labels(
            "Lunar polarization",
            "Wavelengths (nm)",
            "Polarizations (Fraction of unity)",
        )

    def polar_error(self, error: Exception):
        self._unblock_gui()
        raise error


class LimeTBXWidget(QtWidgets.QWidget):
    """
    Main widget of the lime toolbox desktop app.
    """

    def __init__(self, kernels_path: str, eocfi_path: str):
        super().__init__()
        self.setLocale("English")
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        settings_manager = settings.MockSettingsManager()
        self.page = MainSimulationsWidget(
            self.kernels_path, self.eocfi_path, settings_manager
        )
        self.main_layout.addWidget(self.page)

    def propagate_close_event(self):
        pass


class LimeTBXWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._create_menu_bar()

    def _create_actions(self):
        # File actions
        self.eli_from_file_action = QtWidgets.QAction(self)
        self.eli_from_file_action.setText("Calculate &Irradiance from file")
        self.eli_from_file_action.triggered.connect(self.eli_from_file)
        self.elref_from_file_action = QtWidgets.QAction(self)
        self.elref_from_file_action.setText("Calculate &Reflectance from file")
        self.elref_from_file_action.triggered.connect(self.elref_from_file)
        self.polar_from_file_action = QtWidgets.QAction(self)
        self.polar_from_file_action.setText("Calculate &Polarization from file")
        self.polar_from_file_action.triggered.connect(self.polar_from_file)
        self.exit_action = QtWidgets.QAction(self)
        self.exit_action.setText("E&xit")
        self.exit_action.triggered.connect(self.exit)
        # Coefficients actions
        self.download_coefficients_action = QtWidgets.QAction(self)
        self.download_coefficients_action.setText("&Download updated coefficients")
        self.download_coefficients_action.triggered.connect(self.download_coefficients)
        self.select_coefficients_action = QtWidgets.QAction(self)
        self.select_coefficients_action.setText("&Select coefficients")
        self.select_coefficients_action.triggered.connect(self.select_coefficients)
        # Help actions
        self.about_action = QtWidgets.QAction(self)
        self.about_action.setText("&About")
        self.about_action.triggered.connect(self.about)

    def _create_menu_bar(self):
        self._create_actions()
        self.menu_bar = self.menuBar()
        file_menu = QtWidgets.QMenu("&File", self)
        file_menu.addAction(self.eli_from_file_action)
        file_menu.addAction(self.elref_from_file_action)
        file_menu.addAction(self.polar_from_file_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        coeffs_menu = QtWidgets.QMenu("&Coefficients", self)
        coeffs_menu.addAction(self.download_coefficients_action)
        coeffs_menu.addAction(self.select_coefficients_action)
        help_menu = QtWidgets.QMenu("&Help", self)
        help_menu.addAction(self.about_action)
        self.menu_bar.addMenu(file_menu)
        self.menu_bar.addMenu(coeffs_menu)
        self.menu_bar.addMenu(help_menu)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        lime_tbx_w: LimeTBXWidget = self.centralWidget()
        lime_tbx_w.propagate_close_event()
        return super().closeEvent(event)

    # ACTIONS

    def eli_from_file(self):
        pass

    def elref_from_file(self):
        pass

    def polar_from_file(self):
        pass

    def exit(self):
        self.close()

    def download_coefficients(self):
        pass

    def select_coefficients(self):
        pass

    def about(self):
        about_dialog = help.AboutDialog(self)
        about_dialog.exec_()
