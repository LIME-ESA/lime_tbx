"""describe class"""

"""___Built-In Modules___"""
from typing import List, Callable, Union, Tuple
from datetime import datetime
import time

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import settings, output, input
from ..simulation.regular_simulation import regular_simulation
from ..datatypes.datatypes import (
    SpectralResponseFunction,
    IrradianceCoefficients,
    SurfacePoint,
    CustomPoint,
)

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
    point: Union[SurfacePoint, CustomPoint],
    coeffs: IrradianceCoefficients,
    kernels_path: str,
) -> Tuple[List[float], List[float]]:
    rs = regular_simulation.RegularSimulation()
    time.sleep(0.01)  # For some reason without this the GUI doesn't get disabled.
    if isinstance(point, SurfacePoint):
        elis: List[float] = rs.get_eli_from_surface(srf, point, coeffs, kernels_path)
    else:
        elis: List[float] = rs.get_eli_from_custom(srf, point, coeffs)
    wlens = list(srf.spectral_response.keys())
    return wlens, elis, point


def elref_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint],
    coeffs: IrradianceCoefficients,
    kernels_path: str,
) -> Tuple[List[float], List[float]]:
    rs = regular_simulation.RegularSimulation()
    if isinstance(point, SurfacePoint):
        elrefs: List[float] = rs.get_elref_from_surface(
            srf, point, coeffs, kernels_path
        )
    else:
        elrefs: List[float] = rs.get_elref_from_custom(srf, point, coeffs)
    wlens = list(srf.spectral_response.keys())
    return wlens, elrefs, point


class MainSimulationsWidget(QtWidgets.QWidget):
    """
    Widget containing the landing gui, which lets the user calculate the eli, elref and polar
    of a surface point, custom input or satellite input at one moment.
    """

    def __init__(self, kernels_path: str, settings_manager: settings.ISettingsManager):
        super().__init__()
        self.kernels_path = kernels_path
        self.settings_manager = settings_manager
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # input
        self.input_widget = input.InputWidget()
        # buttons
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.eli_button = QtWidgets.QPushButton("ELI")
        self.eli_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.eli_button.clicked.connect(self.show_eli)
        self.elref_button = QtWidgets.QPushButton("ELREF")
        self.elref_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.elref_button.clicked.connect(self.show_elref)
        self.polar_button = QtWidgets.QPushButton("POLAR")
        self.polar_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.polar_button.clicked.connect(self.show_polar)
        self.buttons_layout.addWidget(self.eli_button)
        self.buttons_layout.addWidget(self.elref_button)
        self.buttons_layout.addWidget(self.polar_button)
        # graph
        self.graph = output.GraphWidget(
            "Simulation output", "Wavelengths (nm)", "Units"
        )
        # finish main layout
        self.main_layout.addWidget(self.input_widget)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addWidget(self.graph, 1)

    def _unblock_gui(self):
        self.eli_button.setDisabled(False)
        self.elref_button.setDisabled(False)
        self.polar_button.setDisabled(False)
        self.input_widget.setDisabled(False)
        self.graph.setDisabled(False)

    def _block_gui_loading(self):
        self.eli_button.setDisabled(True)
        self.elref_button.setDisabled(True)
        self.polar_button.setDisabled(True)
        self.input_widget.setDisabled(True)
        self.graph.setDisabled(True)

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
            eli_callback, [srf, point, coeffs, self.kernels_path]
        )
        self._start_thread(self.eli_finished, self.eli_error)

    def eli_finished(
        self, data: Tuple[List[float], List[float], Union[SurfacePoint, CustomPoint]]
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
            elref_callback, [srf, point, coeffs, self.kernels_path]
        )
        self._start_thread(self.elref_finished, self.elref_error)

    def elref_finished(
        self, data: Tuple[List[float], List[float], Union[SurfacePoint, CustomPoint]]
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
        pass


class LimeTBXWidget(QtWidgets.QWidget):
    """
    Main widget of the lime toolbox desktop app.
    """

    def __init__(self, kernels_path: str):
        super().__init__()
        self.setLocale("English")
        self.kernels_path = kernels_path
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        settings_manager = settings.MockSettingsManager()
        self.page = MainSimulationsWidget(self.kernels_path, settings_manager)
        self.main_layout.addWidget(self.page)

    def propagate_close_event(self):
        pass


class LimeTBXWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._create_menu_bar()
    
    def _create_actions(self):
        self.eli_from_file_action = QtWidgets.QAction(self)
        self.eli_from_file_action.setText("Calculate EL&I from file")

    def _create_menu_bar(self):
        self._create_actions()
        self.menu_bar = self.menuBar()
        file_menu = QtWidgets.QMenu("&File", self)
        file_menu.addAction(self.eli_from_file_action)
        coeffs_menu = QtWidgets.QMenu("&Coefficients", self)
        self.menu_bar.addMenu(file_menu)
        self.menu_bar.addMenu(coeffs_menu)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        lime_tbx_w: LimeTBXWidget = self.centralWidget()
        lime_tbx_w.propagate_close_event()
        return super().closeEvent(event)
