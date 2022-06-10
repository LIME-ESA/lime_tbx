"""describe class"""

"""___Built-In Modules___"""
from enum import Enum
from typing import List, Callable, Union, Tuple
from datetime import datetime
import time

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
import numpy as np

"""___NPL Modules___"""
from . import settings, output, input, srf, help
# from ..simulation.regular_simulation import regular_simulation
# from ..simulation.common.common import CommonSimulation
# from ..simulation.esa_satellites import esa_satellites
from ..simulation.comparison import comparison
from ..datatypes.datatypes import (
    LunarObservation,
    PolarizationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    IrradianceCoefficients,
    SurfacePoint,
    CustomPoint,
    CimelCoef,
    SpectralData,
)
from ..eocfi_adapter import eocfi_adapter
from lime_tbx.simulation.lime_simulation import LimeSimulation

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
    def_srf: SpectralResponseFunction,
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: IrradianceCoefficients,
    cimel_coef: CimelCoef,
    lime_simulation: LimeSimulation,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    Union[SurfacePoint, CustomPoint, SatellitePoint],
    List[float],
    SpectralResponseFunction,
    Union[SpectralData, List[SpectralData]]
]:
    """
    Callback that performs the Irradiance operations.

    Parameters
    ----------
    def_srf: SpectralResponseFunction
        SRF that will be used to calculate the first graph
    srf: SpectralResponseFunction
        SRF that will be used to calculate the integrated irradiance
    point: Union[SurfacePoint, CustomPoint, SatellitePoint]
        Point used
    coeffs: IrradianceCoefficients
        Coefficients used by the algorithms in order to calculate the irradiance or reflectance.
    cimel_coef: CimelCoef
        CimelCoef with the CIMEL coefficients and uncertainties.
    kernels_path: str
        Path where the directory with the SPICE kernels is located.
    eocfi_path: str
        Path where the directory with the needed EOCFI data files is located.

    Returns
    -------
    wlens: list of float
        Wavelengths of def_srf
    elis: list of float
        Irradiances related to def_srf
    point: Union[SurfacePoint, CustomPoint, SatellitePoint]
        Point that was used in the calculations.
    ch_irrs: list of float
        Integrated irradiance signals for each srf channel
    srf: SpectralResponseFunction
        SRF used for the integrated irradiance signal calculation.
    uncertainty_data: SpectralData or list of SpectralData
        Calculated uncertainty data.
    """
    lime_simulation.update_model_irr(srf,point,cimel_coef)
    return point,srf,lime_simulation.elis,lime_simulation.elis_cimel, lime_simulation.elis_asd

def elref_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: IrradianceCoefficients,
    cimel_coef: CimelCoef,
    lime_simulation: LimeSimulation,
) -> Tuple[
    List[float],
    List[float],
    Union[SurfacePoint, CustomPoint, SatellitePoint],
    Union[SpectralData, List[SpectralData]],
]:
    """Callback that performs the Reflectance operations.

    Parameters
    ----------
    srf: SpectralResponseFunction
        SRF that will be used to calculate the graph
    point: Union[SurfacePoint, CustomPoint, SatellitePoint]
        Point used
    coeffs: IrradianceCoefficients
        Coefficients used by the algorithms in order to calculate the irradiance or reflectance.
    cimel_coef: CimelCoef
        CimelCoef with the CIMEL coefficients and uncertainties.
    kernels_path: str
        Path where the directory with the SPICE kernels is located.
    eocfi_path: str
        Path where the directory with the needed EOCFI data files is located.

    Returns
    -------
    wlens: list of float
        Wavelengths of def_srf
    elrefs: list of float
        Reflectances related to srf
    point: Union[SurfacePoint, CustomPoint, SatellitePoint]
        Point that was used in the calculations.
    uncertainty_data: SpectralData or list of SpectralData
        Calculated uncertainty data.
    """
    lime_simulation.update_model_refl(srf,point,cimel_coef)
    print(lime_simulation.elref)
    print(lime_simulation.elref_cimel)
    print(lime_simulation.elis_asd)
    return point, lime_simulation.elref, lime_simulation.elref_cimel, lime_simulation.elis_asd


def polar_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: PolarizationCoefficients,
    lime_simulation: LimeSimulation,
) -> Tuple[List[float], List[float], Union[SurfacePoint, CustomPoint, SatellitePoint]]:

    lime_simulation.update_model_pol(srf,point,coeffs)
    return point,lime_simulation.polars


def compare_callback(
    mos: List[LunarObservation],
    srf: SpectralResponseFunction,
    coeffs: IrradianceCoefficients,
    kernels_path: str,
):
    co = comparison.Comparison()
    for mo in mos:
        if not mo.check_valid_srf(srf):
            raise ("SRF file not valid for the chosen Moon observations file.")
    irrs, dts, sps = co.get_simulations(mos, srf, coeffs, kernels_path)
    return irrs, dts, sps, mos, srf


def _start_thread(
    worker: CallbackWorker,
    worker_th: QtCore.QThread,
    finished: Callable,
    error: Callable,
):
    worker.moveToThread(worker_th)
    worker_th.started.connect(worker.run)
    worker.finished.connect(worker_th.quit)
    worker.finished.connect(worker.deleteLater)
    worker.finished.connect(finished)
    worker.exception.connect(worker_th.quit)
    worker.exception.connect(worker.deleteLater)
    worker.exception.connect(error)
    worker_th.finished.connect(worker_th.deleteLater)
    worker_th.start()


class ComparisonPageWidget(QtWidgets.QWidget):
    def __init__(
        self,
        lime_simulation: LimeSimulation,
        settings_manager: settings.ISettingsManager,
    ):
        super().__init__()
        self.lime_simulation = lime_simulation
        self.settings_manager = settings_manager
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.input = input.ComparisonInput(self._callback_compare_input_changed)
        self.compare_button = QtWidgets.QPushButton("Compare")
        self.compare_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.compare_button.clicked.connect(self.compare)
        self.compare_button.setDisabled(True)
        self.output = output.ComparisonOutput()
        self.main_layout.addWidget(self.input)
        self.main_layout.addWidget(self.compare_button)
        self.main_layout.addWidget(self.output)

    def _callback_compare_input_changed(self):
        obss = self.input.get_moon_obs()
        srf = self.input.get_srf()
        if len(obss) == 0 or srf == None:
            self.compare_button.setDisabled(True)
        else:
            self.compare_button.setDisabled(False)

    def _start_thread(self, finished: Callable, error: Callable):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error)

    def _unblock_gui(self):
        self.parentWidget().setDisabled(False)

    def _block_gui_loading(self):
        self.parentWidget().setDisabled(True)

    @QtCore.Slot()
    def compare(self):
        self._block_gui_loading()
        mos = self.input.get_moon_obs()
        srf = self.input.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        self.worker = CallbackWorker(
            compare_callback,
            [mos, srf, coeffs, self.lime_simulation],
        )
        self._start_thread(self.compare_finished, self.compare_error)

    def compare_finished(
        self,
        data: Tuple[
            List[List[float]],
            List[List[datetime]],
            List[List[SurfacePoint]],
            List[LunarObservation],
            SpectralResponseFunction,
        ],
    ):
        irrs = data[0]
        dts = data[1]
        sps = data[2]
        mos = data[3]
        srf = data[4]
        ch_names = srf.get_channels_names()
        self.output.set_channels(ch_names)
        to_remove = []
        for i, ch in enumerate(ch_names):
            obs_irrs = []
            for mo in mos:
                if mo.has_ch_value(ch):
                    obs_irrs.append(mo.ch_irrs[ch])
            if len(dts[i]) > 0:
                self.output.update_plot(i, dts[i], [obs_irrs, irrs[i]], sps[i])
                self.output.update_labels(
                    i,
                    "{} ({} nm)".format(ch, srf.get_channel_from_name(ch).center),
                    "datetimes",
                    "Signal (Wm⁻²nm⁻¹)",
                )
                self.output.update_legends(i, ["Observed Signal", "Simulated Signal"])
            else:
                to_remove.append(ch)
        self.output.remove_channels(to_remove)
        self._unblock_gui()

    def compare_error(self, error: Exception):
        self._unblock_gui()
        error_dialog = QtWidgets.QErrorMessage(self)
        error_dialog.showMessage(str(error))
        raise error


class MainSimulationsWidget(QtWidgets.QWidget):
    """
    Widget containing the landing gui, which lets the user calculate the eli, elref and polar
    of a surface point, custom input or satellite input at one moment.
    """

    def __init__(
        self,
        lime_simulation: LimeSimulation,
        eocfi_path: str,
        settings_manager: settings.ISettingsManager,
    ):
        super().__init__()
        self.lime_simulation = lime_simulation
        self.settings_manager = settings_manager
        self.eocfi = eocfi_adapter.EOCFIConverter(eocfi_path)
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
        # signal widget
        self.signal_widget = output.SignalWidget()
        # finish tab
        self.lower_tabs.addTab(self.graph, "Result")
        self.lower_tabs.addTab(self.srf_widget, "SRF")
        self.lower_tabs.addTab(self.signal_widget, "Signal")
        self.lower_tabs.currentChanged.connect(self.lower_tabs_changed)
        # finish main layout
        self.main_layout.addWidget(self.input_widget)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addWidget(self.lower_tabs, 1)

    def _unblock_gui(self):
        self.parentWidget().setDisabled(False)

    def _block_gui_loading(self):
        self.parentWidget().setDisabled(True)

    def _start_thread(self, finished: Callable, error: Callable):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error)

    @QtCore.Slot()
    def lower_tabs_changed(self, i: int):
        if i == 0:
            self.graph.update_size()
        elif i == 1:
            self.srf_widget.update_size()

    @QtCore.Slot()
    def show_eli(self):
        """
        Calculate and show extraterrestrial lunar irradiances for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        def_srf = self.settings_manager.get_default_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.worker = CallbackWorker(
            eli_callback,
            [def_srf, srf, point, coeffs, cimel_coef,self.lime_simulation],
        )
        self._start_thread(self.eli_finished, self.eli_error)

    def eli_finished(
        self,
        data: Tuple[
            Union[SurfacePoint, CustomPoint, SatellitePoint],
            SpectralResponseFunction,
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData,List[SpectralData]],
            Union[SpectralData,List[SpectralData]],
        ],
    ):
        self._unblock_gui()
        # unc = data[5]
        # if isinstance(unc, list):
        #     wlen = [u.wlen_cimel for u in unc]
        #     cimel_data = [u.data for u in unc]
        #     uncert = [u.uncertainties for u in unc]
        # else:
        #     wlen = unc.wlen_cimel
        #     cimel_data = unc.data
        #     uncert = unc.uncertainties
        # self.graph.update_plot(data[0], data[1], data[2], wlen, cimel_data, uncert)
        self.graph.update_plot(data[2], data[3], data[4])
        self.graph.update_labels(
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²/nm)",
        )
        self.signal_widget.update_signals(data[0], data[1], data[2])

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
        def_srf = self.settings_manager.get_default_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.worker = CallbackWorker(
            elref_callback, [def_srf, point, coeffs, cimel_coef,self.lime_simulation]
        )
        self._start_thread(self.elref_finished, self.elref_error)

    def elref_finished(
        self,
        data: Tuple[
                Union[SurfacePoint,CustomPoint,SatellitePoint],
                Union[SpectralData,List[SpectralData]],
                Union[SpectralData,List[SpectralData]],
                Union[SpectralData,List[SpectralData]]],):
        self._unblock_gui()
        # unc = data[3]
        # if isinstance(unc, list):
        #     wlen = [u.wlen_cimel for u in unc]
        #     cimel_data = [u.data for u in unc]
        #     uncert = [u.uncertainties for u in unc]
        # else:
        #     wlen = unc.wlen_cimel
        #     cimel_data = unc.data
        #     uncert = unc.uncertainties

        self.graph.update_plot(data[1], data[2], data[3])
        self.graph.update_labels(
            "Extraterrestrial Lunar Reflectances",
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
        )
        self.signal_widget.clear_signals()

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
        def_srf = self.settings_manager.get_default_srf()
        coeffs = self.settings_manager.get_polar_coeffs()
        self.worker = CallbackWorker(
            polar_callback, [def_srf, point, coeffs,self.lime_simulation]
        )
        self._start_thread(self.polar_finished, self.polar_error)

    def polar_finished(
        self,data: Tuple[
                Union[SurfacePoint,CustomPoint,SatellitePoint],
                Union[SpectralData,List[SpectralData]]],):
        self._unblock_gui()
        self.graph.update_plot(data[1])
        self.graph.update_labels(
            "Lunar polarization",
            "Wavelengths (nm)",
            "Polarizations (Fraction of unity)",
        )
        self.signal_widget.clear_signals()

    def polar_error(self, error: Exception):
        self._unblock_gui()
        raise error


class LimePagesEnum(Enum):
    SIMULATION = 0
    COMPARISON = 1


class LimeTBXWidget(QtWidgets.QWidget):
    """
    Main widget of the lime toolbox desktop app.
    """

    def __init__(self, kernels_path: str, eocfi_path: str):
        super().__init__()
        self.setLocale("English")
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.lime_simulation = LimeSimulation(eocfi_path,kernels_path)
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.settings_manager = settings.MockSettingsManager()
        self.comparison_page = ComparisonPageWidget(
            self.lime_simulation, self.settings_manager
        )
        self.comparison_page.hide()
        self.main_page = MainSimulationsWidget(
            self.lime_simulation, self.eocfi_path, self.settings_manager
        )
        self.page = self.main_page
        self.main_layout.addWidget(self.page)

    def setDisabled(self, arg__1: bool) -> None:
        self.parentWidget().setDisabled(arg__1)
        return super().setDisabled(arg__1)

    def propagate_close_event(self):
        pass

    def change_page(self, pageEnum: LimePagesEnum):
        self.main_layout.removeWidget(self.page)
        self.page.hide()
        self.page.setParent(None)
        if pageEnum == LimePagesEnum.COMPARISON:
            self.page = self.comparison_page
        else:
            self.page = self.main_page
        self.main_layout.addWidget(self.page)
        self.page.show()


class LimeTBXWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._create_menu_bar()

    def _create_actions(self):
        # File actions
        self.comparison_action = QtWidgets.QAction(self)
        self.comparison_action.setText(
            "Perform &comparisons from a remote sensing instrument"
        )
        self.comparison_action.triggered.connect(self.comparison)
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
        file_menu.addAction(self.comparison_action)
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

    def comparison(self):
        self.comparison_action.setText("Perform &simulations")
        self.comparison_action.triggered.connect(self.simulations)
        lime_tbx_w: LimeTBXWidget = self.centralWidget()
        lime_tbx_w.change_page(LimePagesEnum.COMPARISON)

    def simulations(self):
        self.comparison_action.setText(
            "Perform &comparisons from a remote sensing instrument"
        )
        self.comparison_action.triggered.connect(self.comparison)
        lime_tbx_w: LimeTBXWidget = self.centralWidget()
        lime_tbx_w.change_page(LimePagesEnum.SIMULATION)

    def exit(self):
        self.close()

    def download_coefficients(self):
        pass

    def select_coefficients(self):
        pass

    def about(self):
        about_dialog = help.AboutDialog(self)
        about_dialog.exec_()
