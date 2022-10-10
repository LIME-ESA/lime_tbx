"""describe class"""

"""___Built-In Modules___"""
from enum import Enum
from typing import List, Callable, Union, Tuple
from datetime import datetime, timezone
import os

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import settings, output, input, srf, help
from lime_tbx.filedata import moon, srf as srf_loader
from ..simulation.comparison import comparison
from ..datatypes.datatypes import (
    ComparisonData,
    KernelsPath,
    LGLODComparisonData,
    LGLODData,
    LunarObservation,
    LunarObservationWrite,
    Point,
    PolarizationCoefficients,
    SatellitePoint,
    SatellitePosition,
    SelenographicDataWrite,
    SpectralResponseFunction,
    ApolloIrradianceCoefficients,
    SpectralValidity,
    SurfacePoint,
    CustomPoint,
    ReflectanceCoefficients,
    SpectralData,
)
from ..eocfi_adapter import eocfi_adapter
from lime_tbx.simulation.lime_simulation import ILimeSimulation, LimeSimulation
from .ifaces import IMainSimulationsWidget, noconflict_makecls
from lime_tbx.filedata.lglod_factory import create_lglod_data
from lime_tbx.gui import coefficients, constants
from ..datatypes import logger

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "02/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


_INTERNAL_ERROR_MSG = (
    "Something went wrong while performing the operation. See log for more detail."
)


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
            logger.get_logger().exception(e)
            self.exception.emit(e)


class LimeException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def eli_callback(
    srf: SpectralResponseFunction,
    point: Point,
    coeffs: ApolloIrradianceCoefficients,
    cimel_coef: ReflectanceCoefficients,
    lime_simulation: ILimeSimulation,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    Point,
    List[float],
    SpectralResponseFunction,
    Union[SpectralData, List[SpectralData]],
]:
    """
    Callback that performs the Irradiance operations.

    Parameters
    ----------
    srf: SpectralResponseFunction
        SRF that will be used to calculate the data
    point: Point
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
    point: Point
        Point that was used in the calculations.
    ch_irrs: list of float
        Integrated irradiance signals for each srf channel
    srf: SpectralResponseFunction
        SRF used for the integrated irradiance signal calculation.
    uncertainty_data: SpectralData or list of SpectralData
        Calculated uncertainty data.
    """
    lime_simulation.update_irradiance(srf, point, cimel_coef)
    return (
        point,
        srf,
        lime_simulation.get_elis(),
        lime_simulation.get_elis_cimel(),
        lime_simulation.get_elis_asd(),
        lime_simulation.get_signals(),
    )


def elref_callback(
    srf: SpectralResponseFunction,
    point: Point,
    coeffs: ApolloIrradianceCoefficients,
    cimel_coef: ReflectanceCoefficients,
    lime_simulation: ILimeSimulation,
) -> Tuple[List[float], List[float], Point, Union[SpectralData, List[SpectralData]],]:
    """Callback that performs the Reflectance operations.

    Parameters
    ----------
    srf: SpectralResponseFunction
        SRF that will be used to calculate the graph
    point: Point
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
    point: Point
        Point that was used in the calculations.
    uncertainty_data: SpectralData or list of SpectralData
        Calculated uncertainty data.
    """
    lime_simulation.update_reflectance(srf, point, cimel_coef)
    return (
        point,
        lime_simulation.get_elrefs(),
        lime_simulation.get_elrefs_cimel(),
        lime_simulation.get_elrefs_asd(),
    )


def polar_callback(
    srf: SpectralResponseFunction,
    point: Point,
    coeffs: PolarizationCoefficients,
    lime_simulation: ILimeSimulation,
) -> Tuple[List[float], List[float], Point]:

    lime_simulation.update_polarization(srf, point, coeffs)
    return point, lime_simulation.get_polars()


def compare_callback(
    mos: List[LunarObservation],
    srf: SpectralResponseFunction,
    coeffs: ApolloIrradianceCoefficients,
    cimel_coef: ReflectanceCoefficients,
    lime_simulation: ILimeSimulation,
    kernels_path: KernelsPath,
) -> Tuple[List[ComparisonData], List[LunarObservation], SpectralResponseFunction,]:
    co = comparison.Comparison(kernels_path)
    for mo in mos:
        if not mo.check_valid_srf(srf):
            srf_names = srf.get_channels_names()
            if len(mo.ch_names) == len(srf_names):
                for i in range(len(mo.ch_names)):
                    if mo.ch_names[i] in mo.ch_irrs:
                        mo.ch_irrs[srf_names[i]] = mo.ch_irrs.pop(mo.ch_names[i])
                    mo.ch_names[i] = srf_names[i]
            else:
                raise LimeException(
                    "SRF file not valid for the chosen Moon observations file."
                )
    comparisons = co.get_simulations(mos, srf, cimel_coef, lime_simulation)
    return comparisons, mos, srf


def calculate_all_callback(
    srf: SpectralResponseFunction,
    point: Point,
    coeffs: ApolloIrradianceCoefficients,
    cimel_coef: ReflectanceCoefficients,
    p_coeffs: PolarizationCoefficients,
    lime_simulation: ILimeSimulation,
):
    lime_simulation.update_reflectance(srf, point, cimel_coef)
    lime_simulation.update_irradiance(srf, point, cimel_coef)
    lime_simulation.update_polarization(srf, point, p_coeffs)
    return (point, srf)


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
        lime_simulation: ILimeSimulation,
        settings_manager: settings.ISettingsManager,
        kernels_path: KernelsPath,
    ):
        super().__init__()
        self.lime_simulation = lime_simulation
        self.settings_manager = settings_manager
        self.kernels_path = kernels_path
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.input = input.ComparisonInput(self._callback_compare_input_changed)
        self.compare_button = QtWidgets.QPushButton("Compare")
        self.compare_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.compare_button.clicked.connect(self.compare)
        self.compare_button.setDisabled(True)
        self.stack_layout = QtWidgets.QStackedLayout()
        self.stack_layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)
        self.output = output.ComparisonOutput(self.settings_manager)
        self.spinner = SpinnerPage()
        self.spinner.setVisible(False)
        self.stack_layout.addWidget(self.spinner)
        self.stack_layout.addWidget(self.output)
        self.stack_layout.setCurrentIndex(1)
        self.export_lglod_button = QtWidgets.QPushButton("Export to LGLOD file")
        self.export_lglod_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.export_lglod_button.clicked.connect(self.export_to_lglod)
        self.export_lglod_button.setDisabled(True)
        self.main_layout.addWidget(self.input)
        self.main_layout.addWidget(self.compare_button)
        self.main_layout.addLayout(self.stack_layout)
        self.main_layout.addWidget(self.export_lglod_button)

    def _callback_compare_input_changed(self):
        self.lime_simulation.set_simulation_changed()
        obss = self.input.get_moon_obs()
        srf = self.input.get_srf()
        if len(obss) == 0 or srf == None:
            self.compare_button.setDisabled(True)
        else:
            self.compare_button.setDisabled(False)

    def _start_thread(self, finished: Callable, error: Callable):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error)

    def _set_spinner(self, enabled: bool):
        self.spinner.setVisible(enabled)
        if enabled:
            self.spinner.movie_start()
            self.stack_layout.setCurrentIndex(0)
        else:
            self.spinner.movie_stop()
            self.stack_layout.setCurrentIndex(1)

    def _unblock_gui(self):
        self._set_spinner(False)
        self.parentWidget().setDisabled(False)

    def _block_gui_loading(self):
        self._set_spinner(True)
        self.parentWidget().setDisabled(True)

    def can_save_simulation(self) -> bool:
        return self.export_lglod_button.isEnabled()

    @QtCore.Slot()
    def export_to_lglod(self) -> None:
        lglod = LGLODComparisonData(
            self.comps,
            self.srf.get_channels_names(),
            "TODO",
        )
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export LGLOD", "{}.nc".format("lglod")
        )[0]
        vers = self.settings_manager.get_cimel_coef().version
        if name is not None and name != "":
            try:
                moon.write_comparison(
                    lglod,
                    name,
                    datetime.now().astimezone(timezone.utc),
                    vers,
                    self.kernels_path,
                )
            except Exception as e:
                self.show_error(e)

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    @QtCore.Slot()
    def compare(self):
        self._block_gui_loading()
        mos = self.input.get_moon_obs()
        srf = self.input.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.worker = CallbackWorker(
            compare_callback,
            [mos, srf, coeffs, cimel_coef, self.lime_simulation, self.kernels_path],
        )
        self._start_thread(self.compare_finished, self.compare_error)

    def compare_finished(
        self,
        data: Tuple[
            List[ComparisonData],
            List[LunarObservation],
            SpectralResponseFunction,
        ],
    ):
        comps = data[0]
        mos = data[1]
        srf = data[2]
        self.comps = comps
        self.srf = srf
        ch_names = srf.get_channels_names()
        self.output.set_channels(ch_names)
        to_remove = []
        for i, ch in enumerate(ch_names):
            obs_irrs = []
            for mo in mos:
                if mo.has_ch_value(ch):
                    obs_irrs.append(mo.ch_irrs[ch])
            if len(comps[i].dts) > 0:
                self.output.update_plot(i, comps[i])
                self.output.update_labels(
                    i,
                    "{} ({} nm)".format(ch, srf.get_channel_from_name(ch).center),
                    "datetimes",
                    "Signal (Wm⁻²nm⁻¹)",
                )
                self.output.update_legends(
                    i,
                    [
                        ["Observed Signal", "Simulated Signal"],
                        [],
                        [],
                        ["Relative Differences"],
                    ],
                )
            else:
                to_remove.append(ch)
        for chsrf in srf.channels:
            if chsrf.valid_spectre == SpectralValidity.PARTLY_OUT:
                self.output.set_as_partly(chsrf.id)
        self.output.remove_channels(to_remove)
        self._unblock_gui()
        self.export_lglod_button.setEnabled(True)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(False)

    def load_lglod_comparisons(
        self, lglod: LGLODComparisonData, srf: SpectralResponseFunction
    ):
        self.input.clear_input()
        self.compare_button.setDisabled(True)
        comps = lglod.comparisons
        srf = srf
        self.comps = comps
        self.srf = srf
        ch_names = lglod.ch_names
        self.output.set_channels(lglod.ch_names)
        for i, ch in enumerate(ch_names):
            self.output.update_plot(i, comps[i])
            self.output.update_labels(
                i,
                "{} ({} nm)".format(ch, srf.get_channel_from_name(ch).center),
                "datetimes",
                "Signal (Wm⁻²nm⁻¹)",
            )
            self.output.update_legends(
                i,
                [
                    ["Observed Signal", "Simulated Signal"],
                    [],
                    [],
                    ["Relative Differences"],
                ],
            )
        for chsrf in srf.channels:
            if chsrf.valid_spectre == SpectralValidity.PARTLY_OUT:
                self.output.set_as_partly(chsrf.id)
        self._unblock_gui()
        self.export_lglod_button.setEnabled(True)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(False)

    def handle_operation_error(self, error: Exception):
        logger.get_logger().critical(error)
        if isinstance(error, LimeException):
            self.show_error(error)
        else:
            self.show_error(_INTERNAL_ERROR_MSG)

    def compare_error(self, error: Exception):
        self._unblock_gui()
        self.handle_operation_error(error)
        self.export_lglod_button.setEnabled(False)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(True)


class MainSimulationsWidget(
    QtWidgets.QWidget, IMainSimulationsWidget, metaclass=noconflict_makecls()
):
    """
    Widget containing the landing gui, which lets the user calculate the eli, elref and polar
    of a surface point, custom input or satellite input at one moment.
    """

    def __init__(
        self,
        lime_simulation: ILimeSimulation,
        kernels_path: KernelsPath,
        eocfi_path: str,
        settings_manager: settings.ISettingsManager,
    ):
        super().__init__()
        self.kernels_path = kernels_path
        self._finished_building = False
        self.lime_simulation = lime_simulation
        self.settings_manager = settings_manager
        self.eocfi: eocfi_adapter.IEOCFIConverter = eocfi_adapter.EOCFIConverter(
            eocfi_path
        )
        self.satellites = self.eocfi.get_sat_list()
        self._build_layout()
        self._finished_building = True

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # input
        self.input_widget = input.InputWidget(
            self.satellites,
            self._callback_regular_input_changed,
            self._callback_check_calculable,
        )
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
            self.settings_manager,
            "Simulation output",
            "Wavelengths (nm)",
            "Units",
            parent=self,
        )
        self.graph.update_legend(
            [["interpolated data points"], ["CIMEL data points"], ["errorbars (k=2)"]]
        )
        # srf widget
        self.srf_widget = srf.SRFEditWidget(
            self.settings_manager, self._callback_regular_input_changed
        )
        # signal widget
        self.signal_widget = output.SignalWidget(self.settings_manager)
        # finish tab
        self.lower_tabs.addTab(self.graph, "Result")
        self.lower_tabs.addTab(self.srf_widget, "SRF")
        self.lower_tabs.addTab(self.signal_widget, "Signal")
        self.lower_tabs.currentChanged.connect(self.lower_tabs_changed)
        # Export to LGLOD
        self.export_lglod_button = QtWidgets.QPushButton("Export to LGLOD file")
        self.export_lglod_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.export_lglod_button.clicked.connect(self.export_glod)
        self.export_lglod_button.setDisabled(True)
        self._export_lglod_button_was_disabled = True
        # finish main layout
        self.main_layout.addWidget(self.input_widget)
        self.main_layout.addLayout(self.buttons_layout)
        self.lower_stack = QtWidgets.QStackedLayout()
        self.lower_stack.setStackingMode(QtWidgets.QStackedLayout.StackAll)
        self.loading_spinner = SpinnerPage()
        self.loading_spinner.setVisible(False)
        self.lower_stack.addWidget(self.loading_spinner)
        self.lower_stack.addWidget(self.lower_tabs)
        self.lower_stack.setCurrentIndex(1)
        self.main_layout.addLayout(self.lower_stack, 1)
        self.main_layout.addWidget(self.export_lglod_button)

    def _set_spinner(self, enabled: bool):
        self.loading_spinner.setVisible(enabled)
        if enabled:
            self.loading_spinner.movie_start()
            self.lower_stack.setCurrentIndex(0)
        else:
            self.loading_spinner.movie_stop()
            self.lower_stack.setCurrentIndex(1)

    def _unblock_gui(self):
        self._set_spinner(False)
        self.parentWidget().setDisabled(False)
        self._disable_lglod_export(False)
        self._export_lglod_button_was_disabled = False

    def _block_gui_loading(self):
        self._set_spinner(True)
        self.parentWidget().setDisabled(True)
        self._disable_lglod_export(True)
        self._export_lglod_button_was_disabled = True

    def _disable_lglod_export(self, disable: bool):
        self.export_lglod_button.setDisabled(False)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(disable)

    def _callback_regular_input_changed(self):
        self.lime_simulation.set_simulation_changed()

    def _callback_check_calculable(self):
        calculable = self.input_widget.is_calculable()
        self.eli_button.setEnabled(calculable)
        self.elref_button.setEnabled(calculable)
        self.polar_button.setEnabled(calculable)
        if not self._export_lglod_button_was_disabled:
            self._disable_lglod_export(not calculable)

    def _start_thread(self, finished: Callable, error: Callable):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error)

    def set_export_button_disabled(self, disabled: bool):
        if not self._finished_building:
            return
        if disabled:
            self._block_gui_loading()
        else:
            self._unblock_gui()

    @QtCore.Slot()
    def lower_tabs_changed(self, i: int):
        if i == 0:
            self.graph.update_size()
        elif i == 1:
            self.srf_widget.update_size()

    def can_save_simulation(self) -> bool:
        return self.export_lglod_button.isEnabled()

    def handle_operation_error(self, error: Exception):
        logger.get_logger().critical(error)
        if isinstance(error, LimeException):
            self.show_error(error)
        else:
            self.show_error(_INTERNAL_ERROR_MSG)

    @QtCore.Slot()
    def show_eli(self):
        """
        Calculate and show extraterrestrial lunar irradiances for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.worker = CallbackWorker(
            eli_callback,
            [srf, point, coeffs, cimel_coef, self.lime_simulation],
        )
        self._start_thread(self.eli_finished, self.eli_error)

    def eli_finished(
        self,
        data: Tuple[
            Point,
            SpectralResponseFunction,
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            SpectralData,
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[2], data[3], data[4], data[0])
        self.graph.update_labels(
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²nm⁻¹)",
        )
        self.signal_widget.update_signals(data[0], data[1], data[5])
        self.lower_tabs.setCurrentIndex(0)
        self.lower_tabs.setTabEnabled(2, True)

    def eli_error(self, error: Exception):
        self.handle_operation_error(error)
        self._unblock_gui()

    @QtCore.Slot()
    def show_elref(self):
        """
        Calculate and show extraterrestrial lunar reflectances for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.worker = CallbackWorker(
            elref_callback, [srf, point, coeffs, cimel_coef, self.lime_simulation]
        )
        self._start_thread(self.elref_finished, self.elref_error)

    def clear_signals(self):
        self.signal_widget.clear_signals()
        self.lower_tabs.setTabEnabled(2, False)

    def elref_finished(
        self,
        data: Tuple[
            Point,
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[1], data[2], data[3], data[0])
        self.graph.update_labels(
            "Extraterrestrial Lunar Reflectances",
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
        )
        self.clear_signals()
        self.lower_tabs.setCurrentIndex(0)

    def elref_error(self, error: Exception):
        self.handle_operation_error(error)
        self._unblock_gui()

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
            polar_callback, [srf, point, coeffs, self.lime_simulation]
        )
        self._start_thread(self.polar_finished, self.polar_error)

    def polar_finished(
        self,
        data: Tuple[
            Point,
            Union[SpectralData, List[SpectralData]],
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[1], point=data[0])
        self.graph.update_labels(
            "Lunar polarization",
            "Wavelengths (nm)",
            "Polarizations (Fraction of unity)",
        )
        self.clear_signals()
        self.lower_tabs.setCurrentIndex(0)

    def polar_error(self, error: Exception):
        self.handle_operation_error(error)
        self._unblock_gui()

    def load_observations_finished(
        self,
        point: Point,
    ):
        self.input_widget.set_point(point)
        self.input_widget.set_last_point_to_point()
        self.show_eli()

    @QtCore.Slot()
    def export_glod(self):
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        cimel_coef = self.settings_manager.get_cimel_coef()
        p_coeffs = self.settings_manager.get_polar_coeffs()
        self.worker = CallbackWorker(
            calculate_all_callback,
            [srf, point, coeffs, cimel_coef, p_coeffs, self.lime_simulation],
        )
        self._start_thread(self.calculate_all_finished, self.calculate_all_error)

    def calculate_all_finished(self, data):
        self._unblock_gui()
        point: Point = data[0]
        srf: SpectralResponseFunction = data[1]
        lglod = create_lglod_data(point, srf, self.lime_simulation, self.kernels_path)
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export LGLOD", "{}.nc".format("lglod")
        )[0]
        version = self.settings_manager.get_cimel_coef().version
        if name is not None and name != "":
            try:
                moon.write_obs(
                    lglod, name, datetime.now().astimezone(timezone.utc), version
                )
            except Exception as e:
                self.show_error(e)

    def calculate_all_error(self, error: Exception):
        self.handle_operation_error(error)
        self._unblock_gui()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))


class LimePagesEnum(Enum):
    SIMULATION = 0
    COMPARISON = 1


class SpinnerPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Loading spinner
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _movie_path = os.path.join(_current_dir, constants.SPINNER_PATH)
        self.movie = QtGui.QMovie(_movie_path)
        self.movie.setScaledSize(QtCore.QSize(50, 50))
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.label_spinner = QtWidgets.QLabel()
        self.label_spinner.setMovie(self.movie)
        self.h_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.h_layout)
        self.h_layout.addWidget(self.label_spinner, 1)
        self.h_layout.setAlignment(self.label_spinner, QtGui.Qt.AlignHCenter)

    def movie_start(self):
        self.movie.start()

    def movie_stop(self):
        self.movie.stop()


def load_simulation_callback(path: str, kernels_path: KernelsPath):
    lglod = moon.read_glod_file(path, kernels_path)
    return [lglod]


def check_srf_observation_callback(lglod: LGLODData, srf: SpectralResponseFunction):
    valid = True
    for obs in lglod.observations:
        if not obs.check_valid_srf(srf):
            valid = False
    if not valid:
        error_msg = "SRF file not valid for the observation file."
        error = Exception(error_msg)
        raise error
    return [lglod, srf]


def check_srf_comparison_callback(
    lglod: LGLODComparisonData, srf: SpectralResponseFunction
):
    valid = True
    srf_chans = srf.get_channels_names()
    for ch_name in lglod.ch_names:
        if ch_name not in srf_chans:
            valid = False
    if not valid:
        error_msg = "SRF file not valid for the observation file."
        error = Exception(error_msg)
        raise error
    return [lglod, srf]


def return_args_callback(*args):
    return args


class LimeTBXWidget(QtWidgets.QWidget):
    """
    Main widget of the lime toolbox desktop app.
    """

    def __init__(self, kernels_path: KernelsPath, eocfi_path: str):
        super().__init__()
        self.setLocale("English")
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.lime_simulation = LimeSimulation(eocfi_path, kernels_path)
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.settings_manager = settings.MockSettingsManager()
        self.comparison_page = ComparisonPageWidget(
            self.lime_simulation, self.settings_manager, self.kernels_path
        )
        self.comparison_page.hide()
        self.main_page = MainSimulationsWidget(
            self.lime_simulation,
            self.kernels_path,
            self.eocfi_path,
            self.settings_manager,
        )
        self.page = self.main_page
        self.main_layout.addWidget(self.main_page)

    def setDisabled(self, arg__1: bool) -> None:
        self.parentWidget().setDisabled(arg__1)
        return super().setDisabled(arg__1)

    def propagate_close_event(self):
        pass

    def change_page(self, pageEnum: LimePagesEnum):
        self.page.hide()
        self.main_layout.removeWidget(self.page)
        if pageEnum == LimePagesEnum.COMPARISON:
            self.page = self.comparison_page
        else:
            self.page = self.main_page
        self.main_layout.addWidget(self.page)
        self.page.show()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    def _start_thread(self, finished: Callable, error: Callable):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error)

    def load_observations_finished(
        self, lglod: LGLODData, srf: SpectralResponseFunction
    ):
        if srf == None:
            srf = self.settings_manager.get_default_srf()
        self.worker = CallbackWorker(check_srf_observation_callback, [lglod, srf])
        self._start_thread(
            self._load_observations_finished_2, self._load_observations_finished_error
        )

    def _load_observations_finished_2(self, data):
        lglod = data[0]
        srf = data[1]
        self.main_page.srf_widget.set_srf(srf)
        self.lime_simulation.set_observations(lglod, srf)
        point = self.lime_simulation.get_point()
        self.main_page.load_observations_finished(point)
        self.main_page._unblock_gui()

    def _load_observations_finished_error(self, error: Exception):
        logger.get_logger().critical(error)
        self.show_error(error)
        self.main_page._unblock_gui()

    def load_comparisons_finished(
        self, lglod: LGLODComparisonData, srf: SpectralResponseFunction
    ):
        if srf == None:
            srf = self.settings_manager.get_default_srf()
        self.worker = CallbackWorker(check_srf_comparison_callback, [lglod, srf])
        self._start_thread(
            self._load_comparisons_finished_2, self._load_comparisons_finished_error
        )

    def _load_comparisons_finished_2(self, data):
        lglod = data[0]
        srf = data[1]
        self.comparison_page.load_lglod_comparisons(lglod, srf)
        self.comparison_page._unblock_gui()

    def _load_comparisons_finished_error(self, error: Exception):
        logger.get_logger().critical(error)
        self.show_error(error)
        self.comparison_page._unblock_gui()

    def can_save_simulation(self) -> bool:
        return self.page.can_save_simulation()

    def get_current_page(self) -> Union[MainSimulationsWidget, ComparisonPageWidget]:
        return self.page


class LimeTBXWindow(QtWidgets.QMainWindow):
    def __init__(self, kernels_path: KernelsPath):
        super().__init__()
        self.kernels_path = kernels_path
        self._is_comparing = False
        self._create_menu_bar()

    def _create_actions(self):
        # File actions
        self.save_simulation_action = QtWidgets.QAction(self)
        self.save_simulation_action.setText(
            "&Save simulation to LIME GLOD format file."
        )
        self.save_simulation_action.triggered.connect(self.save_simulation)
        self.save_simulation_action.setDisabled(True)
        self.load_simulation_action = QtWidgets.QAction(self)
        self.load_simulation_action.setText(
            "&Load simulation file stored in a LIME GLOD format file."
        )
        self.load_simulation_action.triggered.connect(self.load_simulation)
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
        self.download_coefficients_action.setDisabled(True)
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
        file_menu.addAction(self.save_simulation_action)
        file_menu.addAction(self.load_simulation_action)
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
        lime_tbx_w = self._get_lime_widget()
        lime_tbx_w.propagate_close_event()
        return super().closeEvent(event)

    def set_save_simulation_action_disabled(self, disable: bool) -> None:
        self.save_simulation_action.setDisabled(disable)

    def _start_thread(self, finished: Callable, error: Callable):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error)

    def _get_lime_widget(self) -> LimeTBXWidget:
        return self.centralWidget()

    # ERROR

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    # ACTIONS

    def save_simulation(self):
        lime_tbx_w = self._get_lime_widget()
        if self._is_comparing:
            lime_tbx_w.comparison_page.export_to_lglod()
        else:
            lime_tbx_w.main_page.export_glod()

    def load_simulation_finished(self, data):
        lglod = data[0]
        if isinstance(lglod, LGLODData):
            self.simulations()
            srf = None
            cancel = False
            if lglod.not_default_srf:
                srf_path = QtWidgets.QFileDialog().getOpenFileName(
                    self, "Select SpectralResponseFunction file"
                )[0]
                if srf_path == "":
                    cancel = True
                else:
                    try:
                        srf = srf_loader.read_srf(srf_path)
                    except Exception as e:
                        cancel = True
                        self.show_error(e)
            if not cancel:
                self.worker = CallbackWorker(return_args_callback, [lglod, srf])
                self._start_thread(
                    self._load_observations_finished, self.load_simulation_error
                )
        else:
            self.comparison()
            srf = None
            srf_path = QtWidgets.QFileDialog().getOpenFileName(
                self, "Select SpectralResponseFunction file"
            )[0]
            if srf_path != "":
                try:
                    srf = srf_loader.read_srf(srf_path)
                except Exception as e:
                    self.show_error(e)
                else:
                    self.worker = CallbackWorker(return_args_callback, [lglod, srf])
                    self._start_thread(
                        self._load_comparisons_finished, self.load_simulation_error
                    )

    def _load_observations_finished(self, data):
        lglod = data[0]
        srf = data[1]
        lime_tbx_w = self._get_lime_widget()
        lime_tbx_w.load_observations_finished(lglod, srf)

    def _load_comparisons_finished(self, data):
        lglod = data[0]
        srf = data[1]
        lime_tbx_w = self._get_lime_widget()
        lime_tbx_w.load_comparisons_finished(lglod, srf)

    def load_simulation_error(self, e: Exception):
        self.show_error(e)

    def load_simulation(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self, "Select GLOD file")[0]
        lime_tbx_w = self._get_lime_widget()
        page = lime_tbx_w.get_current_page()
        if path != "":
            page._block_gui_loading()
            self.worker = CallbackWorker(
                load_simulation_callback, [path, self.kernels_path]
            )
            self._start_thread(
                self.load_simulation_finished, self.load_simulation_error
            )

    def comparison(self):
        if not self._is_comparing:
            self.save_simulation_action.setText(
                "&Save comparison to LIME GLOD format file."
            )
            self.comparison_action.setText("Perform &simulations")
            self.comparison_action.triggered.connect(self.simulations)
            lime_tbx_w = self._get_lime_widget()
            lime_tbx_w.lime_simulation.set_simulation_changed()
            lime_tbx_w.change_page(LimePagesEnum.COMPARISON)
            self._is_comparing = True
            self.set_save_simulation_action_disabled(
                not lime_tbx_w.can_save_simulation()
            )

    def simulations(self):
        if self._is_comparing:
            self.save_simulation_action.setText(
                "&Save simulation to LIME GLOD format file."
            )
            self.comparison_action.setText(
                "Perform &comparisons from a remote sensing instrument"
            )
            self.comparison_action.triggered.connect(self.comparison)
            lime_tbx_w = self._get_lime_widget()
            lime_tbx_w.lime_simulation.set_simulation_changed()
            lime_tbx_w.change_page(LimePagesEnum.SIMULATION)
            self._is_comparing = False
            self.set_save_simulation_action_disabled(
                not lime_tbx_w.can_save_simulation()
            )

    def exit(self):
        self.close()

    def download_coefficients(self):
        pass

    def select_coefficients(self):
        lime_tbx_w = self._get_lime_widget()
        select_coefficients_dialog = coefficients.SelectCoefficientsDialog(
            lime_tbx_w.settings_manager, self
        )
        select_coefficients_dialog.exec_()

    def about(self):
        about_dialog = help.AboutDialog(self)
        about_dialog.exec_()
