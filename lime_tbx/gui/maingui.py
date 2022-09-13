"""describe class"""

"""___Built-In Modules___"""
from enum import Enum
from typing import List, Callable, Union, Tuple
from datetime import datetime

from lime_tbx.datatypes import constants

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
import numpy as np

"""___NPL Modules___"""
from . import settings, output, input, srf, help
from lime_tbx.filedata import moon, srf as srf_loader
from ..simulation.comparison import comparison
from ..datatypes.datatypes import (
    ComparisonData,
    KernelsPath,
    LGLODData,
    LunarObservation,
    LunarObservationWrite,
    Point,
    PolarizationCoefficients,
    SatellitePoint,
    SatellitePosition,
    SpectralResponseFunction,
    ApolloIrradianceCoefficients,
    SurfacePoint,
    CustomPoint,
    ReflectanceCoefficients,
    SpectralData,
)
from ..eocfi_adapter import eocfi_adapter
from lime_tbx.simulation.lime_simulation import ILimeSimulation, LimeSimulation
from .ifaces import IMainSimulationsWidget, noconflict_makecls

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
    def_srf: SpectralResponseFunction
        SRF that will be used to calculate the first graph
    srf: SpectralResponseFunction
        SRF that will be used to calculate the integrated irradiance
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
    lime_simulation.update_irradiance(def_srf, srf, point, cimel_coef)
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
) -> Tuple[List[ComparisonData], List[LunarObservation], SpectralResponseFunction,]:
    co = comparison.Comparison()
    for mo in mos:
        if not mo.check_valid_srf(srf):
            raise Exception("SRF file not valid for the chosen Moon observations file.")
    comparisons = co.get_simulations(mos, srf, cimel_coef, lime_simulation)
    return comparisons, mos, srf


def calculate_all_callback(
    srf: SpectralResponseFunction,
    signals_srf: SpectralResponseFunction,
    point: Point,
    coeffs: ApolloIrradianceCoefficients,
    cimel_coef: ReflectanceCoefficients,
    p_coeffs: PolarizationCoefficients,
    lime_simulation: ILimeSimulation,
):
    lime_simulation.update_reflectance(srf, point, cimel_coef)
    lime_simulation.update_irradiance(srf, signals_srf, point, cimel_coef)
    lime_simulation.update_polarization(srf, point, p_coeffs)
    return (point, signals_srf)


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
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.worker = CallbackWorker(
            compare_callback,
            [mos, srf, coeffs, cimel_coef, self.lime_simulation],
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
        self.output.remove_channels(to_remove)
        self._unblock_gui()

    def compare_error(self, error: Exception):
        self._unblock_gui()
        error_dialog = QtWidgets.QErrorMessage(self)
        error_dialog.showMessage(str(error))
        raise error


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
        eocfi_path: str,
        settings_manager: settings.ISettingsManager,
    ):
        super().__init__()
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
            "Simulation output", "Wavelengths (nm)", "Units", parent=self
        )
        self.graph.update_legend(
            [["interpolated data points"], ["CIMEL data points"], ["errorbars (k=2)"]]
        )
        # srf widget
        self.srf_widget = srf.SRFEditWidget(
            self.settings_manager, self._callback_regular_input_changed
        )
        # signal widget
        self.signal_widget = output.SignalWidget()
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
        # finish main layout
        self.main_layout.addWidget(self.input_widget)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addWidget(self.lower_tabs, 1)
        self.main_layout.addWidget(self.export_lglod_button)

    def _unblock_gui(self):
        self.parentWidget().setDisabled(False)
        self.export_lglod_button.setDisabled(False)

    def _block_gui_loading(self):
        self.export_lglod_button.setDisabled(True)
        self.parentWidget().setDisabled(True)

    def _callback_regular_input_changed(self):
        self.lime_simulation.set_simulation_changed()

    def _callback_check_calculable(self):
        calculable = self.input_widget.is_calculable()
        self.lower_tabs.setEnabled(calculable)
        self.export_lglod_button.setDisabled(not calculable)

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
            [def_srf, srf, point, coeffs, cimel_coef, self.lime_simulation],
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

        self.graph.update_plot(data[2], data[3], data[4], data[0])
        self.graph.update_labels(
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²/nm)",
        )
        self.signal_widget.update_signals(data[0], data[1], data[5])

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
            elref_callback, [def_srf, point, coeffs, cimel_coef, self.lime_simulation]
        )
        self._start_thread(self.elref_finished, self.elref_error)

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
        # unc = data[3]
        # if isinstance(unc, list):
        #     wlen = [u.wlen_cimel for u in unc]
        #     cimel_data = [u.data for u in unc]
        #     uncert = [u.uncertainties for u in unc]
        # else:
        #     wlen = unc.wlen_cimel
        #     cimel_data = unc.data
        #     uncert = unc.uncertainties

        self.graph.update_plot(data[1], data[2], data[3], data[0])
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
            polar_callback, [def_srf, point, coeffs, self.lime_simulation]
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
        self.signal_widget.clear_signals()

    def polar_error(self, error: Exception):
        self._unblock_gui()
        raise error

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
        def_srf = self.settings_manager.get_default_srf()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_irr_coeffs()
        cimel_coef = self.settings_manager.get_cimel_coef()
        p_coeffs = self.settings_manager.get_polar_coeffs()
        self.worker = CallbackWorker(
            calculate_all_callback,
            [def_srf, srf, point, coeffs, cimel_coef, p_coeffs, self.lime_simulation],
        )
        self._start_thread(self.calculate_all_finished, self.calculate_all_error)

    def calculate_all_finished(self, data):
        self._unblock_gui()
        point: Point = data[0]
        srf: SpectralResponseFunction = data[1]
        obs = []
        ch_names = srf.get_channels_names()
        sat_pos_ref = "ITRF93"
        elis = self.lime_simulation.get_elis()
        elis_cimel = self.lime_simulation.get_elis_cimel()
        if not isinstance(elis_cimel, list):
            elis_cimel = [elis_cimel]
        elrefs = self.lime_simulation.get_elrefs()
        elrefs_cimel = self.lime_simulation.get_elrefs_cimel()
        if not isinstance(elrefs_cimel, list):
            elrefs_cimel = [elrefs_cimel]
        polars = self.lime_simulation.get_polars()
        signals = self.lime_simulation.get_signals()
        if isinstance(point, SurfacePoint) or isinstance(point, SatellitePoint):
            dts = point.dt
            if not isinstance(dts, list):
                dts = [dts]
            if not isinstance(elis, list):
                elis = [elis]
            if not isinstance(elrefs, list):
                elrefs = [elrefs]
            if not isinstance(polars, list):
                polars = [polars]
            if isinstance(point, SurfacePoint):
                sat_pos = [
                    SatellitePosition(
                        *comparison.to_xyz(
                            point.latitude, point.longitude, point.altitude
                        )
                    )
                    for _ in dts
                ]
                sat_name = ""
            else:
                sur_points = self.lime_simulation.get_surfacepoints()
                if isinstance(sur_points, SurfacePoint):
                    sur_points = [sur_points]
                sat_pos = [
                    SatellitePosition(
                        *comparison.to_xyz(sp.latitude, sp.longitude, sp.altitude)
                    )
                    for sp in sur_points
                ]
                sat_name = point.name
            for i, dt in enumerate(dts):
                ob = LunarObservationWrite(
                    ch_names,
                    sat_pos_ref,
                    dt,
                    sat_pos[i],
                    elis[i],
                    elrefs[i],
                    polars[i],
                    sat_name,
                )
                obs.append(ob)
            is_not_default_srf = True
            if (
                srf.name == constants.DEFAULT_SRF_NAME
                and len(srf.get_channels_names()) == 1
                and srf.get_channels_names()[0] == constants.DEFAULT_SRF_NAME
            ):
                is_not_default_srf = False
            lglod = LGLODData(
                obs, signals, is_not_default_srf, elis_cimel, elrefs_cimel
            )
            name = QtWidgets.QFileDialog().getSaveFileName(
                self, "Export LGLOD", "{}.nc".format("lglod")
            )[0]
            if name is not None and name != "":
                try:
                    moon.write_obs(lglod, name, datetime.now())
                except Exception as e:
                    raise e

    def calculate_all_error(self, error: Exception):
        self._unblock_gui()
        raise error


class LimePagesEnum(Enum):
    SIMULATION = 0
    COMPARISON = 1


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

    def load_observations_finished(
        self, lglod: LGLODData, srf: SpectralResponseFunction
    ):
        valid = True
        if srf == None:
            srf = self.settings_manager.get_default_srf()
        for obs in lglod.observations:
            if not obs.check_valid_srf(srf):
                valid = False
        if not valid:
            error_msg = "SRF file not valid for the observation file."
            error_dialog = QtWidgets.QErrorMessage(self)
            error_dialog.showMessage(error_msg)
            raise Exception(error_msg)
        else:
            self.main_page.srf_widget.set_srf(srf)
            self.lime_simulation.set_observations(lglod, srf)
            point = self.lime_simulation.get_point()
            self.main_page.load_observations_finished(point)


class LimeTBXWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._create_menu_bar()

    def _create_actions(self):
        # File actions
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
        lime_tbx_w: LimeTBXWidget = self.centralWidget()
        lime_tbx_w.propagate_close_event()
        return super().closeEvent(event)

    # ACTIONS

    def load_simulation(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self, "Select GLOD file")[0]
        if path != "":
            lglod = moon.read_lime_glod(path)
            srf = None
            cancel = False
            if lglod.not_default_srf:
                srf_path = QtWidgets.QFileDialog().getOpenFileName(
                    self, "Select SpectralResponseFunction file"
                )[0]
                if srf_path != "":
                    cancel = True
                else:
                    srf = srf_loader.read_srf(srf_path)
            if not cancel:
                lime_tbx_w: LimeTBXWidget = self.centralWidget()
                lime_tbx_w.load_observations_finished(lglod, srf)

    def comparison(self):
        self.comparison_action.setText("Perform &simulations")
        self.comparison_action.triggered.connect(self.simulations)
        lime_tbx_w: LimeTBXWidget = self.centralWidget()
        lime_tbx_w.lime_simulation.set_simulation_changed()
        lime_tbx_w.change_page(LimePagesEnum.COMPARISON)

    def simulations(self):
        self.comparison_action.setText(
            "Perform &comparisons from a remote sensing instrument"
        )
        self.comparison_action.triggered.connect(self.comparison)
        lime_tbx_w: LimeTBXWidget = self.centralWidget()
        lime_tbx_w.lime_simulation.set_simulation_changed()
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
