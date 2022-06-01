"""describe class"""

"""___Built-In Modules___"""
from enum import Enum
from typing import List, Callable, Union, Tuple
from datetime import datetime
import time

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import settings, output, input, srf, help
from ..simulation.regular_simulation import regular_simulation
from ..simulation.common.common import CommonSimulation
from ..simulation.esa_satellites import esa_satellites
from ..simulation.comparison import comparison
from ..datatypes.datatypes import (
    PolarizationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    IrradianceCoefficients,
    SurfacePoint,
    CustomPoint,
)
from ..eocfi_adapter import eocfi_adapter
import lime_tbx.lime_algorithms.rolo.eli as eli
import lime_tbx.lime_algorithms.rolo.elref as elref
import xarray

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
    cimel_data: xarray,
    kernels_path: str,
    eocfi_path: str,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    Union[SurfacePoint, CustomPoint, SatellitePoint],
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
    kernels_path: str
    eocfi_path: str

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
    """
    rs = regular_simulation.RegularSimulation
    es = esa_satellites.ESASatellites
    time.sleep(0.01)  # For some reason without this the GUI doesn't get disabled.
    elis: Union[List[float], List[List[float]]] = []
    elis_srf: Union[List[float], List[List[float]]] = []
    if isinstance(point, SurfacePoint):
        md=rs.get_md_from_surface(point, kernels_path)
        elis = CommonSimulation.get_eli_from_md(def_srf, md, coeffs)
        elis_srf = CommonSimulation.get_eli_from_md(srf, md, coeffs)
    elif isinstance(point, CustomPoint):
        md = rs.get_md_from_custom(point)
        elis = CommonSimulation.get_eli_from_md(def_srf,md,coeffs)
        elis_srf = CommonSimulation.get_eli_from_md(srf,md,coeffs)
    else:
        elis = es.get_eli_from_satellite(
            def_srf, point, coeffs, kernels_path, eocfi_path
        )
        elis_srf = es.get_eli_from_satellite(
            srf, point, coeffs, kernels_path, eocfi_path
        )
    wlens = def_srf.get_wavelengths()
    ch_irrs = rs.integrate_elis(srf, elis_srf)

    wlen_cimel=cimel_data.wavelength.values
    print(wlen_cimel)
    coeff_cimel=cimel_data.coeff.values
    u_coeff_cimel=cimel_data.u_coeff.values
    elis_cimel=eli.calculate_eli_band(wlen_cimel, md, coeff_cimel)
    u_elis_cimel=eli.calculate_eli_band_unc(wlen_cimel, md, coeff_cimel, u_coeff_cimel)
    return wlens, elis, point, ch_irrs, srf, wlen_cimel, elis_cimel, u_elis_cimel


def elref_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: IrradianceCoefficients,
    cimel_data: xarray.Dataset,
    kernels_path: str,
    eocfi_path: str,
) -> Tuple[List[float], List[float], Union[SurfacePoint, CustomPoint, SatellitePoint]]:
    rs = regular_simulation.RegularSimulation
    es = esa_satellites.ESASatellites
    if isinstance(point, SurfacePoint):
        md=rs.get_md_from_surface(point, kernels_path)
        elrefs: List[float] = rs.get_elref_from_surface(
            srf, point, coeffs, kernels_path
        )
    elif isinstance(point, CustomPoint):
        md = rs.get_md_from_custom(point)
        elrefs: List[float] = rs.get_elref_from_custom(srf, point, coeffs)
    else:
        elrefs: List[float] = es.get_elref_from_satellite(
            srf, point, coeffs, kernels_path, eocfi_path
        )
    wlens = srf.get_wavelengths()

    wlen_cimel = cimel_data.wavelength.values
    coeff_cimel = cimel_data.coeff.values
    u_coeff_cimel = cimel_data.u_coeff.values
    elrefs_cimel = elref.band_moon_disk_reflectance(
                    wlen_cimel,md,coeff_cimel
                )
    u_elrefs_cimel = elref.band_moon_disk_reflectance_unc(
                    wlen_cimel,md,coeff_cimel,u_coeff_cimel
                )
    return wlens, elrefs, point, wlen_cimel, elrefs_cimel, u_elrefs_cimel


def polar_callback(
    srf: SpectralResponseFunction,
    point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    coeffs: PolarizationCoefficients,
    kernels_path: str,
    eocfi_path: str,
) -> Tuple[List[float], List[float], Union[SurfacePoint, CustomPoint, SatellitePoint]]:
    rs = regular_simulation.RegularSimulation
    es = esa_satellites.ESASatellites
    if isinstance(point, SurfacePoint):
        polars: List[float] = rs.get_polarized_from_surface(
            srf, point, coeffs, kernels_path
        )
    elif isinstance(point, CustomPoint):
        polars: List[float] = rs.get_polarized_from_custom(srf, point, coeffs)
    else:
        polars: List[float] = es.get_polarized_from_satellite(
            srf, point, coeffs, kernels_path, eocfi_path
        )
    wlens = srf.get_wavelengths()
    return wlens, polars, point


class ComparisonPageWidget(QtWidgets.QWidget):
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
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.input = input.ComparisonInput()
        self.compare_button = QtWidgets.QPushButton("Compare")
        self.compare_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.compare_button.clicked.connect(self.compare)
        self.output = output.ComparisonOutput()
        self.main_layout.addWidget(self.input)
        self.main_layout.addWidget(self.compare_button)
        self.main_layout.addWidget(self.output)

    @QtCore.Slot()
    def compare(self):
        co = comparison.Comparison()
        mos = self.input.get_moon_obs()
        srf = self.input.get_srf()
        for mo in mos:
            if not mo.check_valid_srf(srf):
                raise Exception(
                    "SRF file not valid for the chosen Moon observations file."
                )
        coeffs = self.settings_manager.get_irr_coeffs()
        irrs, dts = co.get_simulations(mos, srf, coeffs, self.kernels_path)
        ch_names = srf.get_channels_names()
        self.output.set_channels(ch_names)
        to_remove = []
        for i, ch in enumerate(ch_names):
            obs_irrs = []
            for mo in mos:
                if mo.has_ch_value(ch):
                    obs_irrs.append(mo.ch_irrs[ch])
            if len(dts[i]) > 0:
                self.output.update_plot(i, dts[i], [obs_irrs, irrs[i]])
            else:
                to_remove.append(ch)
        self.output.remove_channels(to_remove)


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
        cimel_data = self.settings_manager.get_cimel_data()
        self.worker = CallbackWorker(
            eli_callback,
            [def_srf, srf, point, coeffs, cimel_data, self.kernels_path, self.eocfi_path],
        )
        self._start_thread(self.eli_finished, self.eli_error)

    def eli_finished(
        self,
        data: Tuple[
            List[float],
            Union[List[float], List[List[float]]],
            Union[SurfacePoint, CustomPoint, SatellitePoint],
            Union[List[float], List[List[float]]],
            SpectralResponseFunction,
            List[float],
            Union[List[float],List[List[float]]],
            Union[List[float],List[List[float]]],
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[0], data[1], data[2], data[5], data[6], data[7])
        self.graph.update_labels(
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²/nm)",
        )
        self.signal_widget.update_signals(data[3], data[4], data[2])

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
        cimel_data = self.settings_manager.get_cimel_data()
        self.worker = CallbackWorker(
            elref_callback, [def_srf, point, coeffs, cimel_data, self.kernels_path, self.eocfi_path]
        )
        self._start_thread(self.elref_finished, self.elref_error)

    def elref_finished(
        self,
        data: Tuple[
            List[float],
            Union[List[float], List[List[float]]],
            Union[SurfacePoint, CustomPoint, SatellitePoint],
            List[float],
            Union[List[float],List[List[float]]],
            Union[List[float],List[List[float]]],
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[0], data[1], data[2], data[3], data[4], data[5])
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
            polar_callback, [def_srf, point, coeffs, self.kernels_path, self.eocfi_path]
        )
        self._start_thread(self.polar_finished, self.polar_error)

    def polar_finished(
        self,
        data: Tuple[
            List[float],
            Union[List[float], List[List[float]]],
            Union[SurfacePoint, CustomPoint, SatellitePoint],
            List[float],
            Union[List[float], List[List[float]]],
            Union[List[float], List[List[float]]],
        ],
    ):
        self._unblock_gui()
        self.graph.update_plot(data[0], data[1], data[2])
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
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.settings_manager = settings.MockSettingsManager()
        self.page = MainSimulationsWidget(
            self.kernels_path, self.eocfi_path, self.settings_manager
        )
        self.main_layout.addWidget(self.page)

    def _change_page(self, pageWidget: QtWidgets.QWidget):
        self.main_layout.removeWidget(self.page)
        self.page.setParent(None)
        self.page = pageWidget
        self.main_layout.addWidget(self.page)

    def setDisabled(self, arg__1: bool) -> None:
        self.parentWidget().setDisabled(arg__1)
        return super().setDisabled(arg__1)

    def propagate_close_event(self):
        pass

    def change_page(self, page: LimePagesEnum):
        if page == LimePagesEnum.COMPARISON:
            page = ComparisonPageWidget(
                self.kernels_path, self.eocfi_path, self.settings_manager
            )
        else:
            page = MainSimulationsWidget(
                self.kernels_path, self.eocfi_path, self.settings_manager
            )
        self._change_page(page)


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
