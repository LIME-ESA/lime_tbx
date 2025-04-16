"""describe class"""

"""___Built-In Modules___"""
from enum import Enum
from typing import List, Callable, Union, Tuple, Iterable
from datetime import datetime, timezone
import os
import sys

"""___Third-Party Modules___"""
from qtpy import QtWidgets, QtCore, QtGui

if QtCore.__version__.startswith("6"):  # Qt6 specific code
    from qtpy.QtGui import QAction
else:  # Qt5 specific code
    from qtpy.QtWidgets import QAction
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.presentation.gui import (
    settings,
    output,
    input,
    srf,
    help,
    interpoptions,
    canvas,
    coefficients,
)
from lime_tbx.presentation.gui.ifaces import IMainSimulationsWidget, noconflict_makecls
from lime_tbx.presentation.gui.spinner import SpinnerPage
from lime_tbx.presentation.gui import constants
from lime_tbx.presentation.gui.util import CallbackWorker, start_thread as _start_thread
from lime_tbx.presentation.gui.settings import ISettingsManager
from lime_tbx.application.filedata import moon, srf as srf_loader, lglod as lglodlib
from lime_tbx.application.filedata.lglod_factory import create_lglod_data
from lime_tbx.application.simulation.comparison import comparison
from lime_tbx.application.simulation.comparison.utils import sort_by_mpa
from lime_tbx.application.simulation.lime_simulation import (
    ILimeSimulation,
    LimeSimulation,
)
from lime_tbx.business.eocfi_adapter import eocfi_adapter
from lime_tbx.business.interpolation.interp_data import interp_data
from lime_tbx.common.datatypes import (
    ComparisonData,
    KernelsPath,
    LGLODComparisonData,
    LGLODData,
    LimeException,
    LunarObservation,
    Point,
    PolarisationCoefficients,
    SatellitePoint,
    SpectralResponseFunction,
    SpectralValidity,
    SurfacePoint,
    ReflectanceCoefficients,
    SpectralData,
    MoonData,
    EocfiPath,
)
from lime_tbx.common import logger, constants as logic_constants
from lime_tbx.common.constants import CompFields
from lime_tbx.business.spectral_integration.spectral_integration import get_default_srf

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "02/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


_INTERNAL_ERROR_MSG = (
    "Something went wrong while performing the operation. See log for more detail."
)

_WARN_OUTSIDE_MPA_RANGE = "Warning: The LIME can only give a reliable simulation \
for absolute moon phase angles between 2° and 90°"


def _simplify_mdas_mda(
    mdas: Union[List[MoonData], MoonData, None]
) -> Union[MoonData, None]:
    mda = None
    if mdas:
        if isinstance(mdas, MoonData):
            mda = mdas
        elif len(mdas) == 1:
            mda = mdas[0]
    return mda


def eli_callback(
    srf: SpectralResponseFunction,
    point: Point,
    cimel_coef: ReflectanceCoefficients,
    lime_simulation: ILimeSimulation,
    signal_info: QtCore.Signal,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    Point,
    List[float],
    SpectralResponseFunction,
    Union[SpectralData, List[SpectralData], Union[bool, List[bool]]],
    Union[List[MoonData], MoonData, None],
]:
    """
    Callback that performs the Irradiance operations.

    Parameters
    ----------
    srf: SpectralResponseFunction
        SRF that will be used to calculate the data
    point: Point
        Point used
    cimel_coef: RefkectanceCoefficients
        Coefficients to calculate reflectance.
    lime_simulation: ILimeSimulation
        Lime simulation object that will calculate the data
    signal_info: QtCore.Signal
        QT signal that can emit a signal to UI.

    Returns
    -------
    point: Point
        Point that was used in the calculations.
    srf: SpectralResponseFunction
        SRF used for the integrated irradiance signal calculation.
    elis: list of float
        Irradiances related to def_srf
    ch_irrs: list of float
        Integrated irradiance signals for each srf channel
    uncertainty_data: SpectralData or list of SpectralData
        Calculated uncertainty data.
    inside_mpa_range: bool or list of bool
        Indicates if the different point locations/s are inside the valid mpa range.
    mpa: float or None
        Moon phase angle in degrees in case that it's only one moon phase angle value.
    """
    def_srf = get_default_srf()
    callback_obs = lambda: signal_info.emit("another_refl_irr_simulated")
    lime_simulation.update_irradiance(def_srf, srf, point, cimel_coef, callback_obs)
    mdas = lime_simulation.get_moon_datas()
    return (
        point,
        srf,
        lime_simulation.get_elis(),
        lime_simulation.get_elis_cimel(),
        lime_simulation.get_elis_asd(),
        lime_simulation.get_signals(),
        lime_simulation.are_mpas_inside_mpa_range(),
        mdas,
    )


def elref_callback(
    srf: SpectralResponseFunction,
    point: Point,
    cimel_coef: ReflectanceCoefficients,
    lime_simulation: ILimeSimulation,
    signal_info: QtCore.Signal,
) -> Tuple[
    List[float],
    List[float],
    Point,
    Union[SpectralData, List[SpectralData]],
    Union[bool, List[bool]],
    Union[List[MoonData], MoonData, None],
]:
    """Callback that performs the Reflectance operations.

    Parameters
    ----------
    srf: SpectralResponseFunction
        SRF that will be used to calculate the graph
    point: Point
        Point used
    cimel_coef: CimelCoef
        CimelCoef with the CIMEL coefficients and uncertainties.
    lime_simulation: ILimeSimulation
        Lime Simulation instance

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
    inside_mpa_range: bool or list of bool
        Indicates if the different point locations/s are inside the valid mpa range.
    mpa: float or None
        Moon phase angle in degrees in case that it's only one moon phase angle value.
    """
    def_srf = get_default_srf()
    callback_obs = lambda: signal_info.emit("another_refl_simulated")
    lime_simulation.update_reflectance(def_srf, point, cimel_coef, callback_obs)
    mdas = lime_simulation.get_moon_datas()
    return (
        point,
        lime_simulation.get_elrefs(),
        lime_simulation.get_elrefs_cimel(),
        lime_simulation.get_elrefs_asd(),
        lime_simulation.are_mpas_inside_mpa_range(),
        mdas,
    )


def polar_callback(
    srf: SpectralResponseFunction,
    point: Point,
    coeffs: PolarisationCoefficients,
    lime_simulation: ILimeSimulation,
    signal_info: QtCore.Signal,
) -> Tuple[
    Point,
    Union[SpectralData, List[SpectralData]],
    Union[SpectralData, List[SpectralData]],
    Union[SpectralData, List[SpectralData]],
    Union[bool, List[bool]],
    Union[List[MoonData], MoonData, None],
]:
    def_srf = get_default_srf()
    callback_obs = lambda: signal_info.emit("another_pol_simulated")
    lime_simulation.update_polarisation(def_srf, point, coeffs, callback_obs)
    mdas = lime_simulation.get_moon_datas()
    return (
        point,
        lime_simulation.get_polars(),
        lime_simulation.get_polars_cimel(),
        lime_simulation.get_polars_asd(),
        lime_simulation.are_mpas_inside_mpa_range(),
        mdas,
    )


def compare_callback(
    mos: List[LunarObservation],
    srf: SpectralResponseFunction,
    cimel_coef: ReflectanceCoefficients,
    lime_simulation: ILimeSimulation,
    kernels_path: KernelsPath,
    signal_info: QtCore.Signal,
) -> Tuple[
    List[ComparisonData],
    List[ComparisonData],
    List[LunarObservation],
    SpectralResponseFunction,
]:
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
    callback_obs = lambda: signal_info.emit("another_obs_simulated")
    comparisons = co.get_simulations(
        mos, srf, cimel_coef, lime_simulation, callback_obs
    )
    mpa_comp = sort_by_mpa(comparisons)
    return comparisons, mpa_comp, mos, srf


def calculate_all_callback(
    srf: SpectralResponseFunction,
    point: Point,
    cimel_coef: ReflectanceCoefficients,
    p_coeffs: PolarisationCoefficients,
    lime_simulation: ILimeSimulation,
):
    def_srf = get_default_srf()
    lime_simulation.update_irradiance(def_srf, srf, point, cimel_coef)
    lime_simulation.update_reflectance(def_srf, point, cimel_coef)
    lime_simulation.update_polarisation(def_srf, point, p_coeffs)
    return (point, srf)


def show_comparisons_wlen_callback(
    output: output.ComparisonByWlenOutput,
    comps: List[ComparisonData],
    srf: SpectralResponseFunction,
    version: str,
    settings_manager: ISettingsManager,
    chosen_diffs: CompFields,
) -> Tuple[output.ComparisonOutput, List[str]]:
    comps = [c if c.observed_signal is not None else None for c in comps]
    output.update_plot(comps, srf.get_channels_centers(), False, chosen_diffs)
    statscomps = [c for c in comps if c is not None]
    n_comp_points = np.mean([len(c.diffs_signal.wlens) for c in statscomps])
    data_start = min([min(c.dts) for c in statscomps])
    data_end = max([max(c.dts) for c in statscomps])
    warning_out_mpa_range = ""
    if False in [not np.all(c.ampa_valid_range) for c in statscomps]:
        warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
    sp_name = settings_manager.get_selected_spectrum_name()
    skip = settings_manager.is_skip_uncertainties()
    spectrum_info = f" | Interp. spectrum: {sp_name}"
    output.set_interp_spectrum_name(sp_name)
    output.set_skipped_uncertainties(skip)
    subtitle = (
        f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}"
    )
    _subtitle_date_format = canvas.SUBTITLE_DATE_FORMAT
    subtitle = "{}\nData start: {} | Data end: {}\nMean number of points: {}".format(
        subtitle,
        data_start.strftime(_subtitle_date_format),
        data_end.strftime(_subtitle_date_format),
        n_comp_points,
    )
    output.update_labels(
        "All channels",
        "Wavelength (nm)",
        "Irradiance (Wm⁻²nm⁻¹)",
        subtitle=subtitle,
        redraw=False,
    )
    output.update_legends(
        [
            ["Observed Irradiance", "Simulated Irradiance"],
        ],
        redraw=True,
    )
    return []


def show_comparisons_callback(
    output: output.ComparisonOutput,
    comps: List[ComparisonData],
    xlabel: str,
    srf: SpectralResponseFunction,
    version: str,
    settings_manager: ISettingsManager,
    chosen_diffs: CompFields,
) -> Tuple[output.ComparisonOutput, List[str]]:
    to_remove = _show_comps_output(
        output,
        comps,
        xlabel,
        srf,
        version,
        settings_manager,
        chosen_diffs,
    )
    return output, to_remove


def _callback_read_srf(
    path: str, lglod: Union[LGLODData, LGLODComparisonData]
) -> Tuple[SpectralResponseFunction, Union[LGLODData, LGLODComparisonData]]:
    srf = srf_loader.read_srf(path)
    return (srf, lglod)


def _show_comps_output(
    output: output.ComparisonOutput,
    comps: List[ComparisonData],
    y_label: str,
    srf: SpectralResponseFunction,
    version: str,
    settings_manager: ISettingsManager,
    chosen_diffs: CompFields,
) -> List[str]:
    to_remove = []
    ch_names = srf.get_channels_names()
    for i, ch in enumerate(ch_names):
        if len(comps[i].dts) > 0:
            ch_id = output.get_channel_id(ch)
            output.update_plot(ch_id, comps[i], False, chosen_diffs)
            n_comp_points = len(comps[i].diffs_signal.wlens)
            data_start = min(comps[i].dts)
            data_end = max(comps[i].dts)
            warning_out_mpa_range = ""
            if False in comps[i].ampa_valid_range:
                warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
            sp_name = settings_manager.get_selected_spectrum_name()
            skip = settings_manager.is_skip_uncertainties()
            spectrum_info = f" | Interp. spectrum: {sp_name}"
            output.set_interp_spectrum_name(ch_id, sp_name)
            output.set_skipped_uncertainties(ch_id, skip)
            subtitle = f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}"
            _subtitle_date_format = canvas.SUBTITLE_DATE_FORMAT
            subtitle = "{}\nData start: {} | Data end: {}\nNumber of points: {}".format(
                subtitle,
                data_start.strftime(_subtitle_date_format),
                data_end.strftime(_subtitle_date_format),
                n_comp_points,
            )
            output.update_labels(
                ch_id,
                "{} ({} nm)".format(ch, srf.get_channel_from_name(ch).center),
                y_label,
                "Irradiance (Wm⁻²nm⁻¹)",
                subtitle=subtitle,
                redraw=False,
            )
            output.update_legends(
                ch_id,
                [
                    ["Observed Irradiance", "Simulated Irradiance"],
                ],
                redraw=True,
            )
        else:
            to_remove.append(ch)
    for chsrf in srf.channels:
        if chsrf.id in to_remove:
            continue
        if chsrf.valid_spectre == SpectralValidity.PARTLY_OUT:
            output.set_as_partly(chsrf.id)
    return to_remove


def clear_comparison_callback(srf: SpectralResponseFunction):
    # Try to delete the SRF
    for i in range(len(srf.channels) - 1, -1, -1):
        srf.channels[i].spectral_response = None
        srf.channels[i].spectral_response_inrange = None
        srf.channels[i].valid_spectre = None
        del srf.channels[i]
    return []


class ComparisonPageWidget(QtWidgets.QWidget):
    def __init__(
        self,
        lime_simulation: ILimeSimulation,
        settings_manager: settings.ISettingsManager,
        kernels_path: KernelsPath,
        eocfi_path: EocfiPath,
    ):
        super().__init__()
        self.lime_simulation = lime_simulation
        self.settings_manager = settings_manager
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.workers = []
        self.worker_ths = []
        self._listening_changes_combobox = True
        self.chosen_diffs = CompFields.DIFF_REL
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # Top input pre comparison
        self.top_precomp_layout = QtWidgets.QVBoxLayout()
        self.input = input.ComparisonInput(
            self._callback_compare_input_changed,
            self._callback_compare_button_enable,
            self.kernels_path,
            self.eocfi_path,
        )
        self.top_precomp_layout.addWidget(self.input)
        self.compare_button = QtWidgets.QPushButton("Compare")
        self.compare_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.compare_button.clicked.connect(self.compare)
        self.compare_button.setDisabled(True)
        self.top_precomp_layout.addWidget(self.compare_button)
        self.top_precomp = QtWidgets.QWidget()
        self.top_precomp.setLayout(self.top_precomp_layout)
        self.top_precomp.setContentsMargins(0, 0, 0, 0)
        # Top input post comparison
        self.top_postcomp_layout = QtWidgets.QVBoxLayout()
        self.comp_options_box = QtWidgets.QHBoxLayout()
        self.clear_comparison_button = QtWidgets.QPushButton("New")
        self.clear_comparison_button.setCursor(
            QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        )
        self.clear_comparison_button.clicked.connect(self.clear_comparison_pressed)
        self.comp_options_box.addWidget(self.clear_comparison_button)
        self.comp_options_box.addWidget(QtWidgets.QLabel(), 2)
        self.compare_by_label = QtWidgets.QLabel("Compare by:")
        self.compare_by_field = QtWidgets.QComboBox()
        self.compare_by_field.view().setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.compare_by_field.addItems(
            [
                CompFields.COMP_DATE,
                CompFields.COMP_MPA,
                CompFields.COMP_WLEN,
                CompFields.COMP_WLEN_MEAN,
            ]
        )
        self.compare_by_field.currentTextChanged.connect(
            self._update_from_compare_combo
        )
        self.comp_options_box.addWidget(self.compare_by_label)
        self.comp_options_box.addWidget(self.compare_by_field)
        self.comp_options_box.addWidget(QtWidgets.QLabel(), 1)
        self.difference_by_label = QtWidgets.QLabel("Difference:")
        self.difference_by_field = QtWidgets.QComboBox()
        self.difference_by_field.view().setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.difference_by_field.addItems(
            [
                CompFields.DIFF_NONE,
                CompFields.DIFF_REL,
                CompFields.DIFF_PERC,
            ]
        )
        self.difference_by_field.setCurrentIndex(1)
        self.difference_by_field.currentTextChanged.connect(
            self._update_from_difference_combo
        )
        self.comp_options_box.addWidget(self.difference_by_label)
        self.comp_options_box.addWidget(self.difference_by_field)
        self.top_postcomp_layout.addLayout(self.comp_options_box)
        self.top_postcomp = QtWidgets.QWidget()
        self.top_postcomp.setLayout(self.top_postcomp_layout)
        self.top_postcomp.setContentsMargins(0, 0, 0, 0)
        self.top_postcomp.setVisible(False)
        # Comparison content
        self.stack_layout = QtWidgets.QStackedLayout()
        self.stack_layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)
        self.output = output.ComparisonOutput(self.settings_manager)
        self.output_wlen = output.ComparisonByWlenOutput(self.settings_manager)
        self.output_wlen.setVisible(False)
        self.output_stacklay = QtWidgets.QStackedLayout()
        self.output_stacklay.setStackingMode(QtWidgets.QStackedLayout.StackAll)
        self.output_stacklay.addWidget(self.output)
        self.output_stacklay.addWidget(self.output_wlen)
        self.output_stacklayw = QtWidgets.QWidget()
        self.output_stacklayw.setLayout(self.output_stacklay)
        self.spinner = SpinnerPage()
        self.spinner.setVisible(False)
        self.stack_layout.addWidget(self.spinner)
        self.stack_layout.addWidget(self.output_stacklayw)
        self.stack_layout.setCurrentIndex(1)
        self.export_lglod_button = QtWidgets.QPushButton("Export to NetCDF")
        self.export_lglod_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.export_lglod_button.clicked.connect(self.export_to_lglod)
        self.export_lglod_button.setDisabled(True)
        self.main_layout.addWidget(self.top_precomp)
        self.main_layout.addWidget(self.top_postcomp)
        self.main_layout.addLayout(self.stack_layout)
        self.main_layout.addWidget(self.export_lglod_button)

    def _focus_on_comp_wlen(self, focuswlen: bool):
        self.output_wlen.setVisible(focuswlen)
        self.output.setVisible(not focuswlen)
        if focuswlen:
            self.output_stacklay.setCurrentIndex(1)
            self.output_wlen.refresh_canvas()
        else:
            self.output_stacklay.setCurrentIndex(0)

    def is_focus_on_comp_wlen(self) -> bool:
        return self.output_stacklay.currentIndex() == 1

    def _update_from_compare_combo(self, value: str):
        if self._listening_changes_combobox:
            if value == CompFields.COMP_MPA:
                self.show_compare_mpa()
            elif value == CompFields.COMP_DATE:
                self.show_compare_dts()
            elif value == CompFields.COMP_WLEN:
                self.show_compare_wlen_boxplot()
            elif value == CompFields.COMP_WLEN_MEAN:
                self.show_compare_wlen_mean()

    def _update_from_difference_combo(self, value: str):
        if value in [CompFields.DIFF_NONE, CompFields.DIFF_PERC, CompFields.DIFF_REL]:
            self.chosen_diffs = value
        if self._listening_changes_combobox:
            if value == CompFields.DIFF_PERC:
                self.show_perc_diff()
            elif value == CompFields.DIFF_REL:
                self.show_rel_diff()
            else:
                self.show_no_diff()

    def _callback_compare_button_enable(self, enable: bool):
        self.compare_button.setEnabled(enable)

    def _callback_compare_input_changed(self):
        self.lime_simulation.set_simulation_changed()
        obss = self.input.get_moon_obs()
        srf = self.input.get_srf()
        if len(obss) == 0 or srf == None:
            self.compare_button.setDisabled(True)
        else:
            self.compare_button.setDisabled(False)

    def _start_thread(
        self,
        worker: CallbackWorker,
        finished: Callable,
        error: Callable,
        info: Callable = None,
    ):
        worker_th = QtCore.QThread()
        self.worker_ths.append(worker_th)
        self.workers.append(worker)
        _start_thread(worker, worker_th, finished, error, info)

    def _set_spinner(self, enabled: bool):
        self.spinner.setVisible(enabled)
        if enabled:
            self.spinner.movie_start()
            self.stack_layout.setCurrentIndex(0)
        else:
            self.spinner.set_text("")
            self.spinner.movie_stop()
            self.stack_layout.setCurrentIndex(1)

    def _unblock_gui(self):
        self._set_spinner(False)
        self.parentWidget().setDisabled(False)

    def _block_gui_loading(self):
        self._set_spinner(True)
        self.parentWidget().setDisabled(True)

    def block_gui_drawing_message(self):
        self._block_gui_loading()
        self.spinner.set_text("Drawing graphs...")

    def can_save_simulation(self) -> bool:
        return self.export_lglod_button.isEnabled()

    @QtCore.Slot()
    def export_to_lglod(self) -> None:
        self._block_gui_loading()
        vers = self.settings_manager.get_coef_version_name()
        lglod = LGLODComparisonData(
            self.comps,
            self.srf.get_channels_names(),
            self.data_source,
            self.comparison_spectrum,
            self.skipped_uncs,
            vers,
        )
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export LGLOD", "{}.nc".format("lglod")
        )[0]
        if name is not None and name != "":
            worker = CallbackWorker(
                lglodlib.write_comparison,
                [
                    lglod,
                    name,
                    datetime.now().astimezone(timezone.utc),
                    self.kernels_path,
                ],
            )
            self._start_thread(
                worker, lambda _: self._unblock_gui(), self.export_to_lglod_err
            )
        else:
            self._unblock_gui()

    def export_to_lglod_err(self, error: Exception):
        self.handle_operation_error(error)
        self._unblock_gui()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    @QtCore.Slot()
    def compare(self):
        self._block_gui_loading()
        mos = self.input.get_moon_obs()
        srf = self.input.get_srf()
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.quant_mos = len(mos)
        self.quant_mos_simulated = 0
        self.comparison_spectrum = self.settings_manager.get_selected_spectrum_name()
        worker = CallbackWorker(
            compare_callback,
            [mos, srf, cimel_coef, self.lime_simulation, self.kernels_path],
            True,
        )
        self._start_thread(
            worker, self.compare_finished, self.compare_error, self.compare_info
        )

    def compare_info(self, data: str):
        if (
            self.quant_mos_simulated < self.quant_mos - 1
        ):  # -1 So it gives time to the last message to be shown
            self.spinner.set_text(f"{self.quant_mos_simulated}/{self.quant_mos}")
            self.quant_mos_simulated += 1
        else:
            self.spinner.set_text(f"Finishing comparisons\nand drawing graphs...")
            self.quant_mos_simulated += 1

    def set_show_comparison_input(self, show: bool):
        self.top_precomp.setVisible(show)
        self.top_postcomp.setVisible(not show)
        if show:
            self.output.set_channels([])
            self.output_wlen.clear()
            self._focus_on_comp_wlen(False)

    @QtCore.Slot()
    def clear_comparison_pressed(self):
        self.clear_comp_dialog = QtWidgets.QMessageBox(self)
        self.clear_comp_dialog.setStandardButtons(
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        self.clear_comp_dialog.setIcon(QtWidgets.QMessageBox.Information)
        self.clear_comp_dialog.setText("Are you sure?")
        self.clear_comp_dialog.setInformativeText(
            "This will clear the current comparison"
        )
        self.clear_comp_dialog.setWindowTitle("New comparison")
        self.clear_comp_dialog.accepted.connect(self.clear_comparison_accepted)
        self.clear_comp_dialog.rejected.connect(self.clear_comparison_rejected)
        self.clear_comp_dialog.show()

    @QtCore.Slot()
    def clear_comparison_accepted(self):
        self._listening_changes_combobox = False
        self._block_gui_loading()
        params = [
            self.srf,
        ]
        worker = CallbackWorker(clear_comparison_callback, params)
        self._start_thread(
            worker, self._clear_comparison_finished, self._clear_comparison_error
        )

    def _clear_comparison_finished(self, data):
        self.srf = None
        self.set_show_comparison_input(True)
        if not self.compare_by_field.currentText() == CompFields.COMP_DATE:
            self.compare_by_field.setCurrentText(CompFields.COMP_DATE)
        if not self.difference_by_field.currentText() == CompFields.DIFF_REL:
            self.difference_by_field.setCurrentText(CompFields.DIFF_REL)
        self.input.clear_input()
        self.lime_simulation.clear_srf()
        self.clear_comp_dialog.close()
        self.export_lglod_button.setEnabled(False)
        self.top_postcomp.setVisible(False)
        self._listening_changes_combobox = True
        self._unblock_gui()

    def _clear_comparison_error(self, error: Exception):
        self._unblock_gui()
        self.handle_operation_error(error)
        self.export_lglod_button.setEnabled(False)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(True)

    @QtCore.Slot()
    def clear_comparison_rejected(self):
        self.clear_comp_dialog.close()

    def compare_finished(
        self,
        data: Tuple[
            List[ComparisonData],
            List[ComparisonData],
            List[LunarObservation],
            SpectralResponseFunction,
        ],
    ):
        self.set_show_comparison_input(False)
        comps = data[0]
        mpa_comps = data[1]
        mos = data[2]
        srf = data[3]
        self.comps = comps
        self.data_source = mos[0].data_source
        self.skipped_uncs = self.settings_manager.is_skip_uncertainties()
        self.mpa_comps = mpa_comps
        self.srf = srf
        self.version = self.settings_manager.get_coef_version_name()
        params = [
            self.output,
            self.comps,
            CompFields.COMP_DATE,
            self.srf,
            self.version,
            self.settings_manager,
            self.chosen_diffs,
        ]
        # Channels are set to the output here, as that needs to be done in the main qt thread.
        ch_names = srf.get_channels_names()
        self.output.set_channels(ch_names)
        worker = CallbackWorker(show_comparisons_callback, params)
        self._start_thread(
            worker, self._load_lglod_comparisons_finished, self.compare_error
        )

    def show_compare_dts(self):
        self._focus_on_comp_wlen(False)
        self._block_gui_loading()
        params = [
            self.output,
            self.comps,
            CompFields.COMP_DATE,
            self.srf,
            self.version,
            self.settings_manager,
            self.chosen_diffs,
        ]
        worker = CallbackWorker(show_comparisons_callback, params)
        self._start_thread(
            worker, self._show_comparisons_switch_finished, self.compare_error
        )

    def show_compare_mpa(self):
        self._focus_on_comp_wlen(False)
        self._block_gui_loading()
        params = [
            self.output,
            self.mpa_comps,
            CompFields.COMP_MPA,
            self.srf,
            self.version,
            self.settings_manager,
            self.chosen_diffs,
        ]
        worker = CallbackWorker(show_comparisons_callback, params)
        self._start_thread(
            worker, self._show_comparisons_switch_finished, self.compare_error
        )

    def _show_compare_wlen(self, boxplot: bool):
        self._focus_on_comp_wlen(True)
        self._block_gui_loading()
        self.output_wlen.set_kind(boxplot)
        params = [
            self.output_wlen,
            self.comps,
            self.srf,
            self.version,
            self.settings_manager,
            self.chosen_diffs,
        ]
        worker = CallbackWorker(show_comparisons_wlen_callback, params)
        self._start_thread(
            worker, self._show_comparisons_wlen_finished, self.compare_error
        )

    def show_compare_wlen_boxplot(self):
        self._show_compare_wlen(True)

    def show_compare_wlen_mean(self):
        self._show_compare_wlen(False)

    def show_perc_diff(self):
        self._block_gui_loading()
        focuswlen = self.is_focus_on_comp_wlen()
        self.output.show_percentage(not focuswlen)
        self.output_wlen.show_percentage(focuswlen)
        self._unblock_gui()

    def show_rel_diff(self):
        self._block_gui_loading()
        focuswlen = self.is_focus_on_comp_wlen()
        self.output.show_relative(not focuswlen)
        self.output_wlen.show_relative(focuswlen)
        self._unblock_gui()

    def show_no_diff(self):
        self._block_gui_loading()
        focuswlen = self.is_focus_on_comp_wlen()
        self.output.show_no_diff(not focuswlen)
        self.output_wlen.show_no_diff(focuswlen)
        self._unblock_gui()

    def load_lglod_comparisons(
        self,
        comps: List[ComparisonData],
        mpa_comps: List[ComparisonData],
        srf: SpectralResponseFunction,
        data_source: str,
        skipped_uncs: bool,
        version: str,
    ):
        self.set_show_comparison_input(False)
        self.input.clear_input()
        self.compare_button.setDisabled(True)
        self.comps = comps
        self.data_source = data_source
        self.skipped_uncs = skipped_uncs
        self.mpa_comps = mpa_comps
        self.srf = srf
        self.comparison_spectrum = self.settings_manager.get_selected_spectrum_name()
        self.version = version
        params = [
            self.output,
            self.comps,
            CompFields.COMP_DATE,
            self.srf,
            version,
            self.settings_manager,
            self.chosen_diffs,
        ]
        # Channels are set to the output here, as that needs to be done in the main qt thread.
        # This section blocks the spinning loading widget.
        ch_names = srf.get_channels_names()
        self.output.set_channels(ch_names)
        worker = CallbackWorker(show_comparisons_callback, params)
        self._start_thread(
            worker, self._load_lglod_comparisons_finished, self.compare_error
        )

    def _load_lglod_comparisons_finished(self, data):
        self._focus_on_comp_wlen(False)
        outp: output.ComparisonOutput = data[0]
        outp.remove_channels(data[1])
        outp.check_if_range_visible()
        self._unblock_gui()
        outp.set_current_channel_index(0)
        self.export_lglod_button.setEnabled(True)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(False)
        self.top_postcomp.setVisible(True)

    def _show_comparisons_switch_finished(self, data):
        self._focus_on_comp_wlen(False)
        outp: output.ComparisonOutput = data[0]
        outp.check_if_range_visible()
        self._unblock_gui()
        self.export_lglod_button.setEnabled(True)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(False)
        self.top_postcomp.setVisible(True)

    def _show_comparisons_wlen_finished(self, data):
        self._focus_on_comp_wlen(True)
        self._unblock_gui()
        self.export_lglod_button.setEnabled(True)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(False)
        self.top_postcomp.setVisible(True)

    def handle_operation_error(self, error: Exception):
        if isinstance(error, LimeException):
            self.show_error(error)
        else:
            logger.get_logger().critical(error)
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
        eocfi_path: EocfiPath,
        settings_manager: settings.ISettingsManager,
    ):
        super().__init__()
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self._finished_building = False
        self.lime_simulation = lime_simulation
        self.settings_manager = settings_manager
        self.eocfi: eocfi_adapter.EOCFIConverter = eocfi_adapter.EOCFIConverter(
            eocfi_path,
            kernels_path,
        )
        self.workers = []
        self.worker_ths = []
        self._build_layout()
        self._finished_building = True

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # input
        self.input_widget = input.InputWidget(
            self._callback_regular_input_changed,
            self.update_calculability,
            self.settings_manager.is_skip_uncertainties(),
            self.eocfi_path,
            self.kernels_path,
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
        self.polar_button = QtWidgets.QPushButton("Polarisation")
        self.polar_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.polar_button.clicked.connect(self.show_polar)
        self.buttons_layout.addWidget(self.eli_button)
        self.buttons_layout.addWidget(self.elref_button)
        self.buttons_layout.addWidget(self.polar_button)
        # Lower tab
        self.lower_tabs = QtWidgets.QTabWidget()
        self.lower_tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        # graph
        self.graph = output.SimGraphWidget(
            self.settings_manager,
            "Simulation output",
            "Wavelengths (nm)",
            "Units",
            parent=self,
        )
        self.graph.update_legend(
            [
                [constants.INTERPOLATED_DATA_LABEL],
                [constants.CIMEL_POINT_LABEL],
                [constants.ERRORBARS_LABEL],
            ]
        )
        self.graph.set_xlim(
            logic_constants.CERTAIN_MIN_WLEN, logic_constants.CERTAIN_MAX_WLEN
        )
        # srf widget
        self.srf_widget = srf.SRFEditWidget(
            self.settings_manager,
            self._callback_regular_input_changed,
            self._callback_set_enabled,
        )
        # signal widget
        self.signal_widget = output.SignalWidget(self.settings_manager)
        # finish tab
        self.lower_tabs.addTab(self.graph, "Result")
        self.lower_tabs.addTab(self.srf_widget, "SRF")
        self.lower_tabs.addTab(self.signal_widget, "Signal")
        self.lower_tabs.currentChanged.connect(self.lower_tabs_changed)
        # Export to LGLOD
        self.export_lglod_button = QtWidgets.QPushButton("Export to NetCDF")
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
        self.update_calculability()

    def _callback_set_enabled(self, enabled: bool):
        if enabled:
            self._unblock_gui()
        else:
            self._block_gui_loading()

    def _set_spinner(self, enabled: bool):
        self.loading_spinner.setVisible(enabled)
        if enabled:
            self.loading_spinner.movie_start()
            self.lower_stack.setCurrentIndex(0)
        else:
            self.loading_spinner.set_text("")
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
        self.export_lglod_button.setDisabled(disable)
        window: LimeTBXWindow = self.parentWidget().parentWidget()
        window.set_save_simulation_action_disabled(disable)

    def _callback_regular_input_changed(self):
        self.lime_simulation.set_simulation_changed()

    def block_gui_drawing_message(self):
        self._block_gui_loading()
        self.loading_spinner.set_text("Drawing graphs...")

    def update_calculability(self):
        calculable = self.input_widget.is_calculable()
        self.eli_button.setEnabled(calculable)
        self.elref_button.setEnabled(calculable)
        polar_calculable = self.settings_manager.get_polar_coef().is_calculable()
        self.polar_button.setEnabled(calculable and polar_calculable)
        if not (self._export_lglod_button_was_disabled):
            self._disable_lglod_export(not calculable)

    def _start_thread(
        self,
        worker: CallbackWorker,
        finished: Callable,
        error: Callable,
        info: Callable = None,
    ):
        worker_th = QtCore.QThread()
        self.worker_ths.append(worker_th)
        self.workers.append(worker)
        _start_thread(worker, worker_th, finished, error, info)

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
        if isinstance(error, LimeException):
            self.show_error(error)
        else:
            logger.get_logger().critical(error)
            self.show_error(_INTERNAL_ERROR_MSG)

    @QtCore.Slot()
    def show_eli(self):
        """
        Calculate and show extraterrestrial lunar irradiances for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.quant_elis = 1
        self.quant_elis_sim = 0
        if hasattr(point, "dt") and isinstance(point.dt, list):
            self.quant_elis = len(point.dt)
        self.will_calc_reflectance_prev = (
            self.lime_simulation.will_irradiance_calculate_reflectance_previously(point)
        )
        if self.will_calc_reflectance_prev:
            self.quant_elrefs = self.quant_elis
            self.quant_elrefs_sim = self.quant_elis_sim
        worker = CallbackWorker(
            eli_callback,
            [srf, point, cimel_coef, self.lime_simulation],
            True,
        )
        self._start_thread(worker, self.eli_finished, self.eli_error, self.eli_info)

    def eli_info(self, data: str):
        if (
            self.will_calc_reflectance_prev
            and self.quant_elrefs_sim < self.quant_elrefs
        ):
            self.loading_spinner.set_text(
                f"{self.quant_elrefs_sim}/{self.quant_elrefs} (reflectances)"
            )
            self.quant_elrefs_sim += 1
        elif self.quant_elis_sim < self.quant_elis:
            self.loading_spinner.set_text(f"{self.quant_elis_sim}/{self.quant_elis}")
            self.quant_elis_sim += 1
        else:
            self.loading_spinner.set_text(f"Finishing simulations...")

    def _set_graph_dts(self, pt: Point):
        self.graph.set_dts([])
        if isinstance(pt, SurfacePoint) or isinstance(pt, SatellitePoint):
            if isinstance(pt.dt, list) and len(pt.dt) > 1:
                self.graph.set_dts(pt.dt)

    def eli_finished(
        self,
        data: Tuple[
            Point,
            SpectralResponseFunction,
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            SpectralData,
            Union[bool, List[bool]],
            Union[List[MoonData], MoonData, None],
        ],
    ):
        self._unblock_gui()
        self._set_graph_dts(data[0])
        mpa_text = ""
        mdas = data[7]
        mda = _simplify_mdas_mda(mdas)
        if mda:
            mpa_text = f" | MPA: {mda.mpa_degrees:.3f}°"
        sp_name = interp_data.get_interpolation_spectrum_name()
        spectrum_info = f" | Interp. spectrum: {sp_name}{mpa_text}"
        self.graph.set_interp_spectrum_name(sp_name)
        self.graph.set_mda(mdas)
        is_skip_uncs = self.lime_simulation.is_skipping_uncs()
        self.graph.set_skipped_uncertainties(is_skip_uncs)
        self.graph.update_plot(data[2], data[3], data[4], data[0], redraw=False)
        version = self.settings_manager.get_coef_version_name()
        is_out_mpa_range = (
            not data[6] if not isinstance(data[6], list) else False in data[6]
        )
        warning_out_mpa_range = ""
        if is_out_mpa_range:
            warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
        self.graph.update_labels(
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances (Wm⁻²nm⁻¹)",
            subtitle=f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}",
        )
        self.graph.set_inside_mpa_range(data[6])
        self.signal_widget.set_interp_spectrum_name(
            self.settings_manager.get_selected_spectrum_name()
        )
        if mda:
            self.signal_widget.set_mpa(mda.mpa_degrees)
        self.signal_widget.set_skipped_uncertainties(is_skip_uncs)
        self.signal_widget.update_signals(data[0], data[1], data[5], data[6])
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
        cimel_coef = self.settings_manager.get_cimel_coef()
        self.quant_elrefs = 1
        self.quant_elrefs_sim = 0
        if hasattr(point, "dt") and isinstance(point.dt, list):
            self.quant_elrefs = len(point.dt)
        worker = CallbackWorker(
            elref_callback,
            [srf, point, cimel_coef, self.lime_simulation],
            True,
        )
        self._start_thread(
            worker, self.elref_finished, self.elref_error, self.elref_info
        )

    def clear_signals(self):
        self.signal_widget.clear_signals()
        self.lower_tabs.setTabEnabled(2, False)

    def elref_info(self, data: str):
        if self.quant_elrefs_sim < self.quant_elrefs:
            self.loading_spinner.set_text(
                f"{self.quant_elrefs_sim}/{self.quant_elrefs}"
            )
            self.quant_elrefs_sim += 1
        else:
            self.loading_spinner.set_text(f"Finishing simulations...")

    def elref_finished(
        self,
        data: Tuple[
            Point,
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            Union[bool, List[bool]],
            Union[List[MoonData], MoonData, None],
        ],
    ):
        self._unblock_gui()
        self._set_graph_dts(data[0])
        mpa_text = ""
        mdas = data[5]
        mda = _simplify_mdas_mda(mdas)
        if mda:
            mpa_text = f" | MPA: {mda.mpa_degrees:.3f}°"
        sp_name = interp_data.get_interpolation_spectrum_name()
        spectrum_info = f" | Interp. spectrum: {sp_name}{mpa_text}"
        self.graph.set_interp_spectrum_name(sp_name)
        self.graph.set_mda(mdas)
        self.graph.set_skipped_uncertainties(self.lime_simulation.is_skipping_uncs())
        self.graph.update_plot(data[1], data[2], data[3], data[0], redraw=False)
        version = self.settings_manager.get_coef_version_name()
        is_out_mpa_range = (
            not data[4] if not isinstance(data[4], list) else False in data[4]
        )
        warning_out_mpa_range = ""
        if is_out_mpa_range:
            warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
        self.graph.update_labels(
            "Extraterrestrial Lunar Reflectances",
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
            subtitle=f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}",
        )
        self.graph.set_inside_mpa_range(data[4])
        self.clear_signals()
        self.lower_tabs.setCurrentIndex(0)

    def elref_error(self, error: Exception):
        self.handle_operation_error(error)
        self._unblock_gui()

    @QtCore.Slot()
    def show_polar(self):
        """
        Calculate and show extraterrestrial lunar polarisation for the given input.
        """
        self._block_gui_loading()
        point = self.input_widget.get_point()
        srf = self.settings_manager.get_srf()
        coeffs = self.settings_manager.get_polar_coef()
        self.quant_polars = 1
        self.quant_polars_sim = 0
        if hasattr(point, "dt") and isinstance(point.dt, list):
            self.quant_polars = len(point.dt)
        worker = CallbackWorker(
            polar_callback,
            [srf, point, coeffs, self.lime_simulation],
            True,
        )
        self._start_thread(
            worker, self.polar_finished, self.polar_error, self.polar_info
        )

    def polar_info(self, data: str):
        if self.quant_polars_sim < self.quant_polars:
            self.loading_spinner.set_text(
                f"{self.quant_polars_sim}/{self.quant_polars}"
            )
            self.quant_polars_sim += 1
        else:
            self.loading_spinner.set_text(f"Finishing simulations...")

    def polar_finished(
        self,
        data: Tuple[
            Point,
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            Union[SpectralData, List[SpectralData]],
            Union[bool, List[bool]],
            Union[List[MoonData], MoonData, None],
        ],
    ):
        self._unblock_gui()
        self._set_graph_dts(data[0])
        mpa_text = ""
        mdas = data[5]
        mda = _simplify_mdas_mda(mdas)
        if mda:
            mpa_text = f" | MPA: {mda.mpa_degrees:.3f}°"
        sp_name = interp_data.get_dolp_interpolation_spectrum_name()
        spectrum_info = f" | Interp. spectrum: {sp_name}{mpa_text}"
        self.graph.set_interp_spectrum_name(sp_name)
        self.graph.set_mda(mdas)
        self.graph.set_skipped_uncertainties(self.lime_simulation.is_skipping_uncs())
        self.graph.update_plot(data[1], data[2], data[3], data[0], redraw=False)
        # self.graph.set_max_ylims(-120, 120) # TODO decide if we do this or not
        version = self.settings_manager.get_coef_version_name()
        is_out_mpa_range = (
            not data[4] if not isinstance(data[4], list) else False in data[4]
        )
        warning_out_mpa_range = ""
        if is_out_mpa_range:
            warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
        self.graph.update_labels(
            "Lunar polarisation",
            "Wavelengths (nm)",
            "Degree of Linear Polarisation (%)",
            subtitle=f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}",
        )
        self.graph.set_inside_mpa_range(data[4])
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
        cimel_coef = self.settings_manager.get_cimel_coef()
        p_coeffs = self.settings_manager.get_polar_coef()
        worker = CallbackWorker(
            calculate_all_callback,
            [srf, point, cimel_coef, p_coeffs, self.lime_simulation],
        )
        self._start_thread(
            worker, self.calculate_all_finished, self.calculate_all_error
        )

    def calculate_all_finished(self, data):
        point: Point = data[0]
        srf: SpectralResponseFunction = data[1]
        sp_name = self.settings_manager.get_selected_spectrum_name()
        polar_sp_name = self.settings_manager.get_selected_polar_spectrum_name()
        version = self.settings_manager.get_coef_version_name()
        mds = self.lime_simulation.get_moon_datas()
        if not isinstance(mds, list):
            mds = [mds]
        lglod = create_lglod_data(
            point,
            srf,
            self.lime_simulation,
            self.kernels_path,
            sp_name,
            polar_sp_name,
            version,
            mds,
        )
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export LGLOD", "{}.nc".format("lglod")
        )[0]
        inside_mpa_range = self.lime_simulation.are_mpas_inside_mpa_range()
        if name is not None and name != "":
            worker = CallbackWorker(
                lglodlib.write_obs,
                [
                    lglod,
                    name,
                    datetime.now().astimezone(timezone.utc),
                    inside_mpa_range,
                ],
            )
            self._start_thread(worker, self._unblock_gui, self.calculate_all_error)
        else:
            self._unblock_gui()

    def calculate_all_error(self, error: Exception):
        self.handle_operation_error(error)
        self._unblock_gui()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))


class LimePagesEnum(Enum):
    SIMULATION = 0
    COMPARISON = 1


def load_simulation_callback(path: str, kernels_path: KernelsPath):
    lglod = lglodlib.read_lglod_file(path, kernels_path)
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
    new_channels = []
    for chan in srf_chans:
        if chan in lglod.ch_names:
            new_channels.append(srf.get_channel_from_name(chan))
    new_srf = SpectralResponseFunction(srf.name, new_channels)
    return [lglod, new_srf]


def obtain_sorted_mpa_callback(
    lglod: LGLODComparisonData,
    srf,
):
    comps = lglod.comparisons
    data_source = lglod.sat_name
    skipped_uncs = lglod.skipped_uncs
    mpa_comps = sort_by_mpa(comps)
    return comps, mpa_comps, srf, data_source, skipped_uncs, lglod.version


def return_args_callback(*args):
    return args


class LimeTBXWidget(QtWidgets.QWidget):
    """
    Main widget of the lime toolbox desktop app.
    """

    def __init__(
        self, kernels_path: KernelsPath, eocfi_path: EocfiPath, selected_version: str
    ):
        super().__init__()
        self.setLocale(QtCore.QLocale("English"))
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.settings_manager = settings.SettingsManager(selected_version)
        self.lime_simulation = LimeSimulation(
            eocfi_path, kernels_path, self.settings_manager
        )
        self.workers = []
        self.worker_ths = []
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.comparison_page = ComparisonPageWidget(
            self.lime_simulation,
            self.settings_manager,
            self.kernels_path,
            self.eocfi_path,
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

    def _start_thread(
        self, worker: CallbackWorker, finished: Callable, error: Callable
    ):
        worker_th = QtCore.QThread()
        self.worker_ths.append(worker_th)
        self.workers.append(worker)
        _start_thread(worker, worker_th, finished, error)

    def load_observations_finished(
        self, lglod: LGLODData, srf: SpectralResponseFunction
    ):
        if srf == None:
            srf = self.settings_manager.get_default_srf()
        worker = CallbackWorker(check_srf_observation_callback, [lglod, srf])
        self._start_thread(
            worker,
            self._load_observations_finished_2,
            self._load_observations_finished_error,
        )

    def _load_observations_finished_2(self, data):
        lglod: LGLODData = data[0]
        srf = data[1]
        self.main_page.srf_widget.set_srf(srf)
        self.lime_simulation.set_observations(lglod, srf)
        self.settings_manager.select_interp_spectrum(lglod.spectrum_name)
        self.settings_manager.set_skip_uncertainties(lglod.skipped_uncs)
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
        worker = CallbackWorker(check_srf_comparison_callback, [lglod, srf])
        self._start_thread(
            worker,
            self._load_comparisons_finished_2,
            self._load_comparisons_finished_error,
        )

    def _load_comparisons_finished_2(self, data):
        lglod: LGLODComparisonData = data[0]
        srf = data[1]
        self.settings_manager.select_interp_spectrum(lglod.spectrum_name)
        self.settings_manager.set_skip_uncertainties(lglod.skipped_uncs)
        worker = CallbackWorker(
            obtain_sorted_mpa_callback,
            [
                lglod,
                srf,
            ],
        )
        self._start_thread(
            worker,
            self._load_comparisons_finished_3,
            self._load_comparisons_finished_error,
        )

    def _load_comparisons_finished_3(self, data):
        comps = data[0]
        mpa_comps = data[1]
        srf = data[2]
        data_source = data[3]
        skipped_uncs = data[4]
        version = data[5]
        self.comparison_page.load_lglod_comparisons(
            comps, mpa_comps, srf, data_source, skipped_uncs, version
        )

    def _load_comparisons_finished_error(self, error: Exception):
        logger.get_logger().critical(error)
        self.show_error(error)
        self.comparison_page._unblock_gui()

    def can_save_simulation(self) -> bool:
        return self.page.can_save_simulation()

    def get_current_page(self) -> Union[MainSimulationsWidget, ComparisonPageWidget]:
        return self.page

    def update_calculability(self):
        self.main_page.update_calculability()


def _set_all_messagebox_buttons_pointing_hands():
    """Function that sets qmessage buttons with pointing hands"""
    for w in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(w, QtWidgets.QMessageBox):
            for button in w.buttons():
                button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))


class LimeTBXWindow(QtWidgets.QMainWindow):
    def __init__(self, kernels_path: KernelsPath):
        super().__init__()
        self.kernels_path = kernels_path
        self._is_comparing = False
        self.worker_ths = []
        self.workers = []
        self._create_menu_bar()

    def _create_actions(self):
        # File actions
        self.save_simulation_action = QAction(self)
        self.save_simulation_action.setText(
            "&Save simulation to LIME GLOD format file."
        )
        self.save_simulation_action.triggered.connect(self.save_simulation)
        self.save_simulation_action.setDisabled(True)
        self.load_simulation_action = QAction(self)
        self.load_simulation_action.setText(
            "&Load simulation file stored in a LIME GLOD format file."
        )
        self.load_simulation_action.triggered.connect(self.load_simulation)
        self.comparison_action = QAction(self)
        self.comparison_action.setText(
            "Perform &comparisons from a remote sensing instrument"
        )
        self.comparison_action.triggered.connect(self.comparison)
        self.exit_action = QAction(self)
        self.exit_action.setText("E&xit")
        self.exit_action.triggered.connect(self.exit)
        # Coefficients actions
        self.download_coefficients_action = QAction(self)
        self.download_coefficients_action.setText("&Download updated coefficients")
        self.download_coefficients_action.triggered.connect(self.download_coefficients)
        # self.download_coefficients_action.setDisabled(True)
        self.select_coefficients_action = QAction(self)
        self.select_coefficients_action.setText("&Select coefficients")
        self.select_coefficients_action.triggered.connect(self.select_coefficients)
        # Help actions
        self.about_action = QAction(self)
        self.about_action.setText("&About")
        self.about_action.triggered.connect(self.about)
        self.help_action = QAction(self)
        self.help_action.setText("&Help")
        self.help_action.triggered.connect(self.help)
        # Settings actions
        self.interpolation_action = QAction(self)
        self.interpolation_action.setText("&Interpolation options")
        self.interpolation_action.triggered.connect(self.interpol_options)
        ##
        # Shortcut action
        toggle_fullscreen = QAction("Toggle Fullscreen", self)
        toggle_fullscreen.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        fullscreen_shortcuts = [QtGui.QKeySequence.FullScreen]
        if sys.platform != "win32":
            fullscreen_shortcuts.append(QtGui.QKeySequence(QtCore.Qt.Key_F11))
        toggle_fullscreen.setShortcuts(fullscreen_shortcuts)
        toggle_fullscreen.triggered.connect(self.toggle_fullscreen)
        self.addAction(toggle_fullscreen)

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
        help_menu.addAction(self.help_action)
        settings_menu = QtWidgets.QMenu("&Settings", self)
        settings_menu.addAction(self.interpolation_action)
        self.menu_bar.addMenu(file_menu)
        self.menu_bar.addMenu(coeffs_menu)
        self.menu_bar.addMenu(help_menu)
        self.menu_bar.addMenu(settings_menu)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        QtCore.QTimer.singleShot(0, _set_all_messagebox_buttons_pointing_hands)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Window Close",
            "Are you sure you want to close the application?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            lime_tbx_w = self._get_lime_widget()
            lime_tbx_w.propagate_close_event()
            QtCore.QCoreApplication.quit()
            os.kill(os.getpid(), 9)
            return super().closeEvent(event)
        else:
            event.ignore()

    def set_save_simulation_action_disabled(self, disable: bool) -> None:
        self.save_simulation_action.setDisabled(disable)

    def _start_thread(
        self, worker: CallbackWorker, finished: Callable, error: Callable
    ):
        worker_th = QtCore.QThread()
        self.worker_ths.append(worker_th)
        self.workers.append(worker)
        _start_thread(worker, worker_th, finished, error)

    def _get_lime_widget(self) -> LimeTBXWidget:
        return self.centralWidget()

    # ERROR

    def show_error(self, error: Exception):
        self._get_lime_widget().get_current_page()._unblock_gui()
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    # ACTIONS

    def save_simulation(self):
        lime_tbx_w = self._get_lime_widget()
        if self._is_comparing:
            lime_tbx_w.comparison_page.export_to_lglod()
        else:
            lime_tbx_w.main_page.export_glod()

    def load_simulation_read_srf_finished(self, data):
        srf = data[0]
        lglod = data[1]
        worker = CallbackWorker(return_args_callback, [lglod, srf])
        self._start_thread(
            worker, self._load_observations_finished, self.load_simulation_error
        )

    def load_comparisons_read_srf_finished(self, data):
        srf = data[0]
        lglod = data[1]
        worker = CallbackWorker(return_args_callback, [lglod, srf])
        self._start_thread(
            worker,
            self._load_comparisons_finished,
            self.load_simulation_error,
        )

    def load_simulation_read_srf_error(self, e):
        self.show_error(e)
        lime_tbx_w = self._get_lime_widget()
        lime_tbx_w.get_current_page()._unblock_gui()

    def load_simulation_finished(self, data):
        lglod = data[0]
        lime_tbx_w = self._get_lime_widget()
        page = lime_tbx_w.get_current_page()
        page._unblock_gui()
        if isinstance(lglod, LGLODData):
            self.simulations()
            lime_tbx_w.get_current_page().block_gui_drawing_message()
            if lglod.not_default_srf:
                srf_path = QtWidgets.QFileDialog().getOpenFileName(
                    self, "Select SpectralResponseFunction file"
                )[0]
                if srf_path == "":
                    lime_tbx_w.get_current_page()._unblock_gui()
                else:
                    worker = CallbackWorker(
                        _callback_read_srf,
                        [srf_path, lglod],
                    )
                    self._start_thread(
                        worker,
                        self.load_simulation_read_srf_finished,
                        self.load_simulation_read_srf_error,
                    )
            else:
                lime_tbx_w = self._get_lime_widget()
                lime_tbx_w.load_observations_finished(lglod, None)
        else:
            self.comparison()
            lime_tbx_w.get_current_page().block_gui_drawing_message()
            srf_path = QtWidgets.QFileDialog().getOpenFileName(
                self, "Select SpectralResponseFunction file"
            )[0]
            if srf_path != "":
                worker = CallbackWorker(
                    _callback_read_srf,
                    [srf_path, lglod],
                )
                self._start_thread(
                    worker,
                    self.load_comparisons_read_srf_finished,
                    self.load_simulation_read_srf_error,
                )
            else:
                lime_tbx_w.get_current_page()._unblock_gui()

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
            worker = CallbackWorker(load_simulation_callback, [path, self.kernels_path])
            self._start_thread(
                worker, self.load_simulation_finished, self.load_simulation_error
            )

    def comparison(self):
        if not self._is_comparing:
            self.save_simulation_action.setText(
                "&Save comparison to LIME GLOD format file."
            )
            self.comparison_action.setText("&Perform simulations")
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
        lime_tbx_w = self._get_lime_widget()
        download_coeffs_dialog = coefficients.DownloadCoefficientsDialog(
            lime_tbx_w.settings_manager, self
        )
        download_coeffs_dialog.exec()

    def select_coefficients(self):
        lime_tbx_w = self._get_lime_widget()
        select_coefficients_dialog = coefficients.SelectCoefficientsDialog(
            lime_tbx_w.settings_manager,
            lime_tbx_w.lime_simulation,
            self.update_calculability,
            self,
        )
        select_coefficients_dialog.exec()

    def about(self):
        about_dialog = help.AboutDialog(self)
        about_dialog.exec()

    def help(self):
        help_dialog = help.HelpDialog(self)
        help_dialog.exec()

    def update_calculability(self):
        lime_tbx_w = self._get_lime_widget()
        lime_tbx_w.update_calculability()

    def interpol_options(self):
        lime_tbx_w = self._get_lime_widget()
        interpol_opt_dialog = interpoptions.InterpOptionsDialog(
            lime_tbx_w.settings_manager,
            lime_tbx_w.lime_simulation,
            self._get_lime_widget().main_page.input_widget,
            self,
        )
        interpol_opt_dialog.exec()
