"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List, Tuple
import os

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    ComparisonData,
    Point,
    SatellitePoint,
    SpectralResponseFunction,
    SpectralValidity,
    SurfacePoint,
    CustomPoint,
    SpectralData,
)
from . import constants
from lime_tbx.gui.settings import ISettingsManager
from ..filedata import csv
from .ifaces import IMainSimulationsWidget

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes: Axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


_current_dir = os.path.dirname(os.path.abspath(__file__))
dir_font_path = os.path.dirname(os.path.join(_current_dir, constants.ESAFONT_PATH))
font_dirs = [dir_font_path]
font_files = fm.findSystemFonts(fontpaths=font_dirs, fontext="otf")
for font_file in font_files:
    fm.fontManager.addfont(font_file)
title_font_prop = fm.FontProperties(
    family=["NotesESA", "sans-serif"], weight="bold", size="large"
)
label_font_prop = fm.FontProperties(family=["NotesESA", "sans-serif"], weight="bold")
font_prop = fm.FontProperties(family=["NotesESA", "sans-serif"])


class GraphWidget(QtWidgets.QWidget):
    def __init__(
        self,
        settings_manager: ISettingsManager,
        title="",
        xlabel="",
        ylabel="",
        parent=None,
    ):
        super().__init__(parent)
        self._init_parent = parent
        self.settings_manager = settings_manager
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = []
        self.data = None
        self.cimel_data = None
        self.asd_data = None
        self.point = None
        self.data_compare = None
        self.vertical_lines = []
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # canvas
        self.canvas = MplCanvas(self)
        self.canvas.axes.set_title(self.title, fontproperties=title_font_prop)
        self.canvas.axes.tick_params(labelsize=8)
        version = self.settings_manager.get_cimel_coef().version
        subtitle = "LIME2 coefficients version: {}".format(version)
        ay2 = self.canvas.axes.twiny()
        ay2.set_xlabel(subtitle, fontproperties=font_prop)
        ay2.tick_params(
            axis="x",
            which="both",
            top=False,
            labeltop=False,
        )
        self.canvas.axes.set_xlabel(self.xlabel, fontproperties=label_font_prop)
        self.canvas.axes.set_ylabel(self.ylabel, fontproperties=label_font_prop)
        self._prepare_toolbar()
        self._redraw()
        # save buttons
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.export_button = QtWidgets.QPushButton("Export graph (.png, .jpg, .pdf...)")
        self.export_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.export_button.clicked.connect(self.export_graph)
        self.csv_button = QtWidgets.QPushButton("Export CSV")
        self.csv_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.csv_button.clicked.connect(self.export_csv)
        self.disable_buttons(True)
        self.buttons_layout.addWidget(self.export_button)
        self.buttons_layout.addWidget(self.csv_button)
        # finish main
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas, 1)
        self.main_layout.addLayout(self.buttons_layout)

    def _prepare_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ["Back", "Forward"]
        for ta in self.toolbar.actions():
            ta: QtWidgets.QAction = ta
            if ta.text() in unwanted_buttons:
                self.toolbar.removeAction(ta)
                continue
            icon = ta.icon()
            sizes = icon.availableSizes()
            max_h = max_w = 0
            for i in range(len(sizes)):
                max_h = max(max_h, sizes[i].height())
                max_w = max(max_w, sizes[i].width())
            pixmap: QtGui.QPixmap = icon.pixmap(QtCore.QSize(max_w, max_h))
            tmp = pixmap.toImage()
            color = QtGui.QColor(QtGui.qRgb(232, 232, 228))
            for h in range(tmp.height()):
                for w in range(tmp.width()):
                    color.setAlpha(tmp.pixelColor(w, h).alpha())
                    tmp.setPixelColor(w, h, color)
            pixmap = QtGui.QPixmap.fromImage(tmp)
            ta.setIcon(QtGui.QIcon(pixmap))

    def disable_buttons(self, disable: bool):
        self.export_button.setDisabled(disable)
        self.csv_button.setDisabled(disable)
        if self._init_parent and isinstance(self._init_parent, IMainSimulationsWidget):
            self._init_parent.set_export_button_disabled(disable)

    def set_vertical_lines(self, xs: List[float]):
        self.vertical_lines = xs
        self._redraw()

    def update_plot(
        self,
        data: Union[SpectralData, List[SpectralData]] = None,
        data_cimel: Union[SpectralData, List[SpectralData]] = None,
        data_asd: Union[SpectralData, List[SpectralData]] = None,
        point: Union[Point, List[Point]] = None,
        data_compare: ComparisonData = None,
        redraw: bool = True,
    ):
        self.point = point
        self.data = data
        self.cimel_data = data_cimel
        self.asd_data = data_asd
        self.data_compare = data_compare
        if data is not None:
            self.disable_buttons(False)
        else:
            self.disable_buttons(True)
        if redraw:
            self._redraw()

    def update_labels(self, title: str, xlabel: str, ylabel: str, redraw: bool = True):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        if redraw:
            self._redraw()

    def update_legend(self, legend: List[List[str]], redraw: bool = True):
        """
        Parameters
        ----------
        legend: list of list of str
            Each list represents a group of legends
            Lengeds index:
            0: data
            1: cimel_data
            2: cimel_data errorbars
            3: comparison
        """
        self.legend = legend
        if redraw:
            self._redraw()

    def update_size(self):
        self._redraw()

    def _redraw(self):
        self.canvas.axes.cla()  # Clear the canvas.
        lines = []
        if self.data is not None:
            iter_data = self.data
            if not isinstance(iter_data, list):
                iter_data = [iter_data]
            for i, data in enumerate(iter_data):
                label = ""
                color = []
                if i == 0:
                    color = ["g"]
                if len(self.legend) > 0:
                    if len(self.legend[0]) > i:
                        label = self.legend[0][i]
                    if len(self.legend[0]) > 1:
                        color = []
                marker = ""
                if len(data.data) == 1:
                    marker = "o"
                lines += self.canvas.axes.plot(
                    data.wlens,
                    data.data,
                    *color,
                    marker=marker,
                    label=label,
                )
                if data.uncertainties is not None:
                    self.canvas.axes.fill_between(
                        data.wlens,
                        data.data - 2 * data.uncertainties,
                        data.data + 2 * data.uncertainties,
                        color="green",
                        alpha=0.3,
                    )

            if self.cimel_data:
                iter_data = self.cimel_data
                if not isinstance(iter_data, list):
                    iter_data = [iter_data]
                for i, cimel_data in enumerate(iter_data):
                    label0 = ""
                    label1 = ""
                    if i == 0 and len(self.legend) >= 3:
                        label0 = self.legend[1][0]
                        label1 = self.legend[2][0]
                    extra_lines = []
                    extra_lines += self.canvas.axes.plot(
                        cimel_data.wlens,
                        cimel_data.data,
                        color="orange",
                        ls="none",
                        marker="o",
                        label=label0,
                    )
                    extra_lines += [
                        self.canvas.axes.errorbar(
                            cimel_data.wlens,
                            cimel_data.data,
                            yerr=cimel_data.uncertainties * 2,
                            color="black",
                            capsize=3,
                            ls="none",
                            label=label1,
                        )
                    ]
                    if i == 0:
                        lines += extra_lines

            if self.asd_data:
                if isinstance(self.asd_data, list):
                    asd_data = self.asd_data[0]
                else:
                    asd_data = self.asd_data

                scaling_factor = (
                    asd_data.data[np.where(asd_data.wlens == 500)]
                    / cimel_data.data[np.where(cimel_data.wlens == 500)]
                )
                lines += self.canvas.axes.plot(
                    asd_data.wlens,
                    asd_data.data / scaling_factor,
                    label="ASD data points, scaled to LIME at 500nm",
                )

            data_compare_info = ""
            if self.data_compare:
                iter_data = self.data_compare.diffs_signal
                if not isinstance(iter_data, list):
                    iter_data = [iter_data]
                ax2 = self.canvas.axes.twinx()
                for i, data_comp in enumerate(iter_data):
                    label = ""
                    if len(self.legend) > 3 and len(self.legend[3]) > 0:
                        label = self.legend[3][0]
                    marker = ""
                    if len(data_comp.data) == 1:
                        marker = "o"
                    lines += ax2.plot(
                        data_comp.wlens,
                        data_comp.data,
                        color="pink",
                        marker=marker,
                        label=label,
                    )
                    if data_comp.uncertainties is not None:
                        ax2.fill_between(
                            data_comp.wlens,
                            data_comp.data - 2 * data_comp.uncertainties,
                            data_comp.data + 2 * data_comp.uncertainties,
                            color="pink",
                            alpha=0.3,
                        )
                    ax2.set_ylim(
                        (
                            min(-0.05, min(data_comp.data) - 0.05),
                            max(0.05, max(data_comp.data) + 0.05),
                        )
                    )
                    data_compare_info = "MRD: {:.4f}\nσ: {:.4f}".format(
                        self.data_compare.mean_relative_difference,
                        self.data_compare.standard_deviation_mrd,
                    )
                    lines += self.canvas.axes.plot([], [], " ", label=data_compare_info)
                ax2.set_ylabel(
                    "Relative difference (Fraction of unity)",
                    fontproperties=label_font_prop,
                )
                plt.setp(
                    self.canvas.axes.get_xticklabels(),
                    rotation=30,
                    horizontalalignment="right",
                )
            if len(self.legend) > 0:
                legend_lines = [
                    l for l in lines if not l.get_label().startswith("_child")
                ]
                labels = [l.get_label() for l in legend_lines]
                self.canvas.axes.legend(legend_lines, labels, loc=0, prop=font_prop)

        self.canvas.axes.set_title(self.title, fontproperties=title_font_prop)
        self.canvas.axes.set_xlabel(self.xlabel, fontproperties=label_font_prop)
        self.canvas.axes.set_ylabel(self.ylabel, fontproperties=label_font_prop)
        if self.vertical_lines and len(self.vertical_lines) > 0:
            for val in self.vertical_lines:
                self.canvas.axes.axvline(x=val, color="k", label="LIME Spectrum limit")
        try:
            self.canvas.fig.tight_layout()
        except:
            pass
        self.canvas.draw()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    @QtCore.Slot()
    def export_graph(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export graph (.png, .jpg, .pdf...)", "{}.png".format(self.title)
        )[0]
        self.parentWidget().setDisabled(True)
        self.disable_buttons(True)
        if name is not None and name != "":
            try:
                self.canvas.print_figure(name)
            except Exception as e:
                self.show_error(e)
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)

    @QtCore.Slot()
    def export_csv(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format(self.title)
        )[0]
        self.parentWidget().setDisabled(True)
        self.disable_buttons(True)
        version = self.settings_manager.get_cimel_coef().version
        if name is not None and name != "":
            try:
                if isinstance(self.point, list):
                    csv.export_csv_comparation(
                        self.data,
                        self.ylabel,
                        self.point,
                        name,
                        version,
                    )
                else:
                    csv.export_csv(
                        self.data,
                        self.xlabel,
                        self.ylabel,
                        self.point,
                        name,
                        version,
                    )
            except Exception as e:
                self.show_error(e)
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)


class SignalWidget(QtWidgets.QWidget):
    def __init__(self, settings_manager: ISettingsManager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.groupbox = QtWidgets.QGroupBox()
        self.container_layout = QtWidgets.QVBoxLayout()
        self.data_layout = QtWidgets.QFormLayout()
        self.groupbox.setLayout(self.container_layout)
        self.range_warning = None
        # table
        self.table = QtWidgets.QTableWidget()
        # csv button
        self.button_csv = QtWidgets.QPushButton("Export CSV")
        self.button_csv.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button_csv.clicked.connect(self.export_csv)

        self.container_layout.addLayout(self.data_layout, 1)
        self.data_layout.addWidget(self.table)
        self.container_layout.addStretch()
        self.container_layout.addWidget(self.button_csv)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.groupbox)
        self.main_layout.addWidget(self.scroll_area)
        self.disable_buttons(True)

    def _clear_layout(self):
        while self.table.rowCount() > 0:
            self.table.removeRow(0)
        if self.range_warning:
            self.range_warning.setParent(None)
            self.range_warning = None
        self.disable_buttons(True)

    def update_signals(
        self,
        point: Point,
        srf: SpectralResponseFunction,
        signals: SpectralData,
    ):
        self._clear_layout()
        show_range_info = False
        self.srf = srf
        self.signals = signals

        self.point = point
        head_id_item = QtWidgets.QTableWidgetItem("ID")
        head_center_item = QtWidgets.QTableWidgetItem("Center (nm)")
        self.table.setRowCount(1 + len(signals.data))
        if isinstance(point, CustomPoint):
            self.table.setColumnCount(2 + 2)
            self.table.setItem(0, 2, QtWidgets.QTableWidgetItem("Signal (Wm⁻²nm⁻¹)"))
            self.table.setItem(0, 3, QtWidgets.QTableWidgetItem("Uncertainties"))
        else:
            dts = point.dt
            if not isinstance(dts, list):
                dts = [dts]
            self.table.setColumnCount(len(dts) * 2 + 2)
            for i, dt in enumerate(dts):
                item_title_value = QtWidgets.QTableWidgetItem(
                    "Signal (Wm⁻²nm⁻¹) on {}".format(
                        dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    )
                )
                item_title_uncert = QtWidgets.QTableWidgetItem("Uncertainties")
                self.table.setItem(0, i * 2 + 2, item_title_value)
                self.table.setItem(0, i * 2 + 3, item_title_uncert)
        self.table.setItem(0, 0, head_id_item)
        self.table.setItem(0, 1, head_center_item)
        for i, ch_signals in enumerate(signals.data):
            ch = srf.channels[i]
            ch_uncs = signals.uncertainties[i]
            if not (isinstance(ch_signals, np.ndarray) or isinstance(ch_signals, list)):
                ch_signals = [ch_signals]
                ch_uncs = [ch_uncs]
            id_item = QtWidgets.QTableWidgetItem(str(ch.id))
            center_item = QtWidgets.QTableWidgetItem(str(ch.center))
            self.table.setItem(i + 1, 0, id_item)
            self.table.setItem(i + 1, 1, center_item)
            for j, signal in enumerate(ch_signals):
                if ch.valid_spectre == SpectralValidity.VALID:
                    value = "{}".format(str(signal))
                    unc = "{}".format(str(ch_uncs[j]))
                elif ch.valid_spectre == SpectralValidity.PARTLY_OUT:
                    value = "{} *".format(str(signal))
                    unc = "{} *".format(str(ch_uncs[j]))
                    show_range_info = True
                else:
                    value = "Not available *"
                    unc = "Not available *"
                    show_range_info = True
                value_item = QtWidgets.QTableWidgetItem(value)
                unc_item = QtWidgets.QTableWidgetItem(unc)
                self.table.setItem(i + 1, j * 2 + 2, value_item)
                self.table.setItem(i + 1, j * 2 + 3, unc_item)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        if show_range_info:
            self.range_warning = QtWidgets.QLabel(
                "* The LIME can only give a reliable simulation \
for wavelengths between 350 and 2500 nm"
            )
            self.range_warning.setWordWrap(True)
            self.container_layout.addWidget(self.range_warning)
        self.disable_buttons(False)

    def clear_signals(self):
        self._clear_layout()

    def disable_buttons(self, disable: bool):
        self.button_csv.setDisabled(disable)

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    @QtCore.Slot()
    def export_csv(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format("Signal")
        )[0]
        self.parentWidget().setDisabled(True)
        self.disable_buttons(True)
        version = self.settings_manager.get_cimel_coef().version
        if name is not None and name != "":
            try:
                csv.export_csv_integrated_irradiance(
                    self.srf, self.signals, name, self.point, version
                )
            except Exception as e:
                self.show_error(e)
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)


class ComparisonOutput(QtWidgets.QWidget):
    def __init__(self, settings_manager: ISettingsManager):
        super().__init__()
        self.settings_manager = settings_manager
        self.channels: List[GraphWidget] = []
        self.ch_names = []
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.channel_tabs = QtWidgets.QTabWidget()
        self.channel_tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        self.range_warning = None
        self.main_layout.addWidget(self.channel_tabs)

    def set_channels(self, channels: List[str]):
        while self.channel_tabs.count() > 0:
            self.channel_tabs.removeTab(0)
        for ch in self.channels:
            ch.setParent(None)
        self.channels.clear()
        self.ch_names = []
        for ch in channels:
            channel = GraphWidget(self.settings_manager, ch)
            self.channels.append(channel)
            self.ch_names.append(ch)
            self.channel_tabs.addTab(channel, ch)
        # Remove range warning
        if self.range_warning:
            self.range_warning.setParent(None)
            self.range_warning = None

    def set_as_partly(self, ch_name: str):
        if ch_name in self.ch_names:
            index = self.ch_names.index(ch_name)
            self.channel_tabs.setTabText(index, "{} *".format(ch_name))
            if self.range_warning == None:
                self.range_warning = QtWidgets.QLabel(
                    "* The LIME can only give a reliable simulation \
for wavelengths between 350 and 2500 nm"
                )
                self.range_warning.setWordWrap(True)
                self.main_layout.addWidget(self.range_warning)

    def _check_range_warning_needed(self):
        for i in range(len(self.ch_names)):
            if "*" in self.channel_tabs.tabText(i):
                return
        # Not needed
        if self.range_warning:
            self.range_warning.setParent(None)
            self.range_warning = None

    def remove_channels(self, channels: List[str]):
        for ch_name in channels:
            if ch_name in self.ch_names:
                index = self.ch_names.index(ch_name)
                self.channel_tabs.removeTab(index)
                self.channels[index].setParent(None)
                self.channels.pop(index)
                self.ch_names.pop(index)
        self._check_range_warning_needed()

    def update_plot(self, index: int, comparison: ComparisonData, redraw: bool = True):
        """Update the <index> plot with the given data

        Parameters
        ----------
        index: int
            Plot index (SRF)
        comparison: ComparisonData
        redraw: bool
            Boolean that defines if the plot will be redrawn automatically or not. Default True.
        """
        data = [comparison.observed_signal, comparison.simulated_signal]
        self.channels[index].update_plot(
            data, point=comparison.points, data_compare=comparison, redraw=redraw
        )

    def update_labels(
        self, index: int, title: str, xlabel: str, ylabel: str, redraw: bool = True
    ):
        self.channels[index].update_labels(title, xlabel, ylabel, redraw=redraw)

    def update_legends(self, index: int, legends: List[List[str]], redraw: bool = True):
        """
        Parameters
        ----------
        index: int
            Plot index (SRF)
        legend: list of list of str
            Each list represents a group of legends
            Lengeds index:
            0: data
            1: cimel_data
            2: cimel_data errorbars
            3: comparison
        redraw: bool
            Boolean that defines if the plot will be redrawn automatically or not. Default True.
        """
        self.channels[index].update_legend(legends, redraw=redraw)
