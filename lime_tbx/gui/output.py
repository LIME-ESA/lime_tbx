"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List
from datetime import datetime

import PySide2.QtCore
import PySide2.QtWidgets

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
import matplotlib.backends.backend_pdf  # important import for exporting as pdf
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
import numpy as np
import mplcursors

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    ComparisonData,
    Point,
    SpectralResponseFunction,
    SpectralValidity,
    CustomPoint,
    SpectralData,
)
from lime_tbx.gui.settings import ISettingsManager
from lime_tbx.filedata import csv
from lime_tbx.gui.ifaces import IMainSimulationsWidget
from lime_tbx.gui.canvas import (
    MplCanvas,
    title_font_prop,
    label_font_prop,
    font_prop,
    redraw_canvas,
)
from lime_tbx.gui import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class GraphWidget(QtWidgets.QWidget):
    def __init__(
        self,
        settings_manager: ISettingsManager,
        title="",
        xlabel="",
        ylabel="",
        comparison_x_datetime=True,
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
        self.dts = []
        self.cursor_names = []
        self.mpl_cursor = None
        self.xlim_left = None
        self.xlim_right = None
        self.comparison_x_datetime = comparison_x_datetime
        self.inside_mpa_range = None
        self.interp_spectrum_name = None
        self.skip_uncs = None
        self.compare_percentages = None
        self.ch_names = []
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # canvas
        self.canvas = MplCanvas(self)
        self.canvas.set_title(self.title, fontproperties=title_font_prop)
        self.canvas.axes.tick_params(labelsize=8)
        version = self.settings_manager.get_coef_version_name()
        subtitle = "LIME coefficients version: {}".format(version)
        self.subtitle = subtitle
        self.canvas.set_subtitle(subtitle, fontproperties=font_prop)
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

    def set_xlim(self, left: float = None, right: float = None):
        self.xlim_left = left
        self.xlim_right = right

    def _update_toolbar(self):
        self.toolbar.update()

    def update_plot(
        self,
        data: Union[SpectralData, List[SpectralData]] = None,
        data_cimel: Union[SpectralData, List[SpectralData]] = None,
        data_asd: Union[SpectralData, List[SpectralData]] = None,
        point: Union[Point, List[Point]] = None,
        data_compare: ComparisonData = None,
        redraw: bool = True,
        compare_percentages: bool = False,
    ):
        self.point = point
        self.data = data
        self.cimel_data = data_cimel
        self.asd_data = data_asd
        self.data_compare = data_compare
        self.compare_percentages = compare_percentages
        if data is not None:
            self.disable_buttons(False)
        else:
            self.disable_buttons(True)
        if redraw:
            self._redraw()
            self._update_toolbar()

    def update_labels(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        redraw: bool = True,
        subtitle: str = None,
    ):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        if subtitle != None:
            self.subtitle = subtitle
        if redraw:
            self._redraw()
        self._update_toolbar()

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
        self._update_toolbar()

    def update_size(self):
        self._redraw()

    def set_cursor_names(self, labels: List[str]):
        self.dts = None
        self.cursor_names = labels.copy()

    def set_dts(self, dts: List[datetime]):
        self.dts = dts
        self.cursor_names = [dt.strftime("%Y/%m/%d %H:%M:%S") for dt in dts]

    def _redraw(self):
        self.canvas.axes.cla()  # Clear the canvas.
        lines = redraw_canvas(
            self.canvas,
            self.data,
            self.legend,
            self.cimel_data,
            self.asd_data,
            self.data_compare,
            self.title,
            self.xlabel,
            self.ylabel,
            self.vertical_lines,
            self.interp_spectrum_name,
            self.subtitle,
            self.compare_percentages,
        )
        try:
            self.canvas.fig.tight_layout()
        except:
            pass
        if self.cursor_names:
            cursor_lines = [
                l
                for l in lines
                if l.get_label().startswith("_child")
                or l.get_label() == self.legend[0][0]
            ]
            self.mpl_cursor = mplcursors.cursor(cursor_lines, hover=2)
            func_num_from_label = lambda label: int(int(label[6:]))
            if self.dts:
                func_num_from_label = lambda label: int(int(label[6:]) / 2)

            for l in cursor_lines:
                lab = l.get_label()
                if lab.startswith("_child"):
                    num = func_num_from_label(lab)
                    l.set_label(self.cursor_names[num])

            @self.mpl_cursor.connect("add")
            def _(sel):
                sel.annotation.get_bbox_patch().set(fc="white")
                label = sel.artist.get_label()
                if label.startswith("_child"):
                    num = func_num_from_label(label)
                    label = self.cursor_names[num]
                elif label == constants.INTERPOLATED_DATA_LABEL:
                    label = self.cursor_names[0]
                sel.annotation.set_text(label)

        self.canvas.axes.set_xlim(self.xlim_left, self.xlim_right)
        self.canvas.draw()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.critical(self, "ERROR", str(error))

    @QtCore.Slot()
    def export_graph(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export graph (.png, .jpg, .pdf...)", "{}.png".format(self.title)
        )[0]
        if name is not None and name != "":
            try:
                self.canvas.print_figure(name)
            except Exception as e:
                self.show_error(e)

    def set_inside_mpa_range(self, inside_mpa_range):
        self.inside_mpa_range = inside_mpa_range

    def set_interp_spectrum_name(self, interp_spectrum_name: str):
        self.interp_spectrum_name = interp_spectrum_name

    def set_skipped_uncertainties(self, skip: bool):
        self.skip_uncs = skip

    def set_srf_channel_names(self, ch_names: List[str]):
        self.ch_names = ch_names

    @QtCore.Slot()
    def export_csv(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format(self.title)
        )[0]
        version = self.settings_manager.get_coef_version_name()
        if name is not None and name != "":
            try:
                if isinstance(self.point, list):
                    csv.export_csv_comparison(
                        self.data,
                        self.ylabel,
                        self.point,
                        name,
                        version,
                        self.data_compare,
                        self.interp_spectrum_name,
                        self.skip_uncs,
                        self.comparison_x_datetime,
                    )
                elif self.inside_mpa_range is not None:
                    csv.export_csv_simulation(
                        self.data,
                        self.xlabel,
                        self.ylabel,
                        self.point,
                        name,
                        version,
                        self.inside_mpa_range,
                        self.interp_spectrum_name,
                        self.skip_uncs,
                        self.cimel_data,
                    )
                else:
                    csv.export_csv_srf(
                        self.data,
                        self.ch_names,
                        self.xlabel,
                        self.ylabel,
                        name,
                    )
            except Exception as e:
                self.show_error(e)


class SignalWidget(QtWidgets.QWidget):
    def __init__(self, settings_manager: ISettingsManager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.skip_uncs = None
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

    def set_skipped_uncertainties(self, skip: bool):
        self.skip_uncs = skip

    def update_signals(
        self,
        point: Point,
        srf: SpectralResponseFunction,
        signals: SpectralData,
        inside_mpa_range: Union[bool, List[bool]],
    ):
        self._clear_layout()
        show_range_wlens_info = False
        show_range_mpa_info = False
        self.srf = srf
        self.signals = signals
        self.inside_mpa_range = inside_mpa_range

        self.point = point
        head_id_item = QtWidgets.QTableWidgetItem("ID")
        head_center_item = QtWidgets.QTableWidgetItem("Center (nm)")
        self.table.setRowCount(1 + len(signals.data))
        if isinstance(point, CustomPoint):
            asterisk_if_mpa_out = ""
            if not inside_mpa_range:
                show_range_mpa_info = True
                asterisk_if_mpa_out = " **"
            self.table.setColumnCount(2 + 2)
            self.table.setItem(
                0,
                2,
                QtWidgets.QTableWidgetItem(
                    f"Irradiance (Wm⁻²nm⁻¹){asterisk_if_mpa_out}"
                ),
            )
            self.table.setItem(
                0, 3, QtWidgets.QTableWidgetItem(f"Uncertainties{asterisk_if_mpa_out}")
            )
        else:
            dts = point.dt
            if not isinstance(dts, list):
                dts = [dts]
                inside_mpa_range = [inside_mpa_range]
            self.table.setColumnCount(len(dts) * 2 + 2)
            for i, dt in enumerate(dts):
                asterisk_if_mpa_out = ""
                if not inside_mpa_range[i]:
                    show_range_mpa_info = True
                    asterisk_if_mpa_out = " **"
                item_title_value = QtWidgets.QTableWidgetItem(
                    "Irradiance (Wm⁻²nm⁻¹) on {}{}".format(
                        dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        asterisk_if_mpa_out,
                    )
                )
                item_title_uncert = QtWidgets.QTableWidgetItem(
                    f"Uncertainties{asterisk_if_mpa_out}"
                )
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
                    show_range_wlens_info = True
                else:
                    value = "Not available *"
                    unc = "Not available *"
                    show_range_wlens_info = True
                value_item = QtWidgets.QTableWidgetItem(value)
                unc_item = QtWidgets.QTableWidgetItem(unc)
                self.table.setItem(i + 1, j * 2 + 2, value_item)
                self.table.setItem(i + 1, j * 2 + 3, unc_item)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        warning_msg = ""
        if show_range_wlens_info:
            warning_msg = "* The LIME can only give a reliable simulation \
for wavelengths between 350 and 2500 nm\n"
        if show_range_mpa_info:
            warning_msg += "** The LIME can only give a reliable simulation \
for absolute moon phase angles between 2° and 90°"
        if show_range_wlens_info or show_range_mpa_info:
            self.range_warning = QtWidgets.QLabel(warning_msg)
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

    def set_interp_spectrum_name(self, interp_spectrum_name: str):
        self.interp_spectrum_name = interp_spectrum_name

    @QtCore.Slot()
    def export_csv(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format("Signal")
        )[0]
        self.parentWidget().setDisabled(True)
        self.disable_buttons(True)
        version = self.settings_manager.get_coef_version_name()
        if name is not None and name != "":
            try:
                csv.export_csv_integrated_irradiance(
                    self.srf,
                    self.signals,
                    name,
                    self.point,
                    version,
                    self.inside_mpa_range,
                    self.interp_spectrum_name,
                    self.skip_uncs,
                )
            except Exception as e:
                self.show_error(e)
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)


class ComparisonDualGraphWidget(QtWidgets.QWidget):
    def __init__(self, graph_reldif: GraphWidget, graph_percdif: GraphWidget):
        super().__init__()
        self.graph_reldif = graph_reldif
        self.graph_percdif = graph_percdif
        self.stack_layout = QtWidgets.QStackedLayout(self)
        self.stack_layout.setStackingMode(QtWidgets.QStackedLayout.StackAll)
        self.stack_layout.addWidget(self.graph_reldif)
        self.stack_layout.addWidget(self.graph_percdif)
        self.stack_layout.setCurrentIndex(0)
        self.graph_reldif.setVisible(True)
        self.graph_percdif.setVisible(False)

    def tight_layout(self):
        self.graph_percdif.canvas.fig.tight_layout()
        self.graph_reldif.canvas.fig.tight_layout()

    def show_percentage(self):
        self.graph_reldif.setVisible(False)
        self.graph_percdif.setVisible(True)
        self.stack_layout.setCurrentIndex(1)
        self.graph_percdif.canvas.fig.tight_layout()

    def show_relative(self):
        self.graph_reldif.setVisible(True)
        self.graph_percdif.setVisible(False)
        self.stack_layout.setCurrentIndex(0)
        self.graph_reldif.canvas.fig.tight_layout()

    def clear(self):
        self.graph_reldif.setParent(None)
        self.graph_percdif.setParent(None)
        self.graph_reldif = None
        self.graph_percdif = None

    def update_plot(self, comparison: ComparisonData, redraw: bool = True):
        data = [comparison.observed_signal, comparison.simulated_signal]
        self.graph_reldif.update_plot(
            data, point=comparison.points, data_compare=comparison, redraw=redraw
        )
        self.graph_percdif.update_plot(
            data,
            point=comparison.points,
            data_compare=comparison,
            redraw=redraw,
            compare_percentages=True,
        )

    def update_labels(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        redraw: bool = True,
        subtitle: str = None,
    ):
        self.graph_reldif.update_labels(
            title, xlabel, ylabel, redraw=redraw, subtitle=subtitle
        )
        self.graph_percdif.update_labels(
            title, xlabel, ylabel, redraw=redraw, subtitle=subtitle
        )

    def set_interp_spectrum_name(self, sp_name: str):
        self.graph_reldif.set_interp_spectrum_name(sp_name)
        self.graph_percdif.set_interp_spectrum_name(sp_name)

    def set_skipped_uncertainties(self, skip: bool):
        self.graph_reldif.set_skipped_uncertainties(skip)
        self.graph_percdif.set_skipped_uncertainties(skip)

    def update_legends(self, legends: List[List[str]], redraw: bool = True):
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
        redraw: bool
            Boolean that defines if the plot will be redrawn automatically or not. Default True.
        """
        self.graph_reldif.update_legend(legends, redraw=redraw)
        self.graph_percdif.update_legend(legends, redraw=redraw)


class ComparisonOutput(QtWidgets.QWidget):
    def __init__(self, settings_manager: ISettingsManager, x_datetime: bool):
        super().__init__()
        self.settings_manager = settings_manager
        self.channels: List[ComparisonDualGraphWidget] = []
        self.ch_names = []
        self.x_datetime = x_datetime
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.channel_tabs = QtWidgets.QTabWidget()
        self.channel_tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        self.main_layout.addWidget(self.channel_tabs)
        self.range_warning = QtWidgets.QLabel("")
        self.range_warning.setWordWrap(True)
        self.main_layout.addWidget(self.range_warning)

    def set_channels(self, channels: List[str]):
        while self.channel_tabs.count() > 0:
            self.channel_tabs.removeTab(0)
        for ch in self.channels:
            ch.clear()
            ch.setParent(None)
        self.channels.clear()
        self.ch_names = []
        for ch in channels:
            grel = GraphWidget(
                self.settings_manager, ch, comparison_x_datetime=self.x_datetime
            )
            gperc = GraphWidget(
                self.settings_manager, ch, comparison_x_datetime=self.x_datetime
            )
            channel = ComparisonDualGraphWidget(grel, gperc)
            self.channels.append(channel)
            self.ch_names.append(ch)
            self.channel_tabs.addTab(channel, ch)
        # Remove range warning content
        if self.range_warning:
            self.range_warning.setText("")
        # if self.range_warning:
        #    self.range_warning.setParent(None)
        #    self.range_warning = None

    def set_as_partly(self, ch_name: str):
        if ch_name in self.ch_names:
            index = self.ch_names.index(ch_name)
            self.channel_tabs.setTabText(index, "{} *".format(ch_name))
            msg = "* The LIME can only give a reliable simulation \
for wavelengths between 350 and 2500 nm"
            if self.range_warning == None:
                self.range_warning = QtWidgets.QLabel(msg)
                self.range_warning.setWordWrap(True)
                self.main_layout.addWidget(self.range_warning)
            else:
                self.range_warning.setText(msg)

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
                self.channels[index].clear()
                self.channels[index].setParent(None)
                self.channels.pop(index)
                self.ch_names.pop(index)
        self._check_range_warning_needed()

    def show_relative(self):
        for ch in self.channels:
            ch.show_relative()

    def show_percentage(self):
        for ch in self.channels:
            ch.show_percentage()

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
        self.channels[index].update_plot(comparison, redraw)

    def update_labels(
        self,
        index: int,
        title: str,
        xlabel: str,
        ylabel: str,
        redraw: bool = True,
        subtitle: str = None,
    ):
        self.channels[index].update_labels(
            title, xlabel, ylabel, redraw=redraw, subtitle=subtitle
        )

    def set_interp_spectrum_name(
        self,
        index: int,
        sp_name: str,
    ):
        self.channels[index].set_interp_spectrum_name(sp_name)

    def set_skipped_uncertainties(
        self,
        index: int,
        skip: bool,
    ):
        self.channels[index].set_skipped_uncertainties(skip)

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
        self.channels[index].update_legends(legends, redraw=redraw)

    def get_current_channel_index(self) -> int:
        return self.channel_tabs.currentIndex()

    def set_current_channel_index(self, index: int):
        cui = self.channel_tabs.setCurrentIndex(index)
        self.channels[index].tight_layout()
        return cui
