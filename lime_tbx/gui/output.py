"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List
from datetime import datetime

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
import numpy as np
import mplcursors

"""___LIME_TBX Modules___"""
from ..datatypes.datatypes import (
    ComparisonData,
    Point,
    SpectralResponseFunction,
    SpectralValidity,
    CustomPoint,
    SpectralData,
)
from lime_tbx.gui.settings import ISettingsManager
from ..filedata import csv
from .ifaces import IMainSimulationsWidget
from .canvas import (
    MplCanvas,
    title_font_prop,
    label_font_prop,
    font_prop,
    redraw_canvas,
)

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
        self.mpl_cursor = None
        self.xlim_left = None
        self.xlim_right = None
        self.comparison_x_datetime = comparison_x_datetime
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # canvas
        self.canvas = MplCanvas(self)
        self.canvas.set_title(self.title, fontproperties=title_font_prop)
        self.canvas.axes.tick_params(labelsize=8)
        version = self.settings_manager.get_cimel_coef().version
        subtitle = "LIME2 coefficients version: {}".format(version)
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

    def set_dts(self, dts: List[datetime]):
        self.dts = dts

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
            self.subtitle,
        )
        try:
            self.canvas.fig.tight_layout()
        except:
            pass
        if self.dts:
            cursor_lines = [
                l
                for l in lines
                if l.get_label().startswith("_child")
                or l.get_label() == self.legend[0][0]
            ]
            self.mpl_cursor = mplcursors.cursor(cursor_lines, hover=2)

            @self.mpl_cursor.connect("add")
            def _(sel):
                sel.annotation.get_bbox_patch().set(fc="white")
                label = sel.artist.get_label()
                num = 0
                if label.startswith("_child"):
                    num = int(int(label[6:]) / 2)
                dt: datetime = self.dts[num]
                label = dt.strftime("%Y/%m/%d %H:%M:%S")
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
                        self.comparison_x_datetime,
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
            self.table.setItem(
                0, 2, QtWidgets.QTableWidgetItem("Irradiance (Wm⁻²nm⁻¹)")
            )
            self.table.setItem(0, 3, QtWidgets.QTableWidgetItem("Uncertainties"))
        else:
            dts = point.dt
            if not isinstance(dts, list):
                dts = [dts]
            self.table.setColumnCount(len(dts) * 2 + 2)
            for i, dt in enumerate(dts):
                item_title_value = QtWidgets.QTableWidgetItem(
                    "Irradiance (Wm⁻²nm⁻¹) on {}".format(
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
    def __init__(self, settings_manager: ISettingsManager, x_datetime: bool):
        super().__init__()
        self.settings_manager = settings_manager
        self.channels: List[GraphWidget] = []
        self.ch_names = []
        self.x_datetime = x_datetime
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
            channel = GraphWidget(
                self.settings_manager, ch, comparison_x_datetime=self.x_datetime
            )
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

    def get_current_channel_index(self) -> int:
        return self.channel_tabs.currentIndex()

    def set_current_channel_index(self, index: int):
        return self.channel_tabs.setCurrentIndex(index)
