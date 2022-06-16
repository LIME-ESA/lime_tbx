"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List, Tuple

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import numpy as np

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    Point,
    SatellitePoint,
    SpectralResponseFunction,
    SpectralValidity,
    SurfacePoint,
    CustomPoint,
    SpectralData,
)
from ..filedata import csv

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


class GraphWidget(QtWidgets.QWidget):
    def __init__(self, title="", xlabel="", ylabel=""):
        super().__init__()
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = []
        self.data = None
        self.cimel_data = None
        self.asd_data = None
        self.point = None
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # canvas
        self.canvas = MplCanvas(self)
        self.canvas.axes.set_title(self.title)
        self.canvas.axes.set_xlabel(self.xlabel)
        self.canvas.axes.set_ylabel(self.ylabel)
        self.toolbar = NavigationToolbar(self.canvas, self)
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

    def disable_buttons(self, disable: bool):
        self.export_button.setDisabled(disable)
        self.csv_button.setDisabled(disable)

    def update_plot(
        self,
        data: Union[SpectralData, List[SpectralData]] = None,
        data_cimel: Union[SpectralData, List[SpectralData]] = None,
        data_asd: Union[SpectralData, List[SpectralData]] = None,
        point: Union[Point, List[Point]] = None,
    ):
        self.point = point
        self.data = data
        self.cimel_data = data_cimel
        self.asd_data = data_asd
        if data is not None:
            self.disable_buttons(False)
        else:
            self.disable_buttons(True)
        self._redraw()

    def update_labels(self, title: str, xlabel: str, ylabel: str):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._redraw()

    def update_legend(self, legend: List[List[str]]):
        self.legend = legend
        self._redraw()

    def update_size(self):
        self._redraw()

    def _redraw(self):
        self.canvas.axes.cla()  # Clear the canvas.
        if self.data is not None:
            iter_data = self.data
            if not isinstance(iter_data, list):
                iter_data = [iter_data]
            for i, data in enumerate(iter_data):
                label = ""
                color = ["g"]
                if len(self.legend) > 0:
                    if len(self.legend[0]) > i:
                        label = self.legend[0][i]
                    if len(self.legend[0]) > 1:
                        color = []
                self.canvas.axes.plot(
                    data.wlens,
                    data.data,
                    *color,
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

            if self.asd_data:
                if isinstance(self.asd_data, list):
                    asd_data = self.asd_data[0]
                else:
                    asd_data = self.asd_data
                self.canvas.axes.plot(
                    asd_data.wlens,
                    asd_data.data / 5.0,
                    label="ASD data points / 5",
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
                    self.canvas.axes.plot(
                        cimel_data.wlens,
                        cimel_data.data,
                        color="orange",
                        ls="none",
                        marker="o",
                        label=label0,
                    )
                    self.canvas.axes.errorbar(
                        cimel_data.wlens,
                        cimel_data.data,
                        yerr=cimel_data.uncertainties * 2,
                        color="black",
                        capsize=3,
                        ls="none",
                        label=label1,
                    )
            if len(self.legend) > 0:
                self.canvas.axes.legend()

        self.canvas.axes.set_title(self.title)
        self.canvas.axes.set_xlabel(self.xlabel)
        self.canvas.axes.set_ylabel(self.ylabel)
        try:
            self.canvas.fig.tight_layout()
        except:
            pass
        self.canvas.draw()

    def show_error(self, error: Exception):
        error_dialog = QtWidgets.QErrorMessage(self)
        error_dialog.showMessage(str(error))
        raise error

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
        if isinstance(self.data, np.ndarray) or isinstance(self.data, list):
            x_data = self.data[0].wlens
            y_data = [d.data for d in self.data]
        else:
            x_data = self.data.wlens
            y_data = self.data.data
        if name is not None and name != "":
            try:
                if isinstance(self.point, list):
                    csv.export_csv_comparation(
                        x_data,
                        y_data,
                        self.xlabel,
                        self.ylabel,
                        self.point,
                        name,
                    )
                else:
                    csv.export_csv(
                        x_data,
                        y_data,
                        self.xlabel,
                        self.ylabel,
                        self.point,
                        name,
                    )
            except Exception as e:
                self.show_error(e)
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)


class SignalWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.irrs = signals.data

        self.point = point
        head_id_item = QtWidgets.QTableWidgetItem("ID")
        head_center_item = QtWidgets.QTableWidgetItem("Center (nm)")
        self.table.setRowCount(1 + len(signals.data))
        if isinstance(point, CustomPoint):
            self.table.setColumnCount(2 + 1)
            self.table.setItem(0, 2, QtWidgets.QTableWidgetItem("Signal (Wm⁻²nm⁻¹)"))
        else:
            dts = point.dt
            if not isinstance(dts, list):
                dts = [dts]
            self.table.setColumnCount(len(dts) + 2)
            for i, dt in enumerate(dts):
                item_title_value = QtWidgets.QTableWidgetItem(
                    "Signal (Wm⁻²nm⁻¹) on {}".format(
                        dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    )
                )
                self.table.setItem(0, i + 2, item_title_value)
        self.table.setItem(0, 0, head_id_item)
        self.table.setItem(0, 1, head_center_item)
        print(len(srf.channels), len(signals.data))
        for i, ch_signals in enumerate(signals.data):
            ch = srf.channels[i]
            if not (isinstance(ch_signals, np.ndarray) or isinstance(ch_signals, list)):
                ch_signals = [ch_signals]
            id_item = QtWidgets.QTableWidgetItem(str(ch.id))
            center_item = QtWidgets.QTableWidgetItem(str(ch.center))
            self.table.setItem(i + 1, 0, id_item)
            self.table.setItem(i + 1, 1, center_item)
            for j, signal in enumerate(ch_signals):
                if ch.valid_spectre == SpectralValidity.VALID:
                    value = "{}".format(str(signal))
                elif ch.valid_spectre == SpectralValidity.PARTLY_OUT:
                    value = "{} *".format(str(signal))
                    show_range_info = True
                else:
                    value = "Not available *"
                    show_range_info = True
                value_item = QtWidgets.QTableWidgetItem(value)
                self.table.setItem(i + 1, j + 2, value_item)
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
        error_dialog = QtWidgets.QErrorMessage(self)
        error_dialog.showMessage(str(error))
        raise error

    @QtCore.Slot()
    def export_csv(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format("Signal")
        )[0]
        self.parentWidget().setDisabled(True)
        self.disable_buttons(True)
        if name is not None and name != "":
            try:
                csv.export_csv_integrated_irradiance(
                    self.srf, self.irrs, name, self.point
                )
            except Exception as e:
                self.show_error(e)
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)


class ComparisonOutput(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.channels: List[GraphWidget] = []
        self.ch_names = []
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.channel_tabs = QtWidgets.QTabWidget()
        self.channel_tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        self.main_layout.addWidget(self.channel_tabs)

    def set_channels(self, channels: List[str]):
        while self.channel_tabs.count() > 0:
            self.channel_tabs.removeTab(0)
        for ch in self.channels:
            ch.setParent(None)
        self.channels.clear()
        self.ch_names = []
        for ch in channels:
            channel = GraphWidget(ch)
            self.channels.append(channel)
            self.ch_names.append(ch)
            self.channel_tabs.addTab(channel, ch)

    def remove_channels(self, channels: List[str]):
        for ch_name in channels:
            if ch_name in self.ch_names:
                index = self.ch_names.index(ch_name)
                self.channel_tabs.removeTab(index)
                self.channels[index].setParent(None)
                self.channels.pop(index)
                self.ch_names.pop(index)

    def update_plot(
        self, index: int, data: Union[SpectralData, List[SpectralData]], points: list
    ):
        self.channels[index].update_plot(data, point=points)

    def update_labels(self, index: int, title: str, xlabel: str, ylabel: str):
        self.channels[index].update_labels(title, xlabel, ylabel)

    def update_legends(self, index: int, legends: List[List[str]]):
        self.channels[index].update_legend(legends)
