"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

"""___NPL Modules___"""
from ..datatypes.datatypes import (
    SatellitePoint,
    SpectralResponseFunction,
    SpectralValidity,
    SurfacePoint,
    CustomPoint,
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
        self.x_data = []
        self.y_data = []
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
        # error message
        self.error_message = QtWidgets.QLabel("")
        self.error_message.setWordWrap(True)
        self.error_message.hide()
        # finish main
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas, 1)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addWidget(self.error_message)

    def disable_buttons(self, disable: bool):
        self.export_button.setDisabled(disable)
        self.csv_button.setDisabled(disable)

    def update_plot(
        self,
        x_data: list,
        y_data: list,
        point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.point = point
        if len(x_data) > 0 and len(y_data) > 0:
            self.disable_buttons(False)
        else:
            self.disable_buttons(True)
        self._redraw()

    def update_labels(self, title: str, xlabel: str, ylabel: str):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._redraw()

    def update_size(self):
        self._redraw()

    def _redraw(self):
        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(self.x_data, self.y_data)
        self.canvas.axes.set_title(self.title)
        self.canvas.axes.set_xlabel(self.xlabel)
        self.canvas.axes.set_ylabel(self.ylabel)
        try:
            self.canvas.fig.tight_layout()
        except:
            pass
        self.canvas.draw()

    def clear_error(self):
        self.error_message.hide()

    def show_error(self, msg: str):
        color_red = "#c70000"
        self.error_message.setText(msg)
        self.error_message.setStyleSheet("background-color: {}".format(color_red))
        self.error_message.repaint()
        self.error_message.show()

    @QtCore.Slot()
    def export_graph(self):
        self.clear_error()
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export graph (.png, .jpg, .pdf...)", "{}.png".format(self.title)
        )[0]
        self.parentWidget().setDisabled(True)
        self.disable_buttons(True)
        if name is not None and name != "":
            try:
                self.canvas.print_figure(name)
            except Exception as e:
                self.show_error(str(e))
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)

    @QtCore.Slot()
    def export_csv(self):
        self.clear_error()
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format(self.title)
        )[0]
        self.parentWidget().setDisabled(True)
        self.disable_buttons(True)
        if name is not None and name != "":
            try:
                csv.export_csv(
                    self.x_data, self.y_data, self.xlabel, self.ylabel, self.point, name
                )
            except Exception as e:
                self.show_error(str(e))
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
        # error message
        self.error_message = QtWidgets.QLabel("")
        self.error_message.setWordWrap(True)
        self.error_message.hide()
        # csv button
        self.button_csv = QtWidgets.QPushButton("Export CSV")
        self.button_csv.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button_csv.clicked.connect(self.export_csv)

        self.container_layout.addLayout(self.data_layout)
        self.container_layout.addStretch()
        self.container_layout.addWidget(self.error_message)
        self.container_layout.addWidget(self.button_csv)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.groupbox)
        self.main_layout.addWidget(self.scroll_area)
        self.disable_buttons(True)

    def _clear_layout(self):
        for i in reversed(range(self.data_layout.count())):
            self.data_layout.itemAt(i).widget().setParent(None)
        if self.range_warning:
            self.range_warning.setParent(None)
            self.range_warning = None
        self.disable_buttons(True)

    def update_signals(
        self,
        signals: List[float],
        srf: SpectralResponseFunction,
        point: Union[SurfacePoint, CustomPoint, SatellitePoint],
    ):
        self._clear_layout()
        show_range_info = False
        self.srf = srf
        self.irrs = signals
        self.point = point
        for i, signal in enumerate(signals):
            ch = srf.channels[i]
            title = QtWidgets.QLabel("{} ({} nm):".format(ch.id, ch.center))
            if ch.valid_spectre == SpectralValidity.VALID:
                value = "{} Wm⁻²nm⁻¹".format(str(signal))
            elif ch.valid_spectre == SpectralValidity.PARTLY_OUT:
                value = "{} Wm⁻²nm⁻¹ *".format(str(signal))
                show_range_info = True
            else:
                value = "Not available *"
                show_range_info = True
            value_label = QtWidgets.QLabel(value)
            self.data_layout.addRow(title, value_label)
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

    def clear_error(self):
        self.error_message.hide()

    def show_error(self, msg: str):
        color_red = "#c70000"
        self.error_message.setText(msg)
        self.error_message.setStyleSheet("background-color: {}".format(color_red))
        self.error_message.repaint()
        self.error_message.show()

    @QtCore.Slot()
    def export_csv(self):
        self.clear_error()
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
                self.show_error(str(e))
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)
