"""describe class"""

"""___Built-In Modules___"""
from typing import Union, List, Tuple
from datetime import datetime
from abc import ABC, abstractmethod


"""___Third-Party Modules___"""
from qtpy import QtWidgets, QtCore, QtGui

if QtCore.__version__.startswith("6"):  # Qt6 specific code
    from qtpy.QtGui import QAction
else:  # Qt5 specific code
    from qtpy.QtWidgets import QAction
import matplotlib.backends.backend_pdf  # important import for exporting as pdf
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
import numpy as np
import mplcursors

"""___LIME_TBX Modules___"""
from lime_tbx.common.datatypes import (
    ComparisonData,
    Point,
    SpectralResponseFunction,
    SpectralValidity,
    CustomPoint,
    SpectralData,
    MoonData,
)
from lime_tbx.common import logger
from lime_tbx.presentation.gui.settings import ISettingsManager
from lime_tbx.application.filedata import csv
from lime_tbx.presentation.gui.ifaces import IMainSimulationsWidget, noconflict_makecls
from lime_tbx.presentation.gui.canvas import (
    MplCanvas,
    title_font_prop,
    label_font_prop,
    font_prop,
    redraw_canvas,
    redraw_canvas_compare,
    redraw_canvas_compare_only_diffs,
    redraw_canvas_compare_boxplot,
    redraw_canvas_compare_boxplot_only_diffs,
)
from lime_tbx.presentation.gui import constants
from lime_tbx.common.constants import CompFields
from lime_tbx.application.simulation.comparison.utils import average_comparisons

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class GraphWidget(QtWidgets.QWidget, ABC, metaclass=noconflict_makecls()):
    def _build_layout(self):
        self.is_built = True
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # canvas
        self.canvas = MplCanvas(self)
        self.canvas.set_title(self.title, fontproperties=title_font_prop)
        self.canvas.axes.tick_params(labelsize=8)
        version = self.settings_manager.get_coef_version_name()
        subtitle = "LIME coefficients version: {}".format(version)
        self.subtitle = subtitle
        self.extra_attrs = None
        self.canvas.set_subtitle(subtitle, fontproperties=font_prop)
        self.canvas.axes.set_xlabel(self.xlabel, fontproperties=label_font_prop)
        self.canvas.axes.set_ylabel(self.ylabel, fontproperties=label_font_prop)
        self._prepare_toolbar()
        self._redraw()
        self.disable_buttons(True)
        # finish main
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas, 1)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.canvas.updateGeometry()
        if self._to_update_plot:
            self._update_plot()
        if self._to_update_labels:
            self._update_labels()

    @abstractmethod
    def _redraw(self):
        pass

    def disable_buttons(self, disable: bool):
        self.graph_action.setDisabled(disable)
        self.csv_action.setDisabled(disable)

    def _update_plot(self, redraw=True):
        if self.data is not None:
            self.disable_buttons(False)
        else:
            self.disable_buttons(True)
        if redraw:
            self._redraw()
            self._update_toolbar()

    def showEvent(self, event):
        if not self.is_built:
            self._build_layout()
        super().showEvent(event)

    def _prepare_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self)
        unwanted_buttons = ["Back", "Forward", "Save"]
        for ta in self.toolbar.actions():
            ta: QAction = ta
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
        # Add Export graph and export csv actions
        self.graph_action = QAction(
            "Export\nGraph",
            parent=self.toolbar,
            enabled=False,
            toolTip="Save Figure as a graph (.png, .jpg, .pdf...)",
        )
        self.graph_action.triggered.connect(self.export_graph)
        self.toolbar.insertAction(self.toolbar.actions()[-1], self.graph_action)
        self.csv_action = QAction(
            "Export\nCSV",
            parent=self.toolbar,
            enabled=False,
            toolTip="Export as a CSV file",
        )
        self.csv_action.triggered.connect(self.export_csv)
        self.toolbar.insertAction(self.toolbar.actions()[-1], self.csv_action)
        # Set all toolbar actions with Pointing Hand cursor when hover
        for widg in self.toolbar.findChildren(QtWidgets.QWidget):
            if (
                hasattr(widg, "defaultAction")
                and widg.defaultAction() in self.toolbar.actions()
            ):
                widg.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def set_vertical_lines(self, xs: List[float]):
        self.vertical_lines = xs
        self._redraw()

    def set_xlim(self, left: float = None, right: float = None):
        self.xlim_left = left
        self.xlim_right = right

    def set_max_ylims(self, bottom: float = None, top: float = None):
        self.max_ylim_bottom = bottom
        self.max_ylim_top = top

    def _update_toolbar(self):
        self.toolbar.update()

    def update_labels(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        redraw: bool = True,
        subtitle: str = None,
        extra_attrs: List[Tuple[str, str]] = None,
    ):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        if subtitle != None:
            self.subtitle = subtitle
        self.extra_attrs = extra_attrs
        if self.is_built:
            self._update_labels(redraw)
        else:
            self._to_update_labels = True

    def _update_labels(self, redraw=True):
        if redraw:
            self._redraw()
        self._update_toolbar()

    def update_size(self):
        self._redraw()

    def clear(self):
        if self.is_built:
            self.canvas.clear()

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

    @abstractmethod
    def update_legend(self, legend: List[List[str]], redraw: bool = True):
        pass

    @abstractmethod
    @QtCore.Slot()
    def export_csv(self):
        pass


class SimGraphWidget(GraphWidget):
    def __init__(
        self,
        settings_manager: ISettingsManager,
        title="",
        xlabel="",
        ylabel="",
        parent=None,
        build_layout_ini=True,
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
        self.vertical_lines = []
        self.dts = []
        self.cursor_names = []
        self.mpl_cursor = None
        self.xlim_left = None
        self.xlim_right = None
        self.max_ylim_bottom = None
        self.max_ylim_top = None
        self.inside_mpa_range = None
        self.interp_spectrum_name = None
        self.mda = None
        self.skip_uncs = None
        self.ch_names = []
        self.is_built = False
        self._to_update_plot = False
        self._to_update_labels = False
        if build_layout_ini:
            self._build_layout()

    def disable_buttons(self, disable: bool):
        super().disable_buttons(disable)
        if self._init_parent and isinstance(self._init_parent, IMainSimulationsWidget):
            self._init_parent.set_export_button_disabled(disable)

    def update_plot(
        self,
        data: Union[SpectralData, List[SpectralData]] = None,
        data_cimel: Union[SpectralData, List[SpectralData]] = None,
        data_asd: Union[SpectralData, List[SpectralData]] = None,
        point: Union[Point, List[Point]] = None,
        redraw: bool = True,
    ):
        self.point = point
        self.data = data
        self.cimel_data = data_cimel
        self.asd_data = data_asd
        if self.is_built:
            self._update_plot(redraw)
        else:
            self._to_update_plot = True

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
        """
        self.legend = legend
        if self.is_built:
            if redraw:
                self._redraw()
            self._update_toolbar()
        else:
            self._to_update_labels = True

    def set_cursor_names(self, labels: List[str]):
        self.dts = None
        self.cursor_names = labels.copy()

    def set_dts(self, dts: List[datetime]):
        # TODO: Refactor this, change func name and combine with set_cursor_names
        self.dts = dts
        self.cursor_names = [dt.strftime("%Y/%m/%d %H:%M:%S") for dt in dts]

    def _redraw(self):
        if not self.is_built:
            return
        self.canvas.axes.cla()  # Clear the canvas.
        subtitle = self.subtitle
        if self.extra_attrs:
            subtitle = (
                subtitle
                + "\n"
                + "\n".join(f"{ea[0]}: {ea[1]}" for ea in self.extra_attrs)
            )
        lines = redraw_canvas(
            self.canvas,
            self.data,
            self.legend,
            self.cimel_data,
            self.asd_data,
            self.title,
            self.xlabel,
            self.ylabel,
            self.vertical_lines,
            self.interp_spectrum_name,
            subtitle,
            self.settings_manager.is_show_cimel_points(),
        )
        try:
            self.canvas.fig.tight_layout()
            self.canvas.draw()
        except:
            pass
        if self.cursor_names:
            cursor_lines = [
                l
                for l in lines
                if l.get_label().startswith("_child")
                or l.get_label() == self.legend[0][0]
            ]
            max_cursors = 25
            if len(cursor_lines) > max_cursors:
                cursor_lines = np.array(cursor_lines)[
                    np.round(np.linspace(0, len(cursor_lines) - 1, max_cursors)).astype(
                        int
                    )
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
        bottom = top = None
        current_bottom, current_top = self.canvas.axes.get_ylim()
        if self.max_ylim_bottom and self.max_ylim_bottom > current_bottom:
            bottom = self.max_ylim_bottom
        if self.max_ylim_top and self.max_ylim_top < current_top:
            top = self.max_ylim_top
        self.canvas.axes.set_ylim(bottom, top)
        self.canvas.draw()
        self.update()
        self.canvas.update()

    def set_inside_mpa_range(self, inside_mpa_range):
        self.inside_mpa_range = inside_mpa_range

    def set_interp_spectrum_name(self, interp_spectrum_name: str):
        self.interp_spectrum_name = interp_spectrum_name

    def set_mda(self, mda: Union[List[MoonData], MoonData, None]):
        self.mda = mda

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
                if self.inside_mpa_range is not None:
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
                        self.mda,
                        self.extra_attrs,
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

    def recycle(self, title: str):
        self.title = title
        if self.is_built:
            self.canvas.axes.cla()


class CompGraphWidget(GraphWidget):
    def __init__(
        self,
        settings_manager: ISettingsManager,
        title="",
        xlabel="",
        ylabel="",
        chosen_diffs=CompFields.DIFF_REL,
        parent=None,
        build_layout_ini=True,
    ):
        super().__init__(parent)
        self._init_parent = parent
        self.settings_manager = settings_manager
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = []
        self.data = None
        self.vertical_lines = []
        self.dts = []
        self.cursor_names = []
        self.mpl_cursor = None
        self.xlim_left = None
        self.xlim_right = None
        self.max_ylim_bottom = None
        self.max_ylim_top = None
        self.interp_spectrum_name = None
        self.skip_uncs = None
        self.is_built = False
        self._to_update_plot = False
        self._to_update_labels = False
        self.chosen_diffs = chosen_diffs
        if build_layout_ini:
            self._build_layout()

    def update_plot(
        self,
        data_compare: ComparisonData = None,
        redraw: bool = True,
        chosen_diffs: CompFields = CompFields.DIFF_REL,
    ):
        self.data = data_compare
        self.chosen_diffs = chosen_diffs
        if self.is_built:
            self._update_plot(redraw)
        else:
            self._to_update_plot = True

    def update_legend(self, legend: List[List[str]], redraw: bool = True):
        """
        Parameters
        ----------
        legend: list of list of str
            Each list represents a group of legends
            Lengeds index:
            0: comparison
        """
        self.legend = legend
        if self.is_built:
            if redraw:
                self._redraw()
            self._update_toolbar()
        else:
            self._to_update_labels = True

    def _redraw_canvas_compare(self) -> list:
        return redraw_canvas_compare(
            self.canvas,
            self.data,
            self.legend,
            self.title,
            self.xlabel,
            self.ylabel,
            self.subtitle,
            self.chosen_diffs,
        )

    def _redraw_canvas_compare_only_diffs(self):
        return redraw_canvas_compare_only_diffs(
            self.canvas,
            self.data,
            self.subtitle,
            self.chosen_diffs,
        )

    def _redraw(self):
        if not self.is_built:
            return
        self.canvas.clear()  # Clear the canvas.
        lines = self._redraw_canvas_compare()
        try:
            self.canvas.fig.tight_layout()
            self.canvas.draw()
        except:
            pass

        xll, xlr = self.xlim_left, self.xlim_right
        if self.data and isinstance(self.data, ComparisonData):
            xmin = self.data.observed_signal.wlens.min()
            xmax = self.data.observed_signal.wlens.max()
            xmargin = (xmax - xmin) * 0.05
            if not xll:
                xll = xmin - xmargin
            if not xlr:
                xlr = xmax + xmargin
        self.canvas.axes.set_xlim(xll, xlr)
        bottom = top = None
        current_bottom, current_top = self.canvas.axes.get_ylim()
        if self.max_ylim_bottom and self.max_ylim_bottom > current_bottom:
            bottom = self.max_ylim_bottom
        if self.max_ylim_top and self.max_ylim_top < current_top:
            top = self.max_ylim_top
        self.canvas.axes.set_ylim(bottom, top)
        self.canvas.draw()
        self.update()
        self.canvas.update()

    def change_diff_canvas(self, chosen_diffs: CompFields):
        self.chosen_diffs = chosen_diffs
        if self.is_built:
            self._redraw_canvas_compare_only_diffs()
            try:
                self.canvas.fig.tight_layout()
                self.canvas.draw()
            except Exception as e:
                logger.get_logger().error(e)
        else:
            self._to_update_plot = True

    def set_interp_spectrum_name(self, interp_spectrum_name: str):
        self.interp_spectrum_name = interp_spectrum_name

    def set_skipped_uncertainties(self, skip: bool):
        self.skip_uncs = skip

    @QtCore.Slot()
    def export_csv(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format(self.title)
        )[0]
        version = self.settings_manager.get_coef_version_name()
        if name is not None and name != "":
            try:
                csv.export_csv_comparison(
                    self.data,
                    self.xlabel,
                    self.legend[0],
                    name,
                    version,
                    self.interp_spectrum_name,
                    self.skip_uncs,
                    self.chosen_diffs,
                )
            except Exception as e:
                self.show_error(e)

    def recycle(self, title: str):
        self.title = title
        self.clear()

    def showEvent(self, event):
        super().showEvent(event)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.tight_layout()

    def _on_resize(self, event):
        self.tight_layout()

    def tight_layout(self):
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        self.update()


class SignalWidget(QtWidgets.QWidget):
    def __init__(self, settings_manager: ISettingsManager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.skip_uncs = None
        self.mpa = None
        self.interp_spectrum_name = None
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

    def set_mpa(self, mpa: Union[float, None]):
        self.mpa = mpa

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
                    self.mpa,
                )
            except Exception as e:
                self.show_error(e)
        self.disable_buttons(False)
        self.parentWidget().setDisabled(False)


class ComparisonOutput(QtWidgets.QWidget):
    def __init__(self, settings_manager: ISettingsManager):
        super().__init__()
        self.settings_manager = settings_manager
        self.channels: List[CompGraphWidget] = []
        self.ch_names = []
        self.chosen_diffs = CompFields.DIFF_REL
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
        # TODO make this function faster
        new_channel_tabs = QtWidgets.QTabWidget()
        new_channel_tabs.tabBar().setCursor(QtCore.Qt.PointingHandCursor)
        self.main_layout.replaceWidget(self.channel_tabs, new_channel_tabs)
        self.channel_tabs.setParent(None)
        self.channel_tabs.deleteLater()
        self.channel_tabs = new_channel_tabs

        self.ch_names = []
        build_layout_ini = len(channels) < 15
        for i, ch in enumerate(channels):
            if len(self.channels) > i:
                channel = self.channels[i]
                channel.recycle(
                    ch,
                )
            else:
                channel = CompGraphWidget(
                    self.settings_manager,
                    ch,
                    build_layout_ini=build_layout_ini,
                )
                self.channels.append(channel)
            self.ch_names.append(ch)
            self.channel_tabs.addTab(channel, ch)
        # Remove unused channels
        for _ in range(len(channels), len(self.channels)):
            ch = self.channels.pop(len(channels))
            ch.setParent(None)
            ch.deleteLater()
        # Remove range warning content
        if self.range_warning:
            self.range_warning.setText("")

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

    def check_if_range_visible(self):
        if self.range_warning.text() != "":
            self.range_warning.setVisible(True)
        else:
            self.range_warning.setVisible(False)

    def _check_range_warning_needed(self):
        for i in range(len(self.ch_names)):
            if "*" in self.channel_tabs.tabText(i):
                return
        if self.range_warning:
            self.range_warning.setText("")
            self.range_warning.setVisible(False)

    def remove_channels(self, channels: List[str]):
        for ch_name in channels:
            if ch_name in self.ch_names:
                index = self.ch_names.index(ch_name)
                self.channel_tabs.removeTab(index)
                self.channels[index].setParent(None)
                self.channels.pop(index)
                self.ch_names.pop(index)
        self._check_range_warning_needed()

    def _redraw_new_diffs(self):
        for ch in self.channels:
            ch.change_diff_canvas(self.chosen_diffs)

    def show_relative(self, redraw=True):
        self.chosen_diffs = CompFields.DIFF_REL
        if redraw:
            self._redraw_new_diffs()

    def show_percentage(self, redraw=True):
        self.chosen_diffs = CompFields.DIFF_PERC
        if redraw:
            self._redraw_new_diffs()

    def show_no_diff(self, redraw=True):
        self.chosen_diffs = CompFields.DIFF_NONE
        if redraw:
            self._redraw_new_diffs()

    def update_plot(
        self,
        index: int,
        comparison: ComparisonData,
        redraw: bool = True,
        chosen_diffs: CompFields = CompFields.DIFF_REL,
    ):
        """Update the <index> plot with the given data

        Parameters
        ----------
        index: int
            Plot index (SRF)
        comparison: ComparisonData
        redraw: bool
            Boolean that defines if the plot will be redrawn automatically or not. Default True.
        """
        self.chosen_diffs = chosen_diffs
        self.channels[index].update_plot(comparison, redraw, self.chosen_diffs)

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
            Plot index (SRF band)
        legend: list of list of str
            Each list represents a group of legends
            Lengeds index:
            0: main data legend
        redraw: bool
            Boolean that defines if the plot will be redrawn automatically or not. Default True.
        """
        self.channels[index].update_legend(legends, redraw=redraw)

    def get_current_channel_index(self) -> int:
        return self.channel_tabs.currentIndex()

    def set_current_channel_index(self, index: int):
        cui = self.channel_tabs.setCurrentIndex(index)
        self.channels[index].tight_layout()
        return cui

    def get_channel_names(self) -> List[str]:
        return self.ch_names

    def get_channel_id(self, ch_name: str) -> int:
        return self.get_channel_names().index(ch_name)


class CompBoxPlotGraphWidget(CompGraphWidget):
    def __init__(
        self,
        settings_manager: ISettingsManager,
        wlens: List[float],
        title="",
        xlabel="",
        ylabel="",
        chosen_diffs=CompFields.DIFF_REL,
        parent=None,
        build_layout_ini=True,
    ):
        self.wlens = wlens
        super().__init__(
            settings_manager,
            title,
            xlabel,
            ylabel,
            chosen_diffs,
            parent,
            build_layout_ini,
        )

    def _redraw_canvas_compare(self) -> list:
        return redraw_canvas_compare_boxplot(
            self.canvas,
            self.data,
            self.wlens,
            self.legend,
            self.title,
            self.xlabel,
            self.ylabel,
            self.subtitle,
            self.chosen_diffs,
        )

    def _redraw_canvas_compare_only_diffs(self):
        return redraw_canvas_compare_boxplot_only_diffs(
            self.canvas,
            self.data,
            self.wlens,
            self.legend,
            self.subtitle,
            self.chosen_diffs,
        )

    @QtCore.Slot()
    def export_csv(self):
        name = QtWidgets.QFileDialog().getSaveFileName(
            self, "Export CSV", "{}.csv".format(self.title)
        )[0]
        version = self.settings_manager.get_coef_version_name()
        if name is not None and name != "":
            try:
                csv.export_csv_comparison_bywlen(
                    self.data,
                    self.wlens,
                    self.xlabel,
                    self.legend[0],
                    name,
                    version,
                    self.interp_spectrum_name,
                    self.skip_uncs,
                    self.chosen_diffs,
                )
            except Exception as e:
                self.show_error(e)

    def update_plot(
        self,
        comps: List[ComparisonData],
        wlens: List[float],
        redraw: bool = True,
        chosen_diffs: CompFields = CompFields.DIFF_REL,
    ):
        self.wlens = wlens
        super().update_plot(comps, redraw, chosen_diffs)


class CompWlensGraphWidget(CompGraphWidget):
    def update_plot(
        self,
        comps: List[ComparisonData],
        wlens: List[float],
        redraw: bool = True,
        chosen_diffs: CompFields = CompFields.DIFF_REL,
    ):
        wlens = np.array([w for w, c in zip(wlens, comps) if c is not None])
        comps = [c for c in comps if c is not None]
        self.wlens = wlens
        c = average_comparisons(wlens, comps)
        super().update_plot(c, redraw, chosen_diffs)


class ComparisonByWlenOutput(QtWidgets.QWidget):
    def __init__(self, settings_manager: ISettingsManager):
        super().__init__()
        self.settings_manager = settings_manager
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.stackl = QtWidgets.QStackedLayout()
        self.main_layout.addLayout(self.stackl)
        self.comp_bp = CompBoxPlotGraphWidget(self.settings_manager, [])
        self.comp_normal = CompWlensGraphWidget(self.settings_manager)
        self.stackl.addWidget(self.comp_bp)
        self.stackl.addWidget(self.comp_normal)
        self.stackl.setCurrentIndex(0)

    def _get_current_graph(self):
        if self.stackl.currentIndex() == 0:
            return self.comp_bp
        return self.comp_normal

    def update_plot(
        self,
        comps: List[ComparisonData],
        wlens: List[float],
        redraw: bool = True,
        chosen_diffs: CompFields = CompFields.DIFF_REL,
    ):
        self._get_current_graph().update_plot(comps, wlens, redraw, chosen_diffs)

    def update_labels(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        redraw: bool = True,
        subtitle: str = None,
    ):
        self._get_current_graph().update_labels(
            title, xlabel, ylabel, redraw=redraw, subtitle=subtitle
        )

    def update_legends(self, legends: List[List[str]], redraw: bool = True):
        """
        Parameters
        ----------
        legend: list of list of str
            Each list represents a group of legends
            Lengeds index:
            0: main data legend
        redraw: bool
            Boolean that defines if the plot will be redrawn automatically or not. Default True.
        """
        self._get_current_graph().update_legend(legends, redraw=redraw)

    def set_interp_spectrum_name(
        self,
        sp_name: str,
    ):
        self._get_current_graph().set_interp_spectrum_name(sp_name)

    def set_skipped_uncertainties(
        self,
        skip: bool,
    ):
        self._get_current_graph().set_skipped_uncertainties(skip)

    def _redraw_new_diffs(self):
        self._get_current_graph().change_diff_canvas(self.chosen_diffs)
        self.refresh_canvas()

    def show_relative(self, redraw=True):
        self.chosen_diffs = CompFields.DIFF_REL
        if redraw:
            self._redraw_new_diffs()

    def show_percentage(self, redraw=True):
        self.chosen_diffs = CompFields.DIFF_PERC
        if redraw:
            self._redraw_new_diffs()

    def show_no_diff(self, redraw=True):
        self.chosen_diffs = CompFields.DIFF_NONE
        if redraw:
            self._redraw_new_diffs()

    def clear(self):
        self.comp_bp.clear()
        self.comp_normal.clear()

    def set_kind(self, boxplot: bool):
        if boxplot:
            self.stackl.setCurrentIndex(0)
        else:
            self.stackl.setCurrentIndex(1)
        self.refresh_canvas()

    def _refresh_canvas(self):
        canvas = self._get_current_graph().canvas
        canvas.updateGeometry()
        canvas.draw()
        canvas.flush_events()
        canvas.repaint()
        canvas.update()

    def refresh_canvas(self):
        # Workaround for occasional rendering issues where the canvas does not appear correctly
        # when first shown in a QStackedLayout.
        self._refresh_canvas()
