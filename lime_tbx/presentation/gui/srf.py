"""describe class"""

"""___Built-In Modules___"""
from typing import Callable, Tuple

"""___Third-Party Modules___"""
from qtpy import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from lime_tbx.application.filedata import srf as file_srf
from lime_tbx.common.datatypes import SpectralData, SpectralResponseFunction
from lime_tbx.common import constants
from lime_tbx.presentation.gui import settings, output
from lime_tbx.presentation.gui.util import (
    CallbackWorker,
    start_thread as _start_thread,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "21/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def _callback_read_srf(path: str) -> Tuple[SpectralResponseFunction]:
    srf = file_srf.read_srf(path)
    return (srf,)


class SRFEditWidget(QtWidgets.QWidget):
    def __init__(
        self,
        settings_manager: settings.ISettingsManager,
        changed_callback: Callable,
        enabled_callback: Callable,
    ):
        super().__init__()
        self.combobox_listen = False
        self.settings_manager = settings_manager
        self.loaded_srf = None
        self.changed_callback = changed_callback
        self.enabled_callback = enabled_callback
        self.worker_ths = []
        self.workers = []
        self._build_layout()
        self.update_output_data()
        self.combobox_listen = True

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # Current SRF
        self.selection_layout = QtWidgets.QHBoxLayout()
        self.current_srf_label = QtWidgets.QLabel("Spectral Response Function: ")
        self.combo_srf = QtWidgets.QComboBox()
        self.combo_srf.currentIndexChanged.connect(self.update_from_combobox)
        self.update_combo_srf()
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.load_button.clicked.connect(self.load_srf)
        self.selection_layout.addWidget(self.current_srf_label)
        self.selection_layout.addWidget(self.combo_srf, 1)
        self.selection_layout.addWidget(self.load_button)
        # Graph
        self.graph = output.SimGraphWidget(
            self.settings_manager,
            "SRF",
            "Wavelengths (nm)",
            "Intensity (Fractions of unity)",
        )
        self.graph.set_vertical_lines([constants.MIN_WLEN, constants.MAX_WLEN])
        # Finish main
        self.main_layout.addLayout(self.selection_layout)
        self.main_layout.addWidget(self.graph)

    def update_size(self):
        self.graph.update_size()

    def update_output_data(self):
        srf = self.settings_manager.get_srf()
        srf_data = []
        ch_names = []
        for ch in srf.channels:
            x_data = list(ch.spectral_response.keys())
            y_data = list(ch.spectral_response.values())
            ch_data = SpectralData(x_data, y_data, None, None)
            ch_names.append(ch.id)
            srf_data.append(ch_data)
        self.graph.set_cursor_names(ch_names)
        self.graph.update_plot(srf_data)
        self.graph.set_srf_channel_names(srf.get_channels_names())
        self.changed_callback()

    def update_combo_srf(self):
        self.combobox_listen = False
        self.available_srfs = self.settings_manager.get_available_srfs()
        self.combo_srf.clear()
        self.combo_srf.addItems([srf.name for srf in self.available_srfs])
        srf = self.settings_manager.get_srf()
        index = self.available_srfs.index(srf)
        self.combo_srf.setCurrentIndex(index)
        self.combobox_listen = True

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

    def _load_srf_finished(self, data):
        srf = data[0]
        self.set_srf(srf)
        self.enabled_callback(True)

    def _load_srf_error(self, e):
        self.enabled_callback(True)
        self.show_error(e)

    @QtCore.Slot()
    def load_srf(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        if path != "":
            self.enabled_callback(False)
            worker = CallbackWorker(
                _callback_read_srf,
                [path],
            )
            self._start_thread(worker, self._load_srf_finished, self._load_srf_error)

    @QtCore.Slot()
    def set_srf(self, srf):
        self.loaded_srf = srf
        self.settings_manager.load_srf(srf)
        self.settings_manager.select_srf(-1)
        self.update_combo_srf()
        self.update_output_data()

    @QtCore.Slot()
    def update_from_combobox(self, i: int):
        if self.combobox_listen:
            self.settings_manager.select_srf(i)
            self.update_output_data()
