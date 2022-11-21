"""GUI Widgets related to the coefficients actions."""

"""___Built-In Modules___"""
from typing import Callable, Tuple

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___LIME_TBX Modules___"""
from lime_tbx.gui.settings import ISettingsManager
from lime_tbx.gui.spinner import SpinnerPage
from lime_tbx.gui.util import (
    CallbackWorker,
    start_thread as _start_thread,
    WorkerStopper,
)
from lime_tbx.gui import constants
from lime_tbx.datatypes import logger
from lime_tbx.coefficients.update.update import IUpdate, Update

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "20/09/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class SelectCoefficientsDialog(QtWidgets.QDialog):
    def __init__(self, settings_manager: ISettingsManager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.combobox_listen = True
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle(constants.APPLICATION_NAME)
        self.title_label = QtWidgets.QLabel("Select the coefficients")
        self.main_layout.addWidget(self.title_label)
        self.combo_versions = QtWidgets.QComboBox()
        self.main_layout.addWidget(self.combo_versions)
        self.update_combo_versions()
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_button.clicked.connect(self.save_clicked)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.cancel_button.clicked.connect(self.cancel_clicked)
        self.buttons_layout.addWidget(self.save_button)
        self.buttons_layout.addWidget(self.cancel_button)
        self.main_layout.addLayout(self.buttons_layout)

    def update_combo_versions(self):
        self.combobox_listen = False
        self.available_cimel_coeffs = self.settings_manager.get_available_cimel_coeffs()
        self.combo_versions.clear()
        self.combo_versions.addItems(
            [coeff.version for coeff in self.available_cimel_coeffs]
        )
        coeff = self.settings_manager.get_cimel_coef()
        index = self.available_cimel_coeffs.index(coeff)
        self.combo_versions.setCurrentIndex(index)
        self.combobox_listen = True

    @QtCore.Slot()
    def update_from_combobox(self, i: int):
        if self.combobox_listen:
            self.settings_manager.select_cimel_coeff(i)

    @QtCore.Slot()
    def save_clicked(self):
        index = self.combo_versions.currentIndex()
        self.update_from_combobox(index)
        self.close()

    @QtCore.Slot()
    def cancel_clicked(self):
        self.close()


def _callback_stopper_check_is_running(stopper: WorkerStopper):
    stopper.mutex.lock()
    if stopper.running == False:
        stopper.mutex.unlock()
        return False
    stopper.mutex.unlock()
    return True


def _callback_download(stopper: WorkerStopper) -> Tuple[bool]:
    updater: IUpdate = Update()
    if updater.check_for_updates():
        news, fails = updater.download_coefficients(
            _callback_stopper_check_is_running, [stopper]
        )
        return (True, news, fails)
    return (False, 0, 0)


class DownloadCoefficientsDialog(QtWidgets.QDialog):
    def __init__(self, settings_manager: ISettingsManager, parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self._downloading = False
        self._build_layout()
        self._start_downloading()

    def _build_layout(self):
        self.setWindowFlag(QtCore.Qt.CustomizeWindowHint, True)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle(constants.APPLICATION_NAME)
        self.title_label = QtWidgets.QLabel(
            "\tDownloading coefficients...\t\n",
            alignment=QtCore.Qt.AlignCenter,
        )
        self.main_layout.addWidget(self.title_label)
        self.spinner = SpinnerPage()
        self.spinner.setVisible(False)
        self.main_layout.addWidget(self.spinner)
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton("Ok")
        self.ok_button.setDisabled(True)
        self.ok_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.ok_button.clicked.connect(self.ok_clicked)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.cancel_button.clicked.connect(self.cancel_clicked)
        self.buttons_layout.addWidget(self.ok_button)
        self.buttons_layout.addWidget(self.cancel_button)
        self.main_layout.addLayout(self.buttons_layout)

    def _start_downloading(self):
        self._downloading = True
        self.spinner.setVisible(True)
        self.spinner.movie_start()
        self.worker = CallbackWorker(
            _callback_download,
            [],
            stoppable=True,
        )
        self._start_thread(self.download_finished, self.download_error)

    def _start_thread(self, finished: Callable, error: Callable, info: Callable = None):
        self.worker_th = QtCore.QThread()
        _start_thread(self.worker, self.worker_th, finished, error, info)

    def _finished_loading(self):
        self._downloading = False
        self.spinner.setVisible(False)
        self.spinner.movie_stop()
        self.ok_button.setDisabled(False)

    def download_finished(self, data: Tuple[bool, int, int]):
        updates, news, fails = data
        msg = "Download finished.\nThere were no updates."
        if updates:
            newsstring = f"There was 1 update"
            failsstring = f"it failed"
            if news > 1:
                newsstring = f"There were {news} updates"
                failsstring = f"{fails} of them failed"
            if fails == 0:
                msg = f"Download finished.\n{newsstring}."
            else:
                msg = f"Download finished with errors.\n{newsstring}, {failsstring}."
        self.title_label.setText(msg)
        if updates:
            self.settings_manager.reload_coeffs()
        self._finished_loading()

    def download_error(self, error: Exception):
        self.title_label.setText(
            "Error connecting to the server.\nCheck log for details."
        )
        logger.get_logger().error(str(error))
        self._finished_loading()

    @QtCore.Slot()
    def ok_clicked(self):
        self.close()

    @QtCore.Slot()
    def cancel_clicked(self):
        if self._downloading:
            self.worker.stop()
            self.worker_th.quit()
        self.close()
