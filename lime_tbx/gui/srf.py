"""describe class"""

"""___Built-In Modules___"""
from datetime import datetime
from typing import Union, Tuple, Optional

"""___Third-Party Modules___"""
from typing import Callable
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from ..filedata import srf as file_srf
from ..datatypes.datatypes import (
    SurfacePoint,
    CustomPoint,
)
from . import settings, output

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "21/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class SRFWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        parent: CurrentSRFWidget = self.parentWidget()
        if parent:
            parent.update_from_settings_manager()
            parent = parent.parentWidget()
            if parent:
                parent = parent.parentWidget()
                if parent:
                    parent = parent.parentWidget()
                    if parent:
                        parent.setDisabled(False)
        return super().closeEvent(event)


class SRFEditWidget(QtWidgets.QWidget):
    def __init__(self, settings_manager: settings.ISettingsManager):
        super().__init__()
        self.combobox_listen = False
        self.settings_manager = settings_manager
        self.loaded_srf = None
        self._build_layout()
        self.update_output_data()
        self.combobox_listen = True

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # Current SRF
        self.selection_layout = QtWidgets.QHBoxLayout()
        self.current_srf_label = QtWidgets.QLabel("Current SRF: ")
        self.combo_srf = QtWidgets.QComboBox()
        self.combo_srf.currentIndexChanged.connect(self.update_from_combobox)
        self.update_combo_srf()
        self.selection_layout.addWidget(self.current_srf_label)
        self.selection_layout.addWidget(self.combo_srf)
        # Load SRF
        self.load_button = QtWidgets.QPushButton("Load SRF file")
        self.load_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.load_button.clicked.connect(self.load_srf)
        # Graph
        self.name_label = QtWidgets.QLabel("SRF: ")
        self.name_label.setWordWrap(True)
        self.graph = output.GraphWidget(
            "SRF", "Wavelengths (nm)", "Intensity (Fractions of unity)"
        )
        # Finish main
        self.main_layout.addLayout(self.selection_layout)
        self.main_layout.addWidget(self.load_button)
        self.main_layout.addWidget(self.name_label)
        self.main_layout.addWidget(self.graph)

    def update_output_data(self):
        srf = self.settings_manager.get_srf()
        self.name_label.setText("SRF: {}".format(srf.name))
        x_data = list(srf.spectral_response.keys())
        y_data = list(srf.spectral_response.values())
        self.graph.update_plot(x_data, y_data, None)

    def update_combo_srf(self):
        self.combobox_listen = False
        self.available_srfs = self.settings_manager.get_available_srfs()
        self.combo_srf.clear()
        self.combo_srf.addItems([srf.name for srf in self.available_srfs])
        srf = self.settings_manager.get_srf()
        index = self.available_srfs.index(srf)
        self.combo_srf.setCurrentIndex(index)
        self.combobox_listen = True

    @QtCore.Slot()
    def load_srf(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self)[0]
        srf = file_srf.read_srf(path)
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


class CurrentSRFWidget(QtWidgets.QWidget):
    def __init__(self, settings_manager: settings.ISettingsManager):
        super().__init__()
        self.settings_manager = settings_manager
        self._build_layout()
        self.update_from_settings_manager()

    def _build_layout(self):
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.title = QtWidgets.QLabel("SRF: ")
        self.name_label = QtWidgets.QLabel("")
        self.name_label.setWordWrap(True)
        self.edit_button = QtWidgets.QPushButton("Edit")
        self.edit_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.edit_button.clicked.connect(self.open_edit_srf)
        self.end_space = QtWidgets.QLabel("")
        self.main_layout.addWidget(self.title)
        self.main_layout.addWidget(self.name_label)
        self.main_layout.addWidget(self.edit_button)
        self.main_layout.addWidget(self.end_space, 1)

    def update_from_settings_manager(self):
        name = self.settings_manager.get_srf().name
        if name is not None:
            self.name_label.setText(name)

    @QtCore.Slot()
    def open_edit_srf(self):
        self.edit_srf_window = SRFWindow(self)
        self.edit_srf = SRFEditWidget(self.settings_manager)
        self.edit_srf_window.setCentralWidget(self.edit_srf)
        self.edit_srf_window.show()
        self.parentWidget().parentWidget().parentWidget().setDisabled(True)
        self.edit_srf_window.setDisabled(False)
        self.edit_srf.setDisabled(False)
