"""GUI Widgets related to the coefficients actions."""

"""___Built-In Modules___"""
from typing import List, Callable, Union, Tuple, Optional

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___LIME_TBX Modules___"""
from lime_tbx.gui.settings import ISettingsManager

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
