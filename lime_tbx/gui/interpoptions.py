"""GUI Widgets related to the coefficients actions."""

"""___Built-In Modules___"""
pass

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___LIME_TBX Modules___"""
from lime_tbx.gui.settings import ISettingsManager
from lime_tbx.gui import constants
from lime_tbx.simulation.lime_simulation import ILimeSimulation

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "06/02/2023"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class InterpOptionsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        settings_manager: ISettingsManager,
        lime_simulation: ILimeSimulation,
        parent=None,
    ):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.lime_simulation = lime_simulation
        self.combobox_listen = True
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle(constants.APPLICATION_NAME)
        # select interpolation reference
        self.title_label = QtWidgets.QLabel("Select the interpolation reference")
        self.main_layout.addWidget(self.title_label)
        self.combo_versions = QtWidgets.QComboBox()
        self.main_layout.addWidget(self.combo_versions)
        self.form_layout = QtWidgets.QFormLayout()
        # select output SRF
        self.title_label_SRF = QtWidgets.QLabel("Select the output SRF")
        self.main_layout.addWidget(self.title_label_SRF)
        self.combo_SRF = QtWidgets.QComboBox()
        self.main_layout.addWidget(self.combo_SRF)
        self.form_layout = QtWidgets.QFormLayout()
        # show interp
        self.label_show_interp = QtWidgets.QLabel("Show interpolation spectrum:")
        self.checkbox_show_interp = QtWidgets.QCheckBox()
        self.checkbox_show_interp.setChecked(
            self.settings_manager.is_show_interp_spectrum()
        )
        self.form_layout.setWidget(
            0, QtWidgets.QFormLayout.LabelRole, self.label_show_interp
        )
        self.form_layout.setWidget(
            0, QtWidgets.QFormLayout.FieldRole, self.checkbox_show_interp
        )
        # skip uncertainties
        self.label_skip_uncerts = QtWidgets.QLabel("Skip uncertainties calculation:")
        self.checkbox_skip_uncerts = QtWidgets.QCheckBox()
        self.checkbox_skip_uncerts.setChecked(
            self.settings_manager.is_skip_uncertainties()
        )
        self.form_layout.setWidget(
            1, QtWidgets.QFormLayout.LabelRole, self.label_skip_uncerts
        )
        self.form_layout.setWidget(
            1, QtWidgets.QFormLayout.FieldRole, self.checkbox_skip_uncerts
        )
        self.main_layout.addLayout(self.form_layout)
        self.update_combos()
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

    def update_combos(self):
        self.combobox_listen = False
        # combo interp spectrum
        self.spectra_names = self.settings_manager.get_available_spectra_names()
        self.combo_versions.clear()
        self.combo_versions.addItems(self.spectra_names)
        spname = self.settings_manager.get_selected_spectrum_name()
        index = self.spectra_names.index(spname)
        self.combo_versions.setCurrentIndex(index)
        # combo srf
        self.SRF_names = self.settings_manager.get_available_interp_SRFs()
        self.combo_SRF.clear()
        self.combo_SRF.addItems(self.SRF_names)
        spname = self.settings_manager.get_selected_interp_SRF()
        index = self.SRF_names.index(spname)
        self.combo_SRF.setCurrentIndex(index)
        self.combobox_listen = True

    @QtCore.Slot()
    def update_from_combobox(self, name: str, name_SRF: str):
        if self.combobox_listen:
            self.settings_manager.select_interp_spectrum(name)
            self.settings_manager.select_interp_SRF(name_SRF)
            self.settings_manager.set_show_interp_spectrum(
                self.checkbox_show_interp.isChecked()
            )
            self.settings_manager.set_skip_uncertainties(
                self.checkbox_skip_uncerts.isChecked()
            )
            self.lime_simulation.set_simulation_changed()

    @QtCore.Slot()
    def save_clicked(self):
        name = self.combo_versions.currentText()
        name_SRF = self.combo_SRF.currentText()
        self.update_from_combobox(name, name_SRF)
        self.close()

    @QtCore.Slot()
    def cancel_clicked(self):
        self.close()

    def closeEvent(self, arg__1: QtGui.QCloseEvent) -> None:
        parent = self.parent()
        if parent is not None:
            parent.update_calculability()
        return super().closeEvent(arg__1)
