"""GUI Widgets related to the Help actions"""

"""___Built-In Modules___"""
from typing import List, Callable, Union, Tuple, Optional

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___LIME_TBX Modules___"""
from . import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "24/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent)
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle(constants.APPLICATION_NAME)
        # title
        title = "Lunar Irradiance Model of ESA ToolBox"
        self.title_label = QtWidgets.QLabel(title, alignment=QtCore.Qt.AlignCenter)
        # description
        description = "Lunar Irradiance Model of ESA ToolBox."
        self.description_label = QtWidgets.QLabel(
            description, alignment=QtCore.Qt.AlignCenter
        )
        self.description_label.setWordWrap(True)
        self.version_label = QtWidgets.QLabel(
            "Version: Development", alignment=QtCore.Qt.AlignCenter
        )
        self.main_layout.addWidget(self.title_label)
        self.main_layout.addWidget(self.description_label)
        self.main_layout.addWidget(self.version_label)
