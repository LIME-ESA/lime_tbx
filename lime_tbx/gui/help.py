"""describe class"""

"""___Built-In Modules___"""
from ctypes import alignment
from typing import List, Callable, Union, Tuple, Optional
from datetime import datetime
import time

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import settings, output, input, srf
from ..simulation.regular_simulation import regular_simulation
from ..datatypes.datatypes import (
    PolarizationCoefficients,
    SpectralResponseFunction,
    IrradianceCoefficients,
    SurfacePoint,
    CustomPoint,
)

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
        description = "Lunar Irradiance Model of ESA ToolBox."
        self.description_label = QtWidgets.QLabel(
            description, alignment=QtCore.Qt.AlignCenter
        )
        self.description_label.setWordWrap(True)
        self.version_label = QtWidgets.QLabel(
            "Version: Development", alignment=QtCore.Qt.AlignCenter
        )
        self.main_layout.addWidget(self.description_label)
        self.main_layout.addWidget(self.version_label)
