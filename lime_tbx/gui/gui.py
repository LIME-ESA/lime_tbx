"""describe class"""

"""___Built-In Modules___"""
import sys
import pkgutil

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import constants

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class LimeTBXWidget(QtWidgets.QWidget):
    """
    Main widget of the lime toolbox desktop app.
    """

    def __init__(self, kernels_path: str):
        super().__init__()
        self.kernels_path = kernels_path
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)


class GUI:
    def __init__(self, kernels_path: str):
        self.kernels_path = kernels_path
        app = QtWidgets.QApplication([constants.APPLICATION_NAME])
        window = QtWidgets.QMainWindow()
        main_widget = LimeTBXWidget(kernels_path)
        window.resize(650, 450)
        window.setCentralWidget(main_widget)
        window.show()
        window.setWindowTitle(constants.APPLICATION_NAME)

        qss_bytes = pkgutil.get_data(__name__, constants.MAIN_QSS_PATH)
        window.setStyleSheet(qss_bytes.decode())
        # window.setWindowIcon(QtGui.QIcon(resource_path(constants.ICON_PATH)))
        sys.exit(app.exec_())

    def function1(self, argument1, argument2):
        return argument1 + argument2
