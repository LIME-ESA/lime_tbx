"""describe class"""

"""___Built-In Modules___"""
import sys
import pkgutil

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import constants, guieli, settings

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "02/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
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
        settings_manager = settings.MockSettingsManager()
        self.eli_page = guieli.ELISurfaceWidget(self.kernels_path, settings_manager)
        self.main_layout.addWidget(self.eli_page)


class GUI:
    def __init__(self, kernels_path: str):
        self.kernels_path = kernels_path
        app = QtWidgets.QApplication([constants.APPLICATION_NAME])
        window = QtWidgets.QMainWindow()
        main_widget = LimeTBXWidget(kernels_path)
        # window.resize(400, 400)
        window.setCentralWidget(main_widget)
        window.show()
        window.setWindowTitle(constants.APPLICATION_NAME)

        qss_bytes = pkgutil.get_data(__name__, constants.MAIN_QSS_PATH)
        window.setStyleSheet(qss_bytes.decode())
        # window.setWindowIcon(QtGui.QIcon(resource_path(constants.ICON_PATH)))
        sys.exit(app.exec_())

    def function1(self, argument1, argument2):
        return argument1 + argument2
