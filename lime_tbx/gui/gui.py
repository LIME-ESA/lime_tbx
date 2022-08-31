"""describe class"""

"""___Built-In Modules___"""
import sys
import pkgutil
import os

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui

"""___NPL Modules___"""
from . import constants, maingui

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "02/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class GUI:
    def __init__(self, kernels_path: str, eocfi_path: str):
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        app = QtWidgets.QApplication([constants.APPLICATION_NAME])
        window = maingui.LimeTBXWindow()
        main_widget = maingui.LimeTBXWidget(kernels_path, eocfi_path)
        # window.resize(400, 400)
        window.setCentralWidget(main_widget)
        window.show()
        window.setWindowTitle(constants.APPLICATION_NAME)

        qss_bytes = pkgutil.get_data(__name__, constants.MAIN_QSS_PATH)
        window.setStyleSheet(qss_bytes.decode())
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(_current_dir, constants.LOGO_PATH)
        window.setWindowIcon(QtGui.QIcon(logo_path))
        sys.exit(app.exec_())
