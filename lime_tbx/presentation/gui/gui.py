"""describe class"""

"""___Built-In Modules___"""
import sys
import pkgutil
import os
import traceback

"""___Third-Party Modules___"""
from qtpy import QtWidgets, QtGui, QtCore

"""___LIME_TBX Modules___"""
from lime_tbx.presentation.gui import constants, maingui
from lime_tbx.common.datatypes import KernelsPath, EocfiPath
from lime_tbx.common import logger

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "02/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def _preprocess_qss(qss: str, qss_constants: str):
    consts = qss_constants.split("\n")
    for const in consts:
        key_val = list(map(lambda c: c.strip(), const.split("=")))
        if key_val[0] == "":
            continue
        qss = qss.replace(key_val[0], key_val[1])
    return qss


class LimeApp(QtWidgets.QApplication):
    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            logger.get_logger().error(
                "Exception in event loop:\n%s", traceback.format_exc()
            )
            return False


class GUI:
    def __init__(
        self,
        kernels_path: KernelsPath,
        eocfi_path: EocfiPath,
        selected_version: str = None,
    ):
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        app = LimeApp([constants.APPLICATION_NAME])
        if sys.platform == "linux":
            QtGui.QGuiApplication.setDesktopFileName("limetbx")
            QtGui.QGuiApplication.setApplicationName("LimeTBX")
            QtGui.QGuiApplication.setApplicationDisplayName("LimeTBX")
            QtCore.QCoreApplication.setApplicationName("LimeTBX")
        self._init_fonts()
        window = maingui.LimeTBXWindow(kernels_path)
        main_widget = maingui.LimeTBXWidget(kernels_path, eocfi_path, selected_version)
        window.resize(800, 750)
        window.setCentralWidget(main_widget)
        window.show()
        window.setWindowTitle(constants.APPLICATION_NAME)
        # QSS | Read, and preprocess it with global constants and os constants
        qss = pkgutil.get_data(__name__, constants.MAIN_QSS_PATH).decode()
        if sys.platform == "darwin":
            qss_os_constants = pkgutil.get_data(
                __name__, constants.QSS_DARWIN_CONSTANTS_PATH
            ).decode()
            qss = _preprocess_qss(qss, qss_os_constants)
        qss_constants = pkgutil.get_data(
            __name__, constants.QSS_CONSTANTS_PATH
        ).decode()
        window.setStyleSheet(_preprocess_qss(qss, qss_constants))
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(_current_dir, constants.LOGO_PATH)
        window.setWindowIcon(QtGui.QIcon(logo_path))
        if sys.platform == "win32":
            import ctypes

            myappid = "esa.lime.limetbx"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        else:
            app.setStyle("Fusion")
        sys.exit(app.exec())

    def _init_fonts(self):
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        bold_path = os.path.join(_current_dir, constants.ESABOLDFONT_PATH)
        font_id = QtGui.QFontDatabase.addApplicationFont(bold_path)
        if font_id < 0:
            raise Exception("Error loading fonts.")
        reg_path = os.path.join(_current_dir, constants.ESAFONT_PATH)
        font_id = QtGui.QFontDatabase.addApplicationFont(reg_path)
        if font_id < 0:
            raise Exception("Error loading fonts.")
