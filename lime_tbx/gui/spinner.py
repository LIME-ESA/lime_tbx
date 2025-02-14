"""Page that contains the spinner and an optional message"""

"""___Built-In Modules___"""
import os

"""___Third-Party Modules___"""
from PySide6 import QtWidgets, QtGui, QtCore

"""___LIME_TBX Modules___"""
from lime_tbx.gui import constants


class SpinnerPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Loading spinner
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _movie_path = os.path.join(_current_dir, constants.SPINNER_PATH)
        self.movie = QtGui.QMovie(_movie_path)
        self.movie.setScaledSize(QtCore.QSize(50, 50))
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.label_spinner = QtWidgets.QLabel()
        self.label_spinner.setMovie(self.movie)
        self.label_text = QtWidgets.QLabel()
        self.main_layout.addWidget(QtWidgets.QLabel())
        self.main_layout.addWidget(self.label_spinner)
        self.main_layout.addWidget(self.label_text)
        self.main_layout.addWidget(QtWidgets.QLabel())
        self.main_layout.setAlignment(self.label_spinner, QtGui.Qt.AlignHCenter)
        self.main_layout.setAlignment(self.label_text, QtGui.Qt.AlignHCenter)

    def movie_start(self):
        self.movie.start()

    def movie_stop(self):
        self.movie.stop()

    def set_text(self, text: str):
        self.label_text.setText(text)
