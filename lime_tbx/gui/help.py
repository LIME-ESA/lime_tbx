"""GUI Widgets related to the Help actions"""

"""___Built-In Modules___"""
from typing import List, Callable, Union, Tuple, Optional
import os
import webbrowser

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


_DESCRIPTION: str = (
    """
The LIME Toolbox allows users to simulate lunar observations for any observer
position around the Earth and at any time, from satellite positions of multiple
ESA satellites like ENVISAT, Proba-V, S2, S3 and FLEX, or any satellite position
for which an orbital scenario file is provided by the user in EOCFI compatible
format, and for any observer/solar selenographic latitude and longitude (thus
bypassing the need for their computation from the position/time of the observer).
\\n\\nThis simulations can be performed for any user defined instrument spectral
response pre-stored in a GLOD format file.
\\n\\nIt also allows performing comparisons of lunar observations from a remote
sensing instrument (pre-stored in GLOD format files) to the LIME model output.
""".replace(
        "\n", " "
    )
    .replace("\\n", "\n")
    .strip()
)


class AboutDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent)
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle(constants.APPLICATION_NAME)
        # Title
        title = "Lunar Irradiance Model of ESA ToolBox"
        self.title_label = QtWidgets.QLabel(title, alignment=QtCore.Qt.AlignCenter)
        # Version
        self.version_label = QtWidgets.QLabel(
            f"Version: {constants.VERSION_NAME}", alignment=QtCore.Qt.AlignCenter
        )
        # LIME Logo
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(_current_dir, constants.LOGO_PATH)
        lime_pixmap = QtGui.QPixmap(logo_path).scaledToHeight(220)
        self.lime_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.lime_logo.setPixmap(lime_pixmap)
        # Description
        self.description_label = QtWidgets.QLabel(
            _DESCRIPTION, alignment=QtCore.Qt.AlignLeft
        )
        self.description_label.setWordWrap(True)
        # ESA Logo
        esa_logo_path = os.path.join(_current_dir, constants.ESA_LOGO_PATH)
        esa_pixmap = QtGui.QPixmap(esa_logo_path).scaledToHeight(250)
        self.esa_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.esa_logo.setPixmap(esa_pixmap)
        # Collaborators
        self.collaborators_layout = QtWidgets.QVBoxLayout()
        self.collaborators_text = QtWidgets.QLabel("In collaboration with:")
        ## UVa
        uva_logo_path = os.path.join(_current_dir, constants.UVA_LOGO_PATH)
        uva_pixmap = QtGui.QPixmap(uva_logo_path).scaledToHeight(100)
        self.uva_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.uva_logo.setPixmap(uva_pixmap)
        self.uva_logo.mousePressEvent = self._open_link_uva
        self.uva_logo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        ## NPL
        npl_logo_path = os.path.join(_current_dir, constants.NPL_LOGO_PATH)
        npl_pixmap = QtGui.QPixmap(npl_logo_path).scaledToHeight(100)
        self.npl_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.npl_logo.setPixmap(npl_pixmap)
        self.npl_logo.mousePressEvent = self._open_link_npl
        self.npl_logo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        ## VITO
        vito_logo_path = os.path.join(_current_dir, constants.VITO_LOGO_PATH)
        vito_pixmap = QtGui.QPixmap(vito_logo_path).scaledToHeight(90)
        self.vito_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.vito_logo.setPixmap(vito_pixmap)
        self.vito_logo.mousePressEvent = self._open_link_vito
        self.vito_logo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        ## Finish Collaborators layout
        self.collaborators_img_layout = QtWidgets.QHBoxLayout()
        self.collaborators_img_layout.addWidget(self.uva_logo)
        self.collaborators_img_layout.addWidget(self.npl_logo)
        self.collaborators_img_layout.addWidget(self.vito_logo)
        self.collaborators_layout.addWidget(self.collaborators_text)
        self.frame_collabs_img = QtWidgets.QFrame()
        self.frame_collabs_img.setLayout(self.collaborators_img_layout)
        self.frame_collabs_img.setStyleSheet("background-color: white;")
        self.collaborators_layout.addWidget(self.frame_collabs_img)
        # Finish layout
        self.scroll_layout = QtWidgets.QVBoxLayout()
        self.scroll_layout.addWidget(self.title_label)
        self.scroll_layout.addWidget(self.version_label)
        self.scroll_layout.addWidget(self.lime_logo)
        self.scroll_layout.addWidget(self.description_label)
        self.scroll_layout.addWidget(self.esa_logo)
        self.scroll_layout.addLayout(self.collaborators_layout)
        self.scroll_layout.addStretch()
        self.groupbox = QtWidgets.QGroupBox()
        self.groupbox.setLayout(self.scroll_layout)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.groupbox)
        self.main_layout.addWidget(self.scroll_area)

    @QtCore.Slot()
    def _open_link_uva(self, event):
        self._open_link("https://www.uva.es/")

    @QtCore.Slot()
    def _open_link_npl(self, event):
        self._open_link("https://www.npl.co.uk/")

    @QtCore.Slot()
    def _open_link_vito(self, event):
        self._open_link("https://vito.be/en")

    def _open_link(self, link: str):
        webbrowser.open(link, new=2)
