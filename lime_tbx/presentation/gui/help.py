"""GUI Widgets related to the Help actions"""

"""___Built-In Modules___"""
from typing import Optional, Tuple, List, Union
import os
import sys
from shutil import which
import webbrowser
import subprocess
import re
import functools

"""___Third-Party Modules___"""
from qtpy import QtWidgets, QtCore, QtGui

"""___LIME_TBX Modules___"""
from lime_tbx.presentation.gui import constants
from lime_tbx.common import logger, constants as dtp_constants

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


_CONTACT: str = """
To communicate any problem or error please visit the forum at
<a style=\"color: #00ae9d\" href=\"https://lime.uva.es/forums/\">lime.uva.es/forums</a>
or contact:
<a style=\"color: #00ae9d\" href=\"mailto:lime_tbx@goa.uva.es\">lime_tbx@goa.uva.es</a>
""".replace(
    "\n", " "
).strip()


def _launch_cmd(cmd: str) -> Tuple[str, str]:
    cmd_exec = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    so, serr = cmd_exec.communicate()
    return so, serr


def _try_go_to_link_sensible_browser(link: str) -> bool:
    if which("sensible-browser") is not None:
        logger.get_logger().info("Using sensible-browser to open link")
        cmd = f"sensible-browser {link} >/dev/null 2>&1 &"
        so, serr = _launch_cmd(cmd)
        logger.get_logger().debug(f"Linux executing {cmd}: {so}")
        if serr is not None and len(serr) > 0:
            logger.get_logger().error(f"Returned error: Linux {cmd}: {serr}")
            return False
        return True
    return False


def _get_mimeapps_lines() -> List[str]:
    user_home = os.environ.get("HOME")
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home is None:
        xdg_config_home = os.path.join(user_home, ".config")
    mimeapps_list = os.path.join(xdg_config_home, "mimeapps.list")
    if os.path.exists(mimeapps_list):
        with open(mimeapps_list, "r") as fp:
            lines = fp.readlines()
    else:
        lines = []
    return lines


def _get_default_mimeapps_browser_desktopfile() -> Union[str, None]:
    valid_starts = (
        "text/html",
        "x-scheme-handler/http",
        "x-scheme-handler/https",
        "x-scheme-handler/about",
        "x-scheme-handler/unknown",
    )
    for line in _get_mimeapps_lines():
        if line.startswith(valid_starts):
            start = line.index("=") + 1
            end = line.index("\n") if ";" not in line else line.index(";")
            appname = line[start:end]
            return appname
    else:
        return None


def _get_default_xdg_settings_browser_desktopfile() -> Union[str, None]:
    if which("xdg-settings") is not None:
        so, _ = _launch_cmd("xdg-settings get default-web-browser")
        if so is not None:
            so = so.strip()
            if so.endswith(".desktop"):
                return so
    return None


def _get_default_browser_desktoppath() -> Union[str, None]:
    appname = _get_default_xdg_settings_browser_desktopfile()
    if appname == None:
        appname = _get_default_mimeapps_browser_desktopfile()
    if appname == None:
        return None
    user_home = os.environ.get("HOME")
    desktop_folders = [
        os.path.join(user_home, ".local/share/applications/"),
        "/usr/share/applications/",
        "/usr/local/share/applications/",
    ]
    for folder in desktop_folders:
        if os.access(folder, os.R_OK) and os.path.exists(folder):
            if appname in os.listdir(folder):
                break
    else:
        return None
    appname = os.path.join(folder, appname)
    return appname


def _go_to_link(link: str) -> bool:
    if sys.platform != "linux":
        return webbrowser.open_new_tab(link)
    sensible_worked = _try_go_to_link_sensible_browser(link)
    if sensible_worked:
        return True
    appname = _get_default_browser_desktoppath()
    if appname != None:
        try:
            # cmd = f"nohup $(grep '^Exec' {appname} | head -1 | sed 's/^Exec=//' | sed 's/%.//' | sed 's/^\"//g' | sed 's/\" *$//g') {link} >/dev/null 2>&1 &"
            line = ""
            with open(appname) as fp:
                for line in fp.readlines():
                    if line.startswith("Exec"):
                        break
            line = line.strip()
            browsercmd = functools.reduce(
                lambda a, b: re.sub(b, "", a), [line, "^Exec=", "%.", '^"', '" *$']
            )
            cmd = f"nohup {browsercmd} {link} >/dev/null 2>&1 &"
            so, serr = _launch_cmd(cmd)
            logger.get_logger().debug(f"Linux executing {cmd}: {so}")
            if serr is not None and len(serr) > 0:
                logger.get_logger().warning(
                    f"Returned error messages: Linux {cmd}: {serr}"
                )
            return True
        except Exception as e:
            logger.get_logger().exception(e)
    return False


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
            f"Version: {dtp_constants.VERSION_NAME}", alignment=QtCore.Qt.AlignCenter
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
        # Contact
        self.contact_label = QtWidgets.QLabel(_CONTACT, alignment=QtCore.Qt.AlignLeft)
        self.contact_label.setWordWrap(True)
        self.contact_label.setTextFormat(QtCore.Qt.RichText)
        self.contact_label.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.contact_label.setOpenExternalLinks(True)
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
        self.uva_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.uva_logo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        uva_pixmap = QtGui.QPixmap(uva_logo_path).scaledToHeight(80)
        self.uva_logo.setPixmap(uva_pixmap)
        self.uva_logo.mousePressEvent = self._open_web_uva
        self.uva_logo.setOpenExternalLinks(True)
        ## GOA UVa
        goa_logo_path = os.path.join(_current_dir, constants.GOA_UVA_LOGO_PATH)
        self.goa_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.goa_logo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        goa_pixmap = QtGui.QPixmap(goa_logo_path).scaledToHeight(80)
        self.goa_logo.setPixmap(goa_pixmap)
        self.goa_logo.mousePressEvent = self._open_web_goa_uva
        self.goa_logo.setOpenExternalLinks(True)
        ## NPL
        npl_logo_path = os.path.join(_current_dir, constants.NPL_LOGO_PATH)
        self.npl_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.npl_logo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        npl_pixmap = QtGui.QPixmap(npl_logo_path).scaledToHeight(80)
        self.npl_logo.setPixmap(npl_pixmap)
        self.npl_logo.mousePressEvent = self._open_web_npl
        self.npl_logo.setOpenExternalLinks(True)
        ## VITO
        vito_logo_path = os.path.join(_current_dir, constants.VITO_LOGO_PATH)
        self.vito_logo = QtWidgets.QLabel("", alignment=QtCore.Qt.AlignCenter)
        self.vito_logo.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        vito_pixmap = QtGui.QPixmap(vito_logo_path).scaledToHeight(72)
        self.vito_logo.setPixmap(vito_pixmap)
        self.vito_logo.mousePressEvent = self._open_web_vito
        self.vito_logo.setOpenExternalLinks(True)
        ## Finish Collaborators layout
        self.collaborators_img_layout = QtWidgets.QHBoxLayout()
        self.collaborators_img_layout.addWidget(self.uva_logo)
        self.collaborators_img_layout.addWidget(self.goa_logo)
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
        self.scroll_layout.addWidget(self.contact_label)
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
    def _open_web_uva(self, event):
        _go_to_link("http://www.uva.es/")

    @QtCore.Slot()
    def _open_web_goa_uva(self, event):
        _go_to_link("http://goa.uva.es/")

    @QtCore.Slot()
    def _open_web_npl(self, event):
        _go_to_link("https://www.npl.co.uk/")

    @QtCore.Slot()
    def _open_web_vito(self, event):
        _go_to_link("https://vito.be/en")


_HELP_TEXT = (
    """
<p>
    Welcome to the LIME ToolBox! To check the user guide please visit:
    <a style=\"color: #00ae9d\" href=\"https://calvalportal.ceos.org/lime\">calvalportal.ceos.org/lime</a>
</p>
"""
    + _CONTACT
    + """
<h2>How does it work?</h2>
<p>
    The LIME model computes lunar reflectance for CIMEL wavelengths with
    the specified coefficients. Subsequently, these values undergo interpolation
    across the entire spectrum using the 'ASD' spectrum or 'Apollo 16 + Breccia'.
    Following this, irradiance is determined based on the calculated reflectance,
    and ultimately, the irradiance is integrated for each band in the selected
    Spectral Response Function (SRF).
</p>
<h2>Simulation</h2>
<p>
    Explore the various input types by switching between the top tabs.
</p>
<p>
    Select the 'Irradiance,' 'Reflectance,' or 'Polarisation' buttons to
    display the simulation output corresponding to the provided input.
    The spectrum will be displayed in the Result tab, and if irradiance is selected,
    the integrated irradiances will be presented in the Signal tab.
</p>
<h2>Comparison</h2>
<p>
    To transition from Simulation to Comparison, go to File > Perform Comparisons.
    Here, you can load observation files in GLOD format along with their corresponding
    SRF file and proceed to conduct the desired comparisons.
</p>
<p>
    The comparison can be presented either with respect to datetime or moon phase angle.
    The difference can be expressed as either a percentage difference or a relative
    difference, calculated using the formula '100 * (Observation - Simulation) / Simulation'.
</p>
<h2>Too slow?</h2>
<p>
    Calculating uncertainties for large datasets can be time-consuming,
    and users may not always require this information.
    If this process is causing issues, it can be disabled.
</p>
<p>
    To do so, navigate to Settings > Interpolation Options > Skip Uncertainty Calculations,
    and toggle the switch to activate.
</p>
"""
)


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = ...) -> None:
        super().__init__(parent)
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle(constants.APPLICATION_NAME)
        self.content = QtWidgets.QLabel(_HELP_TEXT)
        self.content.setWordWrap(True)
        self.content.setOpenExternalLinks(True)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.content)
        self.main_layout.addWidget(self.scroll_area)
