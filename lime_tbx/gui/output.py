"""describe class"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "05/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes: Axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class GraphWidget(QtWidgets.QWidget):
    def __init__(self, title = "", xlabel = "", ylabel = ""):
        super().__init__()
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_data = []
        self.y_data = []
        self._build_layout()

    def _build_layout(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # canvas
        self.canvas = MplCanvas(self)
        self.canvas.axes.set_title(self.title)
        self.canvas.axes.set_xlabel(self.xlabel)
        self.canvas.axes.set_ylabel(self.ylabel)
        self._redraw()
        # save button
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.save_button.clicked.connect(self._save_image)
        self.save_button.setDisabled(True)
        # finish main
        self.main_layout.addWidget(self.canvas)
        self.main_layout.addWidget(self.save_button)

    def update_plot(self, x_data: list, y_data: list):
        self.x_data = x_data
        self.y_data = y_data
        if len(x_data) > 0 and len(y_data) > 0:
            self.save_button.setDisabled(False)
        else:
            self.save_button.setDisabled(True)
        self._redraw()

    def update_labels(self, title: str, xlabel: str, ylabel: str):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._redraw()

    def _redraw(self):
        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(self.x_data, self.y_data, "o", markersize=2)
        self.canvas.axes.set_title(self.title)
        self.canvas.axes.set_xlabel(self.xlabel)
        self.canvas.axes.set_ylabel(self.ylabel)
        self.canvas.draw()

    @QtCore.Slot()
    def _save_image(self):
        self._save_jpg()

    def _save_jpg(self):
        self.canvas.print_figure("test.jpg")
