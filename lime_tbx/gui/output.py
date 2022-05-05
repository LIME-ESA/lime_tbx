"""describe class"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
from PySide2 import QtWidgets, QtCore, QtGui
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
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class GraphWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.canvas = MplCanvas(self)
        self.setCentralWidget(self.canvas)

    def update_plot(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(self.x_data, self.y_data, "r")
        # Trigger the canvas to update and redraw.
        self.canvas.draw()
