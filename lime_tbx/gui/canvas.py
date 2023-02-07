"""Module in charge of defining the matplotlib canvas"""

"""___Built-In Modules___"""
from typing import Union, List
import os

"""___Third-Party Modules___"""
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib import font_manager as fm
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np


"""___LIME_TBX Modules___"""
from ..datatypes.datatypes import (
    ComparisonData,
    SpectralData,
)
from . import constants
from lime_tbx.interpolation.interp_data import interp_data


SUBTITLE_DATE_FORMAT = "%Y/%m/%d %H:%M:%S"


_current_dir = os.path.dirname(os.path.abspath(__file__))
dir_font_path = os.path.dirname(os.path.join(_current_dir, constants.ESAFONT_PATH))
font_dirs = [dir_font_path]
font_files = fm.findSystemFonts(fontpaths=font_dirs, fontext="otf")
for font_file in font_files:
    fm.fontManager.addfont(font_file)
title_font_prop = fm.FontProperties(
    family=["NotesESA", "sans-serif"], weight="bold", size="large"
)
label_font_prop = fm.FontProperties(family=["NotesESA", "sans-serif"], weight="bold")
font_prop = fm.FontProperties(family=["NotesESA", "sans-serif"])


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes: Axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.axes_y_2 = None

    def set_title(self, title: str, fontproperties: fm.FontProperties = None):
        self.axes.set_title(title, fontproperties=fontproperties)

    def get_title(self) -> str:
        return self.axes.get_title()

    def set_subtitle(self, subtitle: str, fontproperties: fm.FontProperties = None):
        if self.axes_y_2 == None:
            self.axes_y_2 = self.axes.twiny()
        self.axes_y_2.set_xlabel(subtitle, fontproperties=fontproperties)
        self.axes_y_2.tick_params(
            axis="x",
            which="both",
            top=False,
            labeltop=False,
        )

    def get_subtitle(self) -> str:
        if self.axes_y_2 == None:
            return ""
        return self.axes_y_2.get_xlabel()


def redraw_canvas(
    scanvas: MplCanvas,
    sdata: Union[SpectralData, List[SpectralData], None],
    slegend: List[List[str]],
    scimel_data: Union[SpectralData, List[SpectralData], None],
    sasd_data: Union[SpectralData, List[SpectralData], None],
    sdata_compare: Union[ComparisonData, None],
    stitle: str,
    sxlabel: str,
    sylabel: str,
    svertical_lines: List[float],
    subtitle: str = None,
    sp_name: str = interp_data.SPECTRUM_NAME_ASD,
):
    lines = []
    if sdata is not None:
        iter_data = sdata
        if not isinstance(iter_data, list):
            iter_data = [iter_data]
        for i, data in enumerate(iter_data):
            label = ""
            color = []
            if i == 0:
                color = ["g"]
            if len(slegend) > 0:
                if len(slegend[0]) > i:
                    label = slegend[0][i]
                if len(slegend[0]) > 1:
                    color = []
            marker = ""
            if len(data.data) == 1:
                marker = "o"
            lines += scanvas.axes.plot(
                data.wlens,
                data.data,
                *color,
                marker=marker,
                label=label,
            )
            if data.uncertainties is not None:
                scanvas.axes.fill_between(
                    data.wlens,
                    data.data - 2 * data.uncertainties,
                    data.data + 2 * data.uncertainties,
                    color="green",
                    alpha=0.3,
                )

        if scimel_data:
            iter_data = scimel_data
            if not isinstance(iter_data, list):
                iter_data = [iter_data]
            for i, cimel_data in enumerate(iter_data):
                label0 = ""
                label1 = ""
                if i == 0 and len(slegend) >= 3:
                    label0 = slegend[1][0]
                    label1 = slegend[2][0]
                extra_lines = []
                extra_lines += scanvas.axes.plot(
                    cimel_data.wlens,
                    cimel_data.data,
                    color="orange",
                    ls="none",
                    marker="o",
                    label=label0,
                )
                extra_lines += [
                    scanvas.axes.errorbar(
                        cimel_data.wlens,
                        cimel_data.data,
                        yerr=cimel_data.uncertainties * 2,
                        color="black",
                        capsize=3,
                        ls="none",
                        label=label1,
                    )
                ]
                if i == 0:
                    lines += extra_lines

        if sasd_data:
            if isinstance(sasd_data, list):
                asd_data = sasd_data[0]
            else:
                asd_data = sasd_data

            if np.any(cimel_data.data[cimel_data.wlens == 500] == 0):
                asd_data_final = asd_data.data * 0
            else:
                scaled_wlen = 500
                if len(np.where(asd_data.wlens == scaled_wlen)[0]) == 0:
                    scaled_wlen = 503.017  # Breccia
                scaling_factor = (
                    asd_data.data[np.where(asd_data.wlens == scaled_wlen)]
                    / cimel_data.data[np.where(cimel_data.wlens == 500)]
                )
                asd_data_final = asd_data.data / scaling_factor
            lines += scanvas.axes.plot(
                asd_data.wlens,
                asd_data_final,
                label=f"{sp_name} data, scaled to LIME at 500nm",
            )

        data_compare_info = ""
        if sdata_compare:
            data_comp = sdata_compare.diffs_signal
            ax2 = scanvas.axes.twinx()
            label = ""
            if len(slegend) > 3 and len(slegend[3]) > 0:
                label = slegend[3][0]
            marker = "--"
            if len(data_comp.data) == 1:
                marker = "o"
            lines += ax2.plot(
                data_comp.wlens,
                data_comp.data,
                "k{}".format(marker),
                label=label,
            )
            if data_comp.uncertainties is not None:
                ax2.fill_between(
                    data_comp.wlens,
                    data_comp.data - 2 * data_comp.uncertainties,
                    data_comp.data + 2 * data_comp.uncertainties,
                    color="pink",
                    alpha=0.3,
                )
            ax2.set_ylim(
                (
                    min(-0.05, min(data_comp.data) - 0.05),
                    max(0.05, max(data_comp.data) + 0.05),
                )
            )
            data_compare_info = "MRD: {:.4f}\nσ: {:.4f}\n".format(
                sdata_compare.mean_relative_difference,
                sdata_compare.standard_deviation_mrd,
            )
            lines += scanvas.axes.plot([], [], " ", label=data_compare_info)
            ax2.set_ylabel(
                "Relative difference (Fraction of unity)",
                fontproperties=label_font_prop,
            )
            plt.setp(
                scanvas.axes.get_xticklabels(),
                rotation=30,
                horizontalalignment="right",
            )
        if len(slegend) > 0:
            legend_lines = [l for l in lines if not l.get_label().startswith("_child")]
            labels = [l.get_label() for l in legend_lines]
            scanvas.axes.legend(legend_lines, labels, loc=0, prop=font_prop)

    if subtitle != None:
        scanvas.set_subtitle(subtitle, fontproperties=font_prop)
    scanvas.set_title(stitle, fontproperties=title_font_prop)
    scanvas.axes.set_xlabel(sxlabel, fontproperties=label_font_prop)
    scanvas.axes.set_ylabel(sylabel, fontproperties=label_font_prop)
    if svertical_lines and len(svertical_lines) > 0:
        for val in svertical_lines:
            scanvas.axes.axvline(x=val, color="k", label="LIME Spectrum limit")
    return lines
