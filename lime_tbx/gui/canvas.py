"""Module in charge of defining the matplotlib canvas"""

"""___Built-In Modules___"""
from typing import Union, List, Tuple
import os
from datetime import datetime

"""___Third-Party Modules___"""
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import matplotlib.dates as mdates
import matplotlib.backends.backend_pdf  # important import for exporting as pdf
import matplotlib.ticker
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib import font_manager as fm
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
import mplcursors


"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    ComparisonData,
    SpectralData,
)
from lime_tbx.datatypes.constants import CompFields
from lime_tbx.gui import constants


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
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes: Axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.axes_y_2 = None
        self.axes_x2 = None
        self.boxplot_lines_ax = []
        self.boxplot_lines_ax2 = []

    def set_title(self, title: str, fontproperties: fm.FontProperties = None):
        self.axes.set_title(title, fontproperties=fontproperties)

    def get_title(self) -> str:
        return self.axes.get_title()

    def set_subtitle(self, subtitle: str, fontproperties: fm.FontProperties = None):
        if self.axes_y_2 == None:
            self.axes_y_2 = self.axes.twiny()
        self.axes_y_2.set_title("Subtitle", {"alpha": 1, "size": 0})
        self.axes_y_2.set_xlabel(subtitle, fontproperties=fontproperties)
        self.axes_y_2.tick_params(
            axis="x",
            which="both",
            top=False,
            labeltop=False,
        )

        def make_format_old(twin, old):
            def format_coord(x, y):
                display_coord = twin.transData.transform((x, y))
                inv = old.transData.inverted()
                ax_coord = inv.transform(display_coord)
                return f"x={ax_coord[0]:#5g}  y={ax_coord[1]:#5g}"

            return format_coord

        self.axes_y_2.format_coord = make_format_old(self.axes_y_2, self.axes)

    def get_subtitle(self) -> str:
        if self.axes_y_2 is None:
            return ""
        return self.axes_y_2.get_xlabel()

    def get_twinx(self) -> Axes:
        if self.axes_x2 is None:
            self.axes_x2 = self.axes.twinx()
        self.axes_x2.set_visible(True)

        def combine_formats(twin, old):
            def format_coord(x, y):
                display_coord = twin.transData.transform((x, y))
                inv = old.transData.inverted()
                ax_coord = inv.transform(display_coord)
                x = twin.format_xdata(x)
                return f"x={x}  y1={ax_coord[1]:#5g}  y2={y:#5g}"

            return format_coord

        self.axes_x2.format_coord = combine_formats(self.axes_x2, self.axes)
        return self.axes_x2

    def get_lines(self) -> List[Line2D]:
        lines2 = []
        if self.axes_x2 is not None:
            lines2 = self.axes_x2.get_lines()
        return self.axes.get_lines() + lines2

    def store_ax_boxplot_lines(self, boxes: List[dict]):
        self.boxplot_lines_ax += boxes

    def store_ax2_boxplot_lines(self, boxes: List[dict]):
        self.boxplot_lines_ax2 += boxes

    def get_boxplot_lines(self) -> List[dict]:
        return self.boxplot_lines_ax + self.boxplot_lines_ax2

    def clear_twinx(self):
        if self.axes_x2 is not None:
            self.axes_x2.cla()
            self.boxplot_lines_ax2 = []

    def clear(self):
        self.boxplot_lines_ax = []
        self.clear_twinx()
        self.axes.cla()


_YAXIS_NTICKS = 9
_OBS_COLOR = "#1f77b4"
_SIM_COLOR = "#ff7f0e"
_DIFFS_COLOR = "#545454"


def _redraw_canvas_compare_only_diffs(
    scanvas: MplCanvas,
    sdata_compare: Union[ComparisonData, None],
    chosen_diffs: CompFields = CompFields.DIFF_REL,
) -> Tuple[list, str]:
    lines = []
    data_compare_info = ""
    if sdata_compare is not None:
        data_comp, ylabeltit = sdata_compare.get_diffs_and_label(chosen_diffs)
        scanvas.clear_twinx()
        ax2 = scanvas.get_twinx()
        if data_comp:
            label = ylabeltit
            lines += ax2.plot(
                data_comp.wlens,
                data_comp.data,
                marker="o",
                color=_DIFFS_COLOR,
                label=label,
                markersize=4,
                ls="none",
            )
            if data_comp.uncertainties is not None and data_comp.uncertainties.size > 0:
                ax2.errorbar(
                    data_comp.wlens,
                    data_comp.data,
                    yerr=data_comp.uncertainties * 2,
                    color=_DIFFS_COLOR,
                    capsize=2,
                    ls="none",
                    alpha=0.3,
                )
            ylim = max(list(map(abs, ax2.get_ylim())))
            if chosen_diffs == CompFields.DIFF_PERC:
                ax2.set_ylim((0.0, ylim + 0.5))
                data_compare_info = "MPD: {:.4f}%".format(
                    sdata_compare.mean_perc_difference
                )
            else:
                ax2.set_ylim((-ylim - 0.5, ylim + 0.5))
                data_compare_info = "MRD: {:.4f}% | σ: {:.4f}% | MARD: {:.4f}%".format(
                    sdata_compare.mean_relative_difference,
                    sdata_compare.standard_deviation_mrd,
                    sdata_compare.mean_absolute_relative_difference,
                )
            ax2.set_ylabel(
                ylabeltit,
                fontproperties=label_font_prop,
            )
            ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(_YAXIS_NTICKS))
        else:
            ax2.set_visible(False)
    return lines, data_compare_info


def redraw_canvas_compare_only_diffs(
    scanvas: MplCanvas,
    sdata_compare: Union[ComparisonData, None],
    subtitle: str = None,
    chosen_diffs: CompFields = CompFields.DIFF_REL,
):
    dlines, data_compare_info = _redraw_canvas_compare_only_diffs(
        scanvas, sdata_compare, chosen_diffs
    )
    if isinstance(sdata_compare.diffs_signal.wlens[0], datetime):
        scanvas.axes.xaxis.set_major_formatter(
            mdates.AutoDateFormatter(scanvas.axes.xaxis.get_major_locator())
        )
    if subtitle is None:
        subtitle = data_compare_info
    elif data_compare_info:
        subtitle += f" | {data_compare_info}"
    if subtitle != None:
        scanvas.set_subtitle(subtitle, fontproperties=font_prop)

    legend_lines = [l for l in scanvas.get_lines() if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in legend_lines]
    scanvas.axes.legend(legend_lines, labels, loc=0, prop=font_prop)


def _set_boxplot_color(box: dict, color: str):
    for item in ["boxes", "whiskers", "fliers", "medians", "caps"]:
        plt.setp(box[item], color=color)
    plt.setp(box["boxes"], facecolor=color)
    plt.setp(box["fliers"], markeredgecolor=color)


def _redraw_canvas_compare_boxplot_only_diffs(
    scanvas: MplCanvas,
    sdata_compare: List[ComparisonData],
    positions: List[float],
    width: float,
    chosen_diffs: CompFields = CompFields.DIFF_REL,
) -> Tuple[dict, str]:
    lines = dict()
    data_compare_info = ""
    label = ""
    if sdata_compare is not None:
        diffs = [sd.get_diffs_and_label(chosen_diffs)[0] for sd in sdata_compare]

        scanvas.clear_twinx()
        ax2 = scanvas.get_twinx()
        if diffs and diffs[0]:
            label = sdata_compare[0].get_diffs_and_label(chosen_diffs)[1]
            x = np.array([dff.data for dff in diffs], dtype=object)
            lines = ax2.boxplot(
                x.T,
                notch=True,
                patch_artist=True,
                manage_ticks=False,
                positions=positions,
                widths=width,
            )
            lines["boxes"][0].set_label(label)
            _set_boxplot_color(lines, _DIFFS_COLOR)
            scanvas.store_ax2_boxplot_lines([lines])
            ax2.set_ylabel(
                label,
                fontproperties=label_font_prop,
            )
            ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(_YAXIS_NTICKS))

            ylim = max(list(map(abs, ax2.get_ylim())))
            if chosen_diffs == CompFields.DIFF_PERC:
                ax2.set_ylim((0.0, ylim + 0.5))
                data_compare_info = "MPD: {:.4f}%".format(
                    np.ma.masked_invalid(
                        [sd.mean_perc_difference for sd in sdata_compare]
                    ).mean()
                )
            else:
                ax2.set_ylim((-ylim - 0.5, ylim + 0.5))
                data_compare_info = "MRD: {:.4f}% | σ: {:.4f}% | MARD: {:.4f}%".format(
                    np.ma.masked_invalid(
                        [sd.mean_relative_difference for sd in sdata_compare]
                    ).mean(),
                    np.ma.masked_invalid(
                        [sd.standard_deviation_mrd for sd in sdata_compare]
                    ).mean(),
                    np.ma.masked_invalid(
                        [sd.mean_absolute_relative_difference for sd in sdata_compare]
                    ).mean(),
                )
        else:
            ax2.set_visible(False)
    return lines, data_compare_info


def _calc_boxplot_width(positions: List[float]):
    if len(positions) > 1:
        mindiff = min([abs(p1 - p0) for p0, p1 in zip(positions[:-1], positions[1:])])
        width = mindiff / 2
    else:
        width = 0.5
    return width


def redraw_canvas_compare_boxplot_only_diffs(
    scanvas: MplCanvas,
    sdata_compare: List[ComparisonData],
    wlens: List[float],
    slegend: List[List[str]],
    subtitle: str = None,
    chosen_diffs: CompFields = CompFields.DIFF_REL,
):
    if sdata_compare:
        wlens = np.array([w for w, s in zip(wlens, sdata_compare) if s is not None])
        sdata_compare = [s for s in sdata_compare if s is not None]
    positions = np.array(wlens)
    width = _calc_boxplot_width(positions)
    lines, data_compare_info = _redraw_canvas_compare_boxplot_only_diffs(
        scanvas, sdata_compare, positions, width, chosen_diffs
    )
    legend_lines = [l["boxes"][0] for l in scanvas.get_boxplot_lines()]
    legend_lines = [l for l in legend_lines if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in legend_lines]
    scanvas.axes.legend(legend_lines, labels, loc=0, prop=font_prop)

    if subtitle is None:
        subtitle = data_compare_info
    elif data_compare_info:
        subtitle += f" | {data_compare_info}"
    if subtitle != None:
        scanvas.set_subtitle(subtitle, fontproperties=font_prop)


def redraw_canvas_compare_boxplot(
    scanvas: MplCanvas,
    sdata_compare: List[ComparisonData],
    wlens: List[float],
    slegend: List[List[str]],
    stitle: str,
    sxlabel: str,
    sylabel: str,
    subtitle: str = None,
    chosen_diffs: CompFields = CompFields.DIFF_REL,
):
    lines = []
    if sdata_compare:
        wlens = np.array([w for w, s in zip(wlens, sdata_compare) if s is not None])
        sdata_compare = [s for s in sdata_compare if s is not None]
    if sdata_compare:
        positions = np.array(wlens)
        width = _calc_boxplot_width(positions)
        x = np.array([sd.observed_signal.data for sd in sdata_compare], dtype=object)
        box = scanvas.axes.boxplot(
            x.T,
            notch=True,
            patch_artist=True,
            manage_ticks=False,
            positions=positions - width / 20,
            widths=width,
        )
        lines += [box]
        _set_boxplot_color(box, _OBS_COLOR)
        x = np.array([sd.simulated_signal.data for sd in sdata_compare], dtype=object)
        box = scanvas.axes.boxplot(
            x.T,
            labels=wlens,
            notch=True,
            patch_artist=True,
            manage_ticks=True,
            positions=positions + width / 20,
            widths=width,
        )
        lines += [box]
        scanvas.store_ax_boxplot_lines(lines)
        _set_boxplot_color(box, _SIM_COLOR)

        if slegend and len(slegend[0]) > 1:
            for box, legend in zip(lines, slegend[0][:2]):
                box["boxes"][0].set_label(legend)

        data_compare_info = ""
        if chosen_diffs != CompFields.DIFF_NONE:
            dlines, data_compare_info = _redraw_canvas_compare_boxplot_only_diffs(
                scanvas, sdata_compare, positions, width, chosen_diffs
            )
            lines += [dlines]
        if subtitle is None:
            subtitle = data_compare_info
        elif data_compare_info:
            subtitle += f" | {data_compare_info}"
        plt.setp(
            scanvas.axes.get_xticklabels(),
            rotation=30,
            horizontalalignment="right",
        )
        scanvas.axes.yaxis.set_major_locator(
            matplotlib.ticker.LinearLocator(_YAXIS_NTICKS)
        )

        legend_lines = [l["boxes"][0] for l in scanvas.get_boxplot_lines()]
        legend_lines = [l for l in legend_lines if not l.get_label().startswith("_")]
        labels = [l.get_label() for l in legend_lines]
        scanvas.axes.legend(legend_lines, labels, loc=0, prop=font_prop)
        scanvas.axes.set_xlim(min(wlens) - width, max(wlens) + width)

    if subtitle != None:
        scanvas.set_subtitle(subtitle, fontproperties=font_prop)
    if stitle:
        scanvas.set_title(stitle, fontproperties=title_font_prop)
    scanvas.axes.set_xlabel(sxlabel, fontproperties=label_font_prop)
    scanvas.axes.set_ylabel(sylabel, fontproperties=label_font_prop)
    scanvas.axes.grid()
    return lines


def redraw_canvas_compare(
    scanvas: MplCanvas,
    sdata_compare: Union[ComparisonData, None],
    slegend: List[List[str]],
    stitle: str,
    sxlabel: str,
    sylabel: str,
    subtitle: str = None,
    chosen_diffs: CompFields = CompFields.DIFF_REL,
):
    lines = []
    if sdata_compare is not None:
        sdata = [sdata_compare.observed_signal, sdata_compare.simulated_signal]
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
            marker = "o"
            markersize = 4
            ls = "none"
            newlines = scanvas.axes.plot(
                data.wlens,
                data.data,
                *color,
                marker=marker,
                label=label,
                markersize=markersize,
                ls=ls,
            )
            if data.uncertainties is not None and data.uncertainties.size > 0:
                if np.any(data.uncertainties):
                    # Not show errorbars if they are all 0 (skipping uncerts):
                    scanvas.axes.errorbar(
                        data.wlens,
                        data.data,
                        yerr=data.uncertainties * 2,
                        color=newlines[-1].get_color(),
                        capsize=2,
                        ls="none",
                        alpha=0.3,
                    )
            lines += newlines
        data_compare_info = ""
        if chosen_diffs != CompFields.DIFF_NONE:
            dlines, data_compare_info = _redraw_canvas_compare_only_diffs(
                scanvas, sdata_compare, chosen_diffs
            )
            lines += dlines
        if subtitle is None:
            subtitle = data_compare_info
        elif data_compare_info:
            subtitle += f" | {data_compare_info}"

        plt.setp(
            scanvas.axes.get_xticklabels(),
            rotation=30,
            horizontalalignment="right",
        )
        scanvas.axes.yaxis.set_major_locator(
            matplotlib.ticker.LinearLocator(_YAXIS_NTICKS)
        )
        if isinstance(data.wlens[0], datetime):
            scanvas.axes.xaxis.set_major_formatter(
                mdates.AutoDateFormatter(scanvas.axes.xaxis.get_major_locator())
            )

        legend_lines = [l for l in lines if not l.get_label().startswith("_child")]
        labels = [l.get_label() for l in legend_lines]
        scanvas.axes.legend(legend_lines, labels, loc=0, prop=font_prop)

    if subtitle != None:
        scanvas.set_subtitle(subtitle, fontproperties=font_prop)
    scanvas.set_title(stitle, fontproperties=title_font_prop)
    scanvas.axes.set_xlabel(sxlabel, fontproperties=label_font_prop)
    scanvas.axes.set_ylabel(sylabel, fontproperties=label_font_prop)
    scanvas.axes.grid()

    return lines


def redraw_canvas(
    scanvas: MplCanvas,
    sdata: Union[SpectralData, List[SpectralData], None],
    slegend: List[List[str]],
    scimel_data: Union[SpectralData, List[SpectralData], None],
    sasd_data: Union[SpectralData, List[SpectralData], None],
    stitle: str,
    sxlabel: str,
    sylabel: str,
    svertical_lines: List[float],
    sp_name: str,
    subtitle: str = None,
    show_cimel_data: bool = True,
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
            markersize = None
            ls = None
            newlines = scanvas.axes.plot(
                data.wlens,
                data.data,
                *color,
                marker=marker,
                label=label,
                markersize=markersize,
                ls=ls,
            )
            if data.uncertainties is not None and data.uncertainties.size > 0:
                scanvas.axes.fill_between(
                    data.wlens,
                    data.data - 2 * data.uncertainties,
                    data.data + 2 * data.uncertainties,
                    color=newlines[-1].get_color(),
                    alpha=0.3,
                )
            lines += newlines

        if scimel_data:
            iter_data = scimel_data
            if not isinstance(iter_data, list):
                iter_data = [iter_data]
            cimel_data = iter_data[0]  # needed later in asd
            if show_cimel_data:
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
                    if np.any(cimel_data.uncertainties):
                        # Not show errorbars if they are all 0 (skipping uncerts)
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

            if np.any(cimel_data.data[cimel_data.wlens == 1020] == 0):
                asd_data_final = asd_data.data * 0
            else:
                scaled_wlen = 1020
                if len(np.where(asd_data.wlens == scaled_wlen)[0]) == 0:
                    scaled_wlen = 1009.59  # Breccia
                scaling_factor = (
                    asd_data.data[np.where(asd_data.wlens == scaled_wlen)]
                    / cimel_data.data[np.where(cimel_data.wlens == 1020)]
                )
                asd_data_final = asd_data.data / scaling_factor
            lines += scanvas.axes.plot(
                asd_data.wlens,
                asd_data_final,
                label=f"{sp_name} data, scaled to LIME at 1020nm",
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
        svlines = []
        for val in svertical_lines:
            svlines.append(
                scanvas.axes.axvline(
                    x=val, color="k", label=constants.LIME_SPECTRUM_LIMIT_LABEL
                )
            )
        scanvas.mpl_cursor = mplcursors.cursor(svlines, hover=2)

        @scanvas.mpl_cursor.connect("add")
        def _(sel):
            sel.annotation.get_bbox_patch().set(fc="white")
            label = sel.artist.get_label()
            sel.annotation.set_text(label)

    scanvas.axes.grid()

    return lines
