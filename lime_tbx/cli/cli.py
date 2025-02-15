"""
Command Line Interface (CLI) module for the LIME Toolbox.

This module handles the interpretation of command-line options and executes 
the appropriate actions, including simulations, comparisons, and data exports.

It supports:
- Simulations of lunar irradiance, reflectance, and polarization.
- Comparisons with observational data.
- Output in various formats (CSV, Graph, NetCDF).
- Updating coefficient datasets.
- Managing interpolation and spectral response function settings.

This module serves as the entry point for the command-line execution of LIME TBX.
"""

from datetime import datetime, timezone
from dataclasses import dataclass
from abc import ABC
import traceback
from typing import List, Union, Tuple
import os
from enum import Enum
import glob
import sys
import json

import numpy as np

from lime_tbx.datatypes.datatypes import (
    ComparisonData,
    CustomPoint,
    KernelsPath,
    LGLODComparisonData,
    LimeException,
    LunarObservation,
    Point,
    SatellitePoint,
    SurfacePoint,
    MoonData,
    EocfiPath,
)
from lime_tbx.datatypes import constants, logger
from lime_tbx.datatypes.constants import CompFields
from lime_tbx.gui import settings, constants as gui_constants
from lime_tbx.simulation.lime_simulation import LimeSimulation, ILimeSimulation
from lime_tbx.simulation.comparison import comparison
from lime_tbx.simulation.comparison.utils import sort_by_mpa, average_comparisons
from lime_tbx.filedata import moon, srf as srflib, csv, lglod as lglodlib
from lime_tbx.filedata.lglod_factory import create_lglod_data
from lime_tbx.coefficients.update.update import IUpdate, Update
from lime_tbx.spectral_integration.spectral_integration import get_default_srf
from lime_tbx.interpolation.interp_data import interp_data


_DT_FORMAT = "%Y-%m-%dT%H:%M:%S"
OPTIONS = "hvude:l:s:co:f:t:C:i:"
LONG_OPTIONS = [
    "help",
    "version",
    "update",
    "debug",
    "earth=",
    "lunar=",
    "satellite=",
    "comparison",
    "output=",
    "srf=",
    "timeseries=",
    "coefficients=",
    "interpolation=",
]
_WARN_OUTSIDE_MPA_RANGE = "Warning: The LIME can only give a reliable simulation \
for absolute moon phase angles between 2° and 90°"
_ERROR_RINDEX_BOTH_DOT = "When creating output as CSV or GRAPH files for both DT and MPA, \
the full CSV/GRAPH filepaths must be explictly written, including the extension \
(.csv, .png, ...).\nAnother solution is to select the CSVD/GRAPHD option where one \
only has to specify the output directory path.\nProblematic filepath: "


def eprint(*args, **kwargs):
    """Prints messages to the standard error (stderr).

    This function behaves like the built-in `print`, but directs output
    to `sys.stderr` instead of `sys.stdout`. It supports the same arguments
    as `print`.

    Parameters
    ----------
    *args : Any
        Positional arguments passed to `print`, representing the values to be printed.
    **kwargs : Any
        Keyword arguments passed to `print` (e.g., `sep`, `end`, `flush`).
    """
    print(*args, file=sys.stderr, **kwargs)


class ExportData(ABC):
    """Abstract base class for export data formats.

    Subclasses define specific export types such as CSV, Graphs, and NetCDF.
    """


@dataclass
class ExportCSV(ExportData):
    """Represents CSV file export settings for simulation results.

    Attributes
    ----------
    o_file_refl : str
        Path to the output CSV file for reflectance.
    o_file_irr : str
        Path to the output CSV file for irradiance.
    o_file_polar : str
        Path to the output CSV file for polarization.
    o_file_integrated_irr : str
        Path to the output CSV file for integrated irradiance.
    """

    o_file_refl: str
    o_file_irr: str
    o_file_polar: str
    o_file_integrated_irr: str


@dataclass
class ExportGraph(ExportData):
    """Represents graphical export settings for simulation results.

    Attributes
    ----------
    o_file_refl : str
        Path to the output graph file for reflectance.
    o_file_irr : str
        Path to the output graph file for irradiance.
    o_file_polar : str
        Path to the output graph file for polarization.
    """

    o_file_refl: str
    o_file_irr: str
    o_file_polar: str


COMP_KEYS = ["DT", "MPA", "BOTH", "CHANNEL", "CHANNEL_MEAN"]


class ComparisonKey(Enum):
    """Enumeration for comparison output modes.

    Attributes
    ----------
    DT : int
        Compare based on UTC datetime.
    MPA : int
        Compare based on Moon Phase Angle.
    BOTH : int
        Compare using both datetime and Moon Phase Angle.
    CHANNEL : int
        Compare by spectral channel.
    CHANNEL_MEAN : int
        Compare using averaged spectral channels.
    """

    DT = 0
    MPA = 1
    BOTH = 2
    CHANNEL = 3
    CHANNEL_MEAN = 4


class ExportComparison(ABC):
    """Abstract base class for exporting comparison results.

    This class serves as a parent for different export formats, including CSV
    and graphical representations of comparison data.
    """


@dataclass
class ExportComparisonCSV(ExportComparison):
    """Represents CSV file export settings for comparison results.

    Attributes
    ----------
    comparison_key : ComparisonKey
        The comparison method (e.g., DT, MPA, BOTH, CHANNEL).
    output_files : List[str]
        List of file paths for saving the comparison data.
        Each comparison type has a different amount of files.
    chosen_diff : CompFields
        The difference metric (relative, percentage, or none).
    """

    comparison_key: ComparisonKey
    output_files: List[str]
    chosen_diff: CompFields


@dataclass
class ExportComparisonCSVDir(ExportComparison):
    """Represents CSV directory export settings for comparison results.

    Instead of exporting individual CSV files, this class allows exporting
    all comparison data to a specified directory, and automatically generating
    the file names.

    Attributes
    ----------
    comparison_key : ComparisonKey
        The comparison method (e.g., DT, MPA, BOTH, CHANNEL).
    output_dir : str
        The directory where CSV files will be saved.
    chosen_diff : CompFields
        The difference metric (relative, percentage, or none).
    """

    comparison_key: ComparisonKey
    output_dir: str
    chosen_diff: CompFields


@dataclass
class ExportComparisonGraph(ExportComparison):
    """Represents graphical export settings for comparison results.

    Attributes
    ----------
    comparison_key : ComparisonKey
        The comparison method (e.g., DT, MPA, BOTH, CHANNEL).
    output_files : List[str]
        List of file paths for saving the comparison graphs.
        Each comparison type has a different amount of files.
    chosen_diff : CompFields
        The difference metric (relative, percentage, or none).
    """

    comparison_key: ComparisonKey
    output_files: List[str]
    chosen_diff: CompFields


@dataclass
class ExportComparisonGraphDir(ExportComparison):
    """Represents graphical export settings for comparison results in a directory.

    This allows exporting comparison graphs to a specified directory with a
    chosen file extension, and automatically generating the file names.

    Attributes
    ----------
    extension : str
        File extension for the exported graphs (e.g., "png", "jpg").
    comparison_key : ComparisonKey
        The comparison method (e.g., DT, MPA, BOTH, CHANNEL).
    output_dir : str
        The directory where graphs will be saved.
    chosen_diff : CompFields
        The difference metric (relative, percentage, or none).
    """

    extension: str
    comparison_key: ComparisonKey
    output_dir: str
    chosen_diff: CompFields


@dataclass
class ExportNetCDF(ExportData, ExportComparison):
    """Represents NetCDF file export settings for simulations and comparisons.

    This format allows storing structured, multi-dimensional scientific data.

    Attributes
    ----------
    output_file : str
        The path to the NetCDF output file.
    """

    output_file: str


IMAGE_EXTENSIONS = ["pdf", "jpg", "png", "svg"]
COMP_DIFF_KEYS = ["rel", "perc", "none"]


def print_help():
    """Displays the command-line help message.

    This function prints usage instructions and a description of available options
    for the LIME Toolbox CLI. It explains how to run simulations, comparisons,
    and specify output formats.

    The available options include:
    - `-h, --help`: Show this help message.
    - `-v, --version`: Display the version number.
    - `-u, --update`: Update the coefficients.
    - `-e, --earth`: Perform simulations from a geographic point.
    - `-l, --lunar`: Perform simulations from a selenographic point.
    - `-s, --satellite`: Perform simulations from a satellite.
    - `-c, --comparison`: Perform comparisons using GLOD observation files.
    - `-o, --output`: Define output format and file paths.
    - `-f, --srf`: Select a Spectral Response Function (SRF) file.
    - `-t, --timeseries`: Use a CSV file with multiple datetimes.
    - `-C, --coefficients`: Change the coefficients version used by the toolbox.
    - `-i, --interpolation-settings`: Modify interpolation settings via JSON input.

    The function also provides examples of valid input formats and highlights
    any constraints or dependencies for specific options.
    """
    compsel = "(" + "|".join(COMP_KEYS) + ")"
    imsel = "(" + "|".join(IMAGE_EXTENSIONS) + ")"
    compdiffsel = "(" + "|".join(COMP_DIFF_KEYS) + ")"
    print(
        "The lime toolbox performs simulations of lunar irradiance, reflectance and \
polarisation for a given point and datetime. It also performs comparisons for some given \
observations files in GLOD format.\n"
    )
    print("It won't work unless given only one of the options (-h|-e|-l|-s|-c).")
    print("")
    print("Options:")
    print("  -h, --help\t\t Displays the help message.")
    print("  -v, --version\t\t Displays the version name.")
    print("  -u, --update\t\t Updates the coefficients.")
    print("  -e, --earth\t\t Performs simulations from a geographic point.")
    print(f"\t\t\t -e lat_deg,lon_deg,height_m,{_DT_FORMAT}")
    print("  -l, --lunar\t\t Performs a simulation from a selenographic point.")
    print(
        "\t\t\t -l distance_sun_moon,distance_observer_moon,selen_obs_lat,selen_obs_lon,\
selen_sun_lon,moon_phase_angle"
    )
    print("  -s, --satellite\t Performs simulations from a satellite point.")
    print(f"\t\t\t -s sat_name,{_DT_FORMAT}")
    print(
        "  -c, --comparison\t Performs comparisons from observations files in GLOD format."
    )
    print('\t\t\t -c "input_glod1.nc input_glod2.nc ..."')
    print("  -o, --output\t\t Select the output path and format.")
    print("\t\t\t If it's a simulation:")
    print(f"\t\t\t   GRAPH: -o graph,{imsel},refl,irr,polar")
    print("\t\t\t   CSV: -o csv,refl.csv,irr.csv,polar.csv,integrated_irr.csv")
    print("\t\t\t   LGLOD (netcdf): -o nc,output_lglod.nc")
    print("\t\t\t If it's a comparison:")
    print(f"\t\t\t   GRAPH: -o graph,{imsel},{compsel},{compdiffsel},output")
    print(f"\t\t\t\t output:")
    print(
        f"\t\t\t\t\t Compare not by channel: comparison_channel1,comparison_channel2,..."
    )
    print(f"\t\t\t\t\t Compare by channel: comparison")
    print(f"\t\t\t   CSV: -o csv,{compsel},{compdiffsel},output")
    print(f"\t\t\t\t output:")
    print(
        f"\t\t\t\t\t Compare not by channel: comparison_channel1.csv,comparison_channel2.csv,..."
    )
    print(f"\t\t\t\t\t Compare by channel: comparison.csv")
    print(
        f"\t\t\t   GRAPH directory: -o graphd,{imsel},{compsel},{compdiffsel},comparison_folder"
    )
    print(f"\t\t\t   CSV directory: -o csvd,{compsel},{compdiffsel},comparison_folder")
    print("\t\t\t   LGLOD (netcdf): -o nc,output_lglod.nc")
    print(
        "  -f, --srf\t\t Select the file that contains the Spectral Response Function \
in GLOD format."
    )
    print(
        "  -t, --timeseries\t Select a CSV file with multiple datetimes instead of \
inputing directly only one datetime. Valid only if the main option is -e or -s."
    )
    print(
        "  -C, --coefficients\t Change the coefficients version used by the TBX, \
for this execution and the next ones until it's changed again."
    )
    print(
        "  -i, --interpolation-settings\t Change the interpolation settings. The \
input data shall be a json string containing at least one of the following parameters:"
    )
    print(
        "\t\t\t   interp_spectrum: Sets the interpolation spectrum. The valid \
values are 'ASD' and 'Apollo 16 + Breccia'."
    )
    print(
        "\t\t\t   interp_srf: Sets the output SRF. The valid values are 'asd', \
'interpolated_gaussian' and 'interpolated_triangle'."
    )
    print(
        "\t\t\t   show_interp_spectrum: Sets if the graphs should show the spectrum \
used for interpolation. The valid values are 'True' and 'False'."
    )
    print(
        "\t\t\t   skip_uncertainties: Sets if the ToolBox should skip the \
uncertainties calculations. The valid values are 'True' and 'False'."
    )
    print(
        "\t\t\t   show_cimel_points: Sets if the graphs should show the CIMEL \
anchor points used for interpolation. The valid values are 'True' and 'False'."
    )


def print_version():
    print(constants.VERSION_NAME)


def _get_chosen_diff_from_cli(param: str) -> CompFields:
    chosen_diff = CompFields.DIFF_NONE
    if param == COMP_DIFF_KEYS[0]:
        chosen_diff = CompFields.DIFF_REL
    elif param == COMP_DIFF_KEYS[1]:
        chosen_diff = CompFields.DIFF_PERC
    return chosen_diff


class CLI:
    """Command Line Interface handler for LIME TBX.

    This class processes command-line arguments, performs simulations, and manages
    comparisons. It serves as the main interface for users executing the toolbox from
    the command line.

    Attributes
    ----------
    kernels_path : KernelsPath
        Path to the SPICE kernels required for calculations.
    eocfi_path : EocfiPath
        Path to the EO-CFI libraries used for orbit calculations.
    settings_manager : settings.SettingsManager
        Manages configuration settings, including coefficients and interpolation.
    lime_simulation : ILimeSimulation
        Handles lunar irradiance, reflectance and polarization simulations.
    srf : datatypes.SpectralResponseFunction
        Spectral Response Function (SRF) used for simulations.
    updater : IUpdate
        Handles updates to coefficient datasets.

    Methods
    -------
    load_srf(srf_file)
        Loads a spectral response function (SRF) from a given file.
    calculate_geographic(lat, lon, height, dt, export_data)
        Runs a simulation from a geographic location.
    calculate_satellital(sat_name, dt, export_data)
        Runs a simulation from a satellite's position.
    calculate_selenographic(distance_sun_moon, distance_observer_moon, ...)
        Runs a simulation from a selenographic location.
    calculate_comparisons(input_files, export_data)
        Performs comparisons using observation files in GLOD format.
    update_coefficients()
        Checks for and updates coefficient datasets.
    parse_interp_settings(arg)
        Parses and applies interpolation settings to the settings manager from a JSON string.
    check_sys_args(sysargs)
        Validates system arguments to prevent syntax errors.
    handle_input(opts, args)
        Processes command-line input and dispatches actions.
    """

    def __init__(
        self,
        kernels_path: KernelsPath,
        eocfi_path: EocfiPath,
        selected_version: str = None,
    ):
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.settings_manager = settings.SettingsManager(selected_version)
        self.lime_simulation: ILimeSimulation = LimeSimulation(
            eocfi_path, kernels_path, self.settings_manager
        )
        self.srf = self.settings_manager.get_default_srf()
        self.updater = Update()

    def load_srf(self, srf_file: str):
        if srf_file == "":
            self.srf = self.settings_manager.get_default_srf()
        else:
            self.srf = srflib.read_srf(srf_file)

    def _calculate_irradiance(self, point: Point):
        def_srf = get_default_srf()
        self.lime_simulation.update_irradiance(
            def_srf, self.srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_reflectance(self, point: Point):
        def_srf = get_default_srf()
        self.lime_simulation.update_reflectance(
            def_srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_polarisation(self, point: Point):
        def_srf = get_default_srf()
        self.lime_simulation.update_polarisation(
            def_srf, point, self.settings_manager.get_polar_coef()
        )

    def _calculate_all(self, point: Point):
        self._calculate_reflectance(point)
        self._calculate_irradiance(point)
        self._calculate_polarisation(point)

    def _export_csvs(
        self,
        point: Point,
        ed: ExportCSV,
    ):
        version = self.settings_manager.get_lime_coef().version
        are_mpas_oinside_mpa_range = self.lime_simulation.are_mpas_inside_mpa_range()
        sp_name = self.settings_manager.get_selected_spectrum_name()
        mdas = self.lime_simulation.get_moon_datas()
        mpa = None
        mda = None
        if isinstance(mdas, MoonData):
            mda = mdas
            mpa = mdas.mpa_degrees
        dolp_sp_name = self.settings_manager.get_selected_polar_spectrum_name()
        skip_uncs = self.settings_manager.is_skip_uncertainties()
        csv.export_csv_simulation(
            self.lime_simulation.get_elrefs(),
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
            point,
            ed.o_file_refl,
            version,
            are_mpas_oinside_mpa_range,
            sp_name,
            skip_uncs,
            self.lime_simulation.get_elrefs_cimel(),
            mda,
        )
        csv.export_csv_simulation(
            self.lime_simulation.get_elis(),
            "Wavelengths (nm)",
            "Irradiances (Wm⁻²nm⁻¹)",
            point,
            ed.o_file_irr,
            version,
            are_mpas_oinside_mpa_range,
            sp_name,
            skip_uncs,
            self.lime_simulation.get_elis_cimel(),
            mda,
        )
        csv.export_csv_simulation(
            self.lime_simulation.get_polars(),
            "Wavelengths (nm)",
            "Degree of Linear Polarisation (%)",
            point,
            ed.o_file_polar,
            version,
            are_mpas_oinside_mpa_range,
            dolp_sp_name,
            skip_uncs,
            self.lime_simulation.get_polars_cimel(),
            mda,
        )
        csv.export_csv_integrated_irradiance(
            self.srf,
            self.lime_simulation.get_signals(),
            ed.o_file_integrated_irr,
            point,
            version,
            are_mpas_oinside_mpa_range,
            sp_name,
            skip_uncs,
            mpa,
        )

    def _export_lglod(self, point: Point, output_file: str):
        sp_name = self.settings_manager.get_selected_spectrum_name()
        dolp_sp_name = self.settings_manager.get_selected_polar_spectrum_name()
        version = self.settings_manager.get_lime_coef().version
        mds = self.lime_simulation.get_moon_datas()
        if not isinstance(mds, list):
            mds = [mds]
        lglod = create_lglod_data(
            point,
            self.srf,
            self.lime_simulation,
            self.kernels_path,
            sp_name,
            dolp_sp_name,
            version,
            mds,
        )
        now = datetime.now(timezone.utc)
        inside_mpa_range = self.lime_simulation.are_mpas_inside_mpa_range()
        lglodlib.write_obs(lglod, output_file, now, inside_mpa_range)

    def _export_graph(self, point: Point, ed: ExportGraph):
        from lime_tbx.gui import canvas

        canv = canvas.MplCanvas(width=15, height=10, dpi=200)
        canv.set_title("", fontproperties=canvas.title_font_prop)
        canv.axes.tick_params(labelsize=8)
        version = self.settings_manager.get_lime_coef().version
        inside_mpa_range = self.lime_simulation.are_mpas_inside_mpa_range()
        is_out_mpa_range = (
            not inside_mpa_range
            if not isinstance(inside_mpa_range, list)
            else False in inside_mpa_range
        )
        warning_out_mpa_range = ""
        if is_out_mpa_range:
            warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
        sp_name = self.settings_manager.get_selected_spectrum_name()
        mdas = self.lime_simulation.get_moon_datas()
        mpa = None
        if isinstance(mdas, MoonData):
            mpa = mdas.mpa_degrees
        mpa_text = ""
        if mpa is not None:
            mpa_text = f" | MPA: {mpa:.3f}°"
        spectrum_info = f" | Interp. spectrum: {sp_name}{mpa_text}"
        subtitle = f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}"
        canv.set_subtitle(subtitle, fontproperties=canvas.font_prop)
        canv.axes.set_xlabel("Wavelengths (nm)", fontproperties=canvas.label_font_prop)
        canv.axes.set_ylabel("", fontproperties=canvas.label_font_prop)
        canv.axes.cla()  # Clear the canvas.
        sp_name = self.settings_manager.get_selected_spectrum_name()
        dolp_sp_name = self.settings_manager.get_selected_polar_spectrum_name()
        canvas.redraw_canvas(
            canv,
            self.lime_simulation.get_elrefs(),
            [
                [gui_constants.INTERPOLATED_DATA_LABEL],
                [gui_constants.CIMEL_POINT_LABEL],
                [gui_constants.ERRORBARS_LABEL],
            ],
            self.lime_simulation.get_elrefs_cimel(),
            self.lime_simulation.get_elrefs_asd(),
            None,
            "Extraterrestrial Lunar Reflectances",
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
            None,
            sp_name,
        )
        try:
            canv.print_figure(ed.o_file_refl)
        except Exception as e:
            eprint(
                "Something went wrong while exporting reflectance graph. {}".format(
                    str(e)
                )
            )
            sys.exit(1)
        canv.axes.cla()  # Clear the canvas.
        canvas.redraw_canvas(
            canv,
            self.lime_simulation.get_elis(),
            [
                [gui_constants.INTERPOLATED_DATA_LABEL],
                [gui_constants.CIMEL_POINT_LABEL],
                [gui_constants.ERRORBARS_LABEL],
            ],
            self.lime_simulation.get_elis_cimel(),
            self.lime_simulation.get_elis_asd(),
            None,
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances (Wm⁻²nm⁻¹)",
            None,
            sp_name,
        )
        try:
            canv.print_figure(ed.o_file_irr)
        except Exception as e:
            eprint(
                "Something went wrong while exporting irradiance graph. {}".format(
                    str(e)
                )
            )
            sys.exit(1)
        canv.axes.cla()  # Clear the canvas.
        spectrum_info = f" | Interp. spectrum: {dolp_sp_name}{mpa_text}"
        subtitle = f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}"
        canv.set_subtitle(subtitle, fontproperties=canvas.font_prop)
        canv.axes.cla()  # Clear the canvas.
        canvas.redraw_canvas(
            canv,
            self.lime_simulation.get_polars(),
            [
                [gui_constants.INTERPOLATED_DATA_LABEL],
                [gui_constants.CIMEL_POINT_LABEL],
                [gui_constants.ERRORBARS_LABEL],
            ],
            self.lime_simulation.get_polars_cimel(),
            self.lime_simulation.get_polars_asd(),
            None,
            "Extraterrestrial Lunar Polarisation",
            "Wavelengths (nm)",
            "Degree of Linear Polarisation (%)",
            None,
            dolp_sp_name,
        )
        try:
            canv.print_figure(ed.o_file_polar)
        except Exception as e:
            eprint(
                "Something went wrong while exporting polarisation graph. {}".format(
                    str(e)
                )
            )
            sys.exit(1)
        canv.clear()  # Clear the canvas.

    def _export_comparison_graph(
        self,
        comparison: ComparisonData,
        xlabel: str,
        ylabel: str,
        output_file: str,
        version: str,
        title: str,
        chosen_diffs: CompFields,
    ):
        from lime_tbx.gui import canvas

        canv = canvas.MplCanvas(width=15, height=10, dpi=200)
        canv.set_title("", fontproperties=canvas.title_font_prop)
        canv.axes.tick_params(labelsize=8)
        n_comp_points = len(comparison.diffs_signal.wlens)
        data_start = min(comparison.dts)
        data_end = max(comparison.dts)
        version = self.settings_manager.get_lime_coef().version
        warning_out_mpa_range = ""
        if False in comparison.ampa_valid_range:
            warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
        sp_name = self.settings_manager.get_selected_spectrum_name()
        spectrum_info = f" | Interp. spectrum: {sp_name}"
        subtitle = f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}"
        _subtitle_date_format = canvas.SUBTITLE_DATE_FORMAT
        subtitle = "{}\nData start: {} | Data end: {}\nNumber of points: {}".format(
            subtitle,
            data_start.strftime(_subtitle_date_format),
            data_end.strftime(_subtitle_date_format),
            n_comp_points,
        )
        canv.set_subtitle(subtitle, fontproperties=canvas.font_prop)
        canv.axes.set_xlabel("Wavelengths (nm)", fontproperties=canvas.label_font_prop)
        canv.axes.set_ylabel("", fontproperties=canvas.label_font_prop)
        canv.axes.cla()  # Clear the canvas.
        canvas.redraw_canvas_compare(
            canv,
            comparison,
            [
                ["Observed Irradiance", "Simulated Irradiance"],
                ["Relative Differences", "Percentage Differences"],
            ],
            title,
            xlabel,
            ylabel,
            subtitle,
            chosen_diffs,
        )
        try:
            canv.print_figure(output_file)
        except Exception as e:
            eprint(
                "Something went wrong while exporting comparison graph. {}".format(
                    str(e)
                )
            )
            sys.exit(1)
        canv.axes.cla()  # Clear the canvas.

    def _export_comparison_bywlen_graph(
        self,
        data: List[ComparisonData],
        wlcs: List[float],
        xlabel: str,
        ylabel: str,
        output_file: str,
        version: str,
        chosen_diffs: CompFields,
    ):
        from lime_tbx.gui import canvas

        canv = canvas.MplCanvas(width=15, height=10, dpi=200)
        canv.set_title("", fontproperties=canvas.title_font_prop)
        canv.axes.tick_params(labelsize=8)
        version = self.settings_manager.get_lime_coef().version
        comps = [c if c.observed_signal is not None else None for c in data]
        statscomps = [c for c in comps if c is not None]
        n_comp_points = np.mean([len(c.diffs_signal.wlens) for c in statscomps])
        data_start = min([min(c.dts) for c in statscomps])
        data_end = max([max(c.dts) for c in statscomps])
        warning_out_mpa_range = ""
        if False in [not np.all(c.ampa_valid_range) for c in statscomps]:
            warning_out_mpa_range = f"\n{_WARN_OUTSIDE_MPA_RANGE}"
        sp_name = self.settings_manager.get_selected_spectrum_name()
        spectrum_info = f" | Interp. spectrum: {sp_name}"
        subtitle = f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}"
        _subtitle_date_format = canvas.SUBTITLE_DATE_FORMAT
        subtitle = (
            "{}\nData start: {} | Data end: {}\nMean number of points: {}".format(
                subtitle,
                data_start.strftime(_subtitle_date_format),
                data_end.strftime(_subtitle_date_format),
                n_comp_points,
            )
        )
        canvas.redraw_canvas_compare_boxplot(
            canv,
            data,
            wlcs,
            [
                ["Observed Irradiance", "Simulated Irradiance"],
            ],
            "All channels",
            xlabel,
            ylabel,
            subtitle,
            chosen_diffs,
        )
        try:
            canv.print_figure(output_file)
        except Exception as e:
            eprint(
                "Something went wrong while exporting comparison graph. {}".format(
                    str(e)
                )
            )
            sys.exit(1)
        canv.clear()  # Clear the canvas.

    def _export(self, point: Point, ed: ExportData):
        if isinstance(ed, ExportCSV):
            self._export_csvs(point, ed)
        elif isinstance(ed, ExportNetCDF):
            self._export_lglod(point, ed.output_file)
        elif isinstance(ed, ExportGraph):
            self._export_graph(point, ed)

    def calculate_geographic(
        self,
        lat: float,
        lon: float,
        height: float,
        dt: Union[datetime, List[datetime]],
        export_data: ExportData,
    ):
        point = SurfacePoint(lat, lon, height, dt)
        self._calculate_all(point)
        self._export(point, export_data)

    def calculate_satellital(
        self,
        sat_name: str,
        dt: Union[datetime, List[datetime]],
        export_data: ExportData,
    ):
        point = SatellitePoint(sat_name, dt)
        self._calculate_all(point)
        self._export(point, export_data)

    def calculate_selenographic(
        self,
        distance_sun_moon: float,
        distance_observer_moon: float,
        selen_obs_lat: float,
        selen_obs_lon: float,
        selen_sun_lon: float,
        moon_phase_angle: float,
        export_data: ExportData,
    ):
        point = CustomPoint(
            distance_sun_moon,
            distance_observer_moon,
            selen_obs_lat,
            selen_obs_lon,
            selen_sun_lon,
            abs(moon_phase_angle),
            moon_phase_angle,
        )
        self._calculate_all(point)
        self._export(point, export_data)

    def _add_observation(self, obs: LunarObservation):
        for i, pob in enumerate(self.loaded_moons):
            if obs.dt < pob.dt:
                self.loaded_moons.insert(i, obs)
                return
        self.loaded_moons.append(obs)

    def calculate_comparisons(
        self,
        input_files: List[str],
        ed: ExportComparison,
    ):
        self.loaded_moons: List[LunarObservation] = []
        for path in input_files:
            self._add_observation(
                moon.read_moon_obs(path, self.kernels_path, self.eocfi_path)
            )
        if len(self.loaded_moons) == 0:
            raise LimeException("No observations given. Aborting.")
        mos = self.loaded_moons
        if isinstance(ed, ExportComparisonCSV) or isinstance(ed, ExportComparisonGraph):
            ch_names_obs = {
                ch_name for mo in mos for ch_name in list(mo.ch_irrs.keys())
            }
            if ed.comparison_key not in (
                ComparisonKey.CHANNEL,
                ComparisonKey.CHANNEL_MEAN,
            ) and len(ch_names_obs) > len(ed.output_files):
                raise LimeException(
                    "The amount of export files given is not enough. There are more channels."
                )
        for mo in mos:
            if not mo.check_valid_srf(self.srf):
                srf_names = self.srf.get_channels_names()
                if len(mo.ch_names) == len(srf_names):
                    for i in range(len(mo.ch_names)):
                        if mo.ch_names[i] in mo.ch_irrs:
                            mo.ch_irrs[srf_names[i]] = mo.ch_irrs.pop(mo.ch_names[i])
                        mo.ch_names[i] = srf_names[i]
                else:
                    raise LimeException(
                        "SRF file not valid for the chosen Moon observations file."
                    )
        co = comparison.Comparison(self.kernels_path)
        cimel_coef = self.settings_manager.get_cimel_coef()
        comps = co.get_simulations(mos, self.srf, cimel_coef, self.lime_simulation)
        # EXPORT
        skip_uncs = self.settings_manager.is_skip_uncertainties()
        if isinstance(ed, ExportNetCDF):
            vers = self.settings_manager.get_lime_coef().version
            lglod = LGLODComparisonData(
                comps,
                self.srf.get_channels_names(),
                mos[0].data_source,
                self.settings_manager.get_selected_spectrum_name(),
                skip_uncs,
                vers,
            )
            lglodlib.write_comparison(
                lglod,
                ed.output_file,
                datetime.now().astimezone(timezone.utc),
                self.kernels_path,
            )
        else:
            if isinstance(ed, ExportComparisonCSVDir) or isinstance(
                ed, ExportComparisonGraphDir
            ):
                if not os.path.exists(ed.output_dir):
                    os.makedirs(ed.output_dir)
            version = self.settings_manager.get_lime_coef().version
            ch_names = self.srf.get_channels_names()
            file_index = 0
            is_both = ed.comparison_key == ComparisonKey.BOTH
            sp_name = self.settings_manager.get_selected_spectrum_name()
            if ed.comparison_key in (ComparisonKey.DT, ComparisonKey.BOTH):
                for i, ch in enumerate(ch_names):
                    if len(comps[i].dts) > 0:
                        ylabel = "Irradiance (Wm⁻²nm⁻¹)"
                        ylabels = [f"Observed {ylabel}", f"Simulated {ylabel}"]
                        output = ""
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonGraph
                        ):
                            output = ed.output_files[file_index]
                        elif isinstance(ed, ExportComparisonCSVDir):
                            output = "{}.csv".format(os.path.join(ed.output_dir, ch))
                        elif isinstance(ed, ExportComparisonGraphDir):
                            output = "{}.{}".format(
                                os.path.join(ed.output_dir, ch), ed.extension
                            )
                        if is_both:
                            if "." not in output:
                                raise LimeException(_ERROR_RINDEX_BOTH_DOT + output)
                            idx = output.rindex(".")
                            output = output[:idx] + ".dt" + output[idx:]
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonCSVDir
                        ):
                            xlabel = "UTC Date"
                            csv.export_csv_comparison(
                                comps[i],
                                xlabel,
                                ylabels,
                                output,
                                version,
                                sp_name,
                                skip_uncs,
                                ed.chosen_diff,
                            )
                        else:
                            xlabel = "UTC Date"
                            self._export_comparison_graph(
                                comps[i],
                                xlabel,
                                ylabels,
                                output,
                                version,
                                ch,
                                ed.chosen_diff,
                            )
                        file_index += 1
            file_index = 0
            if ed.comparison_key in (ComparisonKey.MPA, ComparisonKey.BOTH):
                mpa_comps = sort_by_mpa(comps)
                for i, ch in enumerate(ch_names):
                    if len(mpa_comps[i].dts) > 0:
                        ylabel = "Irradiance (Wm⁻²nm⁻¹)"
                        ylabels = [f"Observed {ylabel}", f"Simulated {ylabel}"]
                        output = ""
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonGraph
                        ):
                            output = ed.output_files[file_index]
                        elif isinstance(ed, ExportComparisonCSVDir):
                            output = "{}.csv".format(os.path.join(ed.output_dir, ch))
                        elif isinstance(ed, ExportComparisonGraphDir):
                            output = "{}.{}".format(
                                os.path.join(ed.output_dir, ch), ed.extension
                            )
                        if is_both:
                            if "." not in output:
                                raise LimeException(_ERROR_RINDEX_BOTH_DOT + output)
                            idx = output.rindex(".")
                            output = output[:idx] + ".mpa" + output[idx:]
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonCSVDir
                        ):
                            xlabel = "Moon Phase Angle (degrees)"
                            csv.export_csv_comparison(
                                mpa_comps[i],
                                xlabel,
                                ylabels,
                                output,
                                version,
                                sp_name,
                                skip_uncs,
                                ed.chosen_diff,
                            )
                        else:
                            xlabel = "Moon phase angle (degrees)"
                            self._export_comparison_graph(
                                mpa_comps[i],
                                xlabel,
                                ylabel,
                                output,
                                version,
                                ch,
                                ed.chosen_diff,
                            )
                        file_index += 1
            if ed.comparison_key in (ComparisonKey.CHANNEL, ComparisonKey.CHANNEL_MEAN):
                wlcs = self.srf.get_channels_centers()
                comps = [c if c.observed_signal is not None else None for c in comps]
                wlcs = np.array([w for w, c in zip(wlcs, comps) if c is not None])
                comps = [c for c in comps if c is not None]
                xlabel = "Wavelength (nm)"
                ylabel = "Irradiance (Wm⁻²nm⁻¹)"
                ylabels = [f"Observed {ylabel}", f"Simulated {ylabel}"]
                output = ""
                if isinstance(ed, ExportComparisonCSV) or isinstance(
                    ed, ExportComparisonGraph
                ):
                    output = ed.output_files[0]
                elif isinstance(ed, ExportComparisonCSVDir):
                    output = "{}.csv".format(os.path.join(ed.output_dir, "allchannels"))
                elif isinstance(ed, ExportComparisonGraphDir):
                    output = "{}.{}".format(
                        os.path.join(ed.output_dir, "allchannels"), ed.extension
                    )
                    if ed.comparison_key == ComparisonKey.CHANNEL:
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonCSVDir
                        ):
                            csv.export_csv_comparison_bywlen(
                                comps,
                                wlcs,
                                xlabel,
                                ylabels,
                                output,
                                version,
                                sp_name,
                                skip_uncs,
                                ed.chosen_diff,
                            )
                        else:
                            self._export_comparison_bywlen_graph(
                                comps,
                                wlcs,
                                xlabel,
                                ylabel,
                                output,
                                version,
                                ed.chosen_diff,
                            )
                    else:
                        comp = average_comparisons(wlcs, comps)
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonCSVDir
                        ):
                            csv.export_csv_comparison(
                                comp,
                                xlabel,
                                ylabels,
                                output,
                                version,
                                sp_name,
                                skip_uncs,
                                ed.chosen_diff,
                            )
                        else:
                            self._export_comparison_graph(
                                comp,
                                xlabel,
                                ylabel,
                                output,
                                version,
                                "All channels",
                                ed.chosen_diff,
                            )

    def update_coefficients(self) -> int:
        updater: IUpdate = self.updater
        stopper_checker_true = lambda *_: True
        updates = False
        try:
            if updater.check_for_updates():
                news, fails = updater.download_coefficients(stopper_checker_true, [])
                updates = True
        except Exception as error:
            print("Error connecting to the server.\nCheck log for details.")
            logger.get_logger().error(str(error))
            return 1
        msg = "Download finished.\nThere were no updates."
        if updates:
            newsstring = f"There was 1 update"
            failsstring = f"it failed"
            if news > 1:
                newsstring = f"There were {news} updates"
                failsstring = f"{fails} of them failed"
            if fails == 0:
                msg = f"Download finished.\n{newsstring}."
            else:
                msg = f"Download finished with errors.\n{newsstring}, {failsstring}."
        print(msg)
        if updates:
            self.settings_manager.reload_coeffs()
        return 0

    def parse_interp_settings(self, arg: str) -> int:
        # example: -i '{"interp_spectrum": "ASD", "skip_uncertainties": "False", "show_interp_spectrum": "False", "interp_srf": "interpolated_gaussian"}'
        try:
            interp_settings = json.loads(arg)
        except Exception as e:
            eprint(f"Error parsing the interpolation settings {arg}. Error: {e}")
            return 1
        if "interp_spectrum" in interp_settings:
            interp_spectrum = interp_settings["interp_spectrum"]
            names = self.settings_manager.get_available_spectra_names()
            if interp_spectrum not in names:
                eprint(
                    f"Interpolation spectrum not recognized. Selected: {interp_spectrum}. Available: {names}."
                )
                return 1
            self.settings_manager.select_interp_spectrum(interp_spectrum)
        if "interp_srf" in interp_settings:
            interp_srf = interp_settings["interp_srf"]
            srf_translator = {
                v: k for k, v in interp_data.SRF_DICT_SOLAR_DIALOG_SRF_TYPE.items()
            }
            names = list(srf_translator.keys())
            if interp_srf not in names:
                eprint(
                    f"Interpolation settings output SRF not recognized. Selected: {interp_srf}. Available: {names}."
                )
                return 1
            self.settings_manager.select_interp_SRF(srf_translator[interp_srf])
        if "show_interp_spectrum" in interp_settings:
            show_interp_spectrum = interp_settings["show_interp_spectrum"]
            if show_interp_spectrum not in ("True", "False"):
                eprint(
                    f'Interpolation settings show_interp_spectrum value {show_interp_spectrum} not valid. Must be "True" or "False"'
                )
                return 1
            show_interp_spectrum = show_interp_spectrum == "True"
            self.settings_manager.set_show_interp_spectrum(show_interp_spectrum)
        if "skip_uncertainties" in interp_settings:
            skip_uncertainties = interp_settings["skip_uncertainties"]
            if skip_uncertainties not in ("True", "False"):
                eprint(
                    f'Interpolation settings skip_uncertainties value {skip_uncertainties} not valid. Must be "True" or "False"'
                )
                return 1
            skip_uncertainties = skip_uncertainties == "True"
            self.settings_manager.set_skip_uncertainties(skip_uncertainties)
        if "show_cimel_points" in interp_settings:
            show_cimel_points = interp_settings["show_cimel_points"]
            if show_cimel_points not in ("True", "False"):
                eprint(
                    f'Interpolation settings show_cimel_points value {show_cimel_points} not valid. Must be "True" or "False"'
                )
                return 1
            show_cimel_points = show_cimel_points == "True"
            self.settings_manager.set_show_cimel_points(show_cimel_points)
        return 0

    def check_sys_args(self, sysargs: List[str]) -> int:
        # Check if the user has forgotten one dash, or has set one dash but all together
        if any(
            item.startswith("-") and not item.startswith("--") and len(item) > 2
            for item in sysargs
        ):
            problem_flags = [
                item
                for item in sysargs
                if item.startswith("-")
                and not item[0].startswith("--")
                and len(item) > 2
            ]
            eprint(
                f"The flags must be separated from their argument/s by at least one blank space, \
and the flags set with only one dash '-' only have one letter. Problematic flags: {problem_flags}.\n\
Run 'lime -h' for help."
            )
            return 1
        return 0

    def handle_input(self, opts: List[Tuple[str, str]], args: List[str]) -> int:
        srf_file = ""
        export_data: ExportData = None
        timeseries_file: str = None
        # Check if it's comparison
        is_comparison = any(item[0] in ("-c", "--comparison") for item in opts)
        mod_interp_settings = False
        # find settings data
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_help()
                return 0
            if opt in ("-v", "--version"):
                print_version()
                return 0
            if opt in ("-u", "--update"):
                return self.update_coefficients()
            if opt in ("-o", "--output"):  # Output
                splitted = arg.split(",")
                o_type = splitted[0]
                if o_type == "csv":
                    if not is_comparison:
                        if len(splitted) != 5:
                            eprint("Error: Wrong number of arguments for -o csv,...")
                            return 1
                        export_data = ExportCSV(
                            splitted[1], splitted[2], splitted[3], splitted[4]
                        )
                    else:
                        if len(splitted) < 4:
                            eprint("Error: Wrong number of arguments for -o csv,...")
                            return 1
                        if splitted[1] not in COMP_KEYS:
                            eprint("Error in csv DT|MPA|BOTH parameter.")
                            return 1
                        if splitted[2] not in COMP_DIFF_KEYS:
                            eprint("Error in csv rel|perc parameter.")
                            return 1
                        comp_key = ComparisonKey[splitted[1]]
                        chosen_diff = _get_chosen_diff_from_cli(splitted[2])
                        export_data = ExportComparisonCSV(
                            comp_key, splitted[3:], chosen_diff
                        )
                elif o_type == "graph":
                    if not is_comparison:
                        if len(splitted) != 5:
                            eprint("Error: Wrong number of arguments for -o graph,...")
                            return 1
                        if splitted[1] not in IMAGE_EXTENSIONS:
                            eprint(
                                "Error: Graph format not detected. It must be one of the following: {}.".format(
                                    ",".join(IMAGE_EXTENSIONS)
                                )
                            )
                            return 1
                        filepaths = list(
                            map(lambda s: s + ".{}".format(splitted[1]), splitted[2:])
                        )
                        export_data = ExportGraph(*filepaths)
                    else:
                        if len(splitted) < 5:
                            eprint("Error: Wrong number of arguments for -o graph,...")
                            return 1
                        if splitted[1] not in IMAGE_EXTENSIONS:
                            eprint(
                                "Error: Graph format not detected. It must be one of the following: {}.".format(
                                    ",".join(IMAGE_EXTENSIONS)
                                )
                            )
                            return 1
                        if splitted[2] not in COMP_KEYS:
                            eprint("Error in graph DT|MPA|BOTH parameter.")
                            return 1
                        if splitted[3] not in COMP_DIFF_KEYS:
                            eprint("Error in graph rel|perc|none parameter.")
                            return 1
                        filepaths = list(
                            map(lambda s: s + ".{}".format(splitted[1]), splitted[4:])
                        )
                        comp_key = ComparisonKey[splitted[2]]
                        chosen_diff = _get_chosen_diff_from_cli(splitted[3])
                        export_data = ExportComparisonGraph(
                            comp_key,
                            filepaths,
                            chosen_diff,
                        )
                elif o_type == "nc":
                    if len(splitted) != 2:
                        eprint("Error: Wrong number of arguments for -o nc,...")
                        return 1
                    export_data = ExportNetCDF(splitted[1])
                elif o_type == "csvd":
                    if not is_comparison:
                        eprint("Error: csvd output is only available for comparisons.")
                        return 1
                    if len(splitted) != 4:
                        eprint("Error: Wrong number of arguments for -o csvd,...")
                        return 1
                    if splitted[1] not in COMP_KEYS:
                        eprint("Error in csvd DT|MPA|BOTH parameter.")
                        return 1
                    if splitted[2] not in COMP_DIFF_KEYS:
                        eprint("Error in csvd rel|perc parameter.")
                        return 1
                    comp_key = ComparisonKey[splitted[1]]
                    chosen_diff = _get_chosen_diff_from_cli(splitted[2])
                    export_data = ExportComparisonCSVDir(
                        comp_key, splitted[3], chosen_diff
                    )
                elif o_type == "graphd":
                    if not is_comparison:
                        eprint(
                            "Error: graphd output is only available for comparisons."
                        )
                        return 1
                    if len(splitted) != 5:
                        eprint("Error: Wrong number of arguments for -o graphd,...")
                        return 1
                    if splitted[1] not in IMAGE_EXTENSIONS:
                        eprint(
                            "Error: Graph format not detected. It must be one of the following: {}.".format(
                                ",".join(IMAGE_EXTENSIONS)
                            )
                        )
                        return 1
                    if splitted[2] not in COMP_KEYS:
                        eprint("Error in graphd DT|MPA|BOTH parameter.")
                        return 1
                    if splitted[3] not in COMP_DIFF_KEYS:
                        eprint("Error in graphd rel|perc parameter.")
                        return 1
                    comp_key = ComparisonKey[splitted[2]]
                    chosen_diff = _get_chosen_diff_from_cli(splitted[3])
                    export_data = ExportComparisonGraphDir(
                        splitted[1],
                        comp_key,
                        splitted[4],
                        chosen_diff,
                    )
            elif opt in ("-f", "--srf"):
                srf_file = arg
            elif opt in ("-t", "--timeseries"):
                timeseries_file = arg
            elif opt in ("-C", "--coefficients"):
                names = sorted(
                    [
                        coef.version
                        for coef in self.settings_manager.get_available_coeffs()
                    ]
                )
                if arg not in names:
                    eprint(
                        f"Coefficients version not recognized. Selected: {arg}. Available: {names}."
                    )
                    return 1
                self.settings_manager.select_lime_coeff(names.index(arg))
            elif opt in ("-i", "--interpolation-settings"):
                ret = self.parse_interp_settings(arg)
                if ret != 0:
                    return ret
                mod_interp_settings = True

        # Simulation input
        sim_opts = (
            "-e",
            "--earth",
            "-s",
            "--satellite",
            "-l",
            "--lunar",
            "-c",
            "--comparison",
        )
        num_sim_ops = sum(item[0] in sim_opts for item in opts)
        if mod_interp_settings and num_sim_ops == 0:
            return 0

        if srf_file == "" or os.path.exists(srf_file):
            try:
                self.load_srf(srf_file)
            except Exception as e:
                eprint(
                    "Error: Error loading Spectral Response Function. {}".format(str(e))
                )
                return 1
        else:
            eprint("Error: The given srf path does not exist.")
            return 1

        if export_data == None:
            eprint("Error: The output flag (-o | --output) must be included.")
            return 1
        if num_sim_ops == 0:
            eprint("Error: There must be one of the following flags: (-e|-s|-l|-c|-i).")
            print_help()
            return 1
        elif num_sim_ops > 1:
            eprint(
                "Error: There can only be one of the following flags: (-e|-s|-l|-c)."
            )
            print_help()
            return 1
        timeseries = None
        if timeseries_file != None and any(
            item[0] in ("-e", "--earth", "-s", "--satellite") for item in opts
        ):
            if os.path.exists(timeseries_file):
                try:
                    timeseries = csv.read_datetimes(timeseries_file)
                except Exception as e:
                    eprint("Error reading timeseries file: {}".format(str(e)))
                    return 1
            else:
                eprint("Error: Timeseries file does not exist.")
                return 1
        try:
            for opt, arg in opts:
                if opt in ("-e", "--earth"):  # Earth
                    params_str = arg.split(",")
                    lenpar = len(params_str)
                    if lenpar < 3 or (timeseries_file == None and lenpar != 4):
                        eprint("Error: Wrong number of arguments for -e")
                        return 1
                    params = list(map(float, params_str[:3]))
                    lat = params[0]
                    lon = params[1]
                    height = params[2]
                    if timeseries_file != None:
                        dt = timeseries
                    else:
                        dt = datetime.strptime(
                            params_str[3] + "+00:00", _DT_FORMAT + "%z"
                        )
                    self.calculate_geographic(lat, lon, height, dt, export_data)
                    break
                elif opt in ("-s", "--satellite"):  # Satellite
                    params_str = arg.split(",")
                    lenpar = len(params_str)
                    if lenpar < 1 or (timeseries_file == None and lenpar != 2):
                        eprint("Error: Wrong number of arguments for -s")
                        return 1
                    sat_name = params_str[0]
                    if timeseries_file != None:
                        dt = timeseries
                    else:
                        dt = datetime.strptime(
                            params_str[1] + "+00:00", _DT_FORMAT + "%z"
                        )
                    self.calculate_satellital(sat_name, dt, export_data)
                    break
                elif opt in ("-l", "--lunar"):  # Lunar
                    params_str = arg.split(",")
                    if len(params_str) != 6:
                        eprint("Error: Wrong number of arguments for -l")
                        return 1
                    params = list(map(float, params_str))
                    self.calculate_selenographic(*params, export_data)
                    break
                elif opt in ("-c", "--comparison"):  # Comparison
                    params = args
                    if len(params) == 1:
                        params = params[0].split(" ")
                    input_files = []
                    for param in params:
                        input_files += glob.glob(param)
                    self.calculate_comparisons(input_files, export_data)
                    break
        except LimeException as e:
            eprint("Error: {}".format(str(e)))
            return 1
        except Exception as e:
            trace = traceback.format_exc()
            eprint("Error when performing operations: {}".format(str(e)))
            logger.get_logger().critical(f"Error trace: {trace}")
            return 1
        return 0
