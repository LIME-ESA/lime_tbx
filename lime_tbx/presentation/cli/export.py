"""
Data exporting module for LIME TBX's CLI

After parsing the command and calculating its result, this
module is in charge of exporting the results to the selected
kind of output.
"""

from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Tuple

import numpy as np

from lime_tbx.common.constants import CompFields
from lime_tbx.common.datatypes import (
    ComparisonData,
    EocfiPath,
    KernelsPath,
    Point,
    MoonData,
    SpectralResponseFunction,
)
from lime_tbx.application.filedata import csv, lglod as lglodlib
from lime_tbx.application.filedata.lglod_factory import create_lglod_data
from lime_tbx.presentation.gui import constants as gui_constants
from lime_tbx.presentation.gui.settings import SettingsManager
from lime_tbx.application.simulation.lime_simulation import ILimeSimulation

_WARN_OUTSIDE_MPA_RANGE = "Warning: The LIME can only give a reliable simulation \
for absolute moon phase angles between 2° and 90°"


class ExportError(Exception):
    """Exception that encapsules anything that might go wrong during CLI file export."""


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
    o_file_dolp : str
        Path to the output CSV file for degree of linear polarisation.
    o_file_aolp : str
        Path to the output CSV file for angle of linear polarisation.
    o_file_integrated_irr : str
        Path to the output CSV file for integrated irradiance.
    """

    o_file_refl: str
    o_file_irr: str
    o_file_dolp: str
    o_file_aolp: str
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
    o_file_dolp : str
        Path to the output graph file for degree of linear polarisation.
    o_file_aolp : str
        Path to the output graph file for angle of linear polarisation.
    """

    o_file_refl: str
    o_file_irr: str
    o_file_dolp: str
    o_file_aolp: str


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


class CLIExporter:
    """Handles exporting simulation and comparison results from the CLI.

    This class provides methods for saving data in multiple formats, including
    CSV, Graphs, and NetCDF.
    """

    def __init__(
        self,
        kernels_path: KernelsPath,
        eocfi_path: EocfiPath,
        settings_manager: SettingsManager,
        lime_simulation: ILimeSimulation,
    ):
        """Initializes the CLIExporter instance.

        This constructor sets up the necessary paths, settings manager, and simulation
        instance required for exporting simulation and comparison results.

        Parameters
        ----------
        kernels_path : KernelsPath
            Path to the SPICE kernels required for orbital and observational calculations.
        eocfi_path : EocfiPath
            Path to the EO-CFI data directories used for satellite orbit calculations.
        settings_manager : SettingsManager
            Manages interpolation settings, coefficient versions, and spectrum settings.
        lime_simulation : ILimeSimulation
            Handles the execution of lunar irradiance, reflectance, and polarization simulations,
            and contains the results that will be consumed by the methods of this class.
        """
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.settings_manager = settings_manager
        self.lime_simulation = lime_simulation

    def _export_csvs(
        self,
        point: Point,
        ed: ExportCSV,
        srf: SpectralResponseFunction,
    ):
        """Exports simulation results as CSV files.

        Parameters
        ----------
        point : Point
            The simulated observation point (Earth, Lunar, or Satellite).
        ed : ExportCSV
            The CSV export settings.
        srf : SpectralResponseFunction
            The spectral response function used in the simulation.
        """
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
        aolp_sp_name = self.settings_manager.get_selected_aolp_spectrum_name()
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
            ed.o_file_dolp,
            version,
            are_mpas_oinside_mpa_range,
            dolp_sp_name,
            skip_uncs,
            self.lime_simulation.get_polars_cimel(),
            mda,
        )
        csv.export_csv_simulation(
            self.lime_simulation.get_aolp(),
            "Wavelengths (nm)",
            "Angle of Linear Polarisation (°)",
            point,
            ed.o_file_aolp,
            version,
            are_mpas_oinside_mpa_range,
            aolp_sp_name,
            skip_uncs,
            self.lime_simulation.get_aolp_cimel(),
            mda,
        )
        csv.export_csv_integrated_irradiance(
            srf,
            self.lime_simulation.get_signals(),
            ed.o_file_integrated_irr,
            point,
            version,
            are_mpas_oinside_mpa_range,
            sp_name,
            skip_uncs,
            mpa,
        )

    def _export_lglod(
        self,
        point: Point,
        output_file: str,
        srf: SpectralResponseFunction,
    ):
        """Exports simulation results as NetCDF (L-GLOD format).

        Parameters
        ----------
        point : Point
            The simulated observation point (Earth, Lunar, or Satellite).
        output_file : str
            The file path for the NetCDF output.
        srf : SpectralResponseFunction
            The spectral response function used in the simulation.
        """
        sp_name = self.settings_manager.get_selected_spectrum_name()
        dolp_sp_name = self.settings_manager.get_selected_polar_spectrum_name()
        aolp_sp_name = self.settings_manager.get_selected_aolp_spectrum_name()
        version = self.settings_manager.get_lime_coef().version
        mds = self.lime_simulation.get_moon_datas()
        if not isinstance(mds, list):
            mds = [mds]
        lglod = create_lglod_data(
            point,
            srf,
            self.lime_simulation,
            self.kernels_path,
            sp_name,
            dolp_sp_name,
            aolp_sp_name,
            version,
            mds,
        )
        now = datetime.now(timezone.utc)
        inside_mpa_range = self.lime_simulation.are_mpas_inside_mpa_range()
        lglodlib.write_obs(lglod, output_file, now, inside_mpa_range)

    def _export_graph(self, point: Point, ed: ExportGraph):
        """Exports simulation results as graphs.

        Parameters
        ----------
        point : Point
            The simulated observation point (Earth, Lunar, or Satellite).
        ed : ExportGraph
            The graphical export settings.

        Raises
        ------
        ExportError
            If file writing fails.
        """
        from lime_tbx.presentation.gui import canvas

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
        aolp_sp_name = self.settings_manager.get_selected_aolp_spectrum_name()
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
            raise ExportError(
                f"Something went wrong while exporting reflectance graph. {str(e)}"
            ) from e
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
            raise ExportError(
                f"Something went wrong while exporting irradiance graph. {str(e)}"
            ) from e
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
            canv.print_figure(ed.o_file_dolp)
        except Exception as e:
            raise ExportError(
                f"Something went wrong while exporting polarisation graph. {str(e)}"
            ) from e
        canv.axes.cla()  # Clear the canvas.
        spectrum_info = f" | Interp. spectrum: {aolp_sp_name}{mpa_text}"
        subtitle = f"LIME coefficients version: {version}{spectrum_info}{warning_out_mpa_range}"
        canv.set_subtitle(subtitle, fontproperties=canvas.font_prop)
        canv.axes.cla()  # Clear the canvas.
        canvas.redraw_canvas(
            canv,
            self.lime_simulation.get_aolp(),
            [
                [gui_constants.INTERPOLATED_DATA_LABEL],
                [gui_constants.CIMEL_POINT_LABEL],
                [gui_constants.ERRORBARS_LABEL],
            ],
            self.lime_simulation.get_aolp_cimel(),
            self.lime_simulation.get_aolp_asd(),
            None,
            "Extraterrestrial Lunar Angle of Polarisation",
            "Wavelengths (nm)",
            "Angle of Linear Polarisation (°)",
            None,
            aolp_sp_name,
        )
        try:
            canv.print_figure(ed.o_file_aolp)
        except Exception as e:
            raise ExportError(
                f"Something went wrong while exporting angle of polarisation graph. {str(e)}"
            ) from e
        canv.clear()  # Clear the canvas.

    def export_comparison_graph(
        self,
        comparison: ComparisonData,
        xlabel: str,
        ylabel: str,
        output_file: str,
        version: str,
        title: str,
        chosen_diffs: CompFields,
        date_range: Tuple[datetime, datetime] = None,
    ):
        """Exports comparison results as a graph.

        This method generates a graph comparing observed and simulated irradiance values.
        It includes relative or percentage differences based on the chosen difference metric.

        Parameters
        ----------
        comparison : ComparisonData
            The comparison data containing observed and simulated signals.
        xlabel : str
            Label for the x-axis (e.g., "Wavelength (nm)", "UTC Date").
        ylabel : str
            Label for the y-axis (e.g., "Irradiance (Wm⁻²nm⁻¹)").
        output_file : str
            Path to save the exported graph file.
        version : str
            Version of the LIME coefficients used in the simulation.
        title : str
            Title of the graph.
        chosen_diffs : CompFields
            Specifies whether to show relative differences, percentage differences, or none.
        date_range: Optional, Tuple[datetime, datetime]
            Initial and final datetimes, in case the comparison is not an standard comparison,
            and its dts attribute is None.

        Raises
        ------
        ExportError
            If an error occurs while generating or saving the graph.
        """
        from lime_tbx.presentation.gui import canvas

        canv = canvas.MplCanvas(width=15, height=10, dpi=200)
        canv.set_title("", fontproperties=canvas.title_font_prop)
        canv.axes.tick_params(labelsize=8)
        n_comp_points = len(comparison.diffs_signal.wlens)
        if date_range is None:
            data_start = min(comparison.dts)
            data_end = max(comparison.dts)
        else:
            data_start, data_end = date_range
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
            raise ExportError(
                f"Something went wrong while exporting comparison graph. {str(e)}"
            ) from e
        canv.axes.cla()  # Clear the canvas.

    def export_comparison_bywlen_graph(
        self,
        data: List[ComparisonData],
        wlcs: List[float],
        xlabel: str,
        ylabel: str,
        output_file: str,
        version: str,
        chosen_diffs: CompFields,
    ):
        """Exports comparison results by wavelength as a boxplot graph.

        This method visualizes the comparison of observed and simulated data across
        different wavelengths. It includes a statistical representation of differences.

        Parameters
        ----------
        data : List[ComparisonData]
            List of comparison data for each spectral channel.
        wlcs : List[float]
            List of wavelength centers for each channel.
        xlabel : str
            Label for the x-axis (e.g., "Wavelength (nm)").
        ylabel : str
            Label for the y-axis (e.g., "Irradiance (Wm⁻²nm⁻¹)").
        output_file : str
            Path to save the exported graph file.
        version : str
            Version of the LIME coefficients used in the simulation.
        chosen_diffs : CompFields
            Specifies whether to show relative differences, percentage differences, or none.

        Raises
        ------
        ExportError
            If an error occurs while generating or saving the graph.
        """
        from lime_tbx.presentation.gui import canvas

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
            raise ExportError(
                f"Something went wrong while exporting comparison graph. {str(e)}"
            ) from e
        canv.clear()  # Clear the canvas.

    def export(self, point: Point, ed: ExportData, srf: SpectralResponseFunction):
        """Exports simulation or comparison results based on the selected export format.

        This method determines the appropriate export format (CSV, Graph, NetCDF)
        and calls the corresponding function.

        Parameters
        ----------
        point : Point
            The observation point (Earth, Lunar, or Satellite) for the simulation.
        ed : ExportData
            The export configuration object, determining the file format and parameters.
        srf : SpectralResponseFunction
            The spectral response function used in the simulation.

        Raises
        ------
        ExportError
            If an error occurs while exporting the data.
        """
        if isinstance(ed, ExportCSV):
            self._export_csvs(point, ed, srf)
        elif isinstance(ed, ExportNetCDF):
            self._export_lglod(point, ed.output_file, srf)
        elif isinstance(ed, ExportGraph):
            self._export_graph(point, ed)
