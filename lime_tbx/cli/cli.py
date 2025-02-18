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

# TODO: Refactor module. Split parsing from calculating, multiple submodules. Study migrating
# to a library that already implements arguments handling.

from datetime import datetime, timezone
import traceback
from typing import List, Union, Tuple
import os
import glob
import sys
import json

import numpy as np

from lime_tbx.datatypes.datatypes import (
    CustomPoint,
    KernelsPath,
    LGLODComparisonData,
    LimeException,
    LunarObservation,
    Point,
    SatellitePoint,
    SurfacePoint,
    EocfiPath,
)
from lime_tbx.datatypes import constants, logger
from lime_tbx.datatypes.constants import CompFields
from lime_tbx.gui import settings
from lime_tbx.simulation.lime_simulation import LimeSimulation, ILimeSimulation
from lime_tbx.simulation.comparison import comparison
from lime_tbx.simulation.comparison.utils import sort_by_mpa, average_comparisons
from lime_tbx.filedata import moon, srf as srflib, csv, lglod as lglodlib
from lime_tbx.coefficients.update.update import IUpdate, Update
from lime_tbx.spectral_integration.spectral_integration import get_default_srf
from lime_tbx.interpolation.interp_data import interp_data
from . import export


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
_ERROR_RINDEX_BOTH_DOT = """When creating output as CSV or GRAPH \
files for both DT and MPA, the full CSV/GRAPH filepaths must be \
explictly written, including the extension (.csv, .png, ...).
Another solution is to select the CSVD/GRAPHD option where one \
only has to specify the output directory path.
Problematic filepath: """


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


COMP_KEYS = ["DT", "MPA", "BOTH", "CHANNEL", "CHANNEL_MEAN"]
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
    """Prints the package version name to the standard output."""
    print(constants.VERSION_NAME)


class CLIError(Exception):
    """Exception raised for all errors encountered during command-line argument processing."""


class ParsingError(CLIError):
    """Exception raised for errors encountered during command-line argument parsing.

    This exception is used to indicate issues such as:
    - Invalid or unrecognized argument values.
    - Missing required parameters.
    - Incorrect formatting of input arguments.
    """


def _parse_chosen_diff(param: str) -> CompFields:
    """Parses the difference metric parameter for comparisons.

    Validates if the input parameter is one of the allowed difference keys, present
    in `COMP_DIFF_KEYS`.

    Parameters
    ----------
    param : str
        The input string representing the difference type.

    Returns
    -------
    CompFields
        The corresponding `CompFields` enumeration.

    Raises
    ------
    ParsingError
        If the parameter is invalid.
    """
    if param not in COMP_DIFF_KEYS:
        raise ParsingError(
            f"Invalid difference metric {param}. Expected one of {COMP_DIFF_KEYS}."
        )
    chosen_diff = CompFields.DIFF_NONE
    if param == COMP_DIFF_KEYS[0]:
        chosen_diff = CompFields.DIFF_REL
    elif param == COMP_DIFF_KEYS[1]:
        chosen_diff = CompFields.DIFF_PERC
    return chosen_diff


def _parse_comp_key(param: str) -> export.ComparisonKey:
    """Parses the comparison key from the CLI argument.

    Ensures that the comparison key is a valid mode, which are
    persent in `COMP_KEY`.

    Parameters
    ----------
    param : str
        The input string representing the comparison key.

    Returns
    -------
    export.ComparisonKey
        The corresponding `ComparisonKey` enumeration.

    Raises
    ------
    ParsingError
        If the parameter is invalid.
    """
    if param not in COMP_KEYS:
        raise ParsingError(
            f"Invalid comparison key {param}. Expected one of {COMP_KEYS}."
        )
    comp_key = export.ComparisonKey[param]
    return comp_key


def _parse_check_img_extension(param: str):
    """Validates and returns a supported image format.

    Ensures that the provided file extension is one of the supported image formats,
    which are the ones in `IMAGE_EXTENSIONS`.

    Parameters
    ----------
    param : str
        The input string representing the image format.

    Returns
    -------
    str
        The validated image format.

    Raises
    ------
    ParsingError
        If the format is not supported.
    """
    param = param.strip().lower()
    if param not in IMAGE_EXTENSIONS:
        raise ParsingError(
            f"Error: Graph format not detected. It must be one of the following: {','.join(IMAGE_EXTENSIONS)}."
        )
    return param


def _parse_filepaths_from_img_extension(param: str, filenames: List[str]) -> List[str]:
    """Generates file paths with the given image format extension.

    Appends the validated image extension to each filename in the list.

    Parameters
    ----------
    param : str
        The desired image format (e.g., "png", "jpg").
    filenames : List[str]
        List of base filenames (without extensions).

    Returns
    -------
    List[str]
        List of filenames with the correct extension.

    Raises
    ------
    ParsingError
        If the provided format is invalid.
    """
    param = _parse_check_img_extension(param)
    filepaths = list(map(lambda s: s + f".{param}", filenames))
    return filepaths


def _parse_output_params(arg: str, is_comparison: bool) -> export.ExportData:
    """Parses and validates the output parameters for simulation or comparison.

    Determines the export type based on the user input and returns the appropriate
    `ExportData` subclass. Supports:
    - CSV (`ExportCSV`, `ExportComparisonCSV`)
    - Graph (`ExportGraph`, `ExportComparisonGraph`)
    - NetCDF (`ExportNetCDF`)
    - Directory-based export (`ExportComparisonCSVDir`, `ExportComparisonGraphDir`)

    Parameters
    ----------
    arg : str
        The full CLI argument string containing output specifications.
    is_comparison : bool
        Whether the current operation is a comparison.

    Returns
    -------
    export.ExportData
        The corresponding export configuration object.

    Raises
    ------
    ParsingError
        If the output parameters are invalid.
    """
    splitted = arg.split(",")
    o_type = splitted[0]
    if o_type == "csv":
        if not is_comparison:
            if len(splitted) != 5:
                raise ParsingError("Error: Wrong number of arguments for -o csv,...")
            export_data = export.ExportCSV(
                splitted[1], splitted[2], splitted[3], splitted[4]
            )
        else:
            if len(splitted) < 4:
                raise ParsingError("Error: Wrong number of arguments for -o csv,...")
            comp_key = _parse_comp_key(splitted[1])
            chosen_diff = _parse_chosen_diff(splitted[2])
            export_data = export.ExportComparisonCSV(
                comp_key, splitted[3:], chosen_diff
            )
    elif o_type == "graph":
        if not is_comparison:
            if len(splitted) != 5:
                raise ParsingError("Error: Wrong number of arguments for -o graph,...")
            filepaths = _parse_filepaths_from_img_extension(splitted[1], splitted[2:])
            export_data = export.ExportGraph(*filepaths)
        else:
            if len(splitted) < 5:
                raise ParsingError("Error: Wrong number of arguments for -o graph,...")
            comp_key = _parse_comp_key(splitted[2])
            filepaths = _parse_filepaths_from_img_extension(splitted[1], splitted[4:])
            chosen_diff = _parse_chosen_diff(splitted[3])
            export_data = export.ExportComparisonGraph(
                comp_key,
                filepaths,
                chosen_diff,
            )
    elif o_type == "nc":
        if len(splitted) != 2:
            raise ParsingError("Error: Wrong number of arguments for -o nc,...")
        export_data = export.ExportNetCDF(splitted[1])
    elif o_type == "csvd":
        if not is_comparison:
            raise ParsingError("Error: csvd output is only available for comparisons.")
        if len(splitted) != 4:
            raise ParsingError("Error: Wrong number of arguments for -o csvd,...")
        comp_key = _parse_comp_key(splitted[1])
        chosen_diff = _parse_chosen_diff(splitted[2])
        export_data = export.ExportComparisonCSVDir(comp_key, splitted[3], chosen_diff)
    elif o_type == "graphd":
        if not is_comparison:
            raise ParsingError(
                "Error: graphd output is only available for comparisons."
            )
        if len(splitted) != 5:
            raise ParsingError("Error: Wrong number of arguments for -o graphd,...")
        splitted[1] = _parse_check_img_extension(splitted[1])
        comp_key = _parse_comp_key(splitted[2])
        chosen_diff = _parse_chosen_diff(splitted[3])
        export_data = export.ExportComparisonGraphDir(
            splitted[1],
            comp_key,
            splitted[4],
            chosen_diff,
        )
    return export_data


def _parse_load_timeseries(arg: str, opts: List[Tuple[str, str]]):
    timeseries_file = arg
    timeseries = None
    if any(item[0] in ("-e", "--earth", "-s", "--satellite") for item in opts):
        if os.path.exists(timeseries_file):
            try:
                timeseries = csv.read_datetimes(timeseries_file)
            except Exception as e:
                raise CLIError(f"Error reading timeseries file: {str(e)}") from e
        else:
            raise CLIError("Error: Timeseries file does not exist.")
    return timeseries


class CLI:
    """Command Line Interface handler for LIME TBX.

    This class processes command-line arguments, performs simulations, and manages
    comparisons. It serves as the main interface for users executing the toolbox
    from the command line.

    It contains a LimeSimulation instance that handles and stores the asked calculations
    that then will be retrieved for export.

    Responsibilities:
    - Simulations of lunar irradiance, reflectance, and polarization.
    - Comparisons with observational data.
    - Output in various formats (CSV, Graph, NetCDF).
    - Updating coefficient datasets.
    - Managing interpolation and spectral response function settings.
    """

    def __init__(
        self,
        kernels_path: KernelsPath,
        eocfi_path: EocfiPath,
        selected_version: str = None,
    ):
        """Initializes the CLI instance.

        Parameters
        ----------
        kernels_path : KernelsPath
            Path to the SPICE kernels required for calculations.
        eocfi_path : EocfiPath
            Path to the EO-CFI data directories used for orbit calculations.
        selected_version : str, optional
            Selected coefficients version (default: None).
        """
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.settings_manager = settings.SettingsManager(selected_version)
        self.lime_simulation: ILimeSimulation = LimeSimulation(
            eocfi_path, kernels_path, self.settings_manager
        )
        self.srf = self.settings_manager.get_default_srf()
        self.updater = Update()
        self.exporter = export.CLIExporter(
            self.kernels_path,
            self.eocfi_path,
            self.settings_manager,
            self.lime_simulation,
        )

    def load_srf(self, srf_file: str):
        """Loads the Spectral Response Function (SRF) from a specified file.

        If no file is provided, it loads the default SRF.

        Parameters
        ----------
        srf_file : str
            Path to the SRF file. If an empty string is given, it loads the default SRF.
        """
        if srf_file == "":
            self.srf = self.settings_manager.get_default_srf()
        else:
            self.srf = srflib.read_srf(srf_file)

    def _calculate_irradiance(self, point: Point):
        """Calculates lunar irradiance at a given point.

        Parameters
        ----------
        point : Point
            The geographic, lunar, or satellite location for the simulation.
        """
        def_srf = get_default_srf()
        self.lime_simulation.update_irradiance(
            def_srf, self.srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_reflectance(self, point: Point):
        """Calculates lunar reflectance at a given point.

        Parameters
        ----------
        point : Point
            The geographic, lunar, or satellite location for the simulation.
        """
        def_srf = get_default_srf()
        self.lime_simulation.update_reflectance(
            def_srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_polarisation(self, point: Point):
        """Calculates lunar polarization at a given point.

        Parameters
        ----------
        point : Point
            The geographic, lunar, or satellite location for the simulation.
        """
        def_srf = get_default_srf()
        self.lime_simulation.update_polarisation(
            def_srf, point, self.settings_manager.get_polar_coef()
        )

    def _calculate_all(self, point: Point):
        """Runs all calculations (irradiance, reflectance, and polarization)
        for a given location.

        Parameters
        ----------
        point : Point
            The location where calculations should be performed.
        """
        self._calculate_reflectance(point)
        self._calculate_irradiance(point)
        self._calculate_polarisation(point)

    def calculate_geographic(
        self,
        lat: float,
        lon: float,
        height: float,
        dt: Union[datetime, List[datetime]],
        export_data: export.ExportData,
    ):
        """Runs a simulation from a geographic location, ahnd
        export the results as specified.

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.
        height : float
            Height above sea level in meters.
        dt : datetime or List[datetime]
            The timestamp(s) for the simulation.
        export_data : export.ExportData
            The export configuration.

        Raises
        ------
        ExportError
            If something wrong happens during export.
        """
        point = SurfacePoint(lat, lon, height, dt)
        self._calculate_all(point)
        self.exporter.export(point, export_data, self.srf)

    def calculate_satellital(
        self,
        sat_name: str,
        dt: Union[datetime, List[datetime]],
        export_data: export.ExportData,
    ):
        """Runs a simulation from a satellite's perspective.

        Parameters
        ----------
        sat_name : str
            Name of the satellite.
        dt : datetime or List[datetime]
            The timestamp(s) for the simulation.
        export_data : export.ExportData
            The export configuration.

        Raises
        ------
        ExportError
            If something wrong happens during export.
        """
        point = SatellitePoint(sat_name, dt)
        self._calculate_all(point)
        self.exporter.export(point, export_data, self.srf)

    def calculate_selenographic(
        self,
        distance_sun_moon: float,
        distance_observer_moon: float,
        selen_obs_lat: float,
        selen_obs_lon: float,
        selen_sun_lon: float,
        moon_phase_angle: float,
        export_data: export.ExportData,
    ):
        """Runs a simulation from a selenographic (Moon-based) perspective.

        Parameters
        ----------
        distance_sun_moon : float
            Distance between the Sun and the Moon (meters).
        distance_observer_moon : float
            Distance between the observer and the Moon (meters).
        selen_obs_lat : float
            Observer's latitude on the Moon (degrees).
        selen_obs_lon : float
            Observer's longitude on the Moon (degrees).
        selen_sun_lon : float
            Sub-solar longitude on the Moon (degrees).
        moon_phase_angle : float
            Moon phase angle (degrees).
        export_data : export.ExportData
            The export configuration.

        Raises
        ------
        ExportError
            If something wrong happens during export.
        """
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
        self.exporter.export(point, export_data, self.srf)

    def _add_observation(self, obs: LunarObservation):
        for i, pob in enumerate(self.loaded_moons):
            if obs.dt < pob.dt:
                self.loaded_moons.insert(i, obs)
                return
        self.loaded_moons.append(obs)

    def calculate_comparisons(
        self,
        input_files: List[str],
        ed: export.ExportComparison,
    ):
        """Performs comparisons between simulation results and observational data.

        Parameters
        ----------
        input_files : List[str]
            List of file paths containing observational data.
        ed : export.ExportComparison
            The comparison export configuration.

        Raises
        ------
        LimeException
            If the observation data is not valid.
        """
        self.loaded_moons: List[LunarObservation] = []
        for path in input_files:
            self._add_observation(
                moon.read_moon_obs(path, self.kernels_path, self.eocfi_path)
            )
        if len(self.loaded_moons) == 0:
            raise LimeException("No observations given. Aborting.")
        mos = self.loaded_moons
        if isinstance(ed, export.ExportComparisonCSV) or isinstance(
            ed, export.ExportComparisonGraph
        ):
            ch_names_obs = {
                ch_name for mo in mos for ch_name in list(mo.ch_irrs.keys())
            }
            if ed.comparison_key not in (
                export.ComparisonKey.CHANNEL,
                export.ComparisonKey.CHANNEL_MEAN,
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
        if isinstance(ed, export.ExportNetCDF):
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
            if isinstance(ed, export.ExportComparisonCSVDir) or isinstance(
                ed, export.ExportComparisonGraphDir
            ):
                if not os.path.exists(ed.output_dir):
                    os.makedirs(ed.output_dir)
            version = self.settings_manager.get_lime_coef().version
            ch_names = self.srf.get_channels_names()
            file_index = 0
            is_both = ed.comparison_key == export.ComparisonKey.BOTH
            sp_name = self.settings_manager.get_selected_spectrum_name()
            if ed.comparison_key in (
                export.ComparisonKey.DT,
                export.ComparisonKey.BOTH,
            ):
                for i, ch in enumerate(ch_names):
                    if len(comps[i].dts) > 0:
                        ylabel = "Irradiance (Wm⁻²nm⁻¹)"
                        ylabels = [f"Observed {ylabel}", f"Simulated {ylabel}"]
                        output = ""
                        if isinstance(ed, export.ExportComparisonCSV) or isinstance(
                            ed, export.ExportComparisonGraph
                        ):
                            output = ed.output_files[file_index]
                        elif isinstance(ed, export.ExportComparisonCSVDir):
                            output = "{}.csv".format(os.path.join(ed.output_dir, ch))
                        elif isinstance(ed, export.ExportComparisonGraphDir):
                            output = "{}.{}".format(
                                os.path.join(ed.output_dir, ch), ed.extension
                            )
                        if is_both:
                            if "." not in output:
                                raise LimeException(_ERROR_RINDEX_BOTH_DOT + output)
                            idx = output.rindex(".")
                            output = output[:idx] + ".dt" + output[idx:]
                        if isinstance(ed, export.ExportComparisonCSV) or isinstance(
                            ed, export.ExportComparisonCSVDir
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
                            self.exporter.export_comparison_graph(
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
            if ed.comparison_key in (
                export.ComparisonKey.MPA,
                export.ComparisonKey.BOTH,
            ):
                mpa_comps = sort_by_mpa(comps)
                for i, ch in enumerate(ch_names):
                    if len(mpa_comps[i].dts) > 0:
                        ylabel = "Irradiance (Wm⁻²nm⁻¹)"
                        ylabels = [f"Observed {ylabel}", f"Simulated {ylabel}"]
                        output = ""
                        if isinstance(ed, export.ExportComparisonCSV) or isinstance(
                            ed, export.ExportComparisonGraph
                        ):
                            output = ed.output_files[file_index]
                        elif isinstance(ed, export.ExportComparisonCSVDir):
                            output = "{}.csv".format(os.path.join(ed.output_dir, ch))
                        elif isinstance(ed, export.ExportComparisonGraphDir):
                            output = "{}.{}".format(
                                os.path.join(ed.output_dir, ch), ed.extension
                            )
                        if is_both:
                            if "." not in output:
                                raise LimeException(_ERROR_RINDEX_BOTH_DOT + output)
                            idx = output.rindex(".")
                            output = output[:idx] + ".mpa" + output[idx:]
                        if isinstance(ed, export.ExportComparisonCSV) or isinstance(
                            ed, export.ExportComparisonCSVDir
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
                            self.exporter.export_comparison_graph(
                                mpa_comps[i],
                                xlabel,
                                ylabel,
                                output,
                                version,
                                ch,
                                ed.chosen_diff,
                            )
                        file_index += 1
            if ed.comparison_key in (
                export.ComparisonKey.CHANNEL,
                export.ComparisonKey.CHANNEL_MEAN,
            ):
                wlcs = self.srf.get_channels_centers()
                comps = [c if c.observed_signal is not None else None for c in comps]
                wlcs = np.array([w for w, c in zip(wlcs, comps) if c is not None])
                comps = [c for c in comps if c is not None]
                xlabel = "Wavelength (nm)"
                ylabel = "Irradiance (Wm⁻²nm⁻¹)"
                ylabels = [f"Observed {ylabel}", f"Simulated {ylabel}"]
                output = ""
                if isinstance(ed, export.ExportComparisonCSV) or isinstance(
                    ed, export.ExportComparisonGraph
                ):
                    output = ed.output_files[0]
                elif isinstance(ed, export.ExportComparisonCSVDir):
                    output = "{}.csv".format(os.path.join(ed.output_dir, "allchannels"))
                elif isinstance(ed, export.ExportComparisonGraphDir):
                    output = "{}.{}".format(
                        os.path.join(ed.output_dir, "allchannels"), ed.extension
                    )
                    if ed.comparison_key == export.ComparisonKey.CHANNEL:
                        if isinstance(ed, export.ExportComparisonCSV) or isinstance(
                            ed, export.ExportComparisonCSVDir
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
                            self.exporter.export_comparison_bywlen_graph(
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
                        data_start = min([min(c.dts) for c in comps])
                        data_end = max([max(c.dts) for c in comps])
                        if isinstance(ed, export.ExportComparisonCSV) or isinstance(
                            ed, export.ExportComparisonCSVDir
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
                            self.exporter.export_comparison_graph(
                                comp,
                                xlabel,
                                ylabel,
                                output,
                                version,
                                "All channels",
                                ed.chosen_diff,
                                date_range=(data_start, data_end),
                            )

    def update_coefficients(self) -> int:
        """Checks for and downloads coefficient dataset updates.

        Returns
        -------
        int
            - `0` if the update is successful or no updates are available.
            - `1` if an error occurs during the update process.
        """
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

    def _parse_interp_settings(self, arg: str):
        # example: -i '{"interp_spectrum": "ASD", "skip_uncertainties": "False", "show_interp_spectrum": "False", "interp_srf": "interpolated_gaussian"}'
        try:
            interp_settings = json.loads(arg)
        except Exception as e:
            raise ParsingError(
                f"Error parsing the interpolation settings {arg}. Error: {e}"
            ) from e
        if "interp_spectrum" in interp_settings:
            interp_spectrum = interp_settings["interp_spectrum"]
            names = self.settings_manager.get_available_spectra_names()
            if interp_spectrum not in names:
                raise ParsingError(
                    f"Interpolation spectrum not recognized. Selected: {interp_spectrum}. Available: {names}."
                )
            self.settings_manager.select_interp_spectrum(interp_spectrum)
        if "interp_srf" in interp_settings:
            interp_srf = interp_settings["interp_srf"]
            srf_translator = {
                v: k for k, v in interp_data.SRF_DICT_SOLAR_DIALOG_SRF_TYPE.items()
            }
            names = list(srf_translator.keys())
            if interp_srf not in names:
                raise ParsingError(
                    f"Interpolation settings output SRF not recognized. Selected: {interp_srf}. Available: {names}."
                )
            self.settings_manager.select_interp_SRF(srf_translator[interp_srf])
        if "show_interp_spectrum" in interp_settings:
            show_interp_spectrum = interp_settings["show_interp_spectrum"]
            if show_interp_spectrum not in ("True", "False"):
                raise ParsingError(
                    f'Interpolation settings show_interp_spectrum value {show_interp_spectrum} not valid. Must be "True" or "False"'
                )
            show_interp_spectrum = show_interp_spectrum == "True"
            self.settings_manager.set_show_interp_spectrum(show_interp_spectrum)
        if "skip_uncertainties" in interp_settings:
            skip_uncertainties = interp_settings["skip_uncertainties"]
            if skip_uncertainties not in ("True", "False"):
                raise ParsingError(
                    f'Interpolation settings skip_uncertainties value {skip_uncertainties} not valid. Must be "True" or "False"'
                )
            skip_uncertainties = skip_uncertainties == "True"
            self.settings_manager.set_skip_uncertainties(skip_uncertainties)
        if "show_cimel_points" in interp_settings:
            show_cimel_points = interp_settings["show_cimel_points"]
            if show_cimel_points not in ("True", "False"):
                raise ParsingError(
                    f'Interpolation settings show_cimel_points value {show_cimel_points} not valid. Must be "True" or "False"'
                )
            show_cimel_points = show_cimel_points == "True"
            self.settings_manager.set_show_cimel_points(show_cimel_points)

    def check_sys_args(self, sysargs: List[str]) -> int:
        """Validates system arguments to prevent syntax errors.

        Ensures flags are correctly formatted and separated.

        Parameters
        ----------
        sysargs : List[str]
            The raw command-line arguments.

        Returns
        -------
        int
            - `0` if validation passes.
            - `1` if errors are found.
        """
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

    def _parse_load_srf(self, arg):
        srf_file = arg
        if srf_file == "" or os.path.exists(srf_file):
            try:
                self.load_srf(srf_file)
            except Exception as e:
                raise CLIError(
                    f"Error: Error loading Spectral Response Function. {str(e)}"
                ) from e
        else:
            raise CLIError(f"Error: The given srf path '{srf_file}' does not exist.")

    def handle_input(self, opts: List[Tuple[str, str]], args: List[str]) -> int:
        """Processes command-line options and executes the corresponding actions.

        This function parses command-line arguments, validates inputs, and
        dispatches execution to the appropriate simulation, comparison,
        or configuration functions.

        It supports:
        - Simulations of lunar irradiance, reflectance, and polarization
        from different perspectives (Earth, Lunar, Satellite).
        - Comparisons with observational data in GLOD format.
        - Output in multiple formats (CSV, Graph, NetCDF).
        - Updating coefficient datasets.
        - Managing interpolation and spectral response function settings.

        Parameters
        ----------
        opts : List[Tuple[str, str]]
            A list of command-line options and their corresponding arguments.
            Example: `[('-e', '10.5,20.3,100,2023-02-18T12:00:00')]`
        args : List[str]
            Additional arguments passed after the options, primarily used for
            input file paths in comparisons.

        Returns
        -------
        int
            Exit status code:
            - `0` for success.
            - `1` for errors (invalid input, missing parameters, execution failures).
        """
        export_data: export.ExportData = None
        timeseries = None
        # Check if it's comparison
        is_comparison = any(item[0] in ("-c", "--comparison") for item in opts)
        mod_interp_settings = False
        # find settings data
        try:
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
                    export_data = _parse_output_params(arg, is_comparison)
                elif opt in ("-f", "--srf"):
                    self._parse_load_srf(arg)
                elif opt in ("-t", "--timeseries"):
                    timeseries = _parse_load_timeseries(arg, opts)
                elif opt in ("-C", "--coefficients"):
                    names = sorted(
                        [
                            coef.version
                            for coef in self.settings_manager.get_available_coeffs()
                        ]
                    )
                    if arg not in names:
                        raise ParsingError(
                            f"Coefficients version not recognized. Selected: {arg}. Available: {names}."
                        )
                    self.settings_manager.select_lime_coeff(names.index(arg))
                elif opt in ("-i", "--interpolation-settings"):
                    self._parse_interp_settings(arg)
                    mod_interp_settings = True
        except CLIError as e:
            eprint(str(e))
            return 1
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
        try:
            for opt, arg in opts:
                if opt in ("-e", "--earth"):  # Earth
                    params_str = arg.split(",")
                    lenpar = len(params_str)
                    if lenpar < 3 or (timeseries is None and lenpar != 4):
                        eprint("Error: Wrong number of arguments for -e")
                        return 1
                    params = list(map(float, params_str[:3]))
                    lat = params[0]
                    lon = params[1]
                    height = params[2]
                    if timeseries is not None:
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
                    if lenpar < 1 or (timeseries is None and lenpar != 2):
                        eprint("Error: Wrong number of arguments for -s")
                        return 1
                    sat_name = params_str[0]
                    if timeseries is not None:
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
        except export.ExportError as e:
            eprint(str(e))
            return 1
        except LimeException as e:
            eprint(f"Error: {str(e)}")
            return 1
        except Exception as e:
            trace = traceback.format_exc()
            eprint("Error when performing operations: {}".format(str(e)))
            logger.get_logger().critical(f"Error trace: {trace}")
            return 1
        return 0
