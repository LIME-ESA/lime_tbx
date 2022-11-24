"""___Built-In Modules___"""
from datetime import datetime, timezone
from dataclasses import dataclass
from abc import ABC
from typing import List, Union, Tuple
import os
from enum import Enum
import glob
import sys

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    ComparisonData,
    CustomPoint,
    KernelsPath,
    LGLODComparisonData,
    LimeException,
    LunarObservation,
    Point,
    SatellitePoint,
    SpectralData,
    SurfacePoint,
)
from lime_tbx.gui import settings
from lime_tbx.simulation.lime_simulation import LimeSimulation
from lime_tbx.filedata import moon, srf as srflib, csv
from lime_tbx.filedata.lglod_factory import create_lglod_data
from lime_tbx.simulation.comparison import comparison
from lime_tbx.datatypes import constants

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "01/09/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


_DT_FORMAT = "%Y-%m-%dT%H:%M:%S"
OPTIONS = "hvde:l:s:c:o:f:t:"
LONG_OPTIONS = [
    "help",
    "version",
    "debug",
    "earth=",
    "lunar=",
    "satellite=",
    "comparison=",
    "output=",
    "srf=",
    "timeseries=",
]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class ExportData(ABC):
    pass


@dataclass
class ExportCSV(ExportData):
    o_file_refl: str
    o_file_irr: str
    o_file_polar: str
    o_file_integrated_irr: str


@dataclass
class ExportGraph(ExportData):
    o_file_refl: str
    o_file_irr: str
    o_file_polar: str


COMP_KEYS = ["DT", "MPA", "BOTH"]


class ComparisonKey(Enum):
    DT = 0
    MPA = 1
    BOTH = 2


class ExportComparison(ABC):
    pass


@dataclass
class ExportComparisonCSV(ExportComparison):
    comparison_key: ComparisonKey
    output_files: List[str]


@dataclass
class ExportComparisonCSVDir(ExportComparison):
    comparison_key: ComparisonKey
    output_dir: str


@dataclass
class ExportComparisonGraph(ExportComparison):
    comparison_key: ComparisonKey
    output_files: List[str]


@dataclass
class ExportComparisonGraphDir(ExportComparison):
    extension: str
    comparison_key: ComparisonKey
    output_dir: str


@dataclass
class ExportNetCDF(ExportData, ExportComparison):
    output_file: str


IMAGE_EXTENSIONS = ["pdf", "jpg", "png", "svg"]


def print_help():
    compsel = "(" + "|".join(COMP_KEYS) + ")"
    imsel = "(" + "|".join(IMAGE_EXTENSIONS) + ")"
    print(
        "The lime toolbox performs simulations of lunar irradiance, reflectance and \
polarization for a given point and datetime. It also performs comparisons for some given \
observations files in GLOD format.\n"
    )
    print("It won't work unless given only one of the options (-h|-e|-l|-s|-c).")
    print("")
    print("Options:")
    print("  -h, --help\t\t Displays the help message.")
    print("  -v, --version\t\t Displays the version name.")
    print("  -e, --earth\t\t Performs simulations from a geographic point.")
    print("\t\t\t -e lat_deg,lon_deg,height_m,{}".format(_DT_FORMAT))
    print("  -l, --lunar\t\t Performs a simulation from a selenographic point.")
    print(
        "\t\t\t -l distance_sun_moon,distance_observer_moon,selen_obs_lat,selen_obs_lon,\
selen_sun_lon,moon_phase_angle"
    )
    print("  -s, --satellite\t Performs simulations from a satellite point.")
    print("\t\t\t -s sat_name,{}".format(_DT_FORMAT))
    print(
        "  -c, --comparison\t Performs comparisons from observations files in GLOD format."
    )
    print('\t\t\t -c "input_glod1.nc input_glod2.nc ..."')
    print("  -o, --output\t\t Select the output path and format.")
    print("\t\t\t If it's a simulation:")
    print("\t\t\t   GRAPH: -o graph,{},refl,irr,polar".format(imsel))
    print("\t\t\t   CSV: -o csv,refl.csv,irr.csv,polar.csv,integrated_irr.csv")
    print("\t\t\t   LGLOD (netcdf): -o nc,output_lglod.nc")
    print("\t\t\t If it's a comparison:")
    print(
        "\t\t\t   GRAPH: -o graph,{},{},comparison_channel1,comparison_channel2,...".format(
            imsel, compsel
        )
    )
    print(
        "\t\t\t   CSV: -o csv,{},comparison_channel1.csv,comparison_channel2.csv,...".format(
            compsel
        )
    )
    print(
        "\t\t\t   GRAPH directory: -o graphd,{},{},comparison_folder".format(
            imsel, compsel
        )
    )
    print("\t\t\t   CSV directory: -o csvd,{},comparison_folder".format(compsel))
    print("\t\t\t   LGLOD (netcdf): -o nc,output_lglod.nc")
    print(
        "  -f, --srf\t\t Select the file that contains the Spectral Response Function \
in GLOD format."
    )
    print(
        "  -t, --timeseries\t Select a CSV file with multiple datetimes instead of \
inputing directly only one datetime. Valid only if the main option is -e or -s."
    )


def print_version():
    print(constants.VERSION_NAME)


class CLI:
    def __init__(
        self, kernels_path: KernelsPath, eocfi_path: str, selected_version: str = None
    ):
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.lime_simulation = LimeSimulation(eocfi_path, kernels_path)
        self.settings_manager = settings.SettingsManager(selected_version)
        self.srf = self.settings_manager.get_default_srf()

    def load_srf(self, srf_file: str):
        if srf_file == "":
            self.srf = self.settings_manager.get_default_srf()
        else:
            self.srf = srflib.read_srf(srf_file)

    def _calculate_irradiance(self, point: Point):
        self.lime_simulation.update_irradiance(
            self.srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_reflectance(self, point: Point):
        self.lime_simulation.update_reflectance(
            self.srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_polarization(self, point: Point):
        self.lime_simulation.update_polarization(
            self.srf, point, self.settings_manager.get_polar_coef()
        )

    def _calculate_all(self, point: Point):
        self._calculate_reflectance(point)
        self._calculate_irradiance(point)
        self._calculate_polarization(point)

    def _export_csvs(
        self,
        point: Point,
        ed: ExportCSV,
    ):
        version = self.settings_manager.get_lime_coef().version
        csv.export_csv(
            self.lime_simulation.get_elrefs(),
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
            point,
            ed.o_file_refl,
            version,
        )
        csv.export_csv(
            self.lime_simulation.get_elis(),
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²nm⁻¹)",
            point,
            ed.o_file_irr,
            version,
        )
        csv.export_csv(
            self.lime_simulation.get_polars(),
            "Wavelengths (nm)",
            "Polarizations (Fraction of unity)",
            point,
            ed.o_file_polar,
            version,
        )
        csv.export_csv_integrated_irradiance(
            self.srf,
            self.lime_simulation.get_signals(),
            ed.o_file_integrated_irr,
            point,
            version,
        )

    def _export_lglod(self, point: Point, output_file: str):
        lglod = create_lglod_data(
            point, self.srf, self.lime_simulation, self.kernels_path
        )
        now = datetime.now(timezone.utc)
        version = self.settings_manager.get_lime_coef().version
        moon.write_obs(lglod, output_file, now, version)

    def _export_graph(self, point: Point, ed: ExportGraph):
        from lime_tbx.gui import canvas

        canv = canvas.MplCanvas(width=15, height=10, dpi=200)
        canv.set_title("", fontproperties=canvas.title_font_prop)
        canv.axes.tick_params(labelsize=8)
        version = self.settings_manager.get_lime_coef().version
        subtitle = "LIME2 coefficients version: {}".format(version)
        canv.set_subtitle(subtitle, fontproperties=canvas.font_prop)
        canv.axes.set_xlabel("Wavelengths (nm)", fontproperties=canvas.label_font_prop)
        canv.axes.set_ylabel("", fontproperties=canvas.label_font_prop)
        canv.axes.cla()  # Clear the canvas.
        canvas.redraw_canvas(
            canv,
            self.lime_simulation.get_elrefs(),
            [["interpolated data points"], ["CIMEL data points"], ["errorbars (k=2)"]],
            self.lime_simulation.get_elrefs_cimel(),
            self.lime_simulation.get_elrefs_asd(),
            None,
            "Extraterrestrial Lunar Reflectances",
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
            None,
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
            [["interpolated data points"], ["CIMEL data points"], ["errorbars (k=2)"]],
            self.lime_simulation.get_elis_cimel(),
            self.lime_simulation.get_elis_asd(),
            None,
            "Extraterrestrial Lunar Irradiances",
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²nm⁻¹)",
            None,
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
        canvas.redraw_canvas(
            canv,
            self.lime_simulation.get_polars(),
            [["interpolated data points"], ["CIMEL data points"], ["errorbars (k=2)"]],
            self.lime_simulation.get_polars_cimel(),
            self.lime_simulation.get_polars_asd(),
            None,
            "Extraterrestrial Lunar Polarization",
            "Wavelengths (nm)",
            "Polarizations (Fraction of unity)",
            None,
        )
        try:
            canv.print_figure(ed.o_file_polar)
        except Exception as e:
            eprint(
                "Something went wrong while exporting polarization graph. {}".format(
                    str(e)
                )
            )
            sys.exit(1)
        canv.axes.cla()  # Clear the canvas.

    def _export_comparison_graph(
        self,
        data: List[SpectralData],
        xlabel: str,
        ylabel: str,
        output_file: str,
        version: str,
        comparison: ComparisonData,
        ch: str,
    ):
        from lime_tbx.gui import canvas

        canv = canvas.MplCanvas(width=15, height=10, dpi=200)
        canv.set_title("", fontproperties=canvas.title_font_prop)
        canv.axes.tick_params(labelsize=8)
        subtitle = "LIME2 coefficients version: {}".format(version)
        n_comp_points = len(comparison.diffs_signal.wlens)
        data_start = min(comparison.dts)
        data_end = max(comparison.dts)
        version = self.settings_manager.get_lime_coef().version
        subtitle = "LIME2 coefficients version: {}".format(version)
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
        canvas.redraw_canvas(
            canv,
            data,
            [
                ["Observed Irradiance", "Simulated Irradiance"],
                [],
                [],
                ["Relative Differences"],
            ],
            None,
            None,
            comparison,
            ch,
            xlabel,
            ylabel,
            None,
            subtitle,
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
            self._add_observation(moon.read_moon_obs(path))
        if len(self.loaded_moons) == 0:
            raise LimeException("No observations given. Aborting.")
        mos = self.loaded_moons
        if isinstance(ed, ExportComparisonCSV) or isinstance(ed, ExportComparisonGraph):
            ch_names_obs = {ch_name for mo in mos for ch_name in mo.ch_names}
            if len(ch_names_obs) > len(ed.output_files):
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
        if isinstance(ed, ExportNetCDF):
            lglod = LGLODComparisonData(
                comps,
                self.srf.get_channels_names(),
                "TODO",
            )
            vers = self.settings_manager.get_lime_coef().version
            moon.write_comparison(
                lglod,
                ed.output_file,
                datetime.now().astimezone(timezone.utc),
                vers,
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
            if ed.comparison_key != ComparisonKey.MPA:
                for i, ch in enumerate(ch_names):
                    if len(comps[i].dts) > 0:
                        data = [comps[i].observed_signal, comps[i].simulated_signal]
                        points = comps[i].points
                        ylabel = "Irradiance (Wm⁻²nm⁻¹)"
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
                            idx = output.rindex(".")
                            output = output[:idx] + ".dt" + output[idx:]
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonCSVDir
                        ):
                            csv.export_csv_comparation(
                                data,
                                ylabel,
                                points,
                                output,
                                version,
                            )
                        else:
                            xlabel = "UTC datetime"
                            self._export_comparison_graph(
                                data, xlabel, ylabel, output, version, comps[i], ch
                            )
                        file_index += 1
            file_index = 0
            if ed.comparison_key != ComparisonKey.DT:
                mpa_comps = co.sort_by_mpa(comps)
                for i, ch in enumerate(ch_names):
                    if len(mpa_comps[i].dts) > 0:
                        data = [
                            mpa_comps[i].observed_signal,
                            mpa_comps[i].simulated_signal,
                        ]
                        points = mpa_comps[i].points
                        ylabel = "Irradiance (Wm⁻²nm⁻¹)"
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
                            idx = output.rindex(".")
                            output = output[:idx] + ".mpa" + output[idx:]
                        if isinstance(ed, ExportComparisonCSV) or isinstance(
                            ed, ExportComparisonCSVDir
                        ):
                            csv.export_csv_comparation(
                                data,
                                ylabel,
                                points,
                                output,
                                version,
                                False,
                            )
                        else:
                            xlabel = "Moon phase angle (degrees)"
                            self._export_comparison_graph(
                                data, xlabel, ylabel, output, version, mpa_comps[i], ch
                            )
                        file_index += 1

    def handle_input(self, opts: List[Tuple[str, str]]) -> int:
        srf_file = ""
        export_data: ExportData = None
        timeseries_file: str = None
        # Check if it's comparison
        is_comparison = any(item[0] in ("-c", "--comparison") for item in opts)
        # find settings data
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_help()
                return 0
            if opt in ("-v", "--version"):
                print_version()
                return 0
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
                        if len(splitted) < 3:
                            eprint("Error: Wrong number of arguments for -o csv,...")
                            return 1
                        if splitted[1] not in COMP_KEYS:
                            eprint("Error in csv DT|MPA|BOTH parameter.")
                            return 1
                        comp_key = ComparisonKey[splitted[1]]
                        export_data = ExportComparisonCSV(comp_key, splitted[2:])
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
                        if len(splitted) < 4:
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
                            eprint("Error in csv DT|MPA|BOTH parameter.")
                            return 1
                        filepaths = list(
                            map(lambda s: s + ".{}".format(splitted[1]), splitted[3:])
                        )
                        comp_key = ComparisonKey[splitted[2]]
                        export_data = ExportComparisonGraph(comp_key, filepaths)
                elif o_type == "nc":
                    if len(splitted) != 2:
                        eprint("Error: Wrong number of arguments for -o nc,...")
                        return 1
                    export_data = ExportNetCDF(splitted[1])
                elif o_type == "csvd":
                    if not is_comparison:
                        eprint("Error: csvd output is only available for comparisons.")
                        return 1
                    if len(splitted) != 3:
                        eprint("Error: Wrong number of arguments for -o csvd,...")
                        return 1
                    if splitted[1] not in COMP_KEYS:
                        eprint("Error in csvd DT|MPA|BOTH parameter.")
                        return 1
                    comp_key = ComparisonKey[splitted[1]]
                    export_data = ExportComparisonCSVDir(comp_key, splitted[2])
                elif o_type == "graphd":
                    if not is_comparison:
                        eprint(
                            "Error: graphd output is only available for comparisons."
                        )
                        return 1
                    if len(splitted) != 4:
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
                        eprint("Error in csvd DT|MPA|BOTH parameter.")
                        return 1
                    comp_key = ComparisonKey[splitted[2]]
                    export_data = ExportComparisonGraphDir(
                        splitted[1], comp_key, splitted[3]
                    )
            elif opt in ("-f", "--srf"):
                srf_file = arg
            elif opt in ("-t", "--timeseries"):
                timeseries_file = arg
        if export_data == None:
            eprint("Error: The output flag (-o | --output) must be included.")
            return 1
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
        if num_sim_ops == 0:
            eprint("Error: There must be one of the following flags: (-e|-s|-l|-c).")
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
                    params = arg.split(" ")
                    if len(params) < 1:
                        eprint("Error: Wrong number of arguments for -c")
                        return 1
                    input_files = []
                    for param in params:
                        input_files += glob.glob(param)
                    self.calculate_comparisons(input_files, export_data)
                    break
        except LimeException as e:
            eprint("Error: {}".format(str(e)))
            return 1
        except Exception as e:
            eprint("Error when performing operations: {}".format(str(e)))
            return 1
        return 0
