"""
This module contains the abstractions and interfaces that call EO-CFI libraries and
perform satellite functions.

It exports the following classes:
    * EOCFIConverter - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from typing import Tuple, List
from datetime import datetime, timezone
import os
import platform
import subprocess
import yaml

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules___"""
from lime_tbx.common.datatypes import (
    KernelsPath,
    LimeException,
    OrbitFile,
    Satellite,
    EocfiPath,
)
from lime_tbx.common import logger
from lime_tbx.business.spice_adapter.spice_adapter import SPICEAdapter


ESA_SAT_LIST = "esa_sat_list.yml"
METADATA_FILE = "metadata.yml"
EXE_FILE_SATELLITE_LINUX = "eocfi_c/bin/get_positions_linux"
EXE_FILE_SATELLITE_WINDOWS = "eocfi_c\\bin\\get_positions_win64.exe"
EXE_FILE_SATELLLITE_DARWIN = "eocfi_c/bin/get_positions_darwin"
EXE_FILE_SATELLLITE_DARWIN_ARM = "eocfi_c/bin/get_positions_darwin_arm"


def _get_exe_path() -> str:  # pragma: no cover
    """Determines the executable path for the EO-CFI binary based on the operating system.

    This function selects the appropriate binary for retrieving satellite positions,
    based on whether the system is running Linux, Windows, or macOS (Darwin).

    Returns
    -------
    str
        The absolute path to the EO-CFI executable for the current operating system.
    """
    if platform.system() == "Linux":
        exe_file_satellite = EXE_FILE_SATELLITE_LINUX
    elif platform.system() == "Windows":
        exe_file_satellite = EXE_FILE_SATELLITE_WINDOWS
    else:  # Darwin
        if "ARM" in platform.version().upper():
            exe_file_satellite = EXE_FILE_SATELLLITE_DARWIN  # TODO compile in Mac ARM and add in correct path
        else:
            exe_file_satellite = EXE_FILE_SATELLLITE_DARWIN
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _exe_path = f'"{os.path.join(_current_dir, exe_file_satellite)}"'
    return _exe_path


def _get_file_datetimes(filename: str) -> Tuple[datetime, datetime]:
    """Extracts start and end datetime from an orbit file name.

    EO-CFI orbit files encode the validity period in their filenames.
    This function extracts those timestamps and converts them to `datetime` objects.

    Parameters
    ----------
    filename : str
        The orbit file name, which must follow the format:
        `..._<start_datetime>_<end_datetime>_...`

    Returns
    -------
    Tuple[datetime, datetime]
        - The start datetime of the orbit file.
        - The end datetime of the orbit file, or a default far-future date if unspecified.
    """
    splitted = filename.split("_")
    date0 = datetime.strptime(splitted[-3] + "+00:00", "%Y%m%dT%H%M%S%z")
    if splitted[-2] == "99999999T999999":
        date1 = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    else:
        date1 = datetime.strptime(splitted[-2] + "+00:00", "%Y%m%dT%H%M%S%z")
    return date0, date1


class EOCFIConverter:
    """Class that implements the methods of this module.

    It exports the following functions:
        * get_sat_names() - Obtain the list of satellite names, that are the keys that can be used in
            get_satellite_position
        * get_sat_list() - Obtain a list of the satellite data objects that are available in LIME TBX.
        * get_satellite_position() - Get the geographic satellite position for a concrete datetime.
    """

    def __init__(self, eocfi_path: EocfiPath, kernels_path: KernelsPath):
        """Initializes the EOCFIConverter with EO-CFI and SPICE kernel paths.

        Parameters
        ----------
        eocfi_path : EocfiPath
            Path to EO-CFI libraries used for orbit calculations.
        kernels_path : KernelsPath
            Path to SPICE kernels used for coordinate transformations.
        """
        super().__init__()
        self.eocfi_path = eocfi_path
        self.kernels_path = kernels_path

    def get_sat_names(self) -> List[str]:
        """
        Obtain the list of satellite names, that are the keys that can be used in
        get_satellite_position

        Returns
        -------
        sat_names: list of str
            List of the satellite names
        """
        return list(self._get_sat_list_yaml().keys())

    def _get_custom_sat_list_yaml(self) -> dict:
        """
        Read the custom sat list yaml and return the object.

        Returns
        -------
        sat_list: dict
            Dictionary containing the sat list yaml.
        """
        custom_satl = os.path.join(self.eocfi_path.custom_eocfi_path, ESA_SAT_LIST)
        y = {}
        if os.path.exists(custom_satl):
            fl = open(custom_satl)
            y = yaml.load(fl, Loader=yaml.FullLoader)
            fl.close()
        return y

    def _get_sat_list_yaml(self) -> dict:
        """
        Read the sat list yamls and return the object.

        Returns
        -------
        sat_list: dict
            Dictionary containing the sat list yaml.
        """
        yc = self._get_custom_sat_list_yaml()
        fl = open(os.path.join(self.eocfi_path.main_eocfi_path, ESA_SAT_LIST))
        y = yaml.load(fl, Loader=yaml.FullLoader)
        fl.close()
        # join both dicts
        for satname in yc:
            sat = yc[satname]
            is_tle = False
            if "norad_sat_number" in sat:
                is_tle = True
            if satname not in y:
                y[satname] = sat
            else:
                if is_tle and "norad_sat_number" not in y[satname]:
                    y[satname]["norad_sat_number"] = sat["norad_sat_number"]
                    y[satname]["intdes"] = sat["intdes"]
                    y[satname]["time_file"] = sat["time_file"]
                for orbf in sat["orbit_files"][::-1]:
                    y[satname]["orbit_files"].insert(0, orbf)
                n_files = int(y[satname]["n_files"])
                y[satname]["n_files"] = n_files + sat["n_files"]
        return y

    def get_sat_list(self) -> List[Satellite]:
        """
        Obtain a list of the satellite data objects that are available in LIME TBX.

        Returns
        -------
        sat_list: list of Satellite
            List of Satellites available in LIME TBX.
        """
        sat_yaml = self._get_sat_list_yaml()
        sat_list = []
        for s in sat_yaml:
            name = s
            sat_data: dict = sat_yaml.get(s)
            id = sat_data["id"]
            orbit_files_names = sat_data["orbit_files"]
            norad_key = "norad_sat_number"
            norad = None
            if norad_key in sat_data:
                norad = int(sat_data[norad_key])
            intdes_key = "intdes"
            intdes = None
            if intdes_key in sat_data:
                intdes = sat_data[intdes_key]
            time_file_key = "time_file"
            time_file = None
            if time_file_key in sat_data:
                time_file = sat_data[time_file_key]
            if orbit_files_names == None:
                orbit_files_names = []
            orbit_files = []
            for file in orbit_files_names:
                d0, df = _get_file_datetimes(file)
                orbit_f = OrbitFile(file, d0, df)
                orbit_files.append(orbit_f)
            sat = Satellite(name, id, orbit_files, norad, intdes, time_file)
            sat_list.append(sat)
        return sat_list

    def add_sat(self, sat: Satellite):
        """
        Add satellite data to the LIME TBX database.

        Parameters
        -------
        sat: Satellite
            Satellite data to add to LIME's satellite database.
        """
        is_tle = False
        if sat.norad_sat_number is not None:
            is_tle = True
        sat_yaml = self._get_sat_list_yaml()
        if is_tle and sat.time_file is None:
            sat.time_file = sat_yaml["ENVISAT"]["time_file"]
        if is_tle:
            sat_data = {
                "id": sat.id,
                "norad_sat_number": sat.norad_sat_number,
                "intdes": sat.intdes,
                "time_file": sat.time_file,
                "orbit_files": sat.orbit_files,
                "n_files": len(sat.orbit_files),
            }
        else:
            sat_data = {
                "id": sat.id,
                "orbit_files": sat.orbit_files,
                "n_files": len(sat.orbit_files),
            }
        if sat.name not in sat_yaml:
            sat_yaml[sat.name] = sat_data
        else:
            if is_tle and "norad_sat_number" not in sat_yaml[sat.name]:
                sat_yaml[sat.name]["norad_sat_number"] = sat_data["norad_sat_number"]
                sat_yaml[sat.name]["intdes"] = sat_data["intdes"]
                sat_yaml[sat.name]["time_file"] = sat_data["time_file"]
            for orbf in sat_data["orbit_files"][::-1]:
                sat_yaml[sat.name]["orbit_files"].insert(0, orbf)
            n_files = int(sat_yaml[sat.name]["n_files"])
            sat_yaml[sat.name]["n_files"] = n_files + sat_data["n_files"]
        os.makedirs(self.eocfi_path.custom_eocfi_path, exist_ok=True)
        fl = open(
            os.path.join(self.eocfi_path.custom_eocfi_path, f"{ESA_SAT_LIST}"), "w"
        )
        # If adding a satellite, the satlist will be stored in custom path
        # And this list is read with priority over the other one.
        yaml.dump(sat_yaml, fl, Dumper=yaml.Dumper)
        fl.close()

    def get_satellite_position(
        self, sat: str, dts: List[datetime]
    ) -> List[Tuple[float, float, float]]:
        """
        Get the geographic satellite position for a concrete datetime.

        Parameters
        ----------
        sat: str
            Satellite name. Should be present in get_sat_names
        dts: List of datetime
            Datetimes for which the position will be calculated

        Returns
        -------
        positions: list of tuples of floats
            List of tuples of 3 floats, representing:
            latitude: float
                Geocentric latitude of the satellite
            longitude: float
                Geocentric longitude of the satellite
            height: float
                Height of the satellite over sea level in meters.
        """
        xyzs = self.get_satellite_position_rectangular(sat, dts)
        positions = SPICEAdapter.to_planetographic_multiple(
            xyzs, "EARTH", self.kernels_path.main_kernels_path, dts, "ITRF93"
        )  # EOCFI uses ITRF93
        for llh in positions:
            logger.get_logger().debug(
                f"EOCFI output (lat, lon, height): {llh[0]}, {llh[1]}, {llh[2]}"
            )
        return positions

    def get_satellite_position_rectangular(
        self, sat: str, dts: List[datetime]
    ) -> List[Tuple[float, float, float]]:
        """
        Get the geographic satellite position for a concrete datetime.

        Parameters
        ----------
        sat: str
            Satellite name. Should be present in get_sat_names
        dts: List of datetime
            Datetimes for which the position will be calculated

        Returns
        -------
        positions: list of tuples of floats
            List of tuples of 3 floats, representing xyz in meters, in the ITRF93 frame.
        """
        if sat not in self.get_sat_names():
            raise LimeException(
                f"Satellite '{sat}' is not registered in LIME's satellite list."
            )
        sat: Satellite = [s for s in self.get_sat_list() if s.name == sat][0]
        dt_orb_f = {}
        for dt in dts:
            orb_f = sat.get_best_orbit_file(dt)
            if not orb_f in dt_orb_f:
                dt_orb_f[orb_f] = [dt]
            else:
                dt_orb_f[orb_f].append(dt)
        positions_map = {}
        for orb_f in dt_orb_f:
            positions = self._get_sat_position_one_orbit_file(
                sat, dt_orb_f[orb_f], orb_f
            )
            for i, dt in enumerate(dt_orb_f[orb_f]):
                positions_map[dt] = positions[i]
        positions_ret = []
        for dt in dts:
            positions_ret.append(positions_map[dt])
        return positions_ret

    def _get_sat_position_one_orbit_file(
        self, sat: Satellite, dts: List[datetime], orb_f: OrbitFile
    ) -> List[Tuple[float, float, float]]:
        orbit_path = ""
        if sat.orbit_files:
            if orb_f == None:
                raise LimeException(
                    "The satellite position can't be calculated for a given datetime."
                )
            orbit_path = os.path.join(
                self.eocfi_path.custom_eocfi_path,
                f"data/custom_missions/{orb_f.name}",
            )  # Check first if the custom version exists, giving custom files priority in case they share name.
            if not os.path.exists(orbit_path):
                orbit_path = os.path.join(
                    self.eocfi_path.main_eocfi_path,
                    f"data/missions/{orb_f.name}",
                )
                if not os.path.exists(orbit_path):
                    raise LimeException(
                        "The orbit file {} is missing".format(orbit_path)
                    )
        return self._get_sat_position_orbit_path(sat, dts, orbit_path)

    def _get_sat_position_orbit_path(
        self, sat: Satellite, dts: List[datetime], orbit_path: str
    ) -> List[Tuple[float, float, float]]:
        n_dates = len(dts)
        norad = 0
        intdes = ""
        tle_file = ""
        if sat.norad_sat_number != None:
            norad = sat.norad_sat_number
        if sat.intdes != None:
            intdes = sat.intdes
        if orbit_path.endswith(".txt") or orbit_path.endswith(".TLE"):
            tle_file = orbit_path
            orbit_path = os.path.join(
                self.eocfi_path.main_eocfi_path,
                "data",
                "missions",
                sat.time_file,
            )
        # CALLING EXE BECAUSE SHARED LIBRARY DOESNT WORK
        if platform.system() == "Windows":  # pragma: no cover
            orbit_path = orbit_path.replace("/", "\\")
            if tle_file != "":
                tle_file = tle_file.replace("/", "\\")
        orbit_path = '"{}"'.format(orbit_path)
        _exe_path = _get_exe_path()
        if tle_file == "":
            cmd = "{} {} {} {} {}".format(
                _exe_path,
                0,
                n_dates,
                sat.id,
                orbit_path,
            )
        else:
            tle_file = '"{}"'.format(tle_file)
            cmd = "{} {} {} {} {} {} {} {} {}".format(
                _exe_path,
                1,
                n_dates,
                sat.id,
                norad,
                tle_file,
                orbit_path,
                f'"{sat.name}"',
                intdes,
            )
        for dt in dts:
            cmd = cmd + " {}".format(dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        cmd_exec = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        so, serr = cmd_exec.communicate()
        out_lines = so.splitlines()
        if len(serr) > 0:
            err_msg = "Executing EO CFI: {}".format(serr.rstrip())
            log = logger.get_logger()
            if len(out_lines) == 3 * n_dates:
                log.warning(err_msg)
            else:
                log.error(err_msg)
        sat_positions = []
        if len(out_lines) == 3 * n_dates:
            for i in range(n_dates):
                sat_positions_date = []
                sat_positions_date.append(float(out_lines[i * 3]))
                sat_positions_date.append(float(out_lines[i * 3 + 1]))
                sat_positions_date.append(float(out_lines[i * 3 + 2]))
                sat_positions.append(sat_positions_date)
        else:
            if len(out_lines) == 0:
                raise Exception(
                    "No lines outputed after executing the EOCFI binary. Command executed: {}".format(
                        cmd
                    )
                )
            else:
                raise Exception(
                    "Number of lines unexpected. {}/{}.\n{}".format(
                        str(len(out_lines)), str(3 * n_dates), out_lines
                    )
                )

        positions = [(satpos[0], satpos[1], satpos[2]) for satpos in sat_positions]
        return positions

    def check_data_file_works(
        self, sat: Satellite, dts: List[datetime], orbit_path: str
    ) -> bool:
        """
        Check if the file in orbit_path can be used to calculate the satellite
        position for the datetimes specified in dts.

        Parameters
        ----------
        sat: Satellite
            Satellite for which the file is going to be checked
        dts: List of datetime
            Datetimes for which the position will be calculated
        orbit_path: str
            Path of the data file used for position calculation

        Returns
        -------
        works: bool
            True if the file works, False if it doesn't
        """
        positions = self._get_sat_position_orbit_path(sat, dts, orbit_path)
        if np.any(np.all(np.array(positions) == 0.0, axis=1)):
            return False
        return True

    def get_default_timefile(self) -> str:
        """
        Get the default time file for TLEs when it's not known.

        Returns
        -------
        time_file: str
            Default time_file path value for TLEs
        """
        sats = self._get_sat_list_yaml()
        _def_value = "SENTINEL2B/OSF/S2B_OPER_MPL_ORBSCT_20170309T104400_99999999T999999_0007.EOF"
        sats_with_timefile = ["PLEIADES 1A", "PLEIADES 1B", "PROBA-V", "ENVISAT"]
        for satname in sats_with_timefile:
            if satname in sats and "time_file" in sats[satname]:
                return sats[satname]["time_file"]
        return _def_value
