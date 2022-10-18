"""
This module contains the abstractions and interfaces that call EOCFI and perform satellite functions.

It exports the following classes:
    * IEOCFIConverter - Interface that contains the methods of this module.
    * EOCFIConverter - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
import ctypes as ct
from typing import Tuple, List
from datetime import datetime, timezone
import os
import platform

from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter

"""___Third-Party Modules___"""
import yaml

"""___NPL Modules___"""
from ..datatypes.datatypes import KernelsPath, LimeException, OrbitFile, Satellite

"""___Authorship___"""
__author__ = "Ramiro González Catón"
__created__ = "24/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

ESA_SAT_LIST = "esa_sat_list.yml"
METADATA_FILE = "metadata.yml"
SO_FILE_SATELLITE_LINUX = "eocfi_c/bin/get_positions_linux.so"
SO_FILE_SATELLITE_WINDOWS = "eocfi_c\\bin\\get_positions_win64.dll"
SO_FILE_SATELLLITE_DARWIN = "eocfi_c/bin/get_positions_darwin.so"
EXE_FILE_SATELLITE_LINUX = "eocfi_c/bin/get_positions_linux"
EXE_FILE_SATELLITE_WINDOWS = "eocfi_c\\bin\\get_positions_win64.exe"
EXE_FILE_SATELLLITE_DARWIN = "eocfi_c/bin/get_positions_darwin"

if platform.system() == "Linux":
    so_file_satellite = SO_FILE_SATELLITE_LINUX
    exe_file_satellite = EXE_FILE_SATELLITE_LINUX
elif platform.system() == "Windows":
    so_file_satellite = SO_FILE_SATELLITE_WINDOWS
    exe_file_satellite = EXE_FILE_SATELLITE_WINDOWS
else:
    so_file_satellite = SO_FILE_SATELLLITE_DARWIN
    exe_file_satellite = EXE_FILE_SATELLLITE_DARWIN

_current_dir = os.path.dirname(os.path.abspath(__file__))
_so_path = os.path.join(_current_dir, so_file_satellite)
_exe_path = os.path.join(_current_dir, exe_file_satellite)
eocfi_sat = ct.CDLL(_so_path)


def _make_clist(lst: List[str]):
    """
    Helper function to turn Python list of Unicode strings
    into a ctypes array of byte strings.

    Parameters
    ----------
    lst: list of str
        List of unicode strings

    Returns
    -------
    clst: c_char_p_Array
        ctypes array of byte strings
    """
    return (ct.c_char_p * len(lst))(*[x.encode() for x in lst])


def _get_file_datetimes(filename: str) -> Tuple[datetime, datetime]:
    splitted = filename.split("_")
    date0 = datetime.strptime(splitted[-3] + "+00:00", "%Y%m%dT%H%M%S%z")
    if splitted[-2] == "99999999T999999":
        date1 = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    else:
        date1 = datetime.strptime(splitted[-2] + "+00:00", "%Y%m%dT%H%M%S%z")
    return date0, date1


def _to_mjd2000(dt: datetime) -> float:
    mjd = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    tdelta = dt - mjd
    return tdelta.total_seconds() / 86400


class IEOCFIConverter(ABC):
    @abstractmethod
    def get_sat_names(self) -> List[str]:
        """
        Obtain the list of satellite names, that are the keys that can be used in
        get_satellite_position

        Returns
        -------
        sat_names: list of str
            List of the satellite names
        """
        pass

    @abstractmethod
    def get_sat_list(self) -> List[Satellite]:
        """
        Obtain a list of the satellite data objects that are available in LIME TBX.

        Returns
        -------
        sat_list: list of Satellite
            List of Satellites available in LIME TBX.
        """
        pass

    @abstractmethod
    def get_satellite_position(
        self, sat: str, dt: List[datetime]
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
        pass


class EOCFIConverter(IEOCFIConverter):
    def __init__(self, eocfi_path: str, kernels_path: KernelsPath):
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

    def _get_sat_list_yaml(self) -> dict:
        """
        Read the sat list yaml and return the object.

        Returns
        -------
        sat_list: dict
            Dictionary containing the sat list yaml.
        """
        fl = open(os.path.join(self.eocfi_path, ESA_SAT_LIST))
        y = yaml.load(fl, Loader=yaml.FullLoader)
        fl.close()
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
        if sat not in self.get_sat_names():
            raise Exception("Satellite is not registered in LIME's satellite list.")
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
        is_pred = False
        orbit_path = ""
        if sat.orbit_files:
            if orb_f == None:
                raise LimeException(
                    "The satellite position can't be calculated for a given datetime."
                )
            orbit_path = os.path.join(
                self.eocfi_path,
                f"data/mission_configuration_files/{orb_f.name}",
            )
            if not os.path.exists(orbit_path):
                raise LimeException("The orbit file {} is missing".format(orbit_path))
            if "ORBPRE" in orb_f.name:
                is_pred = True
            # We have to make this a list/array
        fl = open(os.path.join(self.eocfi_path, METADATA_FILE))
        metadata = yaml.load(fl, Loader=yaml.FullLoader)
        fl.close()
        if not is_pred:
            bulletin_name = metadata.get("BULLETIN_IERS")["file_b"]
        else:
            bulletin_name = metadata.get("BULLETIN_IERS")["file_a"]
        bulletin = os.path.join(self.eocfi_path, bulletin_name)

        n_dates = len(dts)
        l_cdt = []
        for dt in dts:
            c_dt = (ct.c_int * 6)(
                *[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
            )
            l_cdt.append(c_dt)
        c_dts = (ct.POINTER(ct.c_int) * n_dates)(*l_cdt)
        sat_positions_arrs = []
        for _ in range(n_dates):
            sat_positions_arrs.append((ct.c_double * 3)(*[0.0, 0.0, 0.0]))
        sat_positions = (ct.POINTER(ct.c_double) * n_dates)(*sat_positions_arrs)
        norad = 0
        intdes = ""
        tle_file = ""
        if sat.norad_sat_number != None:
            norad = sat.norad_sat_number
        if sat.intdes != None:
            intdes = sat.intdes
        if orb_f.name.endswith(".txt") or orb_f.name.endswith(".TLE"):
            tle_file = orbit_path
            orbit_path = os.path.join(
                self.eocfi_path,
                "data",
                "mission_configuration_files",
                sat.time_file,
            )
        if tle_file == "":
            eocfi_sat.get_satellite_position_osf(
                ct.c_long(sat.id),
                ct.c_int(n_dates),
                c_dts,
                ct.c_char_p(orbit_path.encode()),
                sat_positions,
            )
        else:
            if platform.system() == "Windows":
                orbit_path = orbit_path.replace("/", "\\")
                tle_file = tle_file.replace("/", "\\")
            orbit_path = '"{}"'.format(orbit_path)
            tle_file = '"{}"'.format(tle_file)
            # CALLING EXE BECAUSE SHARED LIBRARY DOESNT WORK FOR TLE
            cmd = "{} {} {} {} {} {} {} {}".format(
                _exe_path,
                n_dates,
                sat.id,
                norad,
                tle_file,
                orbit_path,
                sat.name,
                intdes,
            )
            for dt in dts:
                cmd = cmd + " {}".format(dt.strftime("%Y-%m-%dT%H:%M:%S"))
            so = os.popen(cmd).read()
            out_lines = so.splitlines()
            if len(out_lines) == 3 * n_dates:
                for i in range(n_dates):
                    sat_positions[i][0] = ct.c_double(float(out_lines[i * 3]))
                    sat_positions[i][1] = ct.c_double(float(out_lines[i * 3 + 1]))
                    sat_positions[i][2] = ct.c_double(float(out_lines[i * 3 + 2]))
            else:
                raise Exception(
                    "Number of lines unexpected. {}/{}.\n{}".format(
                        str(len(out_lines)), str(3 * n_dates), out_lines
                    )
                )

        positions = []
        for i in range(n_dates):
            x, y, z = (sat_positions[i][0], sat_positions[i][1], sat_positions[i][2])
            lat, lon, hhh = SPICEAdapter.to_planetographic(
                x, y, z, "EARTH", self.kernels_path.main_kernels_path
            )
            print(lat, lon, hhh)
            positions.append((lat, lon, hhh))
        return positions
