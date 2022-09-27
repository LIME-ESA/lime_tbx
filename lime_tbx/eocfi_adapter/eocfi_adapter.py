"""
This module contains the abstractions and interfaces that call EOCFI and perform satellite functions.

It exports the following classes:
    * IEOCFIConverter - Interface that contains the methods of this module.
    * EOCFIConverter - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from ctypes import *
from typing import Tuple, List
from datetime import datetime, timezone
import os
import platform

"""___Third-Party Modules___"""
from numpy.ctypeslib import ndpointer
import yaml
import pyproj

"""___NPL Modules___"""
from ..datatypes.datatypes import OrbitFile, Satellite

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

if platform.system() == "Linux":
    so_file_satellite = SO_FILE_SATELLITE_LINUX
else:
    so_file_satellite = SO_FILE_SATELLITE_WINDOWS

_current_dir = os.path.dirname(os.path.abspath(__file__))
_so_path = os.path.join(_current_dir, so_file_satellite)
eocfi_sat = CDLL(_so_path)


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
    return (c_char_p * len(lst))(*[x.encode() for x in lst])


def _get_file_datetimes(filename: str) -> Tuple[datetime, datetime]:
    splitted = filename.split("_")
    date0 = datetime.strptime(splitted[-3] + "+00:00", "%Y%m%dT%H%M%S%z")
    if splitted[-2] == "99999999T999999":
        date1 = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    else:
        date1 = datetime.strptime(splitted[-2] + "+00:00", "%Y%m%dT%H%M%S%z")
    return date0, date1


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
        self, sat: str, dt: datetime
    ) -> Tuple[float, float, float]:
        """
        Get the geographic satellite position for a concrete datetime.

        Parameters
        ----------
        sat: str
            Satellite name. Should be present in get_sat_names
        dt: datetime
            Datetime for which the position will be calculated

        Returns
        -------
        latitude: float
            Geocentric latitude of the satellite
        longitude: float
            Geocentric longitude of the satellite
        height: float
            Height of the satellite over sea level in meters.
        """
        pass


class EOCFIConverter(IEOCFIConverter):
    def __init__(self, eocfi_path: str):
        super().__init__()
        self.eocfi_path = eocfi_path

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
            if orbit_files_names == None:
                orbit_files_names = []
            orbit_files = []
            for file in orbit_files_names:
                d0, df = _get_file_datetimes(file)
                orbit_f = OrbitFile(file, d0, df)
                orbit_files.append(orbit_f)
            sat = Satellite(name, id, orbit_files)
            sat_list.append(sat)
        return sat_list

    def get_satellite_position(
        self, sat: str, dt: datetime
    ) -> Tuple[float, float, float]:
        """
        Get the geographic satellite position for a concrete datetime.

        Parameters
        ----------
        sat: str
            Satellite name. Should be present in get_sat_names
        dt: datetime
            Datetime for which the position will be calculated

        Returns
        -------
        latitude: float
            Geocentric latitude of the satellite.
        longitude: float
            Geocentric longitude of the satellite.
        height: float
            Height of the satellite over sea level in meters.
        """
        if sat not in self.get_sat_names():
            raise Exception("Satellite is not registered in LIME's satellite list.")
        sat: Satellite = [s for s in self.get_sat_list() if s.name == sat][0]
        orb_f = sat.get_best_orbit_file(dt)
        if sat.orbit_files:
            if orb_f == None:
                raise Exception(
                    "The satellite position can't be calculated for the given datetime."
                )
            # We have to make this a list/array
            orbit_files = [
                os.path.join(
                    self.eocfi_path,
                    f"data/mission_configuration_files/{orb_f.name}",
                )
            ]
        else:
            orbit_files = []
        fl = open(os.path.join(self.eocfi_path, METADATA_FILE))
        metadata = yaml.load(fl, Loader=yaml.FullLoader)
        fl.close()

        bulletin = os.path.join(self.eocfi_path, metadata.get("BULLETIN_IERS")["file"])

        bulletinb_file_init_time = bulletin.encode()

        eocfi_sat.get_satellite_position.restype = ndpointer(dtype=c_double, shape=(3,))
        sat_position = eocfi_sat.get_satellite_position(
            c_long(sat.id),
            c_int(dt.year),
            c_int(dt.month),
            c_int(dt.day),
            c_int(dt.hour),
            c_int(dt.minute),
            c_int(dt.second),
            bulletinb_file_init_time,
            _make_clist(orbit_files),
            c_long(len(orbit_files)),
        )

        transformer = pyproj.Transformer.from_crs(
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        )
        lat, lon, hhh = transformer.transform(
            sat_position[0], sat_position[1], sat_position[2], radians=False
        )
        return lat, lon, hhh
