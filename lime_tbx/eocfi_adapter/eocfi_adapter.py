"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from ctypes import *
from typing import Tuple, List
from datetime import datetime
import os
import pkgutil
import platform

"""___Third-Party Modules___"""
from numpy.ctypeslib import ndpointer
import yaml
import pyproj

"""___NPL Modules___"""
# import here

"""___Authorship___"""
__author__ = "Ramiro González Catón"
__created__ = "24/05/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"

ESA_SAT_LIST = "esa_sat_list.yml"
METADATA_FILE = "metadata.yml"
SO_FILE_SATELLITE_LINUX = "eocfi_c/bin/get_positions.so"

if platform.system() == "Linux":
    so_file_satellite = SO_FILE_SATELLITE_LINUX
else:
    so_file_satellite = ""

_package = pkgutil.get_loader(__name__)
_so_path = os.path.join(os.path.dirname(_package.path), so_file_satellite)
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
        return list(self.get_sat_list().keys())

    def get_sat_list(self) -> dict:
        """
        Read the sat list yaml and return the object.

        Returns
        -------
        sat_list: dict
            Dictionary containing the sat list yaml.
        """
        return yaml.load(
            open(os.path.join(self.eocfi_path, ESA_SAT_LIST)), Loader=yaml.FullLoader
        )

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
        sat_list = self.get_sat_list()
        if sat not in sat_list:
            raise Exception("Satellite is not registered in EOCFI's satellite list.")
        sat_data = sat_list.get(sat)
        sat_id = sat_data["id"]
        sat_n_files = sat_data["n_files"]
        sat_orbit_files = sat_data["orbit_files"]
        # hay que hacer que esto sea un array
        orbit_files = []
        for file in range(sat_n_files):
            orbit_file = os.path.join(
                self.eocfi_path,
                f"data/mission_configuration_files/{sat_orbit_files[file]}",
            )
            # orbit_file = orbit_file.encode()
            orbit_files.append(orbit_file)

        metadata = yaml.load(
            open(os.path.join(self.eocfi_path, METADATA_FILE)), Loader=yaml.FullLoader
        )

        bulletin = os.path.join(self.eocfi_path, metadata.get("BULLETIN_IERS")["file"])

        bulletinb_file_init_time = bulletin.encode()

        eocfi_sat.get_satellite_position.restype = ndpointer(dtype=c_double, shape=(3,))
        sat_position = eocfi_sat.get_satellite_position(
            c_long(sat_id),
            c_int(dt.year),
            c_int(dt.month),
            c_int(dt.day),
            c_int(dt.hour),
            c_int(dt.minute),
            c_int(dt.second),
            bulletinb_file_init_time,
            _make_clist(orbit_files),
            c_long(sat_n_files),
        )

        transformer = pyproj.Transformer.from_crs(
            {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
            {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        )
        lat, lon, hhh = transformer.transform(
            sat_position[0], sat_position[1], sat_position[2], radians=False
        )
        print(lat, lon, hhh)
        return lat, lon, hhh
        # print("Geocentric: ", 360+lat, lon, hhh)
