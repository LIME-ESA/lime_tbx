"""describe class"""

"""___Built-In Modules___"""
import os
from abc import ABC, abstractmethod
import requests

"""___Third-Party Modules___"""
# import here

"""___LIME Modules___"""
from ..access_data import access_data
from ..access_data import programdata

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class IUpdate(ABC):
    @abstractmethod
    def check_for_updates(self) -> bool:
        pass

    @abstractmethod
    def download_coefficients(self):
        pass


class Update(IUpdate):
    def __init__(self):
        self.url = self._get_url()

    def _get_url(self, urlfile_dir: str) -> str:
        filepath = os.path.join(urlfile_dir, "coefficients_server.txt")
        url = ""
        with open(filepath) as fp:
            url = fp.readlines()[0].strip()
        return url

    def check_for_updates(self) -> bool:
        """True if there are updates"""
        urlpath = os.path.join(self.url, "list.txt")
        version_files = requests.get(urlpath).text.split()
        self_files = access_data.get_coefficients_filenames()
        for vf in version_files:
            if vf not in self_files:
                return True
        return False

    def download_coefficients(self):
        urlpath = os.path.join(self.url, "list.txt")
        version_files = requests.get(urlpath).text.split()
        self_files = access_data.get_coefficients_filenames()
        for vf in version_files:
            if vf not in self_files:
                fcontent = requests.get(os.path.join(self.url, vf)).text
                with open(
                    os.path.join(
                        programdata.get_programfiles_folder(),
                        "coeff_data",
                        "versions",
                        vf,
                    ),
                    "w",
                ) as fp:
                    fp.write(fcontent)
