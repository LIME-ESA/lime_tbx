"""Module in charge of downloading the updated coefficients"""

"""___Built-In Modules___"""
import os
from abc import ABC, abstractmethod
import requests
from typing import Callable, Tuple
import re

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.coefficients.access_data import access_data
from lime_tbx.local_storage import programdata
from lime_tbx.filedata import coefficients
from lime_tbx.datatypes import logger

"""___Authorship___"""
__author__ = "Pieter De Vis, Jacob Fahy, Javier Gatón Herguedas, Ramiro González Catón, Carlos Toledano"
__created__ = "01/02/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class IUpdate(ABC):
    @abstractmethod
    def check_for_updates(self, timeout=60) -> bool:
        """
        Checks for any coefficient updates in the server

        Parameters
        ----------
        timeout: int
            Timeout in seconds for the request that will be sent to the server. The default is 60.

        Returns
        -------
        are_there_updates: bool
            True if there are updates
        """
        pass

    @abstractmethod
    def download_coefficients(
        self, stopper_checker: Callable, stopper_args: list
    ) -> Tuple[int, int]:
        """
        Download and update the toolbox coefficients from the server.

        Parameters
        ----------
        stopper_checker: callable
            Function that receives '*stopper_args' as the parameters, and returns
            a boolean value that indicates if the update is still running (True) or if
            it has been stopped (False).
        stopper_args: list
            Arguments of the stopper_checker function.

        Returns
        -------
        quant_news: int
            Amount of new coefficients downloaded.
        quant_fails: int
            Amount of coefficients that couldn't be downloaded or updated correctly.
        """
        pass


class Update(IUpdate):
    def __init__(self):
        self.url = self._get_url(
            os.path.join(programdata.get_programfiles_folder(), "coeff_data")
        )

    def _get_url(self, urlfile_dir: str) -> str:
        filepath = os.path.join(urlfile_dir, "coefficients_server.txt")
        url = ""
        with open(filepath) as fp:
            url = fp.readlines()[0].strip()
        return url

    def check_for_updates(self, timeout=60) -> bool:
        urlpath = self.url
        text = requests.get(urlpath, timeout=timeout).text
        text = re.sub("<\!\-\-(.|\n)*\-\-\>", "", text)  # remove comments
        version_files = text.split()
        version_files = [
            tuple(line.split(","))
            for line in version_files
            if line not in ("<br>", "<br/>")
        ]
        version_files = [vf[1][len(urlpath) :] for vf in version_files]
        self_files = access_data.get_coefficients_filenames()
        for vf in version_files:
            if vf not in self_files:
                return True
        return False

    def download_coefficients(
        self, stopper_checker: Callable, stopper_args: list
    ) -> Tuple[int, int]:
        urlpath = f"{self.url}"
        version_files = requests.get(urlpath).text.split()
        version_files = [
            tuple(line.split(","))
            for line in version_files
            if line not in ("<br>", "<br/>")
        ]
        self_files = access_data.get_coefficients_filenames()
        quant_news = 0
        quant_fails = 0
        for index, vfurl in version_files:
            vf = vfurl[len(urlpath) :]
            is_running = stopper_checker(*stopper_args)
            if not is_running:
                return (quant_news, quant_fails)
            if vf not in self_files:
                quant_news += 1
                fcontent = requests.get(f"{vfurl}").content
                filepath = os.path.join(
                    programdata.get_programfiles_folder(),
                    "coeff_data",
                    "versions",
                    vf,
                )
                with open(
                    filepath,
                    "wb",
                ) as fp:
                    fp.write(fcontent)
                try:
                    coefficients.read_coeff_nc(filepath)
                except Exception as e:
                    quant_fails += 1
                    os.remove(filepath)
                    logger.get_logger().warning(
                        "Wrong coefficient data downloaded. Error: {}".format(e)
                    )
        return (quant_news, quant_fails)
