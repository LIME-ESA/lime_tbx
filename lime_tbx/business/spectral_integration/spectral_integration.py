"""
This module performs spectral integrations over spectral response functions and
more data.

It exports the following classes:
    * ISpectralIntegration - Interface that contains the methods of this module.
    * SpectralIntegration - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from typing import List, Union
from abc import ABC, abstractmethod
import os

"""___Third-Party Modules___"""
import numpy as np
import pandas as pd
import punpy
from matheo.band_integration import band_integration

"""___LIME_TBX Modules___"""
from lime_tbx.business.interpolation.interp_data import interp_data
from lime_tbx.common.datatypes import (
    SpectralResponseFunction,
    SpectralData,
    SRF_fwhm,
    SRFChannel,
)
from lime_tbx.common import constants, logger

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


_ASD_FILE = "assets/asd_fwhm.csv"
_CIMEL_FILE = "assets/responses_1088.csv"
_INTERPOLATED_GAUSSIAN_FILE = "assets/interpolated_model_fwhm_3_1_gaussian.csv"
_INTERPOLATED_TRIANGULAR_FILE = "assets/interpolated_model_fwhm_1_1_triangle.csv"


def get_default_srf():
    dir_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    dir_path = os.path.join(dir_path, "lime_tbx", "spectral_integration", "assets")
    data = np.genfromtxt(
        os.path.join(dir_path, "interpolated_model_fwhm_3_1_gaussian.csv"),
        delimiter=",",
    )
    wavs = data[:, 0]
    spectral_response = {i: 1.0 for i in wavs}
    ch = SRFChannel(
        (constants.MAX_WLEN - constants.MIN_WLEN) / 2,
        constants.DEFAULT_SRF_NAME,
        spectral_response,
    )
    srf = SpectralResponseFunction(constants.DEFAULT_SRF_NAME, [ch])
    return srf


class ISpectralIntegration(ABC):
    """Interface that contains the methods of this module.

    It exports the following functions:
        * integrate_elis() - Integrate the irradiances and obtain the signal.
        * u_integrate_elis() - Calculate the uncertainties of the irradiances integration.
    """

    @abstractmethod
    def integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Integrate the irradiances and obtain the signal.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral response function to integrate to
        elis_lime: SpectralData
            Data to integrate to the srf

        Returns
        -------
        integr_data: list of float or list of list of float
            Integrated signals. List of float if elis_lime.data values are floats,
            list of list of float if they are lists or ndarrays.
        """
        pass

    @abstractmethod
    def u_integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Calculate the uncertainties of the irradiances integration.

        Parameters
        ----------
        srf: SpectralResponseFunction
            Spectral response function to integrate to
        elis_lime: SpectralData
            Data to integrate to the srf

        Returns
        -------
        u_integr_data: list of float or list of list of float
            Uncertainties of the integrated signals. List of float if elis_lime.data
            values are floats, list of list of float if they are lists or ndarrays.
        """
        pass

    @abstractmethod
    def integrate_cimel(
        self, data: np.ndarray, wlens: np.ndarray, cimel_wavs: List[float]
    ) -> np.ndarray:
        """
        Integrate the spectrum of x = wlens and y = data over the CIMEL wavelengths specified in cimel_wavs.
        Internally it obtains the CIMEL spectral response function.

        Parameters
        ----------
        data: np.array of float
            Data that is going to be integrated over the CIMEL bands.
        wlens: np.array of float
            Wavelengths in nm related to the data.
        cimel_wavs: np.array of float
            Nominal wavelengths of the chosen CIMEL bands.

        Returns
        -------
        cimel_integ_data: np.array of float
            Data integrated over the CIMEL bands
        """
        pass


class SpectralIntegration(ISpectralIntegration):
    """Class that implements the methods exported by this module"""

    def __init__(self, MCsteps=1000):
        self.prop = punpy.MCPropagation(MCsteps, parallel_cores=1)
        self.asd_srf = self._read_srf_asd()
        self.cimel_srf = self._read_srf_cimel()
        self.interpolated_srf = self._read_srf_interpolated()

    def _read_srf_asd(self) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = np.genfromtxt(os.path.join(dir_path, _ASD_FILE), delimiter=",")
        srf = SRF_fwhm("asd", data[:, 0], data[:, 1], "gaussian")
        return srf

    def _read_srf_interpolated(self) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        srf_type = interp_data.get_interpolation_srf_as_srf_type()
        if srf_type == "interpolated_gaussian":
            data = np.genfromtxt(
                os.path.join(dir_path, _INTERPOLATED_GAUSSIAN_FILE), delimiter=","
            )
            srf = SRF_fwhm("interpolated", data[:, 0], data[:, 1], "gaussian")
        elif srf_type == "interpolated_triangle":
            data = np.genfromtxt(
                os.path.join(dir_path, _INTERPOLATED_TRIANGULAR_FILE), delimiter=","
            )
            srf = SRF_fwhm("interpolated", data[:, 0], data[:, 1], "triangle")
        elif srf_type == "asd":
            data = np.genfromtxt(os.path.join(dir_path, _ASD_FILE), delimiter=",")
            srf = SRF_fwhm("interpolated", data[:, 0], data[:, 1], "asd")
        else:
            logger.get_logger().error(
                "The selected interpolated SRF file for spectral integration wasn't valid."
                + "Selecting the default gaussian one."
            )
            data = np.genfromtxt(
                os.path.join(dir_path, _INTERPOLATED_GAUSSIAN_FILE), delimiter=","
            )
            srf = SRF_fwhm("interpolated", data[:, 0], data[:, 1], "gaussian")
        return srf

    def set_srf_interpolated(
        self, fwhm, sampling, shape, write=False
    ) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        wavs = np.arange(constants.MIN_WLEN, constants.MAX_WLEN + sampling, sampling)
        fwhms = fwhm * np.ones_like(wavs)
        if write:
            id_str = ("%s_%s_%s" % (fwhm, sampling, shape)).replace(".", "p")
            with open(
                os.path.join(
                    dir_path, "assets/interpolated_model_fwhm_" + id_str + ".csv"
                ),
                "w",
            ) as f:
                for i in range(len(wavs)):
                    f.write("%s,%s\n" % (wavs[i], fwhms[i]))
        self.interpolated_srf = SRF_fwhm("interpolated", wavs, fwhms, shape)

    def _read_srf_cimel(self) -> SpectralResponseFunction:
        """
        read asd fwhm and make SRF
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        srflist = []
        df = pd.read_csv(os.path.join(dir_path, _CIMEL_FILE))
        cimel_wavs = [int(c[2:]) for c in df.columns if c.startswith("w")]
        for cw in cimel_wavs:
            wlens = df[f"w.{cw}"].values
            resp = df[f"r.{cw}"].values
            srf = {w: r for w, r in zip(wlens, resp) if r != 0}
            channel = SRFChannel(cw, str(cw), srf)
            srflist.append(channel)
        return SpectralResponseFunction("cimel", srflist)

    def integrate_cimel(
        self, data: np.ndarray, wlens: np.ndarray, cimel_wavs: List[float]
    ) -> np.ndarray:
        channels = [
            self.cimel_srf.get_channel_from_name(f"{int(wl)}") for wl in cimel_wavs
        ]
        if None in channels:
            raise Exception(
                "Coefficients channels are not present in default CIMEL SRF."
            )
        y = self.integrate_elis_xy(channels, data, wlens)
        return y

    def integrate_solar_asd(
        self,
        data: np.ndarray,
        wlens: np.ndarray,
    ) -> np.ndarray:
        return band_integration.pixel_int(
            data,
            wlens,
            self.asd_srf.get_wavelengths(),
            (self.asd_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape=self.asd_srf.get_shape(),
        )

    def integrate_solar_interpolated_default(
        self, data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        return band_integration.pixel_int(
            data,
            wlens,
            self.interpolated_srf.get_wavelengths(),
            (self.interpolated_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape=self.interpolated_srf.get_shape(),
        )

    def integrate_solar_interpolated_gaussian(
        self, data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        # subtracting a fwhm of 0.1 in quadrature to account for fwhm of TSIS solar irradiance
        return band_integration.pixel_int(
            data,
            wlens,
            self.interpolated_srf.get_wavelengths(),
            (self.interpolated_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape="gaussian",
        )

    def integrate_solar_interpolated_triangle(
        self, data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        return band_integration.pixel_int(
            data,
            wlens,
            self.interpolated_srf.get_wavelengths(),
            (self.interpolated_srf.get_values() ** 2 - 0.01) ** 0.5,
            band_shape="triangle",
        )

    def integrate_elis_xy(
        self, channels: List[SRFChannel], data: np.ndarray, wlens: np.ndarray
    ) -> np.ndarray:
        """

        :param srf:
        :type srf:
        :param data:
        :type data:
        :param wlens:
        :type wlens:
        :return:
        :rtype:
        """
        signals = np.zeros(len(channels))
        for ich, ch in enumerate(channels):
            ch_wlens = np.array([w for w in ch.spectral_response.keys()])
            ch_srf = np.array([ch.spectral_response[k] for k in ch_wlens])
            # TODO shouldnt uncerts be added?
            band_int = band_integration.band_int(data, wlens, ch_srf, ch_wlens)
            if isinstance(band_int, np.ndarray):
                band_int = band_int[0]
            signals[ich] = band_int
        return signals

    def _convolve_srf(
        self, ch_wlens: np.ndarray, ch_srf: np.ndarray, ch_elis: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Convolve ch_srf and ch_elis over ch_wlens, generating the signal.
        """
        divider = np.trapz(ch_srf, ch_wlens)
        if isinstance(divider, np.ndarray):
            divider[divider == 0] = 1
        elif divider == 0:
            divider = 1
        ch_signal = np.trapz(ch_srf * ch_elis, ch_wlens) / divider
        return ch_signal

    def _integrate_elis(self, elis, wlens, all_wlens, *ch_wlenssrf):
        signals = []
        if len(elis) == 0:
            return []
        ch_wlenss = []
        ch_srfs = []
        for i in range(0, len(ch_wlenssrf), 2):
            ch_wlenss.append(ch_wlenssrf[i])
            ch_srfs.append(ch_wlenssrf[i + 1])
        all_interp_elis = np.interp(all_wlens, wlens, elis)
        interm_res_path = (
            None
            if constants.DEBUG_INTERMEDIATE_RESULTS_PATH not in os.environ
            else os.environ[constants.DEBUG_INTERMEDIATE_RESULTS_PATH]
        )
        if interm_res_path:
            np.savetxt(
                f"{interm_res_path}/tests_stefan/irrs_srf_interp_from_irrs_spectrum.csv",
                np.array([all_wlens, all_interp_elis]).T,
                delimiter=",",
                fmt=["%f", "%e"],
            )
        for ch_wlens, ch_srf in zip(ch_wlenss, ch_srfs):
            elis_ids = np.where(np.in1d(all_wlens, ch_wlens))[0]
            ch_elis = all_interp_elis[elis_ids]
            signals.append(self._convolve_srf(ch_wlens, ch_srf, ch_elis))
        return signals

    def integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        wlens = elis_lime.wlens
        elis = elis_lime.data
        # ch_wlenss = []
        # ch_srfs = []
        ch_wlensrfs = []
        all_wlens = np.sort(np.unique(srf.get_wavelengths()))
        for ch in srf.channels:
            ch_wlensrfs.append(np.array(list(ch.spectral_response_inrange.keys())))
            ch_wlensrfs.append(np.array(list(ch.spectral_response_inrange.values())))
            # ch_wlenss.append(np.array(list(ch.spectral_response_inrange.keys())))
            # ch_srfs.append(np.array(list(ch.spectral_response_inrange.values())))
        signals = self._integrate_elis(elis, wlens, all_wlens, *ch_wlensrfs)
        return signals

    def u_integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        u_signals = []
        wlens = np.array(elis_lime.wlens)
        elis = np.array(elis_lime.data)
        u_elis = np.array(elis_lime.uncertainties)
        corr_elis = np.array(elis_lime.err_corr)
        ch_wlensrfs = []
        all_wlens = np.sort(np.unique(srf.get_wavelengths()))
        for ch in srf.channels:
            ch_wlensrfs.append(np.array(list(ch.spectral_response_inrange.keys())))
            ch_wlensrfs.append(np.array(list(ch.spectral_response_inrange.values())))
        # check for empty arrays as they crash the unc. propagation, but they have to be readded at the end
        # (only adding one of each two arrays)
        empties = [len(ch_wlensrfs[i]) == 0 for i in range(0, len(ch_wlensrfs), 2)]
        empties_index = [i for i, x in enumerate(empties) if x]
        ch_wlensrfs = [chwls for chwls in ch_wlensrfs if len(chwls) > 0]
        #
        u_signals: np.ndarray = self.prop.propagate_random(
            self._integrate_elis,
            [elis, wlens, all_wlens, *ch_wlensrfs],
            [u_elis, None, None, *[None for _ in ch_wlensrfs]],
            corr_x=[corr_elis, None, None, *[None for _ in ch_wlensrfs]],
        )
        # fill with 0s the removed positions
        for emp_index in empties_index:
            u_signals = np.insert(u_signals, emp_index, 0)
        return u_signals


"""
    @staticmethod
    def integrate_elis_old(
        srf: SpectralResponseFunction, elis: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:
        signals = []
        wlens = srf.get_wavelengths()
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
            wasnt_lists = True
            elis = [elis]
        for ch in srf.channels:
            tots_eli = [0 for _ in range(len(elis))]
            ch_wlens = list(ch.spectral_response.keys())
            dividends = [0 for _ in range(len(elis))]
            for i, wl in enumerate(ch_wlens):
                interval = 0
                if i > 0:
                    interval += (wl - ch_wlens[i - 1]) / 2
                if i < len(ch_wlens) - 1:
                    interval += (ch_wlens[i + 1] - wl) / 2
                extra_dividend = ch.spectral_response[wl] * interval
                for i, sub_elis in enumerate(elis):
                    eli = sub_elis[wlens.index(wl)]
                    tots_eli[i] += extra_dividend * eli
                    dividends[i] += extra_dividend
            ch_signals = []
            for i in range(len(tots_eli)):
                signal = tots_eli[i] / dividends[i]
                ch_signals.append(signal)
            signals.append(ch_signals)
        if wasnt_lists:
            signals = [s[0] for s in signals]
        return signals"""
