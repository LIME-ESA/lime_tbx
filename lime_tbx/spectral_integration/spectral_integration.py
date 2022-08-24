"""
This module performs spectral intetgrations over spectral response functions and
more data.

It exports the following classes:
    * ISpectralIntegration - Interface that contains the methods of this module.
    * SpectralIntegration - Class that implements the methods exported by this module.
"""

"""___Built-In Modules___"""
from typing import List, Union
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np
import punpy

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    SpectralResponseFunction,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"


class ISpectralIntegration(ABC):
    @abstractmethod
    def integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Integrate the irradiances and obtain the signal.
        """
        pass

    @abstractmethod
    def u_integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        """
        Calculate the uncertainties of the irradiances integration.
        """
        pass


class SpectralIntegration(ISpectralIntegration):
    def __init__(self, MCsteps=1000):
        self.prop = punpy.MCPropagation(MCsteps)

    def _convolve_srf(
        self, ch_wlens: np.ndarray, ch_srf: np.ndarray, ch_elis: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Convolve ch_srf and ch_elis over ch_wlens, generating the signal.
        """
        ch_signal = np.trapz(ch_srf * ch_elis, ch_wlens)
        return ch_signal

    def integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        signals = []
        wlens = srf.get_wavelengths()
        elis = elis_lime.data
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
            wasnt_lists = True
            elis = np.array([elis])
        for ch in srf.channels:
            ch_wlens = np.array(list(ch.spectral_response.keys()))
            ch_srf = np.array(list(ch.spectral_response.values()))
            elis_ids = [wlens.index(wl) for wl in ch_wlens]
            ch_signals = []
            for subelis in elis:
                ch_elis = subelis[elis_ids]
                ch_signals.append(self._convolve_srf(ch_wlens, ch_srf, ch_elis))
            signals.append(ch_signals)
        if wasnt_lists:
            signals = [s[0] for s in signals]
        return signals

    def u_integrate_elis(
        self, srf: SpectralResponseFunction, elis_lime: SpectralData
    ) -> Union[List[float], List[List[float]]]:
        u_signals = []
        wlens = elis_lime.wlens
        elis = elis_lime.data
        u_elis = elis_lime.uncertainties
        if len(elis) == 0:
            return []
        wasnt_lists = False
        if not isinstance(elis[0], list) or not isinstance(elis[0], np.ndarray):
            wasnt_lists = True
            elis = np.array([elis])
        for ch in srf.channels:
            ch_wlens = np.array(list(ch.spectral_response.keys()))
            ch_srf = np.array(list(ch.spectral_response.values()))
            elis_ids = [wlens.index(wl) for wl in ch_wlens]

            u_ch_signals = []
            for subelis in elis:
                ch_elis = subelis[elis_ids]
                u_ch_signals.append(
                    self.prop.propagate_random(
                        self._convolve_srf,
                        [ch_wlens, ch_srf, ch_elis],
                        [None, None, u_elis],
                        corr_x=[None, None, None],
                    )
                )
                u_signals.append(u_ch_signals)
        if wasnt_lists:
            u_signals = [s[0] for s in u_signals]
        return u_signals
