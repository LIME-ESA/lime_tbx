"""
This module calculates the extra-terrestrial lunar disk irradiance.

It exports the following classes:
    * IROLO - Interface that contains the methods of this module.
    * ROLO - Class that implements the methods exported by this module.

It follows equations described in the following papers:
- Kieffer and Stone, 2005: The spectral irradiance of the Moon.
- Barreto et al., 2019: Evaluation of night-time aerosols measurements and lunar irradiance
models in the frame of the first multi-instrument nocturnal intercomparison campaign.
- Roman et al., 2020: Correction of a lunar-irradiance model for aerosol optical depth
retrieval and comparison with a star photometer.
"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.lime_algorithms.rolo import eli, elref
from lime_tbx.datatypes.datatypes import (
    MoonData,
    ReflectanceCoefficients,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


class IROLO(ABC):
    """
    Interface that contains the methods of this module.

    It exports the following functions:
        * get_elis_from_elrefs: Calculates the extra-terrestrial lunar irradiance in Wm⁻²/nm
            from reflectance data.
        * get_elrefs: Calculates the extra-terrestrial lunar reflectance in fractions of unity for some
            given parameters.
    """

    @staticmethod
    @abstractmethod
    def get_elis_from_elrefs(
        elref_spectrum: SpectralData,
        moon_data: MoonData,
        srf_type: str,
        skip_uncs: bool = False,
    ) -> SpectralData:
        """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020,
        without using the Apollo Coefficients.

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in Wm⁻²/nm

        Parameters
        ----------
        elref_spectrum : SpectralData
            Reflectance data.
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance.
        srf_type: str
            SRF type that is going to be used. Can be 'cimel', 'asd' or 'interpolated'.
        skip_uncs: bool
            Flag for skipping the calculation of uncertainties.

        Returns
        -------
        elis: SpectralData
            The extraterrestrial lunar irradiances calculated.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_elrefs(
        coefficients: ReflectanceCoefficients,
        moon_data: MoonData,
        skip_uncs: bool = False,
    ) -> SpectralData:
        """Calculation of Extraterrestrial Lunar Reflectance following Eq 3 in Roman et al., 2020
        for the calculation of the irradiance, without using the Apollo Coefficients.

        Allow users to simulate lunar observation for any observer/solar selenographic
        latitude and longitude.

        Returns the data in fractions of unity.

        Parameters
        ----------
        coefficients : ReflectanceCoefficients
            Needed coefficients for the simulation
        moon_data : MoonData
            Moon data needed to calculate Moon's irradiance.
        skip_uncs: bool
            Flag for skipping the calculation of uncertainties.

        Returns
        -------
        SpectralData
            The extraterrestrial lunar reflectances calculated.
        """
        pass


class ROLO(IROLO):
    """
    Class that implements the methods of this module.
    """

    @staticmethod
    def get_elis_from_elrefs(
        elref_spectrum: SpectralData,
        moon_data: MoonData,
        srf_type: str,
        skip_uncs: bool = False,
    ) -> SpectralData:
        wlens = elref_spectrum.wlens
        elis = eli.calculate_eli_from_elref(
            wlens, moon_data, elref_spectrum.data, srf_type
        )
        if not skip_uncs:
            unc, corr = eli.calculate_eli_from_elref_unc(
                elref_spectrum, moon_data, srf_type
            )
        else:
            unc = np.zeros(elis.shape)
            err_corr_side = len(unc)
            corr = np.zeros((err_corr_side, err_corr_side))
            np.fill_diagonal(corr, 1)
        ds_eli = SpectralData.make_irradiance_ds(wlens, elis, unc, corr)
        return SpectralData(wlens, elis, unc, ds_eli)

    @staticmethod
    def get_elrefs(
        coefficients: ReflectanceCoefficients,
        moon_data: MoonData,
        skip_uncs: bool = False,
    ) -> SpectralData:
        wlens = coefficients.wlens
        elrefs = elref.calculate_elref(coefficients, moon_data)
        if not skip_uncs:
            unc, corr = elref.calculate_elref_unc(coefficients, moon_data)
        else:
            unc = np.zeros(elrefs.shape)
            err_corr_side = len(unc)
            corr = np.zeros((err_corr_side, err_corr_side))
            np.fill_diagonal(corr, 1)
        ds = SpectralData.make_reflectance_ds(wlens, elrefs, unc, corr)
        return SpectralData(wlens, elrefs, unc, ds)
