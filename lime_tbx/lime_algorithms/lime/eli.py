"""
This module calculates the extra-terrestrial lunar irradiance.

It exports the following functions:
    * calculate_eli_from_elref - Calculates the expected extra-terrestrial lunar irradiances
        for some given wavelengths in nanometers. Based on Eq 3 in Roman et al., 2020.
        The reflectance is given as a parameter.
    * calculate_eli_from_elref_unc - Calculates the uncertainties of the calculation of the
        expected extra-terrestrial lunar irradiances for some given wavelengths in nanometers.
        Based on Eq 3 in Roman et al., 2020. The reflectance is given as a parameter.
"""

"""___Built-In Modules___"""
import math
from typing import Union, Tuple

"""___Third-Party Modules___"""
import numpy as np

"""___NPL Modules"""
import punpy

"""___LIME_TBX Modules___"""
from lime_tbx.lime_algorithms.lime import esi
from lime_tbx.datatypes.datatypes import (
    MoonData,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "2022/03/02"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


SOLID_ANGLE_MOON: float = 6.4177e-05
DIST_EARTH_MOON_KM: int = 384400


def _measurement_func_eli(
    a_l: Union[float, np.ndarray],
    omega: float,
    esk: Union[float, np.ndarray],
    dsm: float,
    distance_earth_moon_km: float,
    dom: float,
    geom_factor: Union[float, None],
) -> Union[float, np.ndarray]:
    """
    Final computation of the Eq 3 in Roman et al., 2020, but also letting the user specify the geom_factor
    """
    unnormalized_irr = (a_l * omega * esk) / math.pi
    if geom_factor:
        lunar_irr = unnormalized_irr / geom_factor
    else:
        lunar_irr = (
            unnormalized_irr * ((1 / dsm) ** 2) * (distance_earth_moon_km / dom) ** 2
        )
    return lunar_irr


def J_eli(
    a_l: Union[float, np.ndarray],
    omega: float,
    esk: Union[float, np.ndarray],
    dsm: float,
    distance_earth_moon_km: float,
    dom: float,
):
    Jac_x1 = np.diag(
        ((omega * esk) / math.pi)
        * ((1 / dsm) ** 2)
        * (distance_earth_moon_km / dom) ** 2
    )
    Jac_x3 = np.diag(
        ((omega * a_l) / math.pi)
        * ((1 / dsm) ** 2)
        * (distance_earth_moon_km / dom) ** 2
    )
    Jac = np.concatenate(
        (
            Jac_x1,
            Jac_x3,
        )
    ).T
    return Jac


def calculate_eli_from_elref(
    wavelengths_nm: np.ndarray,
    moon_data: MoonData,
    elrefs: np.ndarray,
    srf_type: str,
) -> np.ndarray:
    """Calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

    Parameters
    ----------
    wavelength_nm : np.array of float
        Wavelengths (in nanometers) of which the extraterrestrial lunar irradiance will be
        calculated.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance
    elrefs : np.ndarray of float
        Reflectances previously calculated
    srf_type: str
        SRF type that wants to be used. Can be 'cimel', 'asd', 'interpolated_gaussian' or 'interpolated_triangle'

    Returns
    -------
    np.ndarray of float
        The extraterrestrial lunar irradiance calculated
    """
    esk = esi.get_esi(srf_type)
    esk = esk[[es in wavelengths_nm for es in esk[:, 0]]][:, 1]
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    lunar_irr = _measurement_func_eli(
        elrefs,
        SOLID_ANGLE_MOON,
        esk,
        dsm,
        DIST_EARTH_MOON_KM,
        dom,
        moon_data.geom_factor,
    )
    return lunar_irr


def calculate_eli_from_elref_unc(
    elref_spectrum: SpectralData,
    moon_data: MoonData,
    srf_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uncertainties of the calculation of Extraterrestrial Lunar Irradiance following Eq 3 in Roman et al., 2020

    Simulates a lunar observation for a wavelength for any observer/solar selenographic
    latitude and longitude. The irradiance is calculated in Wm⁻²/nm.

    Parameters
    ----------
    elref_spectrum: SpectralData
        Previously calculated reflectance data.
    moon_data : MoonData
        Moon data needed to calculate Moon's irradiance.
    srf_type: str
        SRF type that wants to be used. Can be 'cimel', 'asd' or 'interpolated'.

    Returns
    -------
    np.ndarray of float
        The uncertainties calculated
    corr: np.ndarray of float
        The error correlation matrix calculated
    """
    esk = esi.get_esi(srf_type)
    esk = esk[[es in elref_spectrum.wlens for es in esk[:, 0]]][:, 1]
    u_esk = esi.get_u_esi(srf_type)
    u_esk = u_esk[[es in elref_spectrum.wlens for es in u_esk[:, 0]]][:, 1]
    dsm = moon_data.distance_sun_moon
    dom = moon_data.distance_observer_moon

    Jx = J_eli(elref_spectrum.data, SOLID_ANGLE_MOON, esk, dsm, DIST_EARTH_MOON_KM, dom)
    prop = punpy.LPUPropagation()

    def _measure_eli(elref, eske):
        return _measurement_func_eli(
            elref,
            SOLID_ANGLE_MOON,
            eske,
            dsm,
            DIST_EARTH_MOON_KM,
            dom,
            moon_data.geom_factor,
        )

    unc, corr = prop.propagate_standard(
        _measure_eli,
        [
            elref_spectrum.data,
            esk,
        ],
        [elref_spectrum.uncertainties, u_esk],
        corr_x=[
            elref_spectrum.err_corr,
            "rand",  # TODO: It was formerly "syst", check if this change is valid
        ],
        return_corr=True,
        Jx=Jx,
    )
    del prop
    unc = np.where(np.isnan(unc), 0, unc)

    return unc, corr
