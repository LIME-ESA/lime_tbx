"""
This module contains different dataclasses and classes used for the exchange
of data between modules of the package lime-tbx.

It exports the following classes:
    * MoonData - Moon data used in the calculations of the Moon's irradiance.
    * IrradianceCoefficients - Coefficients used in the ROLO algorithm. (ROLO's + Apollo's).
"""

"""___Built-In Modules___"""
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from datetime import datetime
from enum import Enum

"""___Third-Party Modules___"""
import numpy as np
import xarray
import obsarray


"""___LIME Modules___"""
from . import constants
from lime_tbx.datatypes.templates_digital_effects_table import template_refl

@dataclass
class MoonData:
    """
    Moon data needed to calculate Moon's irradiance, probably obtained from NASA's SPICE Toolbox

    Attributes
    ----------
    distance_sun_moon : float
        Distance between the Sun and the Moon (in astronomical units)
    distance_observer_moon : float
        Distance between the Observer and the Moon (in kilometers)
    selen_sun_lon_rad : float
        Selenographic longitude of the Sun (in radians)
    selen_obs_lat : float
        Selenographic latitude of the observer (in degrees)
    selen_obs_lon : float
        Selenographic longitude of the observer (in degrees)
    abs_moon_phase_angle : float
        Absolute Moon phase angle (in degrees)
    mpa_degrees : float
        Moon phase angle (in degrees)
    """

    distance_sun_moon: float
    distance_observer_moon: float
    long_sun_radians: float
    lat_obs: float
    long_obs: float
    absolute_mpa_degrees: float
    mpa_degrees: float


class SpectralValidity(Enum):
    """
    Enum that represents if a channel is inside LIME's spectral range.
    VALID: Fully in the range.
    PARTLY_OUT: Some wavelengths out of range.
    OUT: All wavelengths out of range.
    """

    VALID = 0
    PARTLY_OUT = 1
    OUT = 2


@dataclass
class SRFChannel:
    """
    Dataclass containing the spectral responses and metadata for a SRF Channel

    center: float
        Center wavelength
    id: str
        Identifier of the channel
    spectral_response : dict of float, float
        Set of pairs wavelength, percentage. 100% = 1.0.
    spectral_validity: SpectralValidity
        Information about if the channel is inside LIME's spectral calculation range or not.
    """

    center: float
    id: str
    spectral_response: Dict[float, float]
    valid_spectre: SpectralValidity

    def __init__(
        self,
        center: float,
        id: str,
        spectral_response: Dict[float, float],
        min_wlen: float = constants.MIN_WLEN,
        max_wlen: float = constants.MAX_WLEN,
    ):
        self.center = center
        self.id = id
        self.spectral_response = spectral_response
        spec = list(self.spectral_response.keys())
        if spec[-1] < min_wlen or spec[0] > max_wlen:
            self.valid_spectre = SpectralValidity.OUT
        elif spec[0] < min_wlen or spec[-1] > max_wlen:
            self.valid_spectre = SpectralValidity.PARTLY_OUT
        else:
            self.valid_spectre = SpectralValidity.VALID


@dataclass
class SpectralResponseFunction:
    """
    Dataclass containing the spectral response function, a set of channels with their data.

    Attributes
    ----------
    name : str
        Name of the SRF, the identifier.
    channels: list of SRFChannel
        List of the SRF channels.
    """

    name: str
    channels: List[SRFChannel]

    def get_wavelengths(self) -> List[float]:
        wlens = []
        for ch in self.channels:
            wlens += list(ch.spectral_response.keys())
        return wlens

    def get_values(self) -> List[float]:
        vals = []
        for ch in self.channels:
            vals += list(ch.spectral_response.values())
        return vals

    def get_channels_names(self) -> List[str]:
        return [ch.id for ch in self.channels]

    def get_channel_from_name(self, name: str) -> SRFChannel:
        return [ch for ch in self.channels if ch.id == name][0]


@dataclass
class SurfacePoint:
    """
    Dataclass representing a point on Earth's surface.

    The needed parameters for the calculation from a surface point.

    Attributes
    ----------
    latitude: float
        Geographic latitude in decimal degrees.
    longitude: float
        Geographic longitude in decimal degrees.
    altitude: float
        Altitude over the sea level in meters.
    dt: datetime | list of datetime
        Time or time series at which the lunar data will be calculated.
    """

    latitude: float
    longitude: float
    altitude: float
    dt: Union[datetime, List[datetime]]


@dataclass
class CustomPoint:
    """
    Dataclass representing a point with custom Moon data.

    The needed parameters for the calculation from a custom point.

    Attributes
    ----------
    distance_sun_moon : float
        Distance between the Sun and the Moon (in astronomical units)
    distance_observer_moon : float
        Distance between the Observer and the Moon (in kilometers)
    selen_obs_lat : float
        Selenographic latitude of the observer (in degrees)
    selen_obs_lon : float
        Selenographic longitude of the observer (in degrees)
    selen_sun_lon : float
        Selenographic longitude of the Sun (in radians)
    abs_moon_phase_angle : float
        Absolute Moon phase angle (in degrees)
    moon_phase_angle : float
        Absolute Moon phase angle (in degrees)
    """

    distance_sun_moon: float
    distance_observer_moon: float
    selen_obs_lat: float
    selen_obs_lon: float
    selen_sun_lon: float
    abs_moon_phase_angle: float
    moon_phase_angle: float


@dataclass
class SatellitePoint:
    """
    Dataclass representing a Satellite in a concrete datetime

    Attributes
    ----------
    name : str
        Name of the satellite
    dt : datetime | list of datetime
        Datetime/s that will be computed
    """

    name: str
    dt: Union[datetime, List[datetime]]


class PolarizationCoefficients:
    """
    Coefficients used in the DoLP algorithm.
    """

    def __init__(
        self,
        wavelengths: List[float],
        pos_coeffs: List[Tuple[float, float, float, float]],
        neg_coeffs: List[Tuple[float, float, float, float]],
    ):
        """
        Parameters
        ----------
        wavelengths: list of float
            Wavelengths present in the model, in nanometers
        pos_coeffs: list of tuples of 4 floats
            Positive phase angles related to the given wavelengths
        neg_coeffs: list of tuples of 4 floats
            Negative phase angles related to the given wavelengths
        """
        self.wavelengths = wavelengths
        self.pos_coeffs = pos_coeffs
        self.neg_coeffs = neg_coeffs

    def get_wavelengths(self) -> List[float]:
        """Gets all wavelengths present in the model, in nanometers

        Returns
        -------
        list of float
            A list of floats that are the wavelengths in nanometers, in order
        """
        return self.wavelengths

    def get_coefficients_positive(
        self, wavelength_nm: float
    ) -> Tuple[float, float, float, float]:
        """Gets all positive phase angle coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'a' coefficients for the wavelength
        """
        index = self.get_wavelengths().index(wavelength_nm)
        return self.pos_coeffs[index]

    def get_coefficients_negative(
        self, wavelength_nm: float
    ) -> Tuple[float, float, float, float]:
        """Gets all negative phase angle coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'a' coefficients for the wavelength
        """
        index = self.get_wavelengths().index(wavelength_nm)
        return self.neg_coeffs[index]


class IrradianceCoefficients:
    """
    Coefficients used in the ROLO algorithm. (ROLO's + Apollo's).
    """

    @dataclass
    class CoefficientsWln:
        """
        Coefficients data for a wavelength. It includes only the a, b and d coefficients.

        Attributes
        ----------
        a_coeffs : tuple of 4 floats, corresponding to coefficients a0, a1, a2, and a3
        b_coeffs : tuple of 3 floats, corresponding to coefficients b1, b2, and b3
        d_coeffs : tuple of 3floats, corresponding to coefficients d1, d2, and d3
        """

        __slots__ = ["a_coeffs", "b_coeffs", "d_coeffs"]

        def __init__(self, coeffs: List[float]):
            """
            Parameters
            ----------
            coeffs : list of float
                List of floats consisting of all coefficients. In order: a0, a1, a2, a3, b1, b2, b3,
                d1, d2 and d3.

            Returns
            -------
            CoefficientsWln
                Instance of Coefficients with the correct data
            """
            self.a_coeffs = (coeffs[0], coeffs[1], coeffs[2], coeffs[3])
            self.b_coeffs = (coeffs[4], coeffs[5], coeffs[6])
            self.d_coeffs = (coeffs[7], coeffs[8], coeffs[9])

    def __init__(
        self,
        wavelengths: List[float],
        wlen_coeffs: List[CoefficientsWln],
        c_coeffs: List[float],
        p_coeffs: List[float],
        apollo_coeffs: List[float],
    ):
        self.wavelengths = wavelengths
        self.wlen_coeffs = wlen_coeffs
        self.c_coeffs = c_coeffs
        self.p_coeffs = p_coeffs
        self.apollo_coeffs = apollo_coeffs

    def get_wavelengths(self) -> List[float]:
        """Gets all wavelengths present in the model, in nanometers

        Returns
        -------
        list of float
            A list of floats that are the wavelengths in nanometers, in order
        """
        return self.wavelengths

    def get_coefficients_a(self, wavelength_nm: float) -> List[float]:
        """Gets all 'a' coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'a' coefficients for the wavelength
        """
        index = self.get_wavelengths().index(wavelength_nm)
        return self.wlen_coeffs[index].a_coeffs

    def get_coefficients_b(self, wavelength_nm: float) -> List[float]:
        """Gets all 'b' coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'b' coefficients for the wavelength
        """
        index = self.get_wavelengths().index(wavelength_nm)
        return self.wlen_coeffs[index].b_coeffs

    def get_coefficients_d(self, wavelength_nm: float) -> List[float]:
        """Gets all 'd' coefficients for a concrete wavelength

        Parameters
        ----------
        wavelength_nm : float
            Wavelength in nanometers from which one wants to obtain the coefficients.

        Returns
        -------
        list of float
            A list containing the 'd' coefficients for the wavelength
        """
        index = self.get_wavelengths().index(wavelength_nm)
        return self.wlen_coeffs[index].d_coeffs

    def get_coefficients_c(self) -> List[float]:
        """Gets all 'c' coefficients

        Returns
        -------
        list of float
            A list containing all 'c' coefficients
        """
        return self.c_coeffs

    def get_coefficients_p(self) -> List[float]:
        """Gets all 'p' coefficients

        Returns
        -------
        list of float
            A list containing all 'p' coefficients
        """
        return self.p_coeffs

    def get_apollo_coefficients(self) -> List[float]:
        """Coefficients used for the adjustment of the ROLO model using Apollo spectra.

        Returns
        -------
        list of float
            A list containing all Apollo coefficients
        """
        return self.apollo_coeffs


@dataclass
class OrbitFile:
    """
    Dataclass that represents a Satellite orbit file.

    Attributes
    ----------
    name: str
        Name of the orbit file in the file system.
    dt0: datetime
        First datetime for which the orbit file works.
    dtf: datetime
        Last datetime for which the orbit file works
    """

    name: str
    dt0: datetime
    dtf: datetime


@dataclass
class Satellite:
    """
    Dataclass that represents an ESA Satellite.

    Attributes
    ----------
    name: str
        Satellite name
    id: int
        Satellite id
    orbit_files: list of OrbitFile
        Orbit files of the Satellite
    """

    name: str
    id: int
    orbit_files: List[OrbitFile]

    def get_datetime_range(self) -> Tuple[datetime, datetime]:
        """
        Calculate all the datetime range between its orbit files.

        Returns
        -------
        dt0: datetime
            First datetime for which the Satellite can be simulated.
        dtf: datetime
            Last datetime for which the Satellite can be simulated.
        """
        dt0 = dtf = None
        for orf in self.orbit_files:
            if dt0 == None:
                dt0 = orf.dt0
                dtf = orf.dtf
            else:
                if dt0 > orf.dt0:
                    dt0 = orf.dt0
                if dtf < orf.dtf:
                    dtf = orf.dtf
        return dt0, dtf

    def get_best_orbit_file(self, dt: datetime) -> OrbitFile:
        """
        Obtain the best orbit file for the given datetime.

        Parameters
        ----------
        dt: datetime
            Datetime that is queried.

        Returns
        -------
        orbit_file: OrbitFile
            Selected orbit file for the fiven datetime.
        """
        sel_td = None
        sel_orf = None
        for orf in self.orbit_files:
            if dt >= orf.dt0 and dt <= orf.dtf:
                td = min(dt - orf.dt0, orf.dtf - dt)
                if sel_td == None or td < sel_td:
                    sel_td = td
                    sel_orf = orf
        return sel_orf


@dataclass
class SatellitePosition:
    x: float
    y: float
    z: float


@dataclass
class LunarObservation:
    ch_names: List[str]
    sat_pos_ref: str
    ch_irrs: Dict[str, float]
    dt: datetime
    sat_pos: SatellitePosition

    def get_ch_irradiance(self, name: str) -> float:
        if name not in self.ch_irrs:
            raise "Channel name not in data structure"
        return self.ch_irrs[name]

    def has_ch_value(self, name: str) -> bool:
        return name in self.ch_irrs

    def check_valid_srf(self, srf: SpectralResponseFunction) -> bool:
        for ch in self.ch_names:
            found = False
            for ch_srf in srf.channels:
                if ch_srf.id == ch:
                    found = True
                    break
            if not found:
                return False
        return True

@dataclass
class CimelCoef:
    __slots__ = ["_ds_cimel", "wavelengths", "coeffs", "unc_coeffs"]
    # _ds_cimel : xarray DataSet with the CIMEL coefficients and uncertainties


    @dataclass
    class _CimelCoeffs:
        __slots__ = ["_coeffs", "a_coeffs", "b_coeffs", "c_coeffs", "d_coeffs", "p_coeffs"]

        def __init__(self, coeffs: np.ndarray):
            self._coeffs = coeffs
            self.a_coeffs = coeffs[0:4,:]
            self.b_coeffs = coeffs[4:7,:]
            self.c_coeffs = coeffs[7:11,:]
            self.d_coeffs = coeffs[11:14,:]
            self.p_coeffs = coeffs[14::,:]

    def __init__(self, ds_cimel: xarray.Dataset):
        self._ds_cimel = ds_cimel
        self.wavelengths: np.ndarray = ds_cimel.wavelength.values
        coeffs: np.ndarray = ds_cimel.coeff.values
        self.coeffs = CimelCoef._CimelCoeffs(coeffs)
        u_coeff_cimel: np.ndarray = ds_cimel.u_coeff.values
        self.unc_coeffs = CimelCoef._CimelCoeffs(u_coeff_cimel)

@dataclass
class SpectralData:
    wlen: np.ndarray
    data: np.ndarray
    uncertainties: np.ndarray
    ds: xarray.Dataset

    @staticmethod
    def make_reflectance_ds(wavs,refl,unc=None):
        dim_sizes = {"wavelength":len(wavs)}
        # create dataset
        ds_refl = obsarray.create_ds(template_refl,dim_sizes)

        ds_refl = ds_refl.assign_coords(wavelength=wavs)

        ds_refl.reflectance.values = refl

        if unc:
            ds_refl.u_ran_reflectance.values = unc[0]
            ds_refl.u_sys_reflectance.values = unc[1]
        else:
            ds_refl.u_ran_reflectance.values = refl*0.01
            ds_refl.u_sys_reflectance.values = refl*0.05

        return ds_refl

    @staticmethod
    def make_irradiance_ds(wavs,refl,unc=None):
        dim_sizes = {"wavelength":len(wavs)}
        # create dataset
        ds_irr = obsarray.create_ds(template_refl,dim_sizes)

        ds_irr = ds_irr.assign_coords(wavelength=wavs)

        ds_irr.reflectance.values = refl

        if unc:
            ds_irr.u_ran_reflectance.values = unc[0]
            ds_irr.u_sys_reflectance.values = unc[1]
        else:
            ds_irr.u_ran_reflectance.values = refl*0.01
            ds_irr.u_sys_reflectance.values = refl*0.05

        return ds_irr

    @staticmethod
    def make_polarization_ds(wavs,refl,unc=None):
        dim_sizes = {"wavelength":len(wavs)}
        # create dataset
        ds_pol = obsarray.create_ds(template_refl,dim_sizes)

        ds_pol = ds_pol.assign_coords(wavelength=wavs)

        ds_pol.reflectance.values = refl

        if unc:
            ds_pol.u_ran_reflectance.values = unc[0]
            ds_pol.u_sys_reflectance.values = unc[1]
        else:
            ds_pol.u_ran_reflectance.values = refl*0.01
            ds_pol.u_sys_reflectance.values = refl*0.05

        return ds_pol