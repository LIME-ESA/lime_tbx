"""
This module contains different dataclasses and classes used for the exchange
of data between modules of the package lime-tbx.

It exports the following classes:
    * MoonData - Moon data used in the calculations of the Moon's irradiance.
    * SRFChannel - Spectral responses and metadata for a SRF Channel
    * SpectralResponseFunction - The spectral response function, a set of channels with their data.
    * SRF_fwhm - Dataclass containing the spectral response function, a set of channels with their data.
    * Point - Superclass for all point classes.
    * SurfacePoint - Point on Earth's surface
    * CustomPoint - Point with custom Moon data.
    * SatellitePoint - Point of a Satellite in a concrete datetime
    * OrbitFile - Satellite orbit file.
    * Satellite - ESA Satellite
    * SatellitePosition - A satellite's position
    * LunarObservation - GLOD lunar observation
    * PolarizationCoefficients - Coefficients used in the DoLP algorithm.
    * ReflectanceCoefficients - Dataclass containing the cimel coefficients that will be used in the
        reflectance simulation algorithm.
    * LimeCoefficients - Dataclass containing a PolarizationCoefficients and a ReflectanceCoefficients.
    * SpectralData - Data for a spectrum of wavelengths, with an associated uncertainty each.
    * ComparisonData - Dataclass containing the data outputed from a comparison.
    * KernelsPath - Dataclass containing the needed information in order to find all SPICE kernels.
    * SelenographicDataWrite - Extra data that allowes to define CustomPoints in the GLOD data file.
    * LunarObservationWrite - Dataclass containing the needed information to create a Lunar observation
        in a LGLOD file.
    * LGLODData - Dataclass with the data of a LGLOD simulation file. LGLOD is the GLOD-based format
        used by the toolbox.
    * LGLODComparisonData - Dataclass with the data of a LGLOD comparison file. LGLOD is the
        GLOD-based format used by the toolbox.
    * LimeException - Exception that is raised by the toolbox that is intended to be shown to the user.
    * InterpolationSettings - Representation of the YAML file that contains the interpolation settings data.

It exports the following Enums:
    * SpectralValidity - Enum that represents if a channel is inside LIME's spectral range.
"""

"""___Built-In Modules___"""
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from datetime import datetime
from enum import Enum
from abc import ABC

"""___Third-Party Modules___"""
import numpy as np
import xarray
import ruamel.yaml as ruaml
from ruamel.yaml import yaml_object

"""___NPL Modules___"""
import obsarray

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes import constants
from lime_tbx.datatypes.templates import (
    TEMPLATE_IRR,
    TEMPLATE_POL,
    TEMPLATE_REFL,
    TEMPLATE_SIGNALS,
)


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
        self.spectral_response_inrange = {
            k: v for k, v in spectral_response.items() if min_wlen <= k <= max_wlen
        }
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
        chs = [ch for ch in self.channels if ch.id == name]
        if len(chs) == 0:
            return None
        return chs[0]


@dataclass
class SRF_fwhm:
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
    wav_centre: np.ndarray
    fwhm: np.ndarray
    shape: str

    def get_wavelengths(self) -> List[float]:
        return self.wav_centre

    def get_values(self) -> List[float]:
        return self.fwhm

    def get_shape(self) -> str:
        return self.shape


class Point(ABC):
    """Abstract class representing a point which can be used to generate a MoonData"""

    pass


@dataclass
class SurfacePoint(Point):
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
class CustomPoint(Point):
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
        Moon phase angle (in degrees)
    """

    distance_sun_moon: float
    distance_observer_moon: float
    selen_obs_lat: float
    selen_obs_lon: float
    selen_sun_lon: float
    abs_moon_phase_angle: float
    moon_phase_angle: float


@dataclass
class SatellitePoint(Point):
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


@dataclass(eq=True, frozen=True)
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
    norad_sat_number: int | None
        Number of the satellite in the NORAD Catalog number (Celestrak)
        Only present if the satellite has TLE files.
    intdes: str | None
        International Designator of the object.
        Only present if the satellite has TLE files.
    time_file: str | None
        File used for time initialization.
        Only present if the satellite has TLE files.
    """

    name: str
    id: int
    orbit_files: List[OrbitFile]
    norad_sat_number: Union[int, None]
    intdes: Union[str, None]
    time_file: Union[str, None]

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
    """
    Dataclass containing the information of the position of a SatellitePoint,
    for a specific reference system (usually ITRF93)

    Attributes
    ----------
    x: float
        The x coordinate
    y: float
        The y coordinate
    z: float
        The z coordinate
    """

    x: float
    y: float
    z: float


@dataclass
class LunarObservation:
    """
    Dataclass containing the information for one GLOD-like lunar observation instant.

    Attributes
    ----------
    ch_names: list of str
        Names of the channels present
    sat_pos_ref: str
        Name of the reference system (usually ITRF93)
    ch_irrs: dict of str and float
        Irradiances relative to each channel. The key is the channel name, and the irradiance
        is given in Wm⁻²nm⁻¹.
    dt: datetime
        Datetime of the observation.
    sat_pos: SatellitePosition
        Satellite position at that moment.
    data_source: str
        Data source of the lunar observation.
    """

    ch_names: List[str]
    sat_pos_ref: str
    ch_irrs: Dict[str, float]
    dt: datetime
    sat_pos: SatellitePosition
    data_source: str

    def get_ch_irradiance(self, name: str) -> float:
        if name not in self.ch_irrs:
            raise ValueError("Channel name not in data structure")
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


class ReflectanceCoefficients:
    """
    Set of coefficients from the same version. Used in order to calculate the reflectance
    from mainly CIMEL data.

    Attributes
    ----------
    _ds: xarray.DataSet
        Original source dataset
    wlens: np.ndarray
        Wavelengths present in this coefficient version. Each one of them
    coeffs: _WlenReflCoeffs
        Reflectance Coefficients, with an attribute for every coefficient group, a matrix each.
    unc_coeffs: _WlenReflCoeffs
        Reflectance Coefficients uncertainties, with an attribute for every coefficient group, a matrix each.
    """

    __slots__ = ["_ds", "wlens", "coeffs", "unc_coeffs", "err_corr_coeff"]

    @dataclass
    class _WlenReflCoeffs:
        """
        Subdataclass where coefficients are stored

        x_coeffs structure: [[x0_w0, x0_w1, x0_w2, ...], [x1_w0, x1_w2, x1_w2, ...], ...]
        It's that way so "vector-wise" operations can be performed over every xn parameter without
        having to perform them over every wavelength too.
        """

        __slots__ = [
            "_coeffs",
            "a_coeffs",
            "b_coeffs",
            "c_coeffs",
            "d_coeffs",
            "p_coeffs",
        ]

        def __init__(self, coeffs: np.ndarray):
            self._coeffs = coeffs
            self.a_coeffs = coeffs[0:4, :]
            self.b_coeffs = coeffs[4:7, :]
            self.c_coeffs = coeffs[7:11, :]
            self.d_coeffs = coeffs[11:14, :]
            self.p_coeffs = coeffs[14::, :]

    def __init__(self, _ds: xarray.Dataset):
        self._ds = _ds
        self.wlens: np.ndarray = _ds.wavelength.values
        coeffs: np.ndarray = _ds.coeff.values
        self.coeffs = ReflectanceCoefficients._WlenReflCoeffs(coeffs)
        u_coeff_cimel: np.ndarray = _ds.u_coeff.values * coeffs / 100
        self.unc_coeffs = ReflectanceCoefficients._WlenReflCoeffs(u_coeff_cimel)
        self.err_corr_coeff = _ds.err_corr_coeff.values


class PolarizationCoefficients:
    """
    Coefficients used in the DoLP algorithm.

    If created with numpy arrays, it will return those numpy arrays.
    """

    def __init__(
        self,
        wavelengths: List[float],
        pos_coeffs: List[Tuple[float, float, float, float]],
        pos_unc: List[Tuple[float, float, float, float]],
        p_pos_err_corr_data: List[List[float]],
        neg_coeffs: List[Tuple[float, float, float, float]],
        neg_unc: List[Tuple[float, float, float, float]],
        p_neg_err_corr_data: List[List[float]],
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
        self.wlens = wavelengths
        self.pos_coeffs = pos_coeffs
        self.pos_unc = pos_unc
        self.p_pos_err_corr_data = p_pos_err_corr_data
        self.neg_coeffs = neg_coeffs
        self.neg_unc = neg_unc
        self.p_neg_err_corr_data = p_neg_err_corr_data

    def get_wavelengths(self) -> List[float]:
        """Gets all wavelengths present in the model, in nanometers

        Returns
        -------
        list of float
            A list of floats that are the wavelengths in nanometers, in order
        """
        return self.wlens

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

    def get_uncertainties_positive(
        self, wavelength_nm: float
    ) -> Tuple[float, float, float, float]:
        index = self.get_wavelengths().index(wavelength_nm)
        return self.pos_unc[index]

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

    def get_uncertainties_negative(
        self, wavelength_nm: float
    ) -> Tuple[float, float, float, float]:
        index = self.get_wavelengths().index(wavelength_nm)
        return self.neg_unc[index]

    def is_calculable(self) -> bool:
        return not (np.isnan(self.pos_coeffs).any() or np.isnan(self.neg_coeffs).any())


@dataclass
class LimeCoefficients:
    """
    Coefficients used in the LIME algorithms.

    Attributes
    ----------
    reflectance: ReflectanceCoefficients
        Reflectance coefficients for the LIME model.
    polarization: PolatizationCoefficients
        Polarization coefficients for the DoLP/LIME model.
    version: str
        Name of the version that will be shown to the user.
    """

    reflectance: ReflectanceCoefficients
    polarization: PolarizationCoefficients
    version: str


@dataclass
class SpectralData:
    """
    Data for a spectrum of wavelengths, with an associated uncertainty each.

    Attributes
    ----------
    wlens: np.ndarray
        Spectrum of wavelengths
    data: np.ndarray
        Data associated to the wavelengths (irradiance, reflectance, etc)
    uncertainties: np.ndarray
        Uncertainties associated to the data
    ds: xarray.Dataset
        Dataset used in data generation
    """

    wlens: np.ndarray
    data: np.ndarray
    uncertainties: np.ndarray
    ds: xarray.Dataset

    @staticmethod
    def make_reflectance_ds(
        wavs: np.ndarray,
        refl: np.ndarray,
        unc: np.ndarray = None,
        corr: np.ndarray = None,
    ) -> xarray.Dataset:
        dim_sizes = {"wavelength": len(wavs)}
        if (
            corr is not None
            and not isinstance(corr, np.ndarray)
            and not isinstance(corr, list)
        ):
            corr = np.array([[corr]])
        # create dataset
        ds_refl = obsarray.create_ds(TEMPLATE_REFL, dim_sizes)
        ds_refl = ds_refl.assign_coords(wavelength=wavs)

        ds_refl.reflectance.values = refl
        if unc is not None:
            ds_refl.u_reflectance.values = unc
        if corr is not None:
            ds_refl.err_corr_reflectance.values = corr

        return ds_refl

    @staticmethod
    def make_irradiance_ds(
        wavs: np.ndarray,
        refl: np.ndarray,
        unc: np.ndarray = None,
        corr: np.ndarray = None,
    ) -> xarray.Dataset:
        dim_sizes = {"wavelength": len(wavs)}
        # create dataset
        ds_irr = obsarray.create_ds(TEMPLATE_IRR, dim_sizes)

        ds_irr = ds_irr.assign_coords(wavelength=wavs)

        ds_irr.irradiance.values = refl

        if unc is not None:
            ds_irr.u_irradiance.values = unc
        if corr is not None:
            ds_irr.err_corr_irradiance.values = corr

        return ds_irr

    @staticmethod
    def make_polarization_ds(
        wavs: np.ndarray,
        polarization: np.ndarray,
        unc: np.ndarray = None,
        corr: np.ndarray = None,
    ) -> xarray.Dataset:
        dim_sizes = {"wavelength": len(wavs)}
        # create dataset
        ds_pol = obsarray.create_ds(TEMPLATE_POL, dim_sizes)

        ds_pol = ds_pol.assign_coords(wavelength=wavs)

        ds_pol.polarization.values = polarization

        if unc is not None:
            ds_pol.u_polarization.values = unc
        else:
            ds_pol.u_polarization.values = np.abs(polarization) * 0.05

        if corr is not None:
            ds_pol.err_corr_polarization.values = corr
        else:
            ds_pol.err_corr_polarization.values = np.ones(
                (len(polarization), len(polarization))
            )

        return ds_pol

    @staticmethod
    def make_signals_ds(
        channel_ids: np.ndarray,
        signals: np.ndarray,
        unc: np.ndarray = None,
        corr: np.ndarray = None,
    ) -> xarray.Dataset:
        dim_sizes = {"channels": len(channel_ids), "dts": len(signals[0])}
        # create dataset
        ds_refl = obsarray.create_ds(TEMPLATE_SIGNALS, dim_sizes)

        ds_refl = ds_refl.assign_coords(wavelength=channel_ids)

        ds_refl.signals.values = signals

        if unc is not None:
            ds_refl.u_signals.values = unc
        else:
            ds_refl.u_signals.values = signals * 0.05

        return ds_refl


@dataclass
class ComparisonData:
    """Dataclass containing the data outputed from a comparison.

    The SpectralDatas "wlens" attribute is not a list of float, instead a list
    of datetimes, corresponding to the measurements datetimes.

    The comparison data corresponds to the compared data for multiple datetimes
    for a single channel.

    Attributes
    ----------
    observed_signal: SpectralData
        Real data obtained from the GLOD files.
    simulated_signal: SpectralData
        Simulated data obtained from the model for the same conditions.
    diffs_signal: SpectralData
        Relative differences between the simulated and real data. (sim - real) / real.
    mean_relative_difference: float
        The mean of the relative differences (diffs_signals mean).
    standard_deviation_mrd: float
        Standard deviation of relative differences.
    number_samples: int
        Number of compared instances presnet in the object
    dts: list of datetime
        Datetimes of the different samples. They are also used as the "wlens" attribute
        for the SpectraDatas.
    points: list of SurfacePoint
        Point for every datetime.
    mpas: list of float
        Moon phase angle in degrees for every datetime.
    ampa_valid_range: list of bool
        Flag that indicates if the moon phase angle is in the valid LIME range.
    """

    observed_signal: SpectralData
    simulated_signal: SpectralData
    diffs_signal: SpectralData
    mean_relative_difference: float
    standard_deviation_mrd: float
    number_samples: int
    dts: List[datetime]
    points: List[SurfacePoint]
    mpas: List[float]
    ampa_valid_range: List[bool]


@dataclass
class KernelsPath:
    """Dataclass containing the needed information in order to find all SPICE kernels.

    Attributes
    ----------
    main_kernels_path: str
        Path where the main SPICE kernels are located (can be read-only).
    custom_kernel_path: str
        Path where the custom SPICE kernel will be stored (must be writeable).
    """

    main_kernels_path: str
    custom_kernel_path: str


@dataclass
class SelenographicDataWrite:
    """
    Extra data that allowes to define CustomPoints in the GLOD data file.

    Attributes
    ----------
    distance_sun_moon : float
        Distance between the Sun and the Moon (in astronomical units)
    selen_sun_lon_rad : float
        Selenographic longitude of the Sun (in radians)
    mpa_degrees : float
        Moon phase angle (in degrees)
    """

    distance_sun_moon: float
    selen_sun_lon_rad: float
    mpa_degrees: float


@dataclass
class LunarObservationWrite:
    """Dataclass containing the needed information to create a Lunar observation in a LGLOD file.

    Attributes
    ----------
    ch_names: list of str
        Names of the channels present
    sat_pos_ref: str
        Name of the reference system (usually ITRF93)
    dt: datetime
        Datetime of the observation.
    sat_pos: SatellitePosition
        Satellite position at that moment.
    irrs: SpectralData
        Irradiance data
    refls: SpectralData
        Reflectance data
    polars: SpectralData
        Polarization data
    sat_name: str | None
        Name of the satellite. If None or empty, then it's a SurfacePoint
    selenographic_data: SelenographicDataWrite | None
        If a CustomPoint, data that allowes to define the point. If None then it's not selenographic
        (not a CustomPoint).
    data_source: str
        Data source of the lunar observation.
    """

    ch_names: List[str]
    sat_pos_ref: str
    dt: datetime
    sat_pos: SatellitePosition
    irrs: "SpectralData"
    refls: "SpectralData"
    polars: "SpectralData"
    sat_name: str
    selenographic_data: SelenographicDataWrite
    data_source: str

    def has_ch_value(self, name: str) -> bool:
        return name in self.ch_names

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
class LGLODData:
    """Dataclass with the data of a LGLOD simulation file. LGLOD is the GLOD-based format used by the toolbox.

    Attributes
    ----------
    observations: list of LunarObservationWrite
        Spectral irradiance, reflectance and polarization for all the datetimes.
    signals: SpectralData
        SRF-Integrated irradiance data.
    not_default_srf: bool
        Flag that indicates if the spectral response function used is the default one or not (a custom user-selected one).
    elis_cimel: list of SpectralData
        Irradiance for the cimel.
    elrefs_cimel: list of SpectralData
        Reflectance for the cimel.
    polars_cimel: list of SpectralData
        Polarization for the cimel.
    spectrum_name: str
        Name of the spectrum used for interpolation.
    skipped_uncs: bool
        Flag that indicates if the uncertainties calculation was skipped or not.
    dolp_spectrum_name: str
        Name of the spectrum used for polarisation interpolation.
    """

    observations: List[LunarObservationWrite]
    signals: "SpectralData"
    not_default_srf: bool
    elis_cimel: List["SpectralData"]
    elrefs_cimel: List["SpectralData"]
    polars_cimel: List["SpectralData"]
    spectrum_name: str
    skipped_uncs: bool
    version: str
    dolp_spectrum_name: str


@dataclass
class LGLODComparisonData:
    """Dataclass with the data of a LGLOD comparison file. LGLOD is the GLOD-based format used by the toolbox.

    Attributes
    ----------
    comparisons: list of ComparisonData
        List of the comparison values.
    ch_names: list of str
        List with the names of the channels.
    sat_name: str
        Name of the satellite used for comparison.
    spectrum_name: str
        Name of the spectrum used for interpolation.
    skipped_uncs: bool
        Flag that indicates if the uncertainties calculation was skipped or not.
    """

    comparisons: List[ComparisonData]
    ch_names: List[str]
    sat_name: str
    spectrum_name: str
    skipped_uncs: bool
    version: str


class LimeException(Exception):
    """Exception that is raised by the toolbox that is intended to be shown to the user."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


yaml = ruaml.YAML()


@yaml_object(yaml)
@dataclass
class InterpolationSettings:
    """Representation of the YAML file that contains the interpolation settings data.

    Attributes
    ----------
    interpolation_spectrum: str
        Name (and id) of the spectrum used for interpolation.
    interpolation_spectrum_polarization: str
        Name (and id) of the spectrum used for interpolation for polarization.
    interpolation_SRF: str
        Name (and id) of the spectrum used for SRF interpolation.
    show_interp_spectrum: bool
        Flag that indicates if the interpolation spectrum used is going to be shown in graphs.
    skip_uncertainties: bool
        Flag that indicates if the uncertainties calculation should be skipped.
    """

    interpolation_spectrum: str
    interpolation_spectrum_polarization: str
    interpolation_SRF: str
    show_interp_spectrum: bool
    skip_uncertainties: bool

    def _save_disk(self, path: str):
        with open(path, "w") as file:
            yaml.dump([self], file)

    @staticmethod
    def _load_yaml(path: str) -> "InterpolationSettings":
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        yaml_str = "\n".join([line for line in lines])
        setts: "InterpolationSettings" = yaml.load(yaml_str)[0]
        return setts
