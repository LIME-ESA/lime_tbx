"""___Built-In Modules___"""
from datetime import datetime, timezone
from dataclasses import dataclass
from abc import ABC
from typing import List, Union

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import (
    CustomPoint,
    KernelsPath,
    LGLODData,
    Point,
    SatellitePoint,
    SurfacePoint,
)
from lime_tbx.gui import settings
from lime_tbx.simulation.lime_simulation import LimeSimulation
from lime_tbx.filedata import moon, srf, csv
from lime_tbx.filedata.lglod_factory import create_lglod_data


class ExportData(ABC):
    pass


@dataclass
class ExportCSV(ExportData):
    o_file_refl: str
    o_file_irr: str
    o_file_polar: str


@dataclass
class ExportNetCDF(ExportData):
    output_file: str


class CLI:
    def __init__(self, kernels_path: KernelsPath, eocfi_path: str):
        self.kernels_path = kernels_path
        self.eocfi_path = eocfi_path
        self.lime_simulation = LimeSimulation(eocfi_path, kernels_path)
        self.settings_manager = settings.MockSettingsManager()
        self.srf = self.settings_manager.get_default_srf()

    def load_srf(self, srf_file: str):
        if srf_file == "":
            self.srf = self.settings_manager.get_default_srf()
        else:
            self.srf = srf.read_srf(srf_file)

    def _calculate_irradiance(self, point: Point):
        self.lime_simulation.update_irradiance(
            self.srf, self.srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_reflectance(self, point: Point):
        self.lime_simulation.update_reflectance(
            self.srf, point, self.settings_manager.get_cimel_coef()
        )

    def _calculate_polarization(self, point: Point):
        self.lime_simulation.update_polarization(
            self.srf, point, self.settings_manager.get_polar_coeffs()
        )

    def _calculate_all(self, point: Point):
        self._calculate_reflectance(point)
        self._calculate_irradiance(point)
        self._calculate_polarization(point)

    def _export_csvs(
        self, point: Point, o_file_refl: str, o_file_irr: str, o_file_polar: str
    ):
        version = self.settings_manager.get_cimel_coef().version
        csv.export_csv(
            self.lime_simulation.get_elrefs(),
            "Wavelengths (nm)",
            "Reflectances (Fraction of unity)",
            point,
            o_file_refl,
            version,
        )
        csv.export_csv(
            self.lime_simulation.get_elis(),
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²/nm)",
            point,
            o_file_irr,
            version,
        )
        csv.export_csv(
            self.lime_simulation.get_polars(),
            "Wavelengths (nm)",
            "Polarizations (Fraction of unity)",
            point,
            o_file_polar,
            version,
        )

    def _export_lglod(self, point: Point, output_file: str):
        lglod = create_lglod_data(
            point, self.srf, self.lime_simulation, self.kernels_path
        )
        now = datetime.now(timezone.utc)
        version = self.settings_manager.get_cimel_coef().version
        moon.write_obs(lglod, output_file, now, version)

    def _export(self, point: Point, ed: ExportData):
        if isinstance(ed, ExportCSV):
            self._export_csvs(point, ed.o_file_refl, ed.o_file_irr, ed.o_file_polar)
        elif isinstance(ed, ExportNetCDF):
            self._export_lglod(point, ed.output_file)

    def calculate_geographic(
        self,
        lat: float,
        lon: float,
        height: float,
        dt: Union[datetime, List[datetime]],
        export_data: ExportData,
    ):
        point = SurfacePoint(lat, lon, height, dt)
        self._calculate_all(point)
        self._export(point, export_data)

    def calculate_satellital(
        self,
        sat_name: str,
        dt: Union[datetime, List[datetime]],
        export_data: ExportData,
    ):
        point = SatellitePoint(sat_name, dt)
        self._calculate_all(point)
        self._export(point, export_data)

    def calculate_selenographic(
        self,
        distance_sun_moon: float,
        distance_observer_moon: float,
        selen_obs_lat: float,
        selen_obs_lon: float,
        selen_sun_lon: float,
        moon_phase_angle: float,
        export_data: ExportData,
    ):
        point = CustomPoint(
            distance_sun_moon,
            distance_observer_moon,
            selen_obs_lat,
            selen_obs_lon,
            selen_sun_lon,
            abs(moon_phase_angle),
            moon_phase_angle,
        )
        self._calculate_all(point)
        self._export(point, export_data)
