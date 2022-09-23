"""___Built-In Modules___"""
from datetime import datetime

"""___Third-Party Modules___"""
# import here

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes.datatypes import KernelsPath, SatellitePoint, SurfacePoint
from lime_tbx.gui import settings
from lime_tbx.simulation.lime_simulation import LimeSimulation
from lime_tbx.filedata import srf, csv


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

    def calculate_geographic(
        self, lat: float, lon: float, height: float, dt: datetime, output_file: str
    ):
        point = SurfacePoint(lat, lon, height, dt)
        self.lime_simulation.update_irradiance(
            self.srf, self.srf, point, self.settings_manager.get_cimel_coef()
        )
        csv.export_csv(
            self.lime_simulation.get_elis(),
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²/nm)",
            point,
            output_file,
            self.settings_manager.get_cimel_coef().version,
        )

    def calculate_satellital(self, sat_name: str, dt: datetime, output_file: str):
        point = SatellitePoint(sat_name, dt)
        self.lime_simulation.update_irradiance(
            self.srf, self.srf, point, self.settings_manager.get_cimel_coef()
        )
        csv.export_csv(
            self.lime_simulation.get_elis(),
            "Wavelengths (nm)",
            "Irradiances  (Wm⁻²/nm)",
            point,
            output_file,
            self.settings_manager.get_cimel_coef().version,
        )
