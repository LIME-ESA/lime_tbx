"""This module creates LGLOD datatypes"""

"""___Built-In Modules___"""
# import here

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.simulation.lime_simulation import ILimeSimulation
from lime_tbx.spice_adapter.spice_adapter import SPICEAdapter
from ..datatypes import constants
from ..datatypes.datatypes import (
    Point,
    SurfacePoint,
    SatellitePoint,
    CustomPoint,
    SatellitePosition,
    LunarObservationWrite,
    SelenographicDataWrite,
    SpectralResponseFunction,
    KernelsPath,
    LGLODData,
    SpectralData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "26/09/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def create_lglod_data(
    point: Point,
    srf: SpectralResponseFunction,
    lime_simulation: ILimeSimulation,
    kernels_path: KernelsPath,
    spectrum_name: str,
) -> LGLODData:
    obs = []
    ch_names = srf.get_channels_names()
    sat_pos_ref = constants.EARTH_FRAME
    elis = lime_simulation.get_elis()
    elis_cimel = lime_simulation.get_elis_cimel()
    if not isinstance(elis_cimel, list):
        elis_cimel = [elis_cimel]
    elrefs = lime_simulation.get_elrefs()
    elrefs_cimel = lime_simulation.get_elrefs_cimel()
    if not isinstance(elrefs_cimel, list):
        elrefs_cimel = [elrefs_cimel]
    if not isinstance(elis, list):
        elis = [elis]
    if not isinstance(elrefs, list):
        elrefs = [elrefs]
    if lime_simulation.is_polarization_updated():
        polars = lime_simulation.get_polars()
        polars_cimel = lime_simulation.get_polars_cimel()
        if not isinstance(polars_cimel, list):
            polars_cimel = [polars_cimel]
        if not isinstance(polars, list):
            polars = [polars]
    else:
        polars = [
            SpectralData(
                e.wlens, np.zeros(e.data.shape), np.zeros(e.uncertainties.shape), None
            )
            for e in elrefs
        ]
        polars_cimel = [
            SpectralData(
                e.wlens, np.zeros(e.data.shape), np.zeros(e.uncertainties.shape), None
            )
            for e in elrefs_cimel
        ]
    signals = lime_simulation.get_signals()
    if isinstance(point, SurfacePoint) or isinstance(point, SatellitePoint):
        dts = point.dt
        if not isinstance(dts, list):
            dts = [dts]
        if isinstance(point, SurfacePoint):
            sat_pos = [
                SatellitePosition(
                    *SPICEAdapter.to_rectangular(
                        point.latitude,
                        point.longitude,
                        point.altitude,
                        "EARTH",
                        kernels_path.main_kernels_path,
                    )
                )
                for _ in dts
            ]
            sat_name = ""
        else:
            sur_points = lime_simulation.get_surfacepoints()
            if isinstance(sur_points, SurfacePoint):
                sur_points = [sur_points]
            sat_pos = [
                SatellitePosition(
                    *SPICEAdapter.to_rectangular(
                        sp.latitude,
                        sp.longitude,
                        sp.altitude,
                        "EARTH",
                        kernels_path.main_kernels_path,
                    )
                )
                for sp in sur_points
            ]
            sat_name = point.name
        for i, dt in enumerate(dts):
            ob = LunarObservationWrite(
                ch_names,
                sat_pos_ref,
                dt,
                sat_pos[i],
                elis[i],
                elrefs[i],
                polars[i],
                sat_name,
                None,
                constants.LIME_TBX_DATA_SOURCE,
            )
            obs.append(ob)
    elif isinstance(point, CustomPoint):
        sat_pos = SatellitePosition(
            *SPICEAdapter.to_rectangular(
                point.selen_obs_lat,
                point.selen_obs_lon,
                point.distance_observer_moon * 1000,
                "MOON",
                kernels_path.main_kernels_path,
            )
        )
        sat_name = ""
        sat_pos_ref = constants.MOON_FRAME
        obs = [
            LunarObservationWrite(
                ch_names,
                sat_pos_ref,
                None,
                sat_pos,
                elis[0],
                elrefs[0],
                polars[0],
                sat_name,
                SelenographicDataWrite(
                    point.distance_sun_moon,
                    point.selen_sun_lon,
                    point.moon_phase_angle,
                ),
                constants.LIME_TBX_DATA_SOURCE,
            )
        ]
    is_not_default_srf = True
    if (
        srf.name == constants.DEFAULT_SRF_NAME
        and len(srf.get_channels_names()) == 1
        and srf.get_channels_names()[0] == constants.DEFAULT_SRF_NAME
    ):
        is_not_default_srf = False
    return LGLODData(
        obs,
        signals,
        is_not_default_srf,
        elis_cimel,
        elrefs_cimel,
        polars_cimel,
        spectrum_name,
    )
