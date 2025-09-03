"""This module creates LGLOD datatypes"""

"""___Built-In Modules___"""
from typing import List, Union, Tuple

"""___Third-Party Modules___"""
import numpy as np

"""___LIME_TBX Modules___"""
from lime_tbx.application.simulation.lime_simulation import ILimeSimulation
from lime_tbx.business.spice_adapter.spice_adapter import SPICEAdapter
from lime_tbx.common import constants
from lime_tbx.common.datatypes import (
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
    MoonData,
)

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "26/09/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Production"


def create_lglod_data(
    point: Point,
    srf: SpectralResponseFunction,
    lime_simulation: ILimeSimulation,
    kernels_path: KernelsPath,
    spectrum_name: str,
    dolp_spectrum_name: str,
    aolp_spectrum_name: str,
    coeff_version: str,
    mdas: Union[MoonData, List[MoonData]],
) -> LGLODData:
    """
    Creates a LGLOD object

    Parameters
    ----------
    point: Point
    srf: SpectralResponseFunction
    lime_simulation: ILimeSimulation
    kernels_path: KernelsPath
    spectrum_name: str
        Name of the spectrum used to interpolate Reflectance & Irradiance ('ASD', 'linear')
    dolp_spectrum_name: str
        Name of the spectrum used to interpolate DoLP (usually 'linear')
    aolp_spectrum_name: str
        Name of the spectrum used to interpolate AoLP (usually 'linear')
    coeff_version: str
        Coefficients version name

    Returns
    -------
    lglod: LGLODData
        Lglod data object with the information given and generated.
    """
    obs = []
    skipped_uncs = lime_simulation.is_skipping_uncs()
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
    if not isinstance(mdas, list):
        mdas = [mdas]
    if lime_simulation.is_polarisation_updated():
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
    if lime_simulation.is_aolp_updated():
        aolp = lime_simulation.get_aolp()
        aolp_cimel = lime_simulation.get_aolp_cimel()
        if not isinstance(aolp_cimel, list):
            aolp_cimel = [aolp_cimel]
        if not isinstance(aolp, list):
            aolp = [aolp]
    else:
        aolp = [
            SpectralData(
                e.wlens, np.zeros(e.data.shape), np.zeros(e.uncertainties.shape), None
            )
            for e in elrefs
        ]
        aolp_cimel = [
            SpectralData(
                e.wlens, np.zeros(e.data.shape), np.zeros(e.uncertainties.shape), None
            )
            for e in elrefs_cimel
        ]
    signals = lime_simulation.get_signals()
    if isinstance(point, (SurfacePoint, SatellitePoint)):
        dts = point.dt
        if not isinstance(dts, list):
            dts = [dts]
        if isinstance(point, SurfacePoint):
            llhs = [(point.latitude, point.longitude, point.altitude) for _ in dts]
            rects = SPICEAdapter.to_rectangular_multiple(
                llhs,
                "EARTH",
                kernels_path.main_kernels_path,
                dts,
            )
            sat_pos = [SatellitePosition(*rect) for rect in rects]
            sat_name = ""
        else:
            sur_points = lime_simulation.get_surfacepoints()
            if isinstance(sur_points, SurfacePoint):
                sur_points = [sur_points]
            llhs = [(sp.latitude, sp.longitude, sp.altitude) for sp in sur_points]
            spdts = [sp.dt for sp in sur_points]
            rects = SPICEAdapter.to_rectangular_multiple(
                llhs,
                "EARTH",
                kernels_path.main_kernels_path,
                spdts,
            )
            sat_pos = [SatellitePosition(*rect) for rect in rects]
            sat_name = point.name
        for i, (dt, md) in enumerate(zip(dts, mdas)):
            ob = LunarObservationWrite(
                ch_names,
                sat_pos_ref,
                dt,
                sat_pos[i],
                elis[i],
                elrefs[i],
                polars[i],
                aolp[i],
                sat_name,
                SelenographicDataWrite(
                    md.distance_sun_moon,
                    md.long_sun_radians,
                    md.mpa_degrees,
                    md.lat_obs,
                    md.long_obs,
                    md.distance_observer_moon,
                ),
                constants.LIME_TBX_DATA_SOURCE,
            )
            obs.append(ob)
    else:  # Custom or MultiCustom
        if isinstance(point, CustomPoint):
            pts = [point]
        else:
            pts = point.pts
        obs = []
        sat_pos_ref = constants.MOON_FRAME
        sat_name = ""
        for i, pt in enumerate(pts):
            sat_pos = SatellitePosition(
                *SPICEAdapter.to_rectangular_same_frame(
                    [
                        (
                            pt.selen_obs_lat,
                            pt.selen_obs_lon,
                            pt.distance_observer_moon * 1000,
                        )
                    ],
                    "MOON",
                    kernels_path.main_kernels_path,
                )[0]
            )
            obs += [
                LunarObservationWrite(
                    ch_names,
                    sat_pos_ref,
                    None,
                    sat_pos,
                    elis[i],
                    elrefs[i],
                    polars[i],
                    aolp[i],
                    sat_name,
                    SelenographicDataWrite(
                        pt.distance_sun_moon,
                        pt.selen_sun_lon,
                        pt.moon_phase_angle,
                        pt.selen_obs_lat,
                        pt.selen_obs_lon,
                        pt.distance_observer_moon,
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
        aolp_cimel,
        spectrum_name,
        skipped_uncs,
        coeff_version,
        dolp_spectrum_name,
        aolp_spectrum_name,
    )
