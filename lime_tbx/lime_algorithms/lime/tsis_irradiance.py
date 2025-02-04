"""
This module reads in the tsis data and calculates the band integrated solar
irradiances for various spectral response functions
"""

"""___Built-In Modules___"""
import os
from typing import Dict, Tuple

"""___Third-Party Modules___"""
import numpy as np
from numpy.typing import NDArray
import xarray
import punpy

"""___LIME Modules___"""
from lime_tbx.spectral_integration.spectral_integration import SpectralIntegration

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "2022/03/03"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def _get_tsis_data() -> Dict[float, Tuple[float, float]]:
    """Returns the TSIS-1 solar irradiances at 0.1 nm resolution and 0.025 nm sampling interval

    Returns
    -------
    A dict that has the wavelengths as keys (float), and as values it has tuples of the
    (spectral solar irradiance, uncertainty) values.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ds_tsis = xarray.open_dataset(
        os.path.join(
            dir_path,
            "assets",
            "hybrid_reference_spectrum_p1nm_resolution_c2022-11-30_with_unc.nc",
        ),
        engine="netcdf4",
    )
    wavs = ds_tsis["Vacuum Wavelength"].values
    SSI = ds_tsis.SSI.values
    u_SSI = ds_tsis.SSI_UNC.values
    data = {}
    for i in range(len(wavs)):
        data[wavs[i]] = (SSI[i], u_SSI[i])
    return data


_PARALLEL_CORES = 1


def tsis_cimel(
    solar_y: NDArray[np.float_],
    solar_x: NDArray[np.float_],
    u_solar_y: NDArray[np.float_],
    mc_steps=100,
    parallel_cores=_PARALLEL_CORES,
):
    """
    Calculate TSIS solar irradiances and uncertainties, band integrated to the CIMEL bands
    """
    si = SpectralIntegration()
    prop = punpy.MCPropagation(mc_steps, parallel_cores=parallel_cores)
    cimel_wavs = np.array([440, 500, 675, 870, 1020, 1640, 2130])
    cimel_esi = si.integrate_cimel(solar_y, solar_x, cimel_wavs)
    u_cimel_esi = prop.propagate_random(
        si.integrate_cimel,
        [solar_y, solar_x, cimel_wavs],
        [u_solar_y, None, None],
        return_corr=False,
    )

    return cimel_wavs, cimel_esi, u_cimel_esi


class Wrapper:
    def __init__(self, func):
        self._func = func
        self.calls = 0

    def func(self, *args):
        print(self.calls)
        self.calls += 1
        return self._func(*args)


def tsis_asd(
    solar_y: NDArray[np.float_],
    solar_x: NDArray[np.float_],
    u_solar_y: NDArray[np.float_],
    mc_steps=100,
    parallel_cores=_PARALLEL_CORES,
):
    """
    Calculate TSIS solar irradiances and uncertainties, band integrated to the ASD bands
    """
    si = SpectralIntegration()
    prop = punpy.MCPropagation(mc_steps, parallel_cores=parallel_cores)
    asd_wavs = np.array(si.asd_srf.get_wavelengths())
    w = Wrapper(si.integrate_solar_asd)
    asd_esi = w.func(solar_y, solar_x)
    u_asd_esi = prop.propagate_random(w.func, [solar_y, solar_x], [u_solar_y, None])
    return asd_wavs, asd_esi, u_asd_esi


def tsis_fwhm(
    solar_y: NDArray[np.float_],
    solar_x: NDArray[np.float_],
    u_solar_y: NDArray[np.float_],
    fwhm: float,
    sampling: float,
    shape: str,
    mc_steps=100,
    parallel_cores=_PARALLEL_CORES,
):
    """
    Calculate TSIS solar irradiances and uncertainties, band integrated to bands with
    specified FWHM and shape
    """
    si = SpectralIntegration()
    prop = punpy.MCPropagation(mc_steps, parallel_cores=parallel_cores)
    si.set_srf_interpolated(fwhm, sampling, shape, write=True)
    intp_wavs = np.array(si.interpolated_srf.get_wavelengths())
    if shape == "gaussian":
        w = Wrapper(si.integrate_solar_interpolated_gaussian)
    elif shape == "triangle":
        w = Wrapper(si.integrate_solar_interpolated_triangle)
    else:
        raise ValueError("SRF shape not recognised")
    intp_esi = w.func(solar_y, solar_x)
    u_intp_esi = prop.propagate_random(w.func, [solar_y, solar_x], [u_solar_y, None])
    return intp_wavs, intp_esi, u_intp_esi


_AVAILABLE_FWHM = [3, 1, 0.3, 0.1]
_AVAILABLE_FWHM_SAMPLING = [1, 1, 0.1, 0.1]
_AVAILABLE_FWHM_SHAPE = ["gaussian", "triangle", "gaussian", "triangle"]


def _gen_cimel(solar_y, solar_x, u_solar_y):
    print("Generating TSIS CIMEL")
    cimel_wavs, cimel_esi, u_cimel_esi = tsis_cimel(solar_y, solar_x, u_solar_y)
    with open("assets/tsis_cimel.csv", "w", encoding="utf-8") as f:
        for i, (cwav, cesi, ucesi) in enumerate(
            zip(cimel_wavs, cimel_esi, u_cimel_esi)
        ):
            print(f"{i}/{len(cimel_wavs)}")
            f.write(f"{cwav}, {cesi}, {ucesi}\n")


def _gen_asd(solar_y, solar_x, u_solar_y):
    print("Generating TSIS ASD")
    asd_wavs, asd_esi, u_asd_esi = tsis_asd(solar_y, solar_x, u_solar_y)
    with open("assets/tsis_asd.csv", "w", encoding="utf-8") as f:
        for i, (awav, aesi, uaesi) in enumerate(zip(asd_wavs, asd_esi, u_asd_esi)):
            print(f"{i}/{len(asd_wavs)}")
            f.write(f"{awav}, {aesi}, {uaesi}\n")


def _gen_fwhms(solar_y, solar_x, u_solar_y):
    for i, (fwhm, sampling, shape) in enumerate(
        zip(_AVAILABLE_FWHM, _AVAILABLE_FWHM_SAMPLING, _AVAILABLE_FWHM_SHAPE)
    ):
        if i >= 2:
            # Requires too big of a RAM
            break
        print(f"{i}/{len(_AVAILABLE_FWHM)}")
        print(f"Generating {fwhm}, {sampling}, {shape}")
        intp_wavs, intp_esi, u_intp_esi = tsis_fwhm(
            solar_y,
            solar_x,
            u_solar_y,
            fwhm,
            sampling,
            shape,
        )
        id_str = f"{fwhm}_{sampling}_{shape}".replace(".", "p")
        with open(f"assets/tsis_fwhm_{id_str}.csv", "w", encoding="utf-8") as f:
            for j, (iwav, iesi, uiesi) in enumerate(
                zip(intp_wavs, intp_esi, u_intp_esi)
            ):
                print(f"{j}/{len(intp_wavs)}")
                f.write(f"{iwav},{iesi},{uiesi}\n")


def _gen_files():
    """
    Generate all the required TSIS based solar irradiance files.
    This is time consuming and is precomputed prior to the normal execution of the lime_tbx
    """
    solar_data = _get_tsis_data()
    solar_x = np.array(list(solar_data.keys()))
    solar_y = np.array(list(map(lambda x: x[0], solar_data.values())))
    u_solar_y = np.array(list(map(lambda x: x[1], solar_data.values())))
    _gen_cimel(solar_y, solar_x, u_solar_y)
    _gen_asd(solar_y, solar_x, u_solar_y)
    _gen_fwhms(solar_y, solar_x, u_solar_y)


def main():
    _gen_files()


if __name__ == "__main__":
    main()
