"""
This module reads in the tsis data and calculated the band integrated solar irradiances for various SRF
"""

"""___Built-In Modules___"""
from typing import Dict, Tuple
import os

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


def tsis_cimel(
    solar_y: NDArray[np.float_],
    solar_x: NDArray[np.float_],
    u_solar_y: NDArray[np.float_],
    MCsteps=100,
):
    """
    Calculate TSIS solar irradiances and uncertainties, band integrated to the CIMEL bands
    """
    si = SpectralIntegration()
    prop = punpy.MCPropagation(MCsteps, parallel_cores=1)
    cimel_wavs = np.array([440, 500, 675, 870, 1020, 1640])
    cimel_esi = si.integrate_cimel(solar_y, solar_x)
    u_cimel_esi = prop.propagate_systematic(
        si.integrate_cimel, [solar_y, solar_x], [u_solar_y, None], return_corr=False
    )

    return cimel_wavs, cimel_esi, u_cimel_esi


def tsis_asd(
    solar_y: NDArray[np.float_],
    solar_x: NDArray[np.float_],
    u_solar_y: NDArray[np.float_],
    MCsteps=100,
):
    """
    Calculate TSIS solar irradiances and uncertainties, band integrated to the ASD bands
    """
    si = SpectralIntegration()
    prop = punpy.MCPropagation(MCsteps, parallel_cores=1)
    asd_wavs = np.array(si.asd_srf.get_wavelengths())
    asd_esi = si.integrate_solar_asd(solar_y, solar_x)
    u_asd_esi = prop.propagate_systematic(
        si.integrate_solar_asd, [solar_y, solar_x], [u_solar_y, None]
    )

    return asd_wavs, asd_esi, u_asd_esi


def tsis_fwhm(
    solar_y: NDArray[np.float_],
    solar_x: NDArray[np.float_],
    u_solar_y: NDArray[np.float_],
    fwhm: float,
    sampling: float,
    shape: str,
    MCsteps=3,
):
    """
    Calculate TSIS solar irradiances and uncertainties, band integrated to bands with specified FWHM and shape
    """
    si = SpectralIntegration()
    prop = punpy.MCPropagation(MCsteps, parallel_cores=100)
    si.set_srf_interpolated(fwhm, sampling, shape, write=True)
    intp_wavs = np.array(si.interpolated_srf.get_wavelengths())
    if shape=="gaussian":
        intp_esi = si.integrate_solar_interpolated_gaussian(solar_y, solar_x)
        u_intp_esi = prop.propagate_systematic(
            si.integrate_solar_interpolated_gaussian,
            [solar_y, solar_x],
            [u_solar_y, None],
        )
    elif shape=="triangle":
        intp_esi = si.integrate_solar_interpolated_triangle(solar_y, solar_x)
        u_intp_esi = prop.propagate_systematic(
            si.integrate_solar_interpolated_triangle,
            [solar_y, solar_x],
            [u_solar_y, None],
        )
    else:
        raise ValueError("SRF shape not recognised")

    return intp_wavs, intp_esi, u_intp_esi


_AVAILABLE_FWHM = [3, 1, 0.3, 0.1]
_AVAILABLE_FWHM_SAMPLING = [1, 1, 0.1, 0.1]
_AVAILABLE_FWHM_SHAPE = ["gaussian", "triangle", "gaussian", "triangle"]

def _gen_files():
    """
    generate all the required files. This is time consuming and is precomputed prior to the normal execution of the lime_tbx
    """
    solar_data = _get_tsis_data()
    solar_x = np.array(list(solar_data.keys()))
    solar_y = np.array(list(map(lambda x: x[0], solar_data.values())))
    u_solar_y = np.array(list(map(lambda x: x[1], solar_data.values())))
    cimel_wavs, cimel_esi, u_cimel_esi = tsis_cimel(solar_y, solar_x, u_solar_y)
    # with open("assets/tsis_cimel.csv", "w") as f:
    #     for i in range(len(cimel_wavs)):
    #         print(i)
    #         f.write("%s, %s, %s \n" % (cimel_wavs[i], cimel_esi[i], u_cimel_esi[i]))
    # asd_wavs, asd_esi, u_asd_esi = tsis_asd(solar_y, solar_x, u_solar_y)
    # with open("assets/tsis_asd.csv", "w") as f:
    #     for i in range(len(asd_wavs)):
    #         print(f"{i}/{len(asd_wavs)}")
    #         f.write("%s, %s, %s \n" % (asd_wavs[i], asd_esi[i], u_asd_esi[i]))
    for ifwhm in range(len(_AVAILABLE_FWHM)):
        intp_wavs, intp_esi, u_intp_esi = tsis_fwhm(
            solar_y,
            solar_x,
            u_solar_y,
            _AVAILABLE_FWHM[ifwhm],
            _AVAILABLE_FWHM_SAMPLING[ifwhm],
            _AVAILABLE_FWHM_SHAPE[ifwhm],
        )
        id_str = ("%s_%s_%s" % (_AVAILABLE_FWHM[ifwhm],
            _AVAILABLE_FWHM_SAMPLING[ifwhm],
            _AVAILABLE_FWHM_SHAPE[ifwhm])).replace(".", "p")
        with open("assets/tsis_fwhm_%s.csv"%id_str, "w") as f:
            for i in range(len(intp_wavs)):
                f.write("%s,%s,%s\n" % (intp_wavs[i], intp_esi[i], u_intp_esi[i]))


def main():
    _gen_files()


if __name__ == "__main__":
    main()
