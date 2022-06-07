"""describe class"""

"""___Built-In Modules___"""
from abc import ABC, abstractmethod
from typing import Dict
import pkgutil
import csv
from io import StringIO
import os

"""___Third-Party Modules___"""
# import here
import xarray
import obsarray
import numpy as np

"""___NPL Modules___"""
from lime_tbx.datatypes.datatypes import (
    IrradianceCoefficients,
    PolarizationCoefficients,
)

from lime_tbx.datatypes.templates_digital_effects_table import template_asd

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2022"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

def _get_default_asd_data() -> xarray.Dataset:
    # define dim_size_dict to specify size of arrays

    current_dir=os.path.dirname(os.path.abspath(__file__))
    data=np.genfromtxt(os.path.join(current_dir,"assets/ASD_refl_lunar.txt"))
    wavs=data[:,0]
    refl=data[:,1]

    dim_sizes = {"wavelength":len(wavs)}
    # create dataset
    ds_asd = obsarray.create_ds(template_asd,dim_sizes)

    ds_asd=ds_asd.assign_coords(wavelength=wavs)


    ds_asd.reflectance.values=refl
    ds_asd.u_ran_reflectance.values=refl*0.01
    ds_asd.u_sys_reflectance.values=refl*0.05
    return ds_asd

