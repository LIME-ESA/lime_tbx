"""This module contains the values of LIME Toolbox constants."""

MIN_WLEN = 350
MAX_WLEN = 2500
CERTAIN_MIN_WLEN = 350
CERTAIN_MAX_WLEN = 2500
DEFAULT_SRF_NAME = "Default"
EARTH_FRAME = "J2000"  # This is the frame in rectangular coordinates for earth.
MOON_FRAME = "MOON_ME"  # This is the frame in rectangular coordinates for moon.
VERSION_NAME = "1.1.0"
DEBUG_ENV_NAME = "LIME_DEBUG"
DEV_LOGOUT_ENV_NAME = "LIME_DEVELOPER_LOGGING"
LIME_TBX_DATA_SOURCE = "LIME-TBX"
MAX_LIMIT_REFL_ERR_CORR_ARE_STORED = 25
NUM_CIMEL_WLENS = 6
DEBUG_INTERMEDIATE_RESULTS_PATH = "LIMETBX_INTERMEDIATE_PATH"


class CompFields:
    """Enumeration of the values of all the different comparison string fields"""

    COMP_MPA = "Moon Phase Angle"
    COMP_DATE = "Date"
    COMP_WLEN = "Wavelength (Box Plot)"
    COMP_WLEN_MEAN = "Wavelength (Mean)"
    DIFF_NONE = "None"
    DIFF_REL = "Relative"
    DIFF_PERC = "Percentage"
