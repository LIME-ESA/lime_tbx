"""
Configuration overrides for lime_tbx path detection.

This module defines global variables that allow overriding the default
locations for application data and program files. They are intended to be
set by the user or installer before calling the corresponding getter
functions in `appdata` and `programdata` modules.

Variables:
    APPDATA_OVERRIDE (str | None): If set to an absolute path, the
        `get_appdata_folder()` function will use this path instead of the
        platform-specific default. The path must contain subdirectories
        'kernels' and 'coeff_data'. If `None`, the default is used.
    PROGRAMFILES_OVERRIDE (str | None): If set to an absolute path, the
        `get_programfiles_folder()` function will use this path instead of
        the system-detected one. The path must contain subdirectories
        'kernels', 'eocfi_data', and 'coeff_data'. If `None`, the default
        is used.

Note:
    Setting these variables after the getter functions have already been
    called may have no effect, as some modules import the values at runtime
    (they are read each time the getter is called if imported dynamically).
    It is recommended to set them before any lime_tbx operations.
"""

PROGRAMFILES_OVERRIDE = None
APPDATA_OVERRIDE = None
