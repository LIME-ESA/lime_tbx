"""This module is in charge of logging the output data.

It exports the following functions:
    * get_logger() - Returns the toolbox logger, used for logging messages.
"""

import logging
from datetime import datetime, timezone
import os
import sys

from lime_tbx.persistence.local_storage import appdata
from lime_tbx.common import constants

# SINGLETON
_logger: logging.Logger = None

_FORMAT = "%(levelname)s: [%(filename)s:%(lineno)s - %(funcName)10s() ] %(message)s"
_DATEFORMAT = "%H:%M:%S"


def _get_printout_logger() -> logging.Logger:
    """Creates basic logger that prints out the messages.
    Useful for when a logger is needed but the lime logger can't be instanced yet."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFORMAT))
    logger = logging.getLogger("printout_logger")
    logger.setLevel(logging.WARNING)
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    """Returns the toolbox logger, used for logging messages.

    Returns:
    logger: logging.Logger
        LIME Toolbox logger.
    """
    global _logger
    if _logger == None:
        dtnow = datetime.now(timezone.utc)
        logname = dtnow.strftime("%Y%m%d")
        logname = "log_{}.txt".format(logname)
        logname = os.path.join(
            appdata.get_appdata_folder(_get_printout_logger()), logname
        )
        logging.basicConfig(
            filename=logname, filemode="a", format=_FORMAT, datefmt=_DATEFORMAT
        )
        _logger = logging.getLogger(__name__)
        debug_value = "INFO"
        if constants.DEBUG_ENV_NAME in os.environ:
            debug_value = os.environ[constants.DEBUG_ENV_NAME]
        if isinstance(debug_value, str) and debug_value.upper() == "DEBUG":
            _logger.setLevel(logging.DEBUG)
        else:
            _logger.setLevel(logging.INFO)
        if constants.DEV_LOGOUT_ENV_NAME in os.environ:
            # to print logs in terminal screen add environment variable named "LIME_DEVELOPER_LOGGING"
            _logger.addHandler(logging.StreamHandler())

    return _logger
