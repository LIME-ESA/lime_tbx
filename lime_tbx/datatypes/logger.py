"""This module is in charge of logging the output data."""

"""___Built-In Modules___"""
import logging

"""___Third-Party Modules___"""
# import here


"""___LIME Modules___"""
# import here

# SINGLETON
_logger: logging.Logger = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger == None:
        _logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        _logger.setLevel(logging.DEBUG)
    return _logger
