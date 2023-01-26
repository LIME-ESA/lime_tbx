"""Tests for update module"""

"""___Built-In Modules___"""
import requests
import threading
import os
from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler

"""___Third-Party Modules___"""
import unittest

"""___LIME TBX Modules___"""
from ..update import Update, IUpdate

"""___Authorship___"""
__author__ = "Javier Gatón Herguedas"
__created__ = "29/08/2022"
__maintainer__ = "Javier Gatón Herguedas"
__email__ = "gaton@goa.uva.es"
__status__ = "Development"


def get_updater() -> IUpdate:
    up = Update()
    up.url = "http://localhost:8000/"
    return up


class HTTPHandler(SimpleHTTPRequestHandler):
    """This handler uses server.base_path instead of always using os.getcwd()"""

    def translate_path(self, path):
        path = SimpleHTTPRequestHandler.translate_path(self, path)
        relpath = os.path.relpath(path, os.getcwd())
        fullpath = os.path.join(self.server.base_path, relpath)
        return fullpath


class HTTPServer(BaseHTTPServer):
    """The main server, you pass in base_path which is the path you want to serve requests from"""

    def __init__(self, base_path, server_address, RequestHandlerClass=HTTPHandler):
        self.base_path = base_path
        RequestHandlerClass.log_message = lambda a, b, c, d, e: None
        BaseHTTPServer.__init__(self, server_address, RequestHandlerClass)


class TestUpdateNoServer(unittest.TestCase):
    def test_connection_error(self):
        up = get_updater()
        up.url = "http://localhost:6969"  # Which is not the same
        self.assertRaises(requests.exceptions.ConnectionError, up.check_for_updates, 5)


class TestUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dirname = os.path.join(os.path.dirname(__file__), "../../../../coeff_data")
        cls.httpd = HTTPServer(dirname, ("localhost", 8000))
        cls.t = threading.Thread(
            name="test server proc", target=cls.httpd.serve_forever
        )
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.t.join()

    def test_connection(self):
        up = get_updater()
        are_updates = up.check_for_updates()
        self.assertFalse(are_updates)

    def test_download(self):
        check_val = object()

        def _callback_stopper_check_is_running(obj: TestUpdate, val: object):
            obj.assertEqual(val, check_val)
            return True

        up = get_updater()
        news, fails = up.download_coefficients(
            _callback_stopper_check_is_running, [self, check_val]
        )
        self.assertEqual(0, news)
        self.assertEqual(0, fails)

    def test_download_not_running(self):
        check_val = object()

        def _callback_stopper_check_is_running(obj: TestUpdate, val: object):
            obj.assertEqual(val, check_val)
            return False

        up = get_updater()
        news, fails = up.download_coefficients(
            _callback_stopper_check_is_running, [self, check_val]
        )
        self.assertEqual(0, news)
        self.assertEqual(0, fails)


if __name__ == "__main__":
    unittest.main()
