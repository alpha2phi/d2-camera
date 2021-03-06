from d2_camera.d2 import video_capture
import unittest

from d2_camera import video_capture


class TestCli(unittest.TestCase):

    """CLI tests."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_name(self):
        video_capture()
