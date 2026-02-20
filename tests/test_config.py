"""Tests for pineuro.config — platform detection and display settings."""

from unittest.mock import patch

from pineuro.config import get_platform, is_wayland


class TestGetPlatform:
    def test_windows(self):
        with patch("pineuro.config.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert get_platform() == "windows"

    def test_linux(self):
        with patch("pineuro.config.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert get_platform() == "linux"

    def test_darwin(self):
        with patch("pineuro.config.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert get_platform() == "darwin"

    def test_unknown_defaults_to_linux(self):
        with patch("pineuro.config.sys") as mock_sys:
            mock_sys.platform = "freebsd"
            assert get_platform() == "linux"


class TestIsWayland:
    def test_wayland_session(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            assert is_wayland() is True

    def test_x11_session(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            assert is_wayland() is False

    def test_no_session_type(self):
        with patch.dict("os.environ", {}, clear=True):
            assert is_wayland() is False

    def test_case_insensitive(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "Wayland"}):
            assert is_wayland() is True
