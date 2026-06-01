"""Tests for pineuro.display — monitor detection and selection helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from pineuro.display import (
    MonitorInfo,
    get_monitor_by_position,
    get_monitors,
)


def _fake_si_monitor(*, x=0, y=0, width=1920, height=1080, name="HDMI-1", is_primary=False):
    """Build a stand-in for a screeninfo Monitor namespace object."""
    return SimpleNamespace(x=x, y=y, width=width, height=height, name=name, is_primary=is_primary)


class TestGetMonitorsSortAndAdapt:
    def test_empty_when_screeninfo_returns_nothing(self):
        with patch("screeninfo.get_monitors", return_value=[]):
            assert get_monitors() == []

    def test_screeninfo_missing_returns_empty(self):
        # Force the inner `from screeninfo import get_monitors` to fail.
        with patch.dict("sys.modules", {"screeninfo": None}):
            assert get_monitors() == []

    def test_unexpected_exception_returns_empty(self):
        with patch("screeninfo.get_monitors", side_effect=RuntimeError("boom")):
            assert get_monitors() == []

    def test_sorted_left_to_right_by_x(self):
        raw = [
            _fake_si_monitor(x=1920, name="right"),
            _fake_si_monitor(x=0, name="left"),
            _fake_si_monitor(x=3840, name="far_right"),
        ]
        with patch("screeninfo.get_monitors", return_value=raw):
            mons = get_monitors()
        assert [m.name for m in mons] == ["left", "right", "far_right"]
        assert [m.index for m in mons] == [0, 1, 2]

    def test_falls_back_when_name_is_none(self):
        raw = [_fake_si_monitor(name=None), _fake_si_monitor(x=1920, name=None)]
        with patch("screeninfo.get_monitors", return_value=raw):
            mons = get_monitors()
        assert mons[0].name == "Monitor 0"
        assert mons[1].name == "Monitor 1"

    def test_is_primary_none_becomes_false(self):
        raw = [_fake_si_monitor(is_primary=None)]
        with patch("screeninfo.get_monitors", return_value=raw):
            mons = get_monitors()
        assert mons[0].is_primary is False


class TestGetMonitorByPosition:
    def test_none_when_no_monitors(self):
        with patch("pineuro.display.get_monitors", return_value=[]):
            assert get_monitor_by_position("left") is None

    def test_single_monitor_returned_for_both_positions(self):
        only = MonitorInfo(0, 0, 0, 1920, 1080, "only", False)
        with patch("pineuro.display.get_monitors", return_value=[only]):
            assert get_monitor_by_position("left") is only
            assert get_monitor_by_position("right") is only

    def test_left_returns_leftmost(self):
        a = MonitorInfo(0, 0, 0, 1920, 1080, "a", False)
        b = MonitorInfo(1, 1920, 0, 1920, 1080, "b", False)
        with patch("pineuro.display.get_monitors", return_value=[a, b]):
            assert get_monitor_by_position("left") is a

    def test_right_returns_rightmost(self):
        a = MonitorInfo(0, 0, 0, 1920, 1080, "a", False)
        b = MonitorInfo(1, 1920, 0, 1920, 1080, "b", False)
        with patch("pineuro.display.get_monitors", return_value=[a, b]):
            assert get_monitor_by_position("right") is b

    def test_invalid_label_falls_back_to_leftmost(self):
        a = MonitorInfo(0, 0, 0, 1920, 1080, "a", False)
        b = MonitorInfo(1, 1920, 0, 1920, 1080, "b", False)
        with patch("pineuro.display.get_monitors", return_value=[a, b]):
            assert get_monitor_by_position("centre") is a
