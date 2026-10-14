"""Tests for TaskBase's unfocused key polling on macOS (Quartz) and its use in rating loops.

The stimulus window is expected to be unfocused during a run (the operator works in the
GUI), so TaskBase polls global key state and injects presses into PsychoPy's key buffer.
These tests drive the macOS backend with a fake ``Quartz`` module that exposes only real
Quartz API names (so a misspelled constant fails the import, as it did on real Macs in
v0.10.0-v0.11.2) and a fake ``psychopy`` (no window, no display), so they run on every OS.
"""

import logging
import sys
import types
from unittest.mock import MagicMock

import pytest

# Real macOS virtual keycodes (kVK_ANSI_*), matching TaskBase._PSYCHOPY_TO_MAC_VK.
VK_1 = 0x12
VK_ESCAPE = 0x35


@pytest.fixture
def fake_psychopy(monkeypatch):
    """Install a minimal fake ``psychopy``: ``core.getTime`` and ``event._keyBuffer``."""
    pkg = types.ModuleType("psychopy")
    core = types.ModuleType("psychopy.core")
    core.getTime = lambda: 123.0
    event = types.ModuleType("psychopy.event")
    event._keyBuffer = []
    pkg.core = core
    pkg.event = event
    monkeypatch.setitem(sys.modules, "psychopy", pkg)
    monkeypatch.setitem(sys.modules, "psychopy.core", core)
    monkeypatch.setitem(sys.modules, "psychopy.event", event)
    return event


@pytest.fixture
def fake_quartz(monkeypatch):
    """Install a fake ``Quartz``; set ``.held`` to the keycodes currently down."""
    quartz = types.ModuleType("Quartz")
    quartz.held = set()
    quartz.listen_access = True
    quartz.kCGEventSourceStateCombinedSessionState = 0
    quartz.CGEventSourceKeyState = lambda state_id, vk: vk in quartz.held
    quartz.CGPreflightListenEventAccess = lambda: quartz.listen_access
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    return quartz


@pytest.fixture
def task():
    from pineuro.tasks.task_base import TaskBase

    t = TaskBase()
    t.win = MagicMock()
    return t


class TestQuartzPoller:
    def test_press_injects_once_while_held(self, task, fake_quartz, fake_psychopy):
        fake_quartz.held = {VK_1}
        task._poll_keys_quartz(["1"])
        task._poll_keys_quartz(["1"])  # still held: no auto-repeat
        assert fake_psychopy._keyBuffer == [("1", 0, 123.0)]
        assert task._last_inject_source == "poll_quartz"

    def test_release_then_press_injects_again(self, task, fake_quartz, fake_psychopy):
        seen = []

        def poll_and_consume():
            # The real wait loops call event.getKeys right after polling, which drains the buffer.
            task._poll_keys_quartz(["1"])
            seen.extend(k[0] for k in fake_psychopy._keyBuffer)
            fake_psychopy._keyBuffer.clear()

        fake_quartz.held = {VK_1}
        poll_and_consume()
        fake_quartz.held = set()
        poll_and_consume()
        fake_quartz.held = {VK_1}
        poll_and_consume()
        assert seen == ["1", "1"]

    def test_press_delivered_by_pyglet_is_not_injected_again(self, task, fake_quartz, fake_psychopy):
        # Focused window: pumping pyglet delivers the press itself, so polling must not add a second.
        fake_quartz.held = {VK_1}
        task.win.winHandle.dispatch_events.side_effect = lambda: fake_psychopy._keyBuffer.append(("1", 0, 122.9))
        task._poll_keys_quartz(["1"])
        assert fake_psychopy._keyBuffer == [("1", 0, 122.9)]  # pyglet's delivery only

    def test_escape_is_never_injected(self, task, fake_quartz, fake_psychopy):
        fake_quartz.held = {VK_ESCAPE}
        task._poll_keys_quartz(["escape"])
        assert fake_psychopy._keyBuffer == []

    def test_key_held_at_reset_must_be_released_first(self, task, fake_quartz, fake_psychopy, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        fake_quartz.held = {VK_1}
        task._reset_poll_state()  # '1' is already down when the scale appears
        task._poll_keys_quartz(["1"])
        assert fake_psychopy._keyBuffer == []
        fake_quartz.held = set()
        task._poll_keys_quartz(["1"])
        fake_quartz.held = {VK_1}
        task._poll_keys_quartz(["1"])
        assert [k[0] for k in fake_psychopy._keyBuffer] == ["1"]

    def test_missing_quartz_disables_polling_and_logs_once(self, task, fake_psychopy, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "Quartz", None)  # makes `import Quartz` raise ImportError
        with caplog.at_level(logging.ERROR, logger="pineuro.tasks.task_base"):
            task._poll_keys_quartz(["1"])
            task._poll_keys_quartz(["1"])
        assert fake_psychopy._keyBuffer == []
        assert sum("Quartz is not importable" in r.getMessage() for r in caplog.records) == 1

    def test_backend_logged_once_with_input_monitoring_hint(
        self, task, fake_quartz, fake_psychopy, monkeypatch, caplog
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        fake_quartz.listen_access = False
        with caplog.at_level(logging.INFO, logger="pineuro.tasks.task_base"):
            task._inject_keys_if_unfocused(["1"])
            task._inject_keys_if_unfocused(["1"])
        messages = [r.getMessage() for r in caplog.records]
        assert sum("backend=quartz" in m for m in messages) == 1
        assert any("Input Monitoring" in m for m in messages)


class TestRitRatingPolls:
    """The RIT localizer rating previously read keys only from pyglet, so it needed focus on every OS."""

    def test_rating_polls_unfocused_keys_and_resets_carry_over(self, monkeypatch):
        from pineuro.tasks.rit import RITTask

        polls: list[list[str]] = []

        pkg = types.ModuleType("psychopy")
        core = types.ModuleType("psychopy.core")
        core.Clock = MagicMock
        timer = MagicMock()
        timer.getTime.return_value = 1.0
        core.CountdownTimer = lambda duration: timer
        event = types.ModuleType("psychopy.event")
        event.clearEvents = MagicMock()
        # The confirm press only exists once the poller has run (it is the poller's injection).
        event.getKeys = lambda keyList=None, timeStamped=None: [("0", 0.42)] if polls else []
        pkg.core = core
        pkg.event = event
        monkeypatch.setitem(sys.modules, "psychopy", pkg)
        monkeypatch.setitem(sys.modules, "psychopy.core", core)
        monkeypatch.setitem(sys.modules, "psychopy.event", event)

        task = RITTask.__new__(RITTask)
        task.left_key, task.right_key, task.confirm_key = "1", "2", "0"
        task.rating_timeout = 10.0
        task._running = True
        task.win = MagicMock()
        for name in ("_rating_question", "_rating_left_label", "_rating_right_label"):
            setattr(task, name, MagicMock())
        task._update_marker = MagicMock()
        task._pos_to_x = MagicMock(return_value=0.0)
        task._draw_rating_frame = MagicMock()
        task._capture_and_emit_screenshot = MagicMock()
        task._show_fixation = MagicMock()
        task._reset_poll_state = MagicMock()
        task._inject_keys_if_unfocused = MagicMock(side_effect=lambda keys: polls.append(list(keys)))

        response, rt = task._run_rating("How sad do you feel?", "Not at all", "Extremely")

        task._reset_poll_state.assert_called_once()
        assert polls == [["1", "2", "0", "return"]]
        assert (response, rt) == (3, 0.42)
