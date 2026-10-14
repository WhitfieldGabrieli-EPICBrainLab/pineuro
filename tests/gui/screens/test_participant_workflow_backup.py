"""Behavioral tests for the participant workflow's session-backup wiring.

Same pattern as ``test_participant_workflow.py``: the real ``__init__`` builds a
live Flet layout, so a bare screen (``__new__``) gets only the attributes the
methods under test read. The backup engine underneath is the real one, except
where a test needs to hold a pass open.
"""

import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from pineuro.backup import BackupResult, mirror_session
from pineuro.gui.screens.participant_workflow import ParticipantWorkflowScreen


def _screen(tmp_path: Path, backup: dict | None = None, *, running: bool = False) -> ParticipantWorkflowScreen:
    session_dir = tmp_path / "data" / "sub-01" / "ses-01"
    (session_dir / "logs").mkdir(parents=True)
    (session_dir / "logs" / "session.log").write_text("started\n", encoding="utf-8")
    (session_dir / "collection").mkdir()

    config: dict = {"paths": {"base_dir": str(session_dir)}}
    if backup is not None:
        config["backup"] = backup
    screen = ParticipantWorkflowScreen.__new__(ParticipantWorkflowScreen)
    screen.study_config = SimpleNamespace(session_config=config)
    screen.participant_id = "sub-01"
    screen.session_id = "ses-01"
    screen._participant_base_dir = session_dir
    screen._is_running = running
    screen._disk_space_bar = MagicMock()
    screen._backup_lock = threading.Lock()
    screen._backup_thread = None
    screen._backup_stop = None
    screen._backup_requested_again = False
    screen._last_step_finished_at = None
    return screen


def _enabled(destination: Path, **overrides) -> dict:
    return {"enabled": True, "destination": str(destination), "include_dicoms": True, **overrides}


def _wait_for_backup(screen: ParticipantWorkflowScreen, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        thread = screen._backup_thread
        if thread is None or not thread.is_alive():
            return
        time.sleep(0.01)
    raise AssertionError("backup pass did not finish")


def _last_status(screen: ParticipantWorkflowScreen) -> str:
    return screen._disk_space_bar.set_status.call_args.args[0]


class TestStartBackup:
    def test_backs_up_the_session_and_reports_on_the_bar(self, tmp_path):
        destination = tmp_path / "backup"
        destination.mkdir()
        screen = _screen(tmp_path, _enabled(destination))

        screen._start_backup("after rest_run_1")
        _wait_for_backup(screen)

        assert (destination / "sub-01" / "ses-01" / "logs" / "session.log").exists()
        assert "Backup of sub-01/ses-01" in _last_status(screen)

    def test_never_runs_while_a_step_is_running(self, tmp_path):
        destination = tmp_path / "backup"
        destination.mkdir()
        screen = _screen(tmp_path, _enabled(destination), running=True)

        screen._start_backup("after rest_run_1")

        assert screen._backup_thread is None
        assert not (destination / "sub-01").exists()

    def test_disabled_backup_does_nothing(self, tmp_path):
        screen = _screen(tmp_path, {"enabled": False, "destination": str(tmp_path)})

        screen._start_backup("after rest_run_1")

        assert screen._backup_thread is None

    def test_starting_a_step_pauses_a_pass_in_progress(self, tmp_path, monkeypatch):
        destination = tmp_path / "backup"
        destination.mkdir()
        screen = _screen(tmp_path, _enabled(destination))
        entered = threading.Event()

        def held_open_pass(session_dir, dest, *, include_dicoms, stop_event):
            entered.set()
            stop_event.wait(5.0)  # copying files, until the step starts
            return BackupResult(destination=Path(dest), stopped=stop_event.is_set())

        monkeypatch.setattr("pineuro.gui.screens.participant_workflow.mirror_session", held_open_pass)
        screen._start_backup("after rest_run_1")
        assert entered.wait(5.0)

        screen._stop_backup()  # what _start_step does
        _wait_for_backup(screen)

        assert "paused" in _last_status(screen)


class TestCleanupBackupNote:
    def test_no_note_when_backup_is_off(self, tmp_path):
        assert _screen(tmp_path)._cleanup_backup_note() is None

    def test_warns_when_dicoms_are_not_backed_up(self, tmp_path):
        screen = _screen(tmp_path, _enabled(tmp_path / "backup", include_dicoms=False))

        text, is_warning = screen._cleanup_backup_note()

        assert is_warning and "only copy" in text

    def test_warns_without_a_complete_backup_since_the_last_step(self, tmp_path):
        destination = tmp_path / "backup"
        destination.mkdir()
        screen = _screen(tmp_path, _enabled(destination))
        mirror_session(screen._participant_base_dir, destination)

        screen._last_step_finished_at = datetime.now() + timedelta(minutes=5)  # a step finished after it
        text, is_warning = screen._cleanup_backup_note()
        assert is_warning and "not been fully backed up" in text

        screen._last_step_finished_at = datetime.now() - timedelta(minutes=5)
        text, is_warning = screen._cleanup_backup_note()
        assert not is_warning and text.startswith("Backed up to")
