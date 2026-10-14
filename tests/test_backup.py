"""Tests for pineuro.backup: mirroring a participant-session folder to a backup location."""

import os
import threading
from pathlib import Path

from pineuro.backup import (
    BACKUP_DEFAULTS,
    MANIFEST_NAME,
    backup_destination,
    mirror_session,
    read_manifest,
    validate_backup_config,
)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _session(tmp_path: Path) -> Path:
    """A small session folder: <root>/sub-01/ses-01 with outputs, DICOM copies and scratch."""
    session = tmp_path / "data" / "sub-01" / "ses-01"
    _write(session / "logs" / "session_20261014_101500.log", "started\n")
    _write(session / "feedback" / "02_feedback_output" / "sub-01_ses-01_task-ball_run-01_attempt-01_events.tsv")
    _write(session / "collection" / "task-rest_run-01_attempt-01" / "0001.dcm", "dicom")
    _write(session / "tmp" / "incoming_rest_run1" / "0001.dcm", "scratch")
    return session


def _relative_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME}


class TestMirrorSession:
    def test_mirrors_into_participant_session_layout_without_scratch(self, tmp_path):
        session = _session(tmp_path)
        backup = tmp_path / "backup"
        backup.mkdir()

        result = mirror_session(session, backup)

        assert result.complete
        assert result.destination == backup / "sub-01" / "ses-01"
        assert _relative_files(result.destination) == {
            "logs/session_20261014_101500.log",
            "feedback/02_feedback_output/sub-01_ses-01_task-ball_run-01_attempt-01_events.tsv",
            "collection/task-rest_run-01_attempt-01/0001.dcm",
        }  # tmp/ is never backed up

    def test_dicoms_can_be_left_out(self, tmp_path):
        session = _session(tmp_path)
        backup = tmp_path / "backup"
        backup.mkdir()

        result = mirror_session(session, backup, include_dicoms=False)

        assert not any(f.startswith("collection/") for f in _relative_files(result.destination))

    def test_second_pass_copies_only_changes(self, tmp_path):
        session = _session(tmp_path)
        backup = tmp_path / "backup"
        backup.mkdir()
        first = mirror_session(session, backup)

        log = session / "logs" / "session_20261014_101500.log"
        with open(log, "a", encoding="utf-8") as f:
            f.write("step finished\n")
        second = mirror_session(session, backup)

        assert first.copied == 3
        assert (second.copied, second.skipped) == (1, 2)
        assert (second.destination / "logs" / log.name).read_text(encoding="utf-8") == "started\nstep finished\n"

    def test_never_deletes_at_the_destination(self, tmp_path):
        session = _session(tmp_path)
        backup = tmp_path / "backup"
        backup.mkdir()
        mirror_session(session, backup)

        dicom = session / "collection" / "task-rest_run-01_attempt-01" / "0001.dcm"
        os.remove(dicom)  # e.g. "Cleanup Session" or a discarded attempt
        result = mirror_session(session, backup)

        assert (result.destination / "collection" / "task-rest_run-01_attempt-01" / "0001.dcm").exists()

    def test_unavailable_destination_fails_without_creating_it(self, tmp_path):
        session = _session(tmp_path)
        unplugged = tmp_path / "Volumes" / "BackupDrive"  # mount point of a drive that is not connected

        result = mirror_session(session, unplugged)

        assert not result.complete
        assert "not available" in result.errors[0]
        assert not unplugged.exists()

    def test_refuses_a_destination_inside_the_session(self, tmp_path):
        session = _session(tmp_path)
        nested = session / "feedback"  # <session>/feedback/sub-01/ses-01 would be inside the source

        result = mirror_session(session, nested)

        assert not result.complete
        assert result.copied == 0

    def test_stop_event_interrupts_between_files(self, tmp_path):
        session = _session(tmp_path)
        backup = tmp_path / "backup"
        backup.mkdir()
        stop = threading.Event()
        stop.set()

        result = mirror_session(session, backup, stop_event=stop)

        assert result.stopped and not result.complete
        assert result.copied == 0

    def test_manifest_records_the_pass(self, tmp_path):
        session = _session(tmp_path)
        backup = tmp_path / "backup"
        backup.mkdir()

        mirror_session(session, backup)
        manifest = read_manifest(session, backup)

        assert manifest is not None
        assert manifest["complete"] is True
        assert manifest["copied"] == 3
        assert read_manifest(session, tmp_path / "elsewhere") is None


class TestBackupConfig:
    def test_destination_layout(self):
        assert backup_destination(Path("/data/study/sub-01/ses-02"), Path("/backup")) == Path("/backup/sub-01/ses-02")

    def test_disabled_needs_nothing(self):
        assert validate_backup_config(dict(BACKUP_DEFAULTS), "/data/study") is None

    def test_enabled_requires_a_folder(self):
        error = validate_backup_config({"enabled": True, "destination": "  "}, "/data/study")
        assert error is not None and "no backup folder" in error

    def test_folder_cannot_be_the_data_root(self, tmp_path):
        error = validate_backup_config({"enabled": True, "destination": str(tmp_path)}, tmp_path)
        assert error is not None and "data directory" in error
