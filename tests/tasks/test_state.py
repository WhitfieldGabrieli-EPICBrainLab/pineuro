"""Tests for pineuro.tasks.state — TaskState (JSON-backed) and RunInfo.

Consolidates what used to be split between this file (the save-error path) and the
misnamed test_session.py (there is no tasks/session module — those tests covered
tasks.state). The thorough field-by-field round-trip lives in the validation suite
(tests/validation/test_task_state_persistence.py); the create/record/reset behavior
and the OSError handling below are the default-suite coverage.
"""

import logging
from unittest.mock import patch

import pytest

from pineuro.tasks.state import RunInfo, TaskState


class TestTaskState:
    def test_create_for_task(self, tmp_path):
        state = TaskState.create_for_task(tmp_path, "ball")
        assert state.task_id == "ball"
        assert state.next_run_number == 1
        assert state.feedback_run_count == 0

    def test_save_and_load(self, tmp_path):
        state = TaskState.create_for_task(tmp_path, "ball")
        state.set("custom_key", 42)
        state.save()

        loaded = TaskState.load(tmp_path, "ball")
        assert loaded.get("custom_key") == 42

    def test_record_run(self, tmp_path):
        state = TaskState.create_for_task(tmp_path, "ball")
        state.record_run(run_type="feedback", volumes_processed=100, attempt=1)

        assert state.feedback_run_count == 1
        assert state.next_run_number == 2
        assert state.completed_runs[0].run_type == "feedback"
        assert state.completed_runs[0].volumes_processed == 100

    def test_reset(self, tmp_path):
        state = TaskState.create_for_task(tmp_path, "ball")
        state.record_run(run_type="feedback", volumes_processed=50, attempt=1)
        state.save()

        reset = TaskState.reset(tmp_path, "ball")
        assert reset.feedback_run_count == 0
        assert reset.next_run_number == 1

    def test_save_logs_context_and_reraises_on_oserror(self, tmp_path, caplog):
        """A failed write surfaces task_id + path and propagates -- state is never lost silently."""
        state = TaskState(
            state_dir=tmp_path,
            task_id="_unit_test_unregistered_task",
            task_state={},
            completed_runs=[],
            max_feedback_runs=5,
        )

        with (
            patch("pineuro.tasks.state.open", side_effect=OSError("disk full")),
            caplog.at_level(logging.ERROR, logger="pineuro.tasks.state"),
            pytest.raises(OSError, match="disk full"),
        ):
            state.save()

        assert "_unit_test_unregistered_task" in caplog.text
        assert "Failed to save task state" in caplog.text


class TestRunInfo:
    def test_to_dict_roundtrip(self):
        info = RunInfo(
            run_number=2,
            run_type="transfer",
            volumes_processed=150,
            attempt=1,
            task_state_snapshot={"score": 0.75},
        )
        restored = RunInfo.from_dict(info.to_dict())

        assert restored.run_number == 2
        assert restored.run_type == "transfer"
        assert restored.attempt == 1
        assert restored.task_state_snapshot["score"] == 0.75
