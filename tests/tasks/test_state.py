"""Tests for pineuro.tasks.state -- TaskState persistence error handling."""

import logging
from unittest.mock import patch

import pytest

from pineuro.tasks.state import TaskState


def test_save_logs_context_and_reraises_on_oserror(tmp_path, caplog):
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
