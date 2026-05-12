"""Tests for pineuro.tasks.base — BaseTask thread-safe data, block accumulation, results.

Only the pure-Python surface is unit-tested here. Methods that touch PsychoPy
(setup, update, _run_slider, _show_post_run_questions, cleanup, run) are covered
by integration tests / live task runs.
"""

from unittest.mock import MagicMock

import pytest


class _ConcreteTask:
    """Minimal concrete BaseTask subclass for tests (avoids abstract instantiation)."""


@pytest.fixture
def base_task(mock_coordinator):
    """Construct a real `BaseTask` via a concrete subclass with stub setup/update."""
    from pineuro.tasks.base import BaseTask

    class _StubTask(BaseTask):
        def setup(self):  # pragma: no cover — never invoked here
            pass

        def update(self, data):  # pragma: no cover — never invoked here
            pass

    return _StubTask(coordinator=mock_coordinator())


# =============================================================================
# Thread-safe data access
# =============================================================================


class TestLatestData:
    def test_initial_state(self, base_task):
        data = base_task.get_latest_data()
        assert data["activations"] == {}
        assert data["activation"] == 0.0
        assert data["volume"] == 0
        assert data["time"] == 0.0
        assert data["is_baseline"] is True

    def test_update_data_stores_primary_mask(self, base_task):
        # Primary mask is the first one from coordinator.mask_names (ContA in fixture)
        base_task._update_data(
            activations={"ContA": 1.5, "DefaultA": -0.5},
            volume=10,
            elapsed=20.0,
            is_baseline=False,
            is_outlier=False,
        )
        data = base_task.get_latest_data()
        assert data["activation"] == 1.5  # primary = ContA
        assert data["activations"] == {"ContA": 1.5, "DefaultA": -0.5}
        assert data["volume"] == 10
        assert data["time"] == 20.0
        assert data["is_baseline"] is False

    def test_get_latest_data_returns_copy(self, base_task):
        base_task._update_data({"ContA": 0.5}, volume=1, elapsed=1.0, is_baseline=True, is_outlier=False)
        d1 = base_task.get_latest_data()
        d1["activation"] = 999.0
        d2 = base_task.get_latest_data()
        assert d2["activation"] == 0.5  # mutation didn't leak

    def test_primary_falls_back_to_zero_when_mask_missing(self, base_task):
        # If the primary mask name isn't in the activations dict, fall back to 0.0.
        base_task._update_data({"OtherMask": 9.0}, volume=1, elapsed=1.0, is_baseline=False, is_outlier=False)
        assert base_task.get_latest_data()["activation"] == 0.0


# =============================================================================
# Block accumulation
# =============================================================================


class TestBlockAccumulation:
    def test_starts_disabled(self, base_task):
        assert base_task._accumulating is False
        assert base_task._block_activations == []

    def test_start_resets_and_enables(self, base_task):
        base_task._block_activations = [1.0, 2.0]
        base_task._start_block_accumulation()
        assert base_task._accumulating is True
        assert base_task._block_activations == []

    def test_stop_returns_copy(self, base_task):
        base_task._start_block_accumulation()
        base_task._block_activations.extend([0.1, 0.2, 0.3])
        result = base_task._stop_block_accumulation()
        assert result == [0.1, 0.2, 0.3]
        assert base_task._accumulating is False
        # Mutating the returned copy must not affect the internal buffer
        result.append(99.0)
        assert base_task._block_activations == [0.1, 0.2, 0.3]


# =============================================================================
# Mirror state + run results
# =============================================================================


class TestMirrorState:
    def test_default_is_empty(self, base_task):
        state = base_task.get_mirror_state()
        assert state == {"background": "black", "shapes": []}

    def test_emit_calls_callback(self, base_task):
        cb = MagicMock()
        base_task.set_mirror_callback(cb)
        base_task._emit_mirror_state()
        cb.assert_called_once()
        assert cb.call_args[0][0]["background"] == "black"

    def test_emit_swallows_callback_errors(self, base_task):
        base_task.set_mirror_callback(MagicMock(side_effect=RuntimeError("boom")))
        # Should not raise — failure is logged, not propagated.
        base_task._emit_mirror_state()


class TestRunResults:
    def test_default_uses_current_volume(self, base_task):
        base_task.current_volume = 42
        assert base_task.get_run_results() == {"volumes_processed": 42}

    def test_default_falls_back_to_underscore_attr(self, base_task):
        # No current_volume set; _volumes_processed used as fallback
        base_task._volumes_processed = 17
        assert base_task.get_run_results() == {"volumes_processed": 17}


# =============================================================================
# Session-state recording
# =============================================================================


class TestRecordRunCompletion:
    def test_noop_without_session_state(self, base_task):
        # No session_state set → should silently do nothing
        base_task._record_run_completion()

    def test_records_with_session_state(self, base_task):
        session = MagicMock()
        base_task._session_state = session
        base_task.feedback_mode = True
        base_task._volumes_processed = 100
        base_task._task_state_snapshot = {"hits": 5}

        base_task._record_run_completion()

        session.record_run.assert_called_once_with(
            run_type="feedback",
            volumes_processed=100,
            task_state_snapshot={"hits": 5},
        )

    def test_records_transfer_when_not_feedback_mode(self, base_task):
        session = MagicMock()
        base_task._session_state = session
        base_task.feedback_mode = False
        base_task._record_run_completion()
        assert session.record_run.call_args.kwargs["run_type"] == "transfer"


# =============================================================================
# Metadata + trigger
# =============================================================================


class TestTaskMetadataAccessor:
    def test_returns_none_for_unregistered_task_id(self, base_task):
        base_task.task_id = "definitely_not_a_real_task_xyz"
        assert base_task.get_task_metadata() is None


class TestOnTrigger:
    def test_forwards_to_coordinator(self, base_task):
        base_task._on_trigger()
        base_task.coordinator.send_trigger.assert_called_once()
