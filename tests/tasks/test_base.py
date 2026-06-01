"""Tests for pineuro.tasks.base — BaseTask thread-safe data, block accumulation, results.

Only the pure-Python surface is unit-tested here. Methods that touch PsychoPy
(setup, update, _run_slider, _show_post_run_questions, cleanup, run) are covered
by integration tests / live task runs.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def base_task(stub_task):
    """A real `BaseTask` via a concrete stub subclass (setup/update stubbed).

    See the shared ``stub_task`` factory in tests/tasks/conftest.py.
    """
    return stub_task()


# =============================================================================
# Thread-safe data access
# =============================================================================


class TestLatestData:
    def test_initial_state(self, base_task):
        data = base_task.get_latest_data()
        assert data["activations"] == {}
        assert data["volume"] == 0
        assert data["time"] == 0.0
        assert data["is_baseline"] is True

    def test_update_data_stores_all_activations(self, base_task):
        # Primary mask is the first one from coordinator.mask_names (ContA in fixture)
        base_task._update_data(
            activations={"ContA": 1.5, "DefaultA": -0.5},
            volume=10,
            elapsed=20.0,
            is_baseline=False,
            is_outlier=False,
        )
        data = base_task.get_latest_data()
        assert data["activations"] == {"ContA": 1.5, "DefaultA": -0.5}
        assert data["activations"][base_task.mask_name] == 1.5  # primary = ContA
        assert data["volume"] == 10
        assert data["time"] == 20.0
        assert data["is_baseline"] is False

    def test_get_latest_data_returns_copy(self, base_task):
        base_task._update_data({"ContA": 0.5}, volume=1, elapsed=1.0, is_baseline=True, is_outlier=False)
        d1 = base_task.get_latest_data()
        d1["volume"] = 999
        d2 = base_task.get_latest_data()
        assert d2["volume"] == 1  # mutation didn't leak

    def test_primary_falls_back_to_zero_when_mask_missing(self, base_task):
        # If the primary mask name isn't in the activations dict, callers fall back to 0.0.
        base_task._update_data({"OtherMask": 9.0}, volume=1, elapsed=1.0, is_baseline=False, is_outlier=False)
        assert base_task.get_latest_data()["activations"].get(base_task.mask_name, 0.0) == 0.0


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
# Task state recording (coordinator-sourced run fields)
# =============================================================================


class TestRunRecordFields:
    """Feedback tasks source run-completion coordinates from the coordinator."""

    def _stub_task(self, mock_coordinator, feedback_mode):
        from pineuro.tasks.base import BaseTask

        class _StubTask(BaseTask):
            def setup(self):  # pragma: no cover
                pass

            def update(self, data):  # pragma: no cover
                pass

        coord = mock_coordinator(run_number=2, attempt=3)
        return _StubTask(coordinator=coord, feedback_mode=feedback_mode)

    def test_feedback_run_type_and_coordinator_sourcing(self, mock_coordinator):
        fields = self._stub_task(mock_coordinator, feedback_mode=True)._run_record_fields()
        assert fields["run_type"] == "feedback"
        assert fields["attempt"] == 3  # coordinator.attempt
        assert fields["slot_run_number"] == 2  # coordinator.run_number

    def test_transfer_run_type_when_not_feedback(self, mock_coordinator):
        fields = self._stub_task(mock_coordinator, feedback_mode=False)._run_record_fields()
        assert fields["run_type"] == "transfer"


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
# Task-state recording
# =============================================================================


class TestRecordRunCompletion:
    def test_noop_without_task_state(self, base_task):
        # No task_state set → should silently do nothing
        base_task._record_run_completion()

    def test_records_with_task_state(self, base_task):
        session = MagicMock()
        base_task._task_state = session
        base_task.feedback_mode = True
        base_task._volumes_processed = 100
        base_task._task_state_snapshot = {"hits": 5}

        base_task._record_run_completion()

        session.record_run.assert_called_once_with(
            run_type="feedback",
            volumes_processed=100,
            attempt=1,
            slot_run_number=1,
            task_state_snapshot={"hits": 5},
            merge_into_task_state=True,
        )

    def test_records_transfer_when_not_feedback_mode(self, base_task):
        session = MagicMock()
        base_task._task_state = session
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


# =============================================================================
# Post-run question eligibility
# =============================================================================


class TestEligiblePostRunQuestions:
    """`_eligible_post_run_questions` / `has_pending_post_run_questions` mirror the
    per-question `condition` gating that decides which questions are displayed —
    and which drive whether a manual stop offers to show them.
    """

    def _task(self, mock_coordinator, *, questions, feedback_mode=True, is_last=False):
        from types import SimpleNamespace

        from pineuro.tasks.base import BaseTask

        class _StubTask(BaseTask):
            def setup(self):  # pragma: no cover
                pass

            def update(self, data):  # pragma: no cover
                pass

        task = _StubTask(coordinator=mock_coordinator(), feedback_mode=feedback_mode)
        # `questions=None` simulates an unregistered task (no metadata).
        metadata = None if questions is None else SimpleNamespace(post_run_questions=questions)
        task.get_task_metadata = lambda: metadata
        task._is_last_feedback_run = is_last
        return task

    def _q(self, condition):
        from pineuro.tasks.task_registry import PostRunQuestion

        return PostRunQuestion(text=f"q-{condition}", condition=condition)

    def test_no_metadata(self, mock_coordinator):
        task = self._task(mock_coordinator, questions=None)
        assert task._eligible_post_run_questions() == []
        assert task.has_pending_post_run_questions() is False

    def test_empty_question_list(self, mock_coordinator):
        task = self._task(mock_coordinator, questions=[])
        assert task._eligible_post_run_questions() == []
        assert task.has_pending_post_run_questions() is False

    @pytest.mark.parametrize("feedback_mode", [True, False])
    @pytest.mark.parametrize("is_last", [True, False])
    def test_always_shown_regardless_of_context(self, mock_coordinator, feedback_mode, is_last):
        q = self._q("always")
        task = self._task(mock_coordinator, questions=[q], feedback_mode=feedback_mode, is_last=is_last)
        assert task._eligible_post_run_questions() == [q]
        assert task.has_pending_post_run_questions() is True

    def test_feedback_only_shown_on_feedback_run(self, mock_coordinator):
        q = self._q("feedback_only")
        task = self._task(mock_coordinator, questions=[q], feedback_mode=True)
        assert task._eligible_post_run_questions() == [q]

    def test_feedback_only_hidden_on_transfer_run(self, mock_coordinator):
        q = self._q("feedback_only")
        task = self._task(mock_coordinator, questions=[q], feedback_mode=False)
        assert task._eligible_post_run_questions() == []
        assert task.has_pending_post_run_questions() is False

    @pytest.mark.parametrize(
        ("feedback_mode", "is_last", "expected"),
        [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
    )
    def test_last_feedback_only_needs_feedback_and_last(self, mock_coordinator, feedback_mode, is_last, expected):
        q = self._q("last_feedback_only")
        task = self._task(mock_coordinator, questions=[q], feedback_mode=feedback_mode, is_last=is_last)
        assert task.has_pending_post_run_questions() is expected

    def test_mixed_list_filters_to_eligible_subset(self, mock_coordinator):
        always, fb_only, last_only = self._q("always"), self._q("feedback_only"), self._q("last_feedback_only")
        # Feedback run that is NOT the last: always + feedback_only, but not last_feedback_only.
        task = self._task(
            mock_coordinator,
            questions=[always, fb_only, last_only],
            feedback_mode=True,
            is_last=False,
        )
        assert task._eligible_post_run_questions() == [always, fb_only]
