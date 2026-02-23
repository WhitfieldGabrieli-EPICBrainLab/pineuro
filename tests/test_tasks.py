"""Tests for pineuro.tasks — plugin registry, factory, designs, session state."""

import numpy as np
import pytest

# =============================================================================
# Task registry
# =============================================================================


class TestTaskRegistry:
    def test_list_available_tasks(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert isinstance(tasks, dict)
        # Built-in tasks should be registered
        assert len(tasks) > 0

    def test_builtin_feedback_tasks(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks(role="feedback")
        task_ids = set(tasks.keys())
        # At minimum, ball and thermometer should exist
        assert "ball" in task_ids or "gauge" in task_ids

    def test_get_task_metadata(self):
        from pineuro.tasks.task_registry import get_task_metadata, list_available_tasks

        tasks = list_available_tasks()
        if tasks:
            first_id = next(iter(tasks))
            meta = get_task_metadata(first_id)
            assert meta.task_id == first_id
            assert isinstance(meta.display_name, str)

    def test_get_nonexistent_task_raises(self):
        from pineuro.tasks.task_registry import get_task_metadata

        with pytest.raises((KeyError, ValueError)):
            get_task_metadata("nonexistent_task_xyz")


# =============================================================================
# Task factory
# =============================================================================


class TestTaskFactory:
    def test_get_registered_task_ids(self):
        from pineuro.tasks.factory import get_registered_task_ids

        ids = get_registered_task_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0


# =============================================================================
# BlockDesign
# =============================================================================


class TestBlockDesign:
    def test_compute_regressors_shape(self):
        from pineuro.tasks.designs import BlockDesign

        design = BlockDesign(n_baseline=10)
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        assert regressors.shape == (50, 1)

    def test_baseline_is_zero(self):
        from pineuro.tasks.designs import BlockDesign

        design = BlockDesign(n_baseline=10)
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        # First few volumes should be zero (baseline)
        np.testing.assert_allclose(regressors[:8, 0], 0.0, atol=1e-6)

    def test_active_is_nonzero(self):
        from pineuro.tasks.designs import BlockDesign

        design = BlockDesign(n_baseline=10)
        regressors = design.compute_regressors(n_volumes=100, tr=1.0)

        # Well after baseline + HRF delay, regressor should be non-zero
        assert np.any(regressors[30:, 0] > 0.1)


# =============================================================================
# NuisanceOnlyDesign
# =============================================================================


class TestNuisanceOnlyDesign:
    def test_empty_regressors(self):
        from pineuro.tasks.designs import NuisanceOnlyDesign

        design = NuisanceOnlyDesign()
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        assert regressors.shape == (50, 0)

    def test_n_regressors_is_zero(self):
        from pineuro.tasks.designs import NuisanceOnlyDesign

        design = NuisanceOnlyDesign()
        assert design.n_regressors == 0


# =============================================================================
# EventRelatedDesign
# =============================================================================


class TestEventRelatedDesign:
    def test_compute_regressors_shape(self):
        from pineuro.tasks.designs import EventRelatedDesign

        design = EventRelatedDesign(
            event_onsets={"face": [10.0, 30.0, 50.0], "house": [20.0, 40.0, 60.0]},
            event_durations={"face": 5.0, "house": 5.0},
        )
        regressors = design.compute_regressors(n_volumes=50, tr=2.0)

        assert regressors.shape[0] == 50
        assert regressors.shape[1] == 2  # 2 conditions


# =============================================================================
# SessionState
# =============================================================================


class TestSessionState:
    def test_create_for_task(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        assert state.task_id == "ball"
        assert state.next_run_number == 1
        assert state.feedback_run_count == 0

    def test_save_and_load(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        state.set("custom_key", 42)
        state.save()

        loaded = SessionState.load(tmp_path, "ball")
        assert loaded.get("custom_key") == 42

    def test_record_run(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        state.record_run(run_type="feedback", volumes_processed=100)

        assert state.feedback_run_count == 1
        assert state.next_run_number == 2
        assert state.completed_runs[0].run_type == "feedback"
        assert state.completed_runs[0].volumes_processed == 100

    def test_reset(self, tmp_path):
        from pineuro.tasks.session import SessionState

        state = SessionState.create_for_task(tmp_path, "ball")
        state.record_run(run_type="feedback", volumes_processed=50)
        state.save()

        reset = SessionState.reset(tmp_path, "ball")
        assert reset.feedback_run_count == 0
        assert reset.next_run_number == 1


# =============================================================================
# RunInfo
# =============================================================================


class TestRunInfo:
    def test_construction(self):
        from pineuro.tasks.session import RunInfo

        info = RunInfo(run_number=1, run_type="feedback", volumes_processed=200)
        assert info.run_number == 1
        assert info.run_type == "feedback"

    def test_to_dict_roundtrip(self):
        from pineuro.tasks.session import RunInfo

        info = RunInfo(
            run_number=2,
            run_type="transfer",
            volumes_processed=150,
            task_state_snapshot={"score": 0.75},
        )
        d = info.to_dict()
        restored = RunInfo.from_dict(d)

        assert restored.run_number == 2
        assert restored.run_type == "transfer"
        assert restored.task_state_snapshot["score"] == 0.75
