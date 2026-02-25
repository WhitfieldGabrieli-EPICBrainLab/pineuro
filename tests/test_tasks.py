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

    def test_unknown_config_keys_logged(self, caplog):
        import logging
        from unittest.mock import MagicMock

        from pineuro.tasks import factory as factory_mod
        from pineuro.tasks.factory import get_registered_task_ids

        ids = get_registered_task_ids()
        if not ids:
            pytest.skip("No tasks registered")

        task_id = ids[0]

        # Substitute a mock task class so we don't need PsychoPy running
        original_classes = dict(factory_mod._TASK_CLASSES)
        factory_mod._TASK_CLASSES[task_id] = MagicMock(return_value=MagicMock())

        mock_coordinator = MagicMock()
        mock_coordinator.mask_names = ["ContA", "DefaultA"]

        try:
            with caplog.at_level(logging.WARNING, logger="pineuro.tasks.factory"):
                factory_mod.create_task(
                    task_id=task_id,
                    coordinator=mock_coordinator,
                    config={"mask_names": ["ContA", "DefaultA"], "typo_key_xyz": 999},
                )
            assert any("typo_key_xyz" in msg for msg in caplog.messages)
        finally:
            factory_mod._TASK_CLASSES.clear()
            factory_mod._TASK_CLASSES.update(original_classes)


# =============================================================================
# TaskConfigError and validate_config
# =============================================================================


class TestTaskConfigError:
    def _make_metadata(self):
        from pineuro.tasks.task_registry import TaskMetadata, TaskParameter

        return TaskMetadata(
            task_id="test_task",
            display_name="Test Task",
            description="A test task",
            parameters=[
                TaskParameter(
                    key="alpha",
                    label="Alpha",
                    type="float",
                    default=1.0,
                    min_value=0.0,
                    max_value=10.0,
                ),
                TaskParameter(
                    key="count",
                    label="Count",
                    type="int",
                    default=5,
                    min_value=1,
                ),
            ],
        )

    def test_valid_config_does_not_raise(self):
        meta = self._make_metadata()
        meta.validate_config({"alpha": 5.0, "count": 3})  # should not raise

    def test_missing_keys_skipped(self):
        meta = self._make_metadata()
        meta.validate_config({})  # no keys present → nothing to validate

    def test_single_invalid_param_raises(self):
        from pineuro.tasks.task_registry import TaskConfigError

        meta = self._make_metadata()
        with pytest.raises(TaskConfigError) as exc_info:
            meta.validate_config({"alpha": -1.0})
        assert "alpha" in str(exc_info.value).lower() or len(exc_info.value.errors) == 1

    def test_multiple_invalid_params_collected(self):
        from pineuro.tasks.task_registry import TaskConfigError

        meta = self._make_metadata()
        with pytest.raises(TaskConfigError) as exc_info:
            meta.validate_config({"alpha": -1.0, "count": -5})
        assert len(exc_info.value.errors) == 2

    def test_error_includes_task_id(self):
        from pineuro.tasks.task_registry import TaskConfigError

        meta = self._make_metadata()
        with pytest.raises(TaskConfigError) as exc_info:
            meta.validate_config({"alpha": -999.0})
        assert exc_info.value.task_id == "test_task"

    def test_error_message_lists_problems(self):
        from pineuro.tasks.task_registry import TaskConfigError

        meta = self._make_metadata()
        with pytest.raises(TaskConfigError) as exc_info:
            meta.validate_config({"alpha": -1.0, "count": -5})
        msg = str(exc_info.value)
        assert "2 error" in msg

    def test_cross_parameter_validator_called(self):
        from pineuro.tasks.task_registry import TaskConfigError, TaskMetadata, TaskParameter

        def cross_check(config):
            if config.get("alpha", 0) >= config.get("count", 0):
                raise ValueError("alpha must be less than count")

        meta = TaskMetadata(
            task_id="cross_task",
            display_name="Cross Task",
            description="Tests cross-param validation",
            parameters=[
                TaskParameter(key="alpha", label="Alpha", type="float", default=1.0),
                TaskParameter(key="count", label="Count", type="int", default=5),
            ],
            cross_parameter_validator=cross_check,
        )

        with pytest.raises(TaskConfigError):
            meta.validate_config({"alpha": 10.0, "count": 2})

    def test_choices_validation(self):
        from pineuro.tasks.task_registry import TaskConfigError, TaskMetadata, TaskParameter

        meta = TaskMetadata(
            task_id="choice_task",
            display_name="Choice Task",
            description="Tests choices validation",
            parameters=[
                TaskParameter(
                    key="mode",
                    label="Mode",
                    type="str",
                    default="fast",
                    choices=["fast", "slow"],
                ),
            ],
        )

        # Valid choice
        meta.validate_config({"mode": "fast"})

        # Invalid choice
        with pytest.raises(TaskConfigError):
            meta.validate_config({"mode": "invalid_mode"})


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
