"""Tests for pineuro.tasks.task_registry — registry, TaskMetadata, validate_config."""

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
# TaskConfigError and TaskMetadata.validate_config
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
# TaskMetadata.compute_max_run_seconds field
# =============================================================================


class TestTaskMetadataField:
    """Tests for the generic TaskMetadata.compute_max_run_seconds field."""

    def test_compute_max_run_seconds_defaults_to_none(self):
        from pineuro.tasks.task_registry import TaskMetadata

        meta = TaskMetadata(
            task_id="dummy",
            display_name="Dummy",
            description="Test task without duration calculator",
            parameters=[],
        )
        assert meta.compute_max_run_seconds is None

    def test_rest_task_has_no_duration_calculator(self):
        """Fixed-duration tasks (like rest) don't set the callable."""
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rest")
        assert meta.compute_max_run_seconds is None
