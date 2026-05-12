"""Tests for pineuro.tasks.fixation — registry registration and constructor."""


class TestFixationMetadata:
    def test_registered(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert "fixation" in tasks

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("fixation")
        assert meta.task_id == "fixation"
        assert "localizer" in meta.roles
        assert meta.n_networks == 0

    def test_has_duration_parameter(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("fixation")
        param_keys = [p.key for p in meta.parameters]
        assert "duration" in param_keys


class TestFixationConstructor:
    def test_duration_stored(self, tmp_path):
        from pineuro.tasks.fixation import FixationTask

        task = FixationTask(output_dir=tmp_path, participant_id="sub-01", run=1, duration=120.0)
        assert task.duration == 120.0

    def test_default_duration(self, tmp_path):
        from pineuro.tasks.fixation import FixationTask

        task = FixationTask(output_dir=tmp_path, participant_id="sub-01")
        # Default from FIXATION_METADATA is 200.0
        assert task.duration == 200.0

    def test_run_number_propagates(self, tmp_path):
        from pineuro.tasks.fixation import FixationTask

        task = FixationTask(output_dir=tmp_path, participant_id="sub-01", run=3)
        assert task.run_number == 3
