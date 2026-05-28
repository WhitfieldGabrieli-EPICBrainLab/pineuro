"""Tests for pineuro.tasks.rest — registry registration and constructor."""


class TestRestMetadata:
    def test_registered(self):
        from pineuro.tasks.task_registry import list_available_tasks

        tasks = list_available_tasks()
        assert "rest" in tasks

    def test_metadata_fields(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rest")
        assert meta.task_id == "rest"
        assert "localizer" in meta.roles
        assert meta.n_networks == 0

    def test_has_duration_parameter(self):
        from pineuro.tasks.task_registry import get_task_metadata

        meta = get_task_metadata("rest")
        param_keys = [p.key for p in meta.parameters]
        assert "duration" in param_keys


class TestRestConstructor:
    def test_duration_stored(self, tmp_path):
        from pineuro.tasks.rest import RestTask

        task = RestTask(
            output_dir=tmp_path, participant_id="sub-01", session_id="ses-01", attempt=1, run=1, duration=120.0
        )
        assert task.duration == 120.0

    def test_default_duration(self, tmp_path):
        from pineuro.tasks.rest import RestTask

        task = RestTask(output_dir=tmp_path, participant_id="sub-01", session_id="ses-01", attempt=1)
        # Default from REST_METADATA is 200.0
        assert task.duration == 200.0

    def test_run_number_propagates(self, tmp_path):
        from pineuro.tasks.rest import RestTask

        task = RestTask(output_dir=tmp_path, participant_id="sub-01", session_id="ses-01", attempt=1, run=3)
        assert task.run_number == 3
