"""Tests for pineuro.tasks.protocol — CoordinatorProtocol and TaskProtocol runtime checks."""

from unittest.mock import MagicMock


class TestProtocols:
    def test_coordinator_protocol_importable(self):
        from pineuro.tasks.protocol import CoordinatorProtocol

        # Annotated attributes live in __annotations__, not as class attrs
        annotations = CoordinatorProtocol.__annotations__
        assert "subject" in annotations
        assert "n_volumes" in annotations

    def test_task_protocol_importable(self):
        from pineuro.tasks.protocol import TaskProtocol

        # Methods are defined on Protocol class, accessible via hasattr
        assert hasattr(TaskProtocol, "setup")
        assert hasattr(TaskProtocol, "update")
        assert hasattr(TaskProtocol, "run")
        assert hasattr(TaskProtocol, "cleanup")

    def test_coordinator_protocol_is_runtime_checkable(self):
        from pineuro.tasks.protocol import CoordinatorProtocol

        mock = MagicMock()
        mock.subject = "sub-01"
        mock.task = "ball"
        mock.run_number = 1
        mock.acq = None
        mock.n_volumes = 100
        mock.mask_names = ["DMN"]
        mock.is_complete = MagicMock(return_value=False)
        mock.check_error = MagicMock()
        mock.was_stopped = MagicMock(return_value=False)
        mock.should_show_questions = MagicMock(return_value=True)
        mock.send_trigger = MagicMock()

        assert isinstance(mock, CoordinatorProtocol)

    def test_task_protocol_is_runtime_checkable(self):
        from pineuro.tasks.protocol import TaskProtocol

        mock = MagicMock()
        mock.setup = MagicMock()
        mock.update = MagicMock()
        mock.run = MagicMock()
        mock.cleanup = MagicMock()
        mock.get_run_results = MagicMock(return_value={})
        mock.get_mirror_state = MagicMock(return_value={})
        mock.get_volume_log_suffix = MagicMock(return_value="")

        assert isinstance(mock, TaskProtocol)
