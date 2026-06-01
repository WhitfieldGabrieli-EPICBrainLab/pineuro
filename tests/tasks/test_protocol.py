"""Tests for pineuro.tasks.protocol — CoordinatorProtocol and TaskProtocol surface.

Only the protocol *surface* is asserted here (the members consumers rely on). The
``isinstance(x, Protocol)`` runtime check is not tested: ``@runtime_checkable`` only
verifies member presence, and a ``MagicMock`` auto-creates every attribute, so the
check is satisfied by construction and proves nothing.
"""


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
