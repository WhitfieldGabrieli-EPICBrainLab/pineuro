"""Shared fixtures for pineuro.tasks tests.

Provides a `mock_coordinator` fixture that satisfies `CoordinatorProtocol`
without requiring the heavy `NeurofeedbackCoordinator` to be instantiated,
and a `task_task_state` factory for tests that need a saved TaskState.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_coordinator():
    """Factory fixture: MagicMock that satisfies `CoordinatorProtocol`.

    Defaults can be overridden by passing keyword arguments to the factory.

    Usage::

        def test_something(mock_coordinator):
            coord = mock_coordinator()                       # defaults
            coord = mock_coordinator(n_volumes=200,
                                     mask_names=["ContA"])   # overrides
    """

    def _make(**overrides):
        from pineuro.tasks.protocol import CoordinatorProtocol

        defaults = {
            "subject": "sub-01",
            "task": "ball",
            "run_number": 1,
            "attempt": 1,
            "n_volumes": 100,
            "mask_names": ["ContA", "DefaultA"],
        }
        defaults.update(overrides)

        coord = MagicMock(spec=CoordinatorProtocol)
        for attr, value in defaults.items():
            setattr(coord, attr, value)
        coord.is_complete.return_value = False
        coord.was_stopped.return_value = False
        coord.should_show_questions.return_value = True
        return coord

    return _make


@pytest.fixture
def task_task_state(tmp_path):
    """Factory fixture: create a `TaskState` rooted at a temp directory.

    Usage::

        def test_something(task_task_state):
            state = task_task_state("ball")
            state.record_run(run_type="feedback", volumes_processed=100)
    """

    def _make(task_id="ball", base_dir=None):
        from pineuro.tasks.state import TaskState

        return TaskState.create_for_task(base_dir or tmp_path, task_id)

    return _make


@pytest.fixture
def stub_task(mock_coordinator):
    """Factory: a minimal concrete ``BaseTask`` (``setup``/``update`` stubbed).

    ``BaseTask`` is abstract; this is the smallest concrete subclass, so tests can
    exercise its pure-Python surface (thread-safe data, block accumulation, run
    records, post-run-question gating, trigger-key normalization) without PsychoPy.
    Pass a ``coordinator`` (defaults to ``mock_coordinator()``) and any ``BaseTask``
    kwargs (``feedback_mode``, ``trigger_keys``, ...).

    Usage::

        def test_x(stub_task):
            task = stub_task(feedback_mode=True)
    """

    def _make(*, coordinator=None, **kwargs):
        from pineuro.tasks.base import BaseTask

        class _StubTask(BaseTask):
            def setup(self):  # pragma: no cover -- never invoked in unit tests
                pass

            def update(self, data):  # pragma: no cover -- never invoked in unit tests
                pass

        return _StubTask(coordinator=coordinator if coordinator is not None else mock_coordinator(), **kwargs)

    return _make


@pytest.fixture
def write_tsv():
    """Factory: write a TSV from a header row + data rows; returns the Path."""

    def _make(path, header, rows):
        lines = ["\t".join(header)] + ["\t".join(str(c) for c in row) for row in rows]
        path.write_text("\n".join(lines) + "\n")
        return path

    return _make
