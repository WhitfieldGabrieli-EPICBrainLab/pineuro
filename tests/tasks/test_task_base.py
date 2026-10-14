"""Tests for pineuro.tasks.task_base — scanner trigger-key resolution and simulator shutdown.

`TaskBase` is the single place trigger keys are resolved: it applies the shared
default and normalizes the user-facing symbol ``=`` to PsychoPy's event name
``equal`` (what ``event.getKeys`` and the VK/keysym maps use). Leaf tasks no
longer declare ``trigger_keys`` — they inherit this via ``**kwargs`` through
``BaseTask``, which is the contract a third-party task plugin also relies on.
Unfocused key polling is covered in ``test_key_polling.py``.
"""

import threading


class TestTaskBaseTriggerKeys:
    """Default + ``=``→``equal`` normalization, resolved in TaskBase.__init__."""

    def test_default_uses_symbol_and_normalizes(self):
        from pineuro.tasks.task_base import TaskBase

        # Default is the user-facing ["t", "+", "5", "="]; "=" is normalized to "equal".
        assert TaskBase().trigger_keys == ["t", "+", "5", "equal"]

    def test_equals_symbol_normalized_to_psychopy_name(self):
        from pineuro.tasks.task_base import TaskBase

        assert TaskBase(trigger_keys=["="]).trigger_keys == ["equal"]
        assert TaskBase(trigger_keys=["t", "+", "5", "="]).trigger_keys == ["t", "+", "5", "equal"]

    def test_psychopy_name_passes_through(self):
        from pineuro.tasks.task_base import TaskBase

        # An un-migrated config that still carries "equal" keeps working.
        assert TaskBase(trigger_keys=["equal"]).trigger_keys == ["equal"]

    def test_empty_falls_back_to_default(self):
        from pineuro.tasks.task_base import TaskBase

        assert TaskBase(trigger_keys=[]).trigger_keys == ["t", "+", "5", "equal"]

    def test_other_keys_untouched(self):
        from pineuro.tasks.task_base import TaskBase

        assert TaskBase(trigger_keys=["5"]).trigger_keys == ["5"]


class TestTriggerKeysInheritedViaKwargs:
    """Leaf tasks don't declare ``trigger_keys``; it flows through ``**kwargs``.

    Mirrors the real construction path (factory injects ``trigger_keys`` as a
    runtime parameter -> task ``**kwargs`` -> ``BaseTask`` pops it -> ``TaskBase``).
    """

    def _stub_task_cls(self):
        from pineuro.tasks.base import BaseTask

        class _StubTask(BaseTask):
            def setup(self):  # pragma: no cover — never invoked here
                pass

            def update(self, data):  # pragma: no cover — never invoked here
                pass

        return _StubTask

    def test_symbol_flows_through_and_is_normalized(self, mock_coordinator):
        task = self._stub_task_cls()(coordinator=mock_coordinator(), trigger_keys=["="])
        assert task.trigger_keys == ["equal"]

    def test_default_applied_when_not_supplied(self, mock_coordinator):
        task = self._stub_task_cls()(coordinator=mock_coordinator())
        assert task.trigger_keys == ["t", "+", "5", "equal"]


class _TriggerWaitingSimulator:
    """Simulator stand-in that blocks, like a manual-trigger wait, until request_stop()."""

    clear_target = False

    def __init__(self):
        self._stop = threading.Event()
        self.trigger_event = None
        self.trigger_key = None
        self.start_time = None

    def copy_volumes_iter(self):
        while not self._stop.wait(0.01):
            pass
        yield from ()

    def request_stop(self):
        self._stop.set()


class TestSimulatorShutdown:
    """A stop before the scanner trigger used to leave the simulator thread waiting forever."""

    def test_stop_releases_a_simulator_waiting_for_its_trigger(self):
        from pineuro.tasks.task_base import TaskBase

        task = TaskBase(simulator=_TriggerWaitingSimulator())
        task._running = True
        task._start_simulator_thread()
        # Daemon: it can never keep the task subprocess alive after the task ends.
        assert task._simulator_thread.daemon

        task.stop()
        task._simulator_thread.join(timeout=2.0)

        assert not task._simulator_thread.is_alive()
