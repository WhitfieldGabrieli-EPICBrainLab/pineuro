"""Tests for pineuro.tasks.subprocess_runner.

The child entrypoint is exercised IN-PROCESS (not spawned) with stubbed
coordinator/task/factory so we can assert the IPC message protocol, the stop
bridge, and the mirror forwarder without a real PsychoPy window or scanner.
The one exception spawns a real child: the stop-signal deadlock it guards
against needs a waiter inside a process that has already exited.
"""

import multiprocessing
import pickle
import queue
import threading
import time

from pineuro.tasks import subprocess_runner as sr
from pineuro.tasks.subprocess_runner import (
    FeedbackStepPayload,
    LocalizerStepPayload,
    _MirrorForwarder,
    _StopBridge,
    run_step_in_subprocess,
)

# ---------------------------------------------------------------------------
# Payloads must be picklable (spawn pickles them into the child).
# ---------------------------------------------------------------------------


def test_feedback_payload_is_picklable():
    payload = FeedbackStepPayload(
        kind="feedback",
        feedback_kwargs={"mask_paths": ["/m.nii.gz"], "n_volumes": 150, "tr": 1.2},
        factory_config={"mask_names": ["net"], "feedback_mode": True},
        task_id="ball",
        output_dir="/out",
        feedback_mode=True,
        is_last_feedback=False,
        max_feedback_runs=2,
        sham_replay=None,
    )
    assert pickle.loads(pickle.dumps(payload)) == payload


def test_localizer_payload_is_picklable():
    payload = LocalizerStepPayload(
        kind="localizer",
        task_id="rest",
        output_dir="/out/preprocessing",
        participant_id="sub-01",
        session_id="ses-01",
        config={},
        task_state_dir="/out/preprocessing/task_state_rest",
        run=1,
        attempt=1,
        watch_dir="/tmp/incoming",
        monitor_position="right",
        trigger_keys=["t", "+", "5", "="],
        simulator_source_dir=None,
        simulator_target_dir="/tmp/incoming",
        simulator_tr=1.2,
        simulator_speed=1.0,
        simulator_trigger_mode="manual",
    )
    assert pickle.loads(pickle.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# Stubs for the child entrypoint.
# ---------------------------------------------------------------------------


class _StubCoordinator:
    def __init__(self):
        self.n_processed = 5
        self._error = None
        self._activation_callback = None
        self.stopped_with = None

    def check_error(self):
        pass

    def was_stopped(self):
        return False

    def stop(self, show_questions=False):
        self.stopped_with = show_questions


class _StubThread:
    def join(self):
        pass


class _StubTask:
    def __init__(self, *, results=None, post_run_questions=True):
        self._results = results if results is not None else {"volumes_processed": 5, "n_episodes": 2}
        self._pq = post_run_questions
        self.stopped = False

    def _update_data(self, *a, **k):
        pass

    def has_pending_post_run_questions(self):
        return self._pq

    def set_mirror_callback(self, cb):
        # Store only — emitting here would race async MSG_MIRROR against MSG_DONE.
        self._mirror_cb = cb

    def run(self):
        pass

    def get_run_results(self):
        return self._results

    def stop(self):
        self.stopped = True


class _StubTaskState:
    max_feedback_runs = 2

    @staticmethod
    def load(*a, **k):
        return _StubTaskState()

    def save(self):
        pass


def _feedback_payload():
    return FeedbackStepPayload(
        kind="feedback",
        feedback_kwargs={},
        factory_config={},
        task_id="ball",
        output_dir="/out",
        feedback_mode=True,
        is_last_feedback=True,
        max_feedback_runs=2,
        sham_replay=None,
    )


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class _Value:
    """Stand-in for a multiprocessing Value('b')."""

    def __init__(self, value=False):
        self.value = value


# ---------------------------------------------------------------------------
# Child entrypoint message protocol.
# ---------------------------------------------------------------------------


def test_run_step_feedback_emits_result_and_done(monkeypatch):
    coord = _StubCoordinator()
    monkeypatch.setattr(
        "pineuro.feedback.run_feedback_task",
        lambda **kw: {"coordinator": coord, "thread": _StubThread(), "simulator": None},
    )
    monkeypatch.setattr("pineuro.tasks.create_task", lambda **kw: _StubTask())
    monkeypatch.setattr("pineuro.tasks.state.TaskState", _StubTaskState)

    q: queue.Queue = queue.Queue()
    run_step_in_subprocess(_feedback_payload(), q, threading.Event(), _Value())

    tags = [m[0] for m in _drain(q)]
    assert sr.MSG_POST_RUN_QUESTIONS in tags
    assert sr.MSG_RESULT in tags
    assert tags[-1] == sr.MSG_DONE
    # Result must be enqueued before DONE so the parent has it when DONE arrives.
    assert tags.index(sr.MSG_RESULT) < tags.index(sr.MSG_DONE)


def test_run_step_feedback_forwards_exception_as_error(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr("pineuro.feedback.run_feedback_task", _boom)

    q: queue.Queue = queue.Queue()
    run_step_in_subprocess(_feedback_payload(), q, threading.Event(), _Value())

    msgs = _drain(q)
    errors = [m for m in msgs if m[0] == sr.MSG_ERROR]
    assert errors, "expected an error message"
    _, exc_type, tb = errors[0]
    assert exc_type == "RuntimeError"
    assert "scanner exploded" in tb
    assert msgs[-1][0] == sr.MSG_DONE


def test_run_step_feedback_zero_volume_guard(monkeypatch):
    coord = _StubCoordinator()
    monkeypatch.setattr(
        "pineuro.feedback.run_feedback_task",
        lambda **kw: {"coordinator": coord, "thread": _StubThread(), "simulator": None},
    )
    # Task reports zero processed volumes and the coordinator was NOT stopped -> guard fires.
    monkeypatch.setattr(
        "pineuro.tasks.create_task",
        lambda **kw: _StubTask(results={"volumes_processed": 0}),
    )
    monkeypatch.setattr("pineuro.tasks.state.TaskState", _StubTaskState)

    q: queue.Queue = queue.Queue()
    run_step_in_subprocess(_feedback_payload(), q, threading.Event(), _Value())

    errors = [m for m in _drain(q) if m[0] == sr.MSG_ERROR]
    assert errors and "0 volumes" in errors[0][2]


# ---------------------------------------------------------------------------
# Stop bridge.
# ---------------------------------------------------------------------------


def test_stop_bridge_stops_coordinator_and_task():
    stop_event = threading.Event()
    bridge = _StopBridge(stop_event, _Value(True))
    coord = _StubCoordinator()
    task = _StubTask()
    bridge.bind(coordinator=coord, task=task)

    stop_event.set()
    deadline = time.time() + 2.0
    while time.time() < deadline and (coord.stopped_with is None or not task.stopped):
        time.sleep(0.01)

    assert coord.stopped_with is True  # show_questions value propagated
    assert task.stopped is True


def test_stop_bridge_acts_on_shared_flag():
    flag = sr.SharedFlag(multiprocessing.get_context("spawn"))
    bridge = _StopBridge(flag, _Value(False))
    task = _StubTask()
    bridge.bind(task=task)

    flag.set()
    deadline = time.time() + 2.0
    while time.time() < deadline and not task.stopped:
        time.sleep(0.01)

    assert task.stopped is True


def test_stop_bridge_close_ends_polling_without_acting():
    flag = sr.SharedFlag(multiprocessing.get_context("spawn"))
    bridge = _StopBridge(flag, _Value(False))
    task = _StubTask()
    bridge.bind(task=task)

    bridge.close()
    bridge._thread.join(timeout=2.0)

    assert not bridge._thread.is_alive()
    flag.set()  # a stop after the step finished has nothing to act on
    time.sleep(0.1)
    assert task.stopped is False


def test_stop_bridge_close_releases_task_on_caller_thread():
    """Regression: the bridge thread dropped the last task reference at exit.

    The task's PsychoPy stimuli were then finalized on that thread, where
    ``TextStim.__del__`` calls ``glDeleteLists`` with no GL context (a macOS
    segfault at the end of RIT). close() must free the task on its caller.
    """
    freed_on: list[int] = []

    class _TrackedTask(_StubTask):
        def __del__(self):
            freed_on.append(threading.get_ident())

    flag = sr.SharedFlag(multiprocessing.get_context("spawn"))
    bridge = _StopBridge(flag, _Value(False))
    bridge.bind(task=_TrackedTask())  # the bridge holds the only reference

    bridge.close()

    assert not bridge._thread.is_alive()
    assert bridge._task is None and bridge._coordinator is None
    assert freed_on == [threading.get_ident()]


def test_stop_bridge_close_releases_stop_waiting_for_bind():
    flag = sr.SharedFlag(multiprocessing.get_context("spawn"))
    bridge = _StopBridge(flag, _Value(False))
    flag.set()  # a stop that arrives before bind()
    time.sleep(0.1)

    start = time.monotonic()
    bridge.close()

    assert time.monotonic() - start < 1.0
    assert not bridge._thread.is_alive()


def test_parent_stop_signal_does_not_block_after_child_exits():
    """Regression: the parent's stop signal hung forever once the child had exited.

    With a multiprocessing.Event, the child's stop-bridge thread registered as a
    waiter for the whole run; after a localizer child ended on its own, the
    parent's Event.set() waited forever for that dead waiter, so the step never
    finished and the GUI had to be restarted.
    """
    ctx = multiprocessing.get_context("spawn")
    flag = sr.SharedFlag(ctx)
    child_code = (
        "import time\n"
        "from pineuro.tasks.subprocess_runner import _StopBridge\n"
        "_StopBridge(flag, None)\n"  # polls the flag on a daemon thread, as in a real task child
        "time.sleep(0.3)\n"
    )
    child = ctx.Process(target=exec, args=(child_code, {"flag": flag}))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 0

    setter = threading.Thread(target=flag.set, daemon=True)
    setter.start()
    setter.join(timeout=2.0)

    assert not setter.is_alive(), "setting the stop flag blocked after the child exited"
    assert flag.is_set()


# ---------------------------------------------------------------------------
# Mirror forwarder: latest-wins, never blocks/raises when the slot is full.
# ---------------------------------------------------------------------------


def test_mirror_forwarder_submit_never_raises_when_flooded():
    out: queue.Queue = queue.Queue()
    fwd = _MirrorForwarder(out.put)
    # Flood far faster than the forwarder can drain; submit must never raise.
    for i in range(500):
        fwd.submit({"frame": i})
    fwd.close()
    time.sleep(0.05)
    # The forwarder delivered *some* frames (coalesced) without deadlocking.
    delivered = _drain(out)
    assert all(isinstance(s, dict) for s in delivered)
