"""Run a single PsychoPy task step (localizer or feedback) in a fresh child process.

Every task step that opens a PsychoPy window is executed in its own spawned
process so that pyglet / OpenGL / Cocoa global state can never leak from one
window to the next. Re-creating a PsychoPy window in a long-lived process is the
root cause of the macOS ``ObjCInstance ... initWithWindow_`` crash and the
Windows "second run crashes" failures; a one-window-per-process model makes the
whole class of stale-graphics-state bugs impossible (the OS reclaims everything
on child exit).

The child rebuilds the coordinator and task from a **picklable** payload using
the existing high-level APIs (:func:`pineuro.feedback.run_feedback_task`,
:func:`pineuro.tasks.create_task` / :func:`create_localizer_task`) — the same
config-replay pattern used by the mask-extraction scripts — and streams progress,
mirror, log, result and error messages back to the parent over a queue.

The parent side (the spawn, the message-drain thread, stop signalling) lives in
``gui/components/pipeline_runner.py``.
"""

from __future__ import annotations

import contextlib
import logging
import queue as _queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Child -> parent message tags. Every message is a tuple whose first element is
# one of these constants.
# ---------------------------------------------------------------------------
MSG_PROGRESS = "progress"  # (MSG_PROGRESS, str)
MSG_MIRROR = "mirror"  # (MSG_MIRROR, dict)        latest task visual state
MSG_LOG = "log"  # (MSG_LOG, levelno, name, message)
MSG_RESULT = "result"  # (MSG_RESULT, dict)        task.get_run_results() / {"events_path": ...}
MSG_ERROR = "error"  # (MSG_ERROR, type_name, traceback_str)
MSG_POST_RUN_QUESTIONS = "post_run_questions"  # (MSG_POST_RUN_QUESTIONS, bool)
MSG_DONE = "done"  # (MSG_DONE,)               always sent last


@dataclass
class FeedbackStepPayload:
    """Everything the child needs to replay a feedback step.

    ``feedback_kwargs`` is the exact kwargs dict the parent would have passed to
    :func:`run_feedback_task` (minus the live ``sham_provider`` object, which is
    rebuilt in the child from ``sham_replay``). All values are picklable.
    """

    kind: str  # always "feedback"
    feedback_kwargs: dict[str, Any]
    factory_config: dict[str, Any]
    task_id: str
    output_dir: str
    feedback_mode: bool
    is_last_feedback: bool
    max_feedback_runs: int
    sham_replay: dict[str, Any] | None


@dataclass
class LocalizerStepPayload:
    """Everything the child needs to run a localizer task.

    Collection (``collect_volumes``) stays in the parent and reads the DICOMs the
    child's simulator writes to ``watch_dir`` on disk; only the PsychoPy task and
    its simulator run in the child.
    """

    kind: str  # always "localizer"
    task_id: str
    output_dir: str
    participant_id: str
    session_id: str
    config: dict[str, Any]
    task_state_dir: str
    run: int
    attempt: int
    watch_dir: str
    monitor_position: str
    trigger_keys: list[str]
    simulator_source_dir: str | None
    simulator_target_dir: str
    simulator_tr: float
    simulator_speed: float
    simulator_trigger_mode: str


# ---------------------------------------------------------------------------
# Child-side helpers (run inside the spawned process).
# ---------------------------------------------------------------------------


class _QueueLogHandler(logging.Handler):
    """Forward ``pineuro`` log records to the parent so they reach the one
    session log file (single writer) and the live log console."""

    def __init__(self, send: Any) -> None:
        super().__init__(level=logging.INFO)
        self._send = send

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            self._send(MSG_LOG, record.levelno, record.name, record.getMessage())


class _MirrorForwarder:
    """Decouple the task render loop from IPC drain rate.

    The task's mirror callback writes the latest visual state into a single-slot
    queue (latest-wins, never blocks the render loop); a daemon thread forwards
    whatever the parent can keep up with. Stale frames are dropped, not queued.
    """

    _STOP = object()

    def __init__(self, send_mirror: Any) -> None:
        self._send_mirror = send_mirror
        self._slot: _queue.Queue[Any] = _queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, name="child-mirror-fwd", daemon=True)
        self._thread.start()

    def submit(self, state: dict[str, Any]) -> None:
        try:
            self._slot.put_nowait(state)
        except _queue.Full:
            with contextlib.suppress(_queue.Empty):
                self._slot.get_nowait()
            with contextlib.suppress(_queue.Full):
                self._slot.put_nowait(state)

    def _run(self) -> None:
        while True:
            item = self._slot.get()
            if item is self._STOP:
                break
            with contextlib.suppress(Exception):
                self._send_mirror(item)

    def close(self) -> None:
        # Drop any pending frame and inject the stop sentinel.
        with contextlib.suppress(_queue.Empty):
            self._slot.get_nowait()
        with contextlib.suppress(_queue.Full):
            self._slot.put_nowait(self._STOP)


class _StopBridge:
    """Bridge the cross-process stop Event to the in-child coordinator/task.

    The parent sets ``stop_event`` (and ``show_questions_value`` beforehand); this
    thread translates that into the same ``coordinator.stop(...)`` / ``task.stop()``
    calls the in-process stop path used to make.
    """

    def __init__(self, stop_event: Any, show_questions_value: Any) -> None:
        self._stop_event = stop_event
        self._show_q = show_questions_value
        self._coordinator: Any = None
        self._task: Any = None
        self._bound = threading.Event()
        self._thread = threading.Thread(target=self._wait, name="child-stop-bridge", daemon=True)
        self._thread.start()

    def bind(self, *, coordinator: Any = None, task: Any = None) -> None:
        self._coordinator = coordinator
        self._task = task
        self._bound.set()

    def _wait(self) -> None:
        self._stop_event.wait()
        # A stop may arrive before the coordinator/task exist; give bind() a moment.
        self._bound.wait(timeout=10.0)
        show_q = bool(self._show_q.value) if self._show_q is not None else False
        if self._coordinator is not None:
            with contextlib.suppress(Exception):
                self._coordinator.stop(show_questions=show_q)
        if self._task is not None:
            with contextlib.suppress(Exception):
                self._task.stop()


def _build_sham_provider(sham_replay: dict[str, Any] | None) -> Any:
    if not sham_replay:
        return None
    from pineuro.sham import ReplayActivationProvider

    return ReplayActivationProvider(
        feedbacklog_path=Path(sham_replay["feedbacklog_path"]),
        mask_names=list(sham_replay["mask_names"]),
    )


def _run_feedback_child(
    payload: FeedbackStepPayload,
    send: Any,
    mirror_callback: Any,
    stop_bridge: _StopBridge,
) -> dict[str, Any]:
    """Replay a feedback step: build coordinator + task, run, collect results.

    Mirrors the in-process wiring that used to live in ``pipeline_runner`` right
    after the ``run_feedback_task`` call.
    """
    from pineuro.feedback import run_feedback_task
    from pineuro.tasks import create_task
    from pineuro.tasks.state import TaskState

    kwargs = dict(payload.feedback_kwargs)
    kwargs["sham_provider"] = _build_sham_provider(payload.sham_replay)

    session_info = run_feedback_task(**kwargs)
    coordinator = session_info["coordinator"]
    coordinator_thread = session_info["thread"]
    simulator = session_info.get("simulator")

    task_state = TaskState.load(Path(payload.output_dir), task_id=payload.task_id)
    if task_state.max_feedback_runs != payload.max_feedback_runs:
        task_state.max_feedback_runs = payload.max_feedback_runs
        task_state.save()

    task = create_task(
        task_id=payload.task_id,
        coordinator=coordinator,
        config=payload.factory_config,
        task_state=task_state,
        output_dir=payload.output_dir,
        simulator=simulator,
    )
    # Coordinator pushes new activations straight to the task (in-process here).
    coordinator._activation_callback = task._update_data
    task._is_last_feedback_run = payload.feedback_mode and payload.is_last_feedback
    task.set_mirror_callback(mirror_callback)

    # Let the parent's Stop dialog know whether post-run questions would show.
    with contextlib.suppress(Exception):
        send(MSG_POST_RUN_QUESTIONS, bool(task.has_pending_post_run_questions()))

    stop_bridge.bind(coordinator=coordinator, task=task)

    # Blocks on THIS process's main thread (creates the PsychoPy window). A crash
    # here is logged but non-fatal — tasks can throw during end-of-run cleanup
    # after a successful run; the coordinator-error and 0-volume guards below
    # decide whether the step actually failed.
    try:
        task.run()
    except Exception:
        logger.error("Feedback task crashed during run:\n%s", traceback.format_exc())
    coordinator_thread.join()
    coordinator.check_error()  # re-raises a coordinator error (e.g. MoCo mismatch)

    results = task.get_run_results()
    volumes_processed = results.get("volumes_processed", coordinator.n_processed)
    # Defense in depth: a run that processed nothing produced no feedback. Fail it
    # (unless the user stopped it) so silent-0-volume causes don't look like success.
    if volumes_processed == 0 and not coordinator.was_stopped() and coordinator._error is None:
        raise RuntimeError(
            "Feedback run processed 0 volumes. No feedback was produced. Check that the "
            "scanner is sending volumes to the watch directory and that the motion "
            "correction mode matches the incoming data."
        )
    return results


def _run_localizer_child(
    payload: LocalizerStepPayload,
    mirror_callback: Any,
    stop_bridge: _StopBridge,
    task_done_event: Any,
) -> dict[str, Any]:
    """Run a localizer PsychoPy task (with its simulator) to completion."""
    from pineuro.simulator import VolumeSimulator
    from pineuro.tasks.factory import create_localizer_task
    from pineuro.tasks.state import TaskState

    simulator = None
    if payload.simulator_source_dir:
        simulator = VolumeSimulator(
            source_dir=payload.simulator_source_dir,
            target_dir=payload.simulator_target_dir,
            tr=payload.simulator_tr,
            speed=payload.simulator_speed,
            clear_target=True,
            trigger_mode=payload.simulator_trigger_mode,  # type: ignore[arg-type]
        )

    task_state = TaskState.load(Path(payload.task_state_dir), task_id=payload.task_id)
    task = create_localizer_task(
        task_id=payload.task_id,
        output_dir=payload.output_dir,
        participant_id=payload.participant_id,
        session_id=payload.session_id,
        config=payload.config,
        task_state=task_state,
        run=payload.run,
        attempt=payload.attempt,
        watch_dir=payload.watch_dir,
        simulator=simulator,
        monitor_position=payload.monitor_position,
        trigger_keys=payload.trigger_keys,
    )
    # trigger_event is intentionally NOT set: collect_volumes (in the parent) uses
    # its first-volume-arrival fallback for the elapsed timer.
    task.set_mirror_callback(mirror_callback)
    stop_bridge.bind(task=task)

    events_path = None
    try:
        # A crash here is logged but non-fatal (matches the old in-process
        # behavior): collection in the parent may still have produced usable
        # DICOMs. Setup failures above (simulator/task/state) DO propagate.
        events_path = task.run()
    except Exception:
        logger.error("Localizer task crashed during run:\n%s", traceback.format_exc())
    finally:
        # Signal collect_volumes (parent) to drain remaining volumes and stop.
        if task_done_event is not None:
            task_done_event.set()
    return {"events_path": str(events_path) if events_path else None}


def run_step_in_subprocess(
    payload: FeedbackStepPayload | LocalizerStepPayload,
    msg_queue: Any,
    stop_event: Any,
    show_questions_value: Any,
    task_done_event: Any = None,
) -> None:
    """Spawn target: run ONE task step to completion in this fresh process.

    Parameters
    ----------
    payload
        :class:`FeedbackStepPayload` or :class:`LocalizerStepPayload`.
    msg_queue
        ``multiprocessing`` Queue for child -> parent messages (see ``MSG_*``).
    stop_event
        ``multiprocessing`` Event set by the parent to request a stop.
    show_questions_value
        ``multiprocessing`` Value('b') the parent sets before ``stop_event`` to
        request post-run questions on a feedback stop.
    task_done_event
        ``multiprocessing`` Event the localizer child sets when its paradigm
        finishes (ignored for feedback).
    """

    def send(tag: str, *args: Any) -> None:
        with contextlib.suppress(Exception):
            msg_queue.put((tag, *args))

    # Route all pineuro logs to the parent (single log-file writer; no local
    # console/file handlers in the child). State is restored in the finally so
    # this leaves no global side effects (the in-process unit tests rely on that;
    # a real spawned child discards it on exit either way).
    pineuro_logger = logging.getLogger("pineuro")
    prev_level = pineuro_logger.level
    prev_propagate = pineuro_logger.propagate
    pineuro_logger.setLevel(logging.INFO)
    log_handler = _QueueLogHandler(send)
    pineuro_logger.addHandler(log_handler)
    pineuro_logger.propagate = False

    mirror_fwd = _MirrorForwarder(lambda state: send(MSG_MIRROR, state))
    stop_bridge = _StopBridge(stop_event, show_questions_value)

    try:
        send(MSG_PROGRESS, "Starting task process...")
        if payload.kind == "feedback":
            assert isinstance(payload, FeedbackStepPayload)
            results = _run_feedback_child(payload, send, mirror_fwd.submit, stop_bridge)
        elif payload.kind == "localizer":
            assert isinstance(payload, LocalizerStepPayload)
            results = _run_localizer_child(payload, mirror_fwd.submit, stop_bridge, task_done_event)
        else:
            raise ValueError(f"Unknown payload kind: {payload.kind!r}")
        send(MSG_RESULT, results)
    except Exception as exc:  # forward ANY failure (incl. the 0-volume guard) to the parent
        send(MSG_ERROR, type(exc).__name__, traceback.format_exc())
    finally:
        # Backstop: always release the parent's collect_volumes, even if the
        # localizer child crashed during setup before task.run() could set it —
        # otherwise the parent would wait forever for volumes that never arrive.
        if task_done_event is not None:
            with contextlib.suppress(Exception):
                task_done_event.set()
        # Order matters: result/error are enqueued above before DONE, so the
        # parent has them when it sees DONE.
        mirror_fwd.close()
        send(MSG_DONE)
        with contextlib.suppress(Exception):
            log_handler.flush()
        # Restore the global logger state we mutated (no leak into the test process).
        pineuro_logger.removeHandler(log_handler)
        pineuro_logger.propagate = prev_propagate
        pineuro_logger.setLevel(prev_level)
