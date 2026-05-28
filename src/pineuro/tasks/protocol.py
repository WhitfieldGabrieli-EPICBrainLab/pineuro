"""Protocols defining the task/coordinator contract for third-party task authors.

Use these protocols to type-check custom task implementations without importing
the concrete classes (which pull in PsychoPy and numpy at import time).

Classes
-------
CoordinatorProtocol : Protocol
    Attributes and methods that task classes may access from the coordinator
    passed to their constructor.  Tasks must NOT call coordinator methods from
    inside PsychoPy drawing code; see the Threading Model note below.

TaskProtocol : Protocol
    The minimal interface a custom task class must satisfy to integrate with
    the pineuro framework.  Inherit from ``BaseTask`` rather than implementing
    this protocol directly — it is provided as documentation and for static
    type-checking of third-party implementations.

Threading Model
---------------
Two threads interact with each task during a feedback task:

Coordinator thread (background)
    Runs incremental GLM and activation computation.  Calls
    ``task._update_data()`` after each processed volume.  Writes
    ``_latest_data`` (protected by ``BaseTask._lock``) and reads
    ``_accumulating`` / ``_block_activations`` (lock-free, coordinator-thread
    only).

Task thread (main thread, required by PsychoPy)
    Runs ``run()`` → ``setup()`` → ``update()`` loop.  Reads ``_latest_data``
    via ``get_latest_data()`` (acquires ``_lock``).  Writes ``_accumulating``
    and ``_block_activations`` via ``_start_block_accumulation()`` /
    ``_stop_block_accumulation()`` (lock-free, task-thread only).

Rule: **Only call PsychoPy APIs from the task (main) thread.**  Never call
PsychoPy drawing functions from the coordinator thread.

Example
-------
Type-check a third-party task without a full pineuro install::

    from pineuro.tasks.protocol import CoordinatorProtocol, TaskProtocol

    class MyTask:
        def setup(self) -> None: ...
        def update(self, data: dict) -> None: ...
        def run(self) -> None: ...
        def cleanup(self) -> None: ...
        def get_run_results(self) -> dict: return {}
        def get_mirror_state(self) -> dict: return {"background": "black", "shapes": []}
        def get_volume_log_suffix(self) -> str: return ""
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CoordinatorProtocol(Protocol):
    """Read-only view of the coordinator that tasks may access.

    Tasks receive a coordinator instance in their constructor.  They may read
    these attributes and call these methods at any time from the **task thread**
    only.

    Attributes
    ----------
    subject : str
        Subject label (e.g. ``'sub-01'``).
    task : str
        Task label registered in the task registry (e.g. ``'ball'``).
    run_number : int
        Current run number (1-based).
    acq : int or None
        Acquisition number for retry tracking (1, 2, 3, ...).
    n_volumes : int
        Total number of volumes expected in this run.
    mask_names : list[str]
        Ordered list of mask/network names (e.g. ``['ContA', 'DefaultA']``).
    """

    subject: str
    task: str
    run_number: int
    acq: int
    n_volumes: int
    mask_names: list[str]

    def is_complete(self) -> bool:
        """Return True when all volumes have been processed."""
        ...

    def check_error(self) -> None:
        """Re-raise any error that occurred in the coordinator thread.

        Call once per frame inside the update loop.  Raises the stored
        exception so it propagates to the task thread.
        """
        ...

    def was_stopped(self) -> bool:
        """Return True if the session was manually stopped by the user."""
        ...

    def should_show_questions(self) -> bool:
        """Return True if post-run questions should be displayed.

        Returns False when the session was stopped mid-run (questions skipped).
        """
        ...

    def send_trigger(self) -> None:
        """Notify the coordinator that the scanner trigger has been received."""
        ...


@runtime_checkable
class TaskProtocol(Protocol):
    """Minimal interface a custom task class must implement.

    Inherit from ``BaseTask`` rather than implementing this protocol directly.
    This protocol exists for documentation and static type-checking of
    third-party task classes that do not inherit from ``BaseTask``.

    Override Guide
    --------------
    Must override (abstract):
        ``setup()``
            Create PsychoPy window and stimuli.  Do **not** call super().
        ``update(data)``
            Draw each frame.  Do **not** call super().

    Should override (call super()):
        ``cleanup()``
            Release resources.  Call ``super().cleanup()`` at the **end**.
        ``get_run_results()``
            Return run metrics.  Merge with ``super().get_run_results()``.
        ``_record_run_completion()``
            Custom session-state recording.  Call ``super()`` first.

    May override (no super() needed):
        ``get_mirror_state()``
            Return vector state for GUI mirror.
        ``get_volume_log_suffix()``
            String appended to coordinator volume log line.
        ``_show_post_run_questions()``
            Replace the question display entirely.

    Do not override:
        ``run()``
            Main loop; framework-controlled.
        ``_update_data()``
            Coordinator callback — only override if ``super()`` is called first.
        ``get_latest_data()``
            Thread-safe data accessor.
    """

    def setup(self) -> None:
        """Initialize PsychoPy window and stimuli.

        Called once before the run loop starts.  Create the window, visual
        stimuli, and any other resources needed for the task.
        """
        ...

    def update(self, data: dict[str, Any]) -> None:
        """Update display based on latest activation data.

        Called every display frame with latest data from the coordinator.

        Parameters
        ----------
        data : dict
            Contains at minimum:

            ``'activation'`` : float
                Current activation value for the primary mask (z-score units;
                see ``ACTIVATION_UNITS`` in ``pineuro.feedback``).
            ``'activations'`` : dict[str, float]
                All mask activations keyed by mask name.
            ``'volume'`` : int
                Current volume number (0-based).
            ``'time'`` : float
                Elapsed time in seconds.
            ``'is_baseline'`` : bool
                ``True`` during the baseline period.
        """
        ...

    def run(self) -> None:
        """Main task loop — do not override."""
        ...

    def cleanup(self) -> None:
        """Release resources after the task completes."""
        ...

    def get_run_results(self) -> dict[str, Any]:
        """Return task-specific results from the completed run.

        The framework calls this after ``run()`` completes.  Override to
        include task-specific metrics (e.g. hit counts, scale factor).

        Returns
        -------
        dict
            Arbitrary key/value pairs with run metrics.
        """
        ...

    def get_mirror_state(self) -> dict[str, Any]:
        """Return current visual state for GUI mirroring.

        Called each frame to update the experimenter mirror display.
        All coordinates should be normalised to [0, 1].

        Returns
        -------
        dict
            ``{'background': str, 'shapes': list[dict]}`` where each shape
            dict has a ``'type'`` key and type-specific properties.
            Supported types: ``'circle'``, ``'line'``, ``'rect'``, ``'text'``.
        """
        ...

    def get_volume_log_suffix(self) -> str:
        """Return an optional string appended to the coordinator volume log.

        Called from the coordinator thread after ``_update_data()`` returns.
        Must be fast and thread-safe.
        """
        ...
