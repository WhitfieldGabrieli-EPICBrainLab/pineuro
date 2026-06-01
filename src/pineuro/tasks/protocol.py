"""Protocol defining the coordinator contract for third-party task authors.

Use this protocol to type-check custom task implementations without importing
the concrete classes (which pull in PsychoPy and numpy at import time).

Classes
-------
CoordinatorProtocol : Protocol
    Attributes and methods that task classes may access from the coordinator
    passed to their constructor.  Tasks must NOT call coordinator methods from
    inside PsychoPy drawing code; see the Threading Model note below.

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
Type-check a coordinator consumer without a full pineuro install::

    from pineuro.tasks.protocol import CoordinatorProtocol

    def describe(coord: CoordinatorProtocol) -> str:
        return f"{coord.subject} {coord.task} run {coord.run_number}"
"""

from typing import Protocol, runtime_checkable


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
    attempt : int or None
        Attempt number for retry tracking (1, 2, 3, ...).
    n_volumes : int
        Total number of volumes expected in this run.
    mask_names : list[str]
        Ordered list of mask/network names (e.g. ``['ContA', 'DefaultA']``).
    """

    subject: str
    task: str
    run_number: int
    attempt: int
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
