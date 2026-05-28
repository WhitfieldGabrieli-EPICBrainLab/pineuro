"""
Task state management for multi-run neurofeedback tasks.

Summary
-------
Persists adaptive task state across runs via JSON in the session directory.
Session state is config-driven: tasks declare their state fields in
TaskMetadata, and TaskState stores them dynamically via get()/set().

Components
----------
- TaskState: Persistent state container with save/load/reset helpers
- RunInfo: Metadata for completed runs

Workflow
--------
- Create TaskState via create_for_task() with task-specific defaults
- Or load existing task_state.json via load()
- Access state fields via get()/set()
- Save updated state to disk after each run

Dependencies
------------
json (stdlib), dataclasses, pathlib
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Metadata for a completed run.

    Stores task-agnostic run information plus task-specific state snapshot.
    """

    run_number: int
    run_type: str  # 'transfer' or 'feedback'
    volumes_processed: int
    attempt: int
    # Task-specific state at end of run (e.g., hit_counters, scale_factor)
    task_state_snapshot: dict[str, Any] = field(default_factory=dict)

    # Slot-based tracking: slot_run_number is the fixed run slot (1 or 2).
    slot_run_number: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "run_number": self.run_number,
            "run_type": self.run_type,
            "volumes_processed": self.volumes_processed,
            "task_state_snapshot": self.task_state_snapshot,
            "attempt": self.attempt,
        }
        if self.slot_run_number is not None:
            data["slot_run_number"] = self.slot_run_number
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunInfo":
        """Create from dictionary."""
        return cls(
            run_number=data["run_number"],
            run_type=data["run_type"],
            volumes_processed=data.get("volumes_processed", 0),
            attempt=data["attempt"],
            task_state_snapshot=data.get("task_state_snapshot", {}),
            slot_run_number=data.get("slot_run_number"),
        )


@dataclass
class TaskState:
    """Persistent state for multi-run neurofeedback tasks.

    Now config-driven: tasks declare their state fields in TaskMetadata,
    and this class stores them dynamically in task_state dict.

    Parameters
    ----------
    feedback_dir : Path
        Directory where task_state.json will be stored
    task_id : str
        Identifier of the task type (e.g., "ball", "thermometer")
    task_state : dict
        Dynamic storage for task-specific state fields
    completed_runs : list[RunInfo]
        Metadata for all completed runs
    max_feedback_runs : int
        Total expected number of feedback runs (for detecting last run)

    Examples
    --------
    >>> # Create new session state with task-specific defaults
    >>> state = TaskState.create_for_task(feedback_dir, "my_task")
    >>> state.get("threshold")  # Access task state
    0.5
    >>> state.set("threshold", 0.8)  # Update task state

    >>> # Or load existing state
    >>> state = TaskState.load(feedback_dir, task_id="my_task")

    >>> state.set("scale_factor", 15.0)  # Update a field
    """

    feedback_dir: Path
    task_id: str = "ball"
    task_state: dict[str, Any] = field(default_factory=dict)
    completed_runs: list[RunInfo] = field(default_factory=list)
    max_feedback_runs: int = 5

    # Slot-based tracking: maps slot_run_number to the task_state snapshot
    # at the START of that slot (before any runs). Enables retries to restore
    # the correct initial state for a slot.
    slot_base_state: dict[int, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default task state from metadata if empty."""
        if not self.task_state:
            try:
                from .task_registry import get_task_metadata

                metadata = get_task_metadata(self.task_id)
                self.task_state = {f.key: f.default for f in metadata.task_state_fields}
            except (ValueError, ImportError):
                pass

    # -------------------------------------------------------------------------
    # Generic state access methods
    # -------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a task state field.

        Parameters
        ----------
        key : str
            Field name (e.g., "threshold", "n_blocks_completed")
        default : Any, optional
            Default value if field doesn't exist

        Returns
        -------
        Any
            Field value or default
        """
        return self.task_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a task state field.

        Parameters
        ----------
        key : str
            Field name
        value : Any
            New value
        """
        self.task_state[key] = value

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------

    @property
    def state_file(self) -> Path:
        """Path to task_state.json file."""
        return self.feedback_dir / "task_state.json"

    @property
    def next_run_number(self) -> int:
        """Auto-derived run number based on completed runs."""
        return len(self.completed_runs) + 1

    def next_run_number_for_type(self, run_type: str) -> int:
        """Get the next run number for a specific run type.

        Parameters
        ----------
        run_type : str
            'transfer' or 'feedback'

        Returns
        -------
        int
            Next run number for this specific run type (1-indexed)
        """
        count = sum(1 for r in self.completed_runs if r.run_type == run_type)
        return count + 1

    @property
    def last_run_type(self) -> str | None:
        """Type of the last completed run, or None if no runs completed."""
        if self.completed_runs:
            return self.completed_runs[-1].run_type
        return None

    @property
    def feedback_run_count(self) -> int:
        """Number of feedback runs completed so far."""
        return sum(1 for r in self.completed_runs if r.run_type == "feedback")

    @property
    def is_last_feedback_run(self) -> bool:
        """Whether we have just completed the last feedback run.

        Returns True when feedback_run_count >= max_feedback_runs,
        meaning all feedback runs are complete. This is used to determine
        when to show "last_feedback_only" post-run questions.
        """
        return self.feedback_run_count >= self.max_feedback_runs

    def get_state_for_slot(self, slot_run_number: int, attempt: int = 1) -> dict[str, Any]:
        """Get the appropriate task state for a run slot.

        For retries (attempt > 1), returns the task state from before that slot
        was first started. For first attempts (attempt == 1), returns the current
        task state and records it as the base for this slot.

        Parameters
        ----------
        slot_run_number : int
            The run slot number (1 or 2)
        attempt : int
            The attempt number (1, 2, 3, ...)

        Returns
        -------
        dict
            The task state to use for this run
        """
        if attempt > 1 and slot_run_number in self.slot_base_state:
            base = self.slot_base_state[slot_run_number]
            logger.info(f"Retry (attempt={attempt}): using base state for slot {slot_run_number}")
            return base.copy()
        else:
            if slot_run_number not in self.slot_base_state:
                self.slot_base_state[slot_run_number] = self.task_state.copy()
                logger.info(f"First attempt at slot {slot_run_number}: recorded base state")
                self.save()
            return self.task_state.copy()

    def has_run_for_slot(self, slot_run_number: int) -> bool:
        """Check if a run has been recorded for a given slot.

        Parameters
        ----------
        slot_run_number : int
            The run slot number (1 or 2)

        Returns
        -------
        bool
            True if a run has been recorded for this slot
        """
        return any(r.slot_run_number == slot_run_number for r in self.completed_runs)

    def get_run_for_slot(self, slot_run_number: int) -> RunInfo | None:
        """Get the run info for a given slot, if it exists.

        Parameters
        ----------
        slot_run_number : int
            The run slot number (1 or 2)

        Returns
        -------
        RunInfo or None
            The run info for this slot, or None if no run recorded
        """
        for r in self.completed_runs:
            if r.slot_run_number == slot_run_number:
                return r
        return None

    # -------------------------------------------------------------------------
    # Persistence methods
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Save current state to task_state.json."""
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "task_id": self.task_id,
            "task_state": self.task_state,
            "completed_runs": [r.to_dict() for r in self.completed_runs],
            "max_feedback_runs": self.max_feedback_runs,
            "slot_base_state": {str(k): v for k, v in self.slot_base_state.items()},
        }

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Session state saved to {self.state_file}")

    @classmethod
    def create_for_task(cls, feedback_dir: Path, task_id: str) -> "TaskState":
        """Create session state with defaults from task metadata.

        Parameters
        ----------
        feedback_dir : Path
            Directory for session state file
        task_id : str
            Task identifier (e.g., "ball", "thermometer")

        Returns
        -------
        TaskState
            New session state initialized with task-specific defaults
        """
        from .task_registry import get_task_metadata

        feedback_dir = Path(feedback_dir)

        try:
            metadata = get_task_metadata(task_id)
            defaults = {f.key: f.default for f in metadata.task_state_fields}
        except ValueError:
            logger.warning(f"Task '{task_id}' not registered, using empty state defaults")
            defaults = {}

        return cls(
            feedback_dir=feedback_dir,
            task_id=task_id,
            task_state=defaults,
        )

    @classmethod
    def load(cls, feedback_dir: Path, task_id: str = "unknown") -> "TaskState":
        """Load state from task_state.json, or create new if not exists.

        Handles both old format (direct fields) and new format (task_state dict).

        Parameters
        ----------
        feedback_dir : Path
            Directory containing (or that will contain) task_state.json
        task_id : str
            Task identifier for creating fresh state

        Returns
        -------
        TaskState
            Loaded state, or fresh state if file doesn't exist
        """
        feedback_dir = Path(feedback_dir)
        state_file = feedback_dir / "task_state.json"

        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)

            # Handle both old and new formats
            if "task_state" in data:
                # New format: use task_state dict
                task_state = data["task_state"]
            else:
                # Old format: migrate legacy fields using task metadata
                loaded_task_id = data.get("task_id", task_id)
                try:
                    from .task_registry import get_task_metadata

                    metadata = get_task_metadata(loaded_task_id)
                    _known_keys = {f.key for f in metadata.task_state_fields}
                    task_state = {f.key: data.get(f.key, f.default) for f in metadata.task_state_fields}
                    logger.info(f"Migrated legacy session state for '{loaded_task_id}': {list(task_state.keys())}")
                except (ValueError, ImportError):
                    # Task not registered: pull any non-framework keys as best-effort
                    framework_keys = {
                        "task_id",
                        "completed_runs",
                        "max_feedback_runs",
                        "slot_base_state",
                        "slot_base_scale_factors",
                    }
                    task_state = {k: v for k, v in data.items() if k not in framework_keys}
                    logger.warning(
                        f"Task '{loaded_task_id}' not registered, migrated all extra keys: {list(task_state.keys())}"
                    )

            # Load slot_base_state (convert string keys back to int)
            slot_base_state_raw = data.get("slot_base_state", {})
            slot_base_state = {int(k): v for k, v in slot_base_state_raw.items()}

            state = cls(
                feedback_dir=feedback_dir,
                task_id=data.get("task_id", task_id),
                task_state=task_state,
                completed_runs=[RunInfo.from_dict(r) for r in data.get("completed_runs", [])],
                max_feedback_runs=data.get("max_feedback_runs", 5),
                slot_base_state=slot_base_state,
            )
            logger.info(f"Loaded session state from {state_file}")
            logger.info(
                f"  Task: {state.task_id}, {len(state.completed_runs)} completed, "
                f"next: run {state.next_run_number}, "
                f"feedback: {state.feedback_run_count}/{state.max_feedback_runs}"
            )
            return state
        else:
            logger.info(f"No existing session state at {state_file}, creating fresh state")
            return cls.create_for_task(feedback_dir, task_id)

    @classmethod
    def reset(cls, feedback_dir: Path, task_id: str = "unknown") -> "TaskState":
        """Reset session state by deleting existing state file.

        Parameters
        ----------
        feedback_dir : Path
            Directory containing task_state.json
        task_id : str
            Task identifier for fresh state

        Returns
        -------
        TaskState
            Fresh session state
        """
        feedback_dir = Path(feedback_dir)
        state_file = feedback_dir / "task_state.json"

        if state_file.exists():
            state_file.unlink()
            logger.info(f"Reset session state: deleted {state_file}")

        return cls.create_for_task(feedback_dir, task_id)

    # -------------------------------------------------------------------------
    # Run recording
    # -------------------------------------------------------------------------

    def record_run(
        self,
        run_type: str,
        volumes_processed: int,
        attempt: int,
        task_state_snapshot: dict[str, Any] | None = None,
        # Slot-based tracking parameters
        slot_run_number: int | None = None,
    ) -> None:
        """Record a completed run and update task state.

        Tasks should apply their own adaptive logic (e.g., difficulty
        adjustment) to the snapshot *before* calling this method. This
        method simply records the run and updates persistent task_state
        from the snapshot.

        Parameters
        ----------
        run_type : str
            'transfer' or 'feedback'
        volumes_processed : int
            Number of volumes processed in this run
        attempt : int
            The attempt number (1, 2, 3, ...). Used for retry tracking.
        task_state_snapshot : dict, optional
            Snapshot of task state at end of run. All keys in this dict
            will be merged into the persistent task_state.
        slot_run_number : int, optional
            The run slot number (1 or 2). If provided, tracks runs by slot.
        """
        if task_state_snapshot is None:
            task_state_snapshot = {}

        # Determine if this is a retry (attempt > 1 for same slot)
        is_retry = False
        existing_run_idx = None
        if slot_run_number is not None:
            for idx, r in enumerate(self.completed_runs):
                if r.slot_run_number == slot_run_number:
                    is_retry = True
                    existing_run_idx = idx
                    break

        # Determine run_number for display purposes
        if slot_run_number is not None:
            run_number = slot_run_number
        else:
            run_number = self.next_run_number

        run_info = RunInfo(
            run_number=run_number,
            run_type=run_type,
            volumes_processed=volumes_processed,
            task_state_snapshot=task_state_snapshot,
            slot_run_number=slot_run_number,
            attempt=attempt,
        )

        if is_retry and existing_run_idx is not None:
            old_run = self.completed_runs[existing_run_idx]
            logger.info(f"Replacing run for slot {slot_run_number} (attempt {old_run.attempt} -> {attempt})")
            self.completed_runs[existing_run_idx] = run_info
        else:
            self.completed_runs.append(run_info)

        # Update persistent task_state from snapshot
        self.task_state.update(task_state_snapshot)

        # Filter to only metadata-declared keys (prevents stale keys from
        # old task types or pre-refactor data from lingering)
        try:
            from .task_registry import get_task_metadata

            metadata = get_task_metadata(self.task_id)
            known_keys = {f.key for f in metadata.task_state_fields}
            # Also keep any keys from the current snapshot (task may use
            # ad-hoc keys not declared in metadata)
            known_keys.update(task_state_snapshot.keys())
            stale = set(self.task_state.keys()) - known_keys
            if stale:
                for k in stale:
                    del self.task_state[k]
                logger.info(f"Removed stale task state keys: {stale}")
        except (ValueError, ImportError):
            pass  # Task not registered -- keep everything

        self.save()

        logger.info(f"Recorded run {run_info.run_number} ({run_type})")
        if slot_run_number is not None:
            logger.debug(f"  Slot: {slot_run_number}, Attempt: {attempt}")
        logger.debug(f"  Task state: {self.task_state}")
