"""
Session state management for multi-run neurofeedback sessions.

Summary
-------
Persists adaptive task state across runs via JSON in the session directory.
Session state is now config-driven: tasks declare their state fields in
TaskMetadata, and SessionState stores them dynamically.

Components
----------
- SessionState: Persistent state container with save/load/reset helpers
- RunInfo: Metadata for completed runs

Workflow
--------
- Create SessionState via create_for_task() with task-specific defaults
- Or load existing session_state.json via load()
- Access state fields via get()/set() or backwards-compatible properties
- Save updated state to disk after each run

Dependencies
------------
json (stdlib), dataclasses, pathlib

Notes
-----
Backwards compatible with existing BallTask code that accesses scale_factor,
hit_counters, and circle_radii as direct attributes.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Metadata for a completed run.

    Stores task-agnostic run information plus task-specific state snapshot.
    """
    run_number: int
    run_type: str  # 'transfer' or 'feedback'
    volumes_processed: int
    # Task-specific state at end of run (e.g., hit_counters, scale_factor)
    task_state_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Legacy fields for backwards compatibility with existing RunInfo data
    hit_counters: Optional[list[int]] = None
    final_scale_factor: Optional[float] = None
    final_circle_radii: Optional[list[float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            'run_number': self.run_number,
            'run_type': self.run_type,
            'volumes_processed': self.volumes_processed,
            'task_state_snapshot': self.task_state_snapshot,
        }
        # Include legacy fields if present (for backwards compatibility)
        if self.hit_counters is not None:
            data['hit_counters'] = self.hit_counters
        if self.final_scale_factor is not None:
            data['final_scale_factor'] = self.final_scale_factor
        if self.final_circle_radii is not None:
            data['final_circle_radii'] = self.final_circle_radii
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunInfo":
        """Create from dictionary (handles both old and new formats)."""
        return cls(
            run_number=data['run_number'],
            run_type=data['run_type'],
            volumes_processed=data.get('volumes_processed', 0),
            task_state_snapshot=data.get('task_state_snapshot', {}),
            hit_counters=data.get('hit_counters'),
            final_scale_factor=data.get('final_scale_factor'),
            final_circle_radii=data.get('final_circle_radii'),
        )


@dataclass
class SessionState:
    """Persistent state for multi-run neurofeedback sessions.

    Now config-driven: tasks declare their state fields in TaskMetadata,
    and this class stores them dynamically in task_state dict.

    Parameters
    ----------
    session_dir : Path
        Directory where session_state.json will be stored
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
    >>> state = SessionState.create_for_task(session_dir, "ball")
    >>> state.get("scale_factor")  # Access task state
    10.0
    >>> state.set("scale_factor", 12.5)  # Update task state

    >>> # Or load existing state
    >>> state = SessionState.load(session_dir)

    >>> # Backwards-compatible property access (for BallTask)
    >>> state.scale_factor = 15.0  # Same as state.set("scale_factor", 15.0)
    """

    session_dir: Path
    task_id: str = "ball"  # Default for backwards compatibility
    task_state: Dict[str, Any] = field(default_factory=dict)
    completed_runs: list[RunInfo] = field(default_factory=list)
    max_feedback_runs: int = 5

    # Minimum circle radius (don't shrink below this) - kept for BallTask
    MIN_CIRCLE_RADIUS: float = field(default=0.05, repr=False)

    def __post_init__(self):
        """Initialize default task state if empty."""
        # Ensure ball task defaults are present for backwards compatibility
        if not self.task_state:
            self.task_state = {
                'scale_factor': 10.0,
                'hit_counters': [0, 0],
                'circle_radii': [0.15, 0.15],
            }

    # -------------------------------------------------------------------------
    # Generic state access methods
    # -------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a task state field.

        Parameters
        ----------
        key : str
            Field name (e.g., "scale_factor", "hit_counters")
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
    # Backwards-compatible property accessors for BallTask
    # -------------------------------------------------------------------------

    @property
    def scale_factor(self) -> float:
        """Current scale factor for ball movement (adaptive difficulty)."""
        return self.get("scale_factor", 10.0)

    @scale_factor.setter
    def scale_factor(self, value: float):
        self.set("scale_factor", value)

    @property
    def hit_counters(self) -> list[int]:
        """Cumulative hit counts for each target [top, bottom]."""
        return self.get("hit_counters", [0, 0])

    @hit_counters.setter
    def hit_counters(self, value: list[int]):
        self.set("hit_counters", value)

    @property
    def circle_radii(self) -> list[float]:
        """Current circle radii (shrink on hits)."""
        return self.get("circle_radii", [0.15, 0.15])

    @circle_radii.setter
    def circle_radii(self, value: list[float]):
        self.set("circle_radii", value)

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------

    @property
    def state_file(self) -> Path:
        """Path to session_state.json file."""
        return self.session_dir / "session_state.json"

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
    def last_run_type(self) -> Optional[str]:
        """Type of the last completed run, or None if no runs completed."""
        if self.completed_runs:
            return self.completed_runs[-1].run_type
        return None

    @property
    def feedback_run_count(self) -> int:
        """Number of feedback runs completed so far."""
        return sum(1 for r in self.completed_runs if r.run_type == 'feedback')

    @property
    def is_last_feedback_run(self) -> bool:
        """Whether the next feedback run will be the last one.

        Returns True when feedback_run_count == max_feedback_runs - 1,
        meaning the current/next run is the final feedback run.
        """
        return self.feedback_run_count >= self.max_feedback_runs - 1

    # -------------------------------------------------------------------------
    # Persistence methods
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Save current state to session_state.json."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'task_id': self.task_id,
            'task_state': self.task_state,
            'completed_runs': [r.to_dict() for r in self.completed_runs],
            'max_feedback_runs': self.max_feedback_runs,
            # Also save legacy fields for backwards compatibility
            'scale_factor': self.scale_factor,
            'hit_counters': self.hit_counters,
            'circle_radii': self.circle_radii,
        }

        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Session state saved to {self.state_file}")

    @classmethod
    def create_for_task(cls, session_dir: Path, task_id: str) -> "SessionState":
        """Create session state with defaults from task metadata.

        Parameters
        ----------
        session_dir : Path
            Directory for session state file
        task_id : str
            Task identifier (e.g., "ball", "thermometer")

        Returns
        -------
        SessionState
            New session state initialized with task-specific defaults
        """
        from .task_registry import get_task_metadata

        session_dir = Path(session_dir)

        try:
            metadata = get_task_metadata(task_id)
            defaults = {f.key: f.default for f in metadata.session_state_fields}
        except ValueError:
            # Task not registered, use empty defaults
            logger.warning(f"Task '{task_id}' not registered, using empty state defaults")
            defaults = {}

        # Ensure ball task defaults for backwards compatibility
        if task_id == "ball" and not defaults:
            defaults = {
                'scale_factor': 10.0,
                'hit_counters': [0, 0],
                'circle_radii': [0.15, 0.15],
            }

        return cls(
            session_dir=session_dir,
            task_id=task_id,
            task_state=defaults,
        )

    @classmethod
    def load(cls, session_dir: Path, task_id: str = "ball") -> "SessionState":
        """Load state from session_state.json, or create new if not exists.

        Handles both old format (direct fields) and new format (task_state dict).

        Parameters
        ----------
        session_dir : Path
            Directory containing (or that will contain) session_state.json
        task_id : str, optional
            Task identifier for creating fresh state (default: "ball")

        Returns
        -------
        SessionState
            Loaded state, or fresh state if file doesn't exist
        """
        session_dir = Path(session_dir)
        state_file = session_dir / "session_state.json"

        if state_file.exists():
            with open(state_file, 'r') as f:
                data = json.load(f)

            # Handle both old and new formats
            if 'task_state' in data:
                # New format: use task_state dict
                task_state = data['task_state']
            else:
                # Old format: migrate legacy fields to task_state
                task_state = {
                    'scale_factor': data.get('scale_factor', 10.0),
                    'hit_counters': data.get('hit_counters', [0, 0]),
                    'circle_radii': data.get('circle_radii', [0.15, 0.15]),
                }

            state = cls(
                session_dir=session_dir,
                task_id=data.get('task_id', task_id),
                task_state=task_state,
                completed_runs=[
                    RunInfo.from_dict(r) for r in data.get('completed_runs', [])
                ],
                max_feedback_runs=data.get('max_feedback_runs', 5),
            )
            logger.info(f"Loaded session state from {state_file}")
            logger.info(f"  Task: {state.task_id}")
            logger.info(f"  Completed runs: {len(state.completed_runs)}")
            logger.info(f"  Next run number: {state.next_run_number}")
            logger.info(f"  Task state keys: {list(state.task_state.keys())}")
            logger.info(f"  Feedback runs: {state.feedback_run_count}/{state.max_feedback_runs}")
            return state
        else:
            logger.info(f"No existing session state at {state_file}, creating fresh state")
            return cls.create_for_task(session_dir, task_id)

    @classmethod
    def reset(cls, session_dir: Path, task_id: str = "ball") -> "SessionState":
        """Reset session state by deleting existing state file.

        Parameters
        ----------
        session_dir : Path
            Directory containing session_state.json
        task_id : str, optional
            Task identifier for fresh state (default: "ball")

        Returns
        -------
        SessionState
            Fresh session state
        """
        session_dir = Path(session_dir)
        state_file = session_dir / "session_state.json"

        if state_file.exists():
            state_file.unlink()
            logger.info(f"Reset session state: deleted {state_file}")

        return cls.create_for_task(session_dir, task_id)

    # -------------------------------------------------------------------------
    # Run recording (with backwards-compatible BallTask support)
    # -------------------------------------------------------------------------

    def record_run(
        self,
        run_type: str,
        volumes_processed: int,
        task_state_snapshot: Optional[Dict[str, Any]] = None,
        # Legacy parameters for BallTask backwards compatibility
        hit_counters: Optional[list[int]] = None,
        scale_factor: Optional[float] = None,
        circle_radii: Optional[list[float]] = None,
        min_hits_threshold: int = 3,
        max_hits_threshold: int = 5,
        scale_adjust_factor: float = 0.25,
    ) -> None:
        """Record a completed run and update state.

        For BallTask, also applies adaptive difficulty adjustment to scale_factor:
        - If total hits < min_hits_threshold: increase scale_factor
        - If either network > max_hits_threshold: decrease scale_factor

        Parameters
        ----------
        run_type : str
            'transfer' or 'feedback'
        volumes_processed : int
            Number of volumes processed in this run
        task_state_snapshot : dict, optional
            Snapshot of task state at end of run (generic tasks)
        hit_counters : list[int], optional
            Hit counts from this run (BallTask legacy)
        scale_factor : float, optional
            Final scale factor at end of run (BallTask legacy)
        circle_radii : list[float], optional
            Final circle radii at end of run (BallTask legacy)
        min_hits_threshold : int, default=3
            If total hits < this, increase scale_factor
        max_hits_threshold : int, default=5
            If either network > this, decrease scale_factor
        scale_adjust_factor : float, default=0.25
            Fraction to adjust scale_factor by (0.25 = 25%)
        """
        # Build task state snapshot
        if task_state_snapshot is None:
            task_state_snapshot = {}

        # Include legacy fields in snapshot if provided
        if hit_counters is not None:
            task_state_snapshot['hit_counters'] = hit_counters.copy()
        if scale_factor is not None:
            task_state_snapshot['scale_factor'] = scale_factor
        if circle_radii is not None:
            task_state_snapshot['circle_radii'] = circle_radii.copy()

        run_info = RunInfo(
            run_number=self.next_run_number,
            run_type=run_type,
            volumes_processed=volumes_processed,
            task_state_snapshot=task_state_snapshot,
            # Legacy fields
            hit_counters=hit_counters.copy() if hit_counters else None,
            final_scale_factor=scale_factor,
            final_circle_radii=circle_radii.copy() if circle_radii else None,
        )

        self.completed_runs.append(run_info)

        # BallTask-specific: update cumulative hit counters
        if hit_counters is not None:
            current_hits = self.hit_counters
            self.hit_counters = [
                current_hits[i] + hit_counters[i]
                for i in range(min(len(current_hits), len(hit_counters)))
            ]

        if circle_radii is not None:
            self.circle_radii = circle_radii.copy()

        # BallTask-specific: adaptive difficulty adjustment
        if hit_counters is not None and scale_factor is not None and run_type == 'feedback':
            total_hits = sum(hit_counters)
            max_single_network_hits = max(hit_counters)

            old_scale = scale_factor
            if max_single_network_hits > max_hits_threshold:
                new_scale = scale_factor * (1 - scale_adjust_factor)
                logger.info(f"  Adaptive difficulty: {max_single_network_hits} hits > {max_hits_threshold} max → "
                           f"decreasing scale_factor {old_scale:.1f} → {new_scale:.1f}")
            elif total_hits < min_hits_threshold:
                new_scale = scale_factor * (1 + scale_adjust_factor)
                logger.info(f"  Adaptive difficulty: {total_hits} total hits < {min_hits_threshold} min → "
                           f"increasing scale_factor {old_scale:.1f} → {new_scale:.1f}")
            else:
                new_scale = scale_factor
                logger.info(f"  Adaptive difficulty: hits in range → keeping scale_factor at {new_scale:.1f}")

            self.scale_factor = new_scale
        elif scale_factor is not None:
            self.scale_factor = scale_factor
            if run_type == 'transfer':
                logger.info(f"  Transfer run - no adaptive difficulty adjustment")

        self.save()

        logger.info(f"Recorded run {run_info.run_number} ({run_type})")
        if hit_counters is not None:
            logger.info(f"  Hits this run: {hit_counters}")
            logger.info(f"  Cumulative hits: {self.hit_counters}")
