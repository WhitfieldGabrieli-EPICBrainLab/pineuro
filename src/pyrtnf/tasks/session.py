"""
Session state management for multi-run neurofeedback sessions.

Provides persistent state across separate task runs (e.g., Transfer → Feedback × N → Transfer).
State is saved to JSON at session root directory and loaded at the start of each run.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    """Metadata for a completed run."""
    run_number: int
    run_type: str  # 'transfer' or 'feedback'
    hit_counters: list[int]
    final_scale_factor: float
    final_circle_radii: list[float]
    volumes_processed: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "RunInfo":
        return cls(**data)


@dataclass
class SessionState:
    """Persistent state for multi-run neurofeedback sessions.
    
    Tracks state across separate task runs, allowing each run to be a fresh
    Python process while maintaining continuity (e.g., adaptive difficulty,
    cumulative hit counts).
    
    Parameters
    ----------
    session_dir : Path
        Directory where session_state.json will be stored
    scale_factor : float
        Current scale factor for ball movement (adaptive difficulty)
    hit_counters : list[int]
        Cumulative hit counts for each target [top, bottom]
    circle_radii : list[float]
        Current circle radii (shrink on hits)
    completed_runs : list[RunInfo]
        Metadata for all completed runs
    max_feedback_runs : int
        Total expected number of feedback runs in the session (for detecting last run)
        
    Examples
    --------
    >>> # At start of run
    >>> state = SessionState.load(session_dir)
    >>> task = BallTask(coordinator, session_state=state, ...)
    >>> task.run()
    >>> # At end of run (called automatically by task.cleanup)
    >>> task.save_state(state)
    
    >>> # Reset session for new participant
    >>> SessionState.reset(session_dir)
    """
    
    session_dir: Path
    scale_factor: float = 10.0
    hit_counters: list[int] = field(default_factory=lambda: [0, 0])
    circle_radii: list[float] = field(default_factory=lambda: [0.15, 0.15])
    completed_runs: list[RunInfo] = field(default_factory=list)
    max_feedback_runs: int = 5
    
    # Minimum circle radius (don't shrink below this)
    MIN_CIRCLE_RADIUS: float = field(default=0.05, repr=False)
    
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
    
    def save(self) -> None:
        """Save current state to session_state.json."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            'scale_factor': self.scale_factor,
            'hit_counters': self.hit_counters,
            'circle_radii': self.circle_radii,
            'completed_runs': [r.to_dict() for r in self.completed_runs],
            'max_feedback_runs': self.max_feedback_runs,
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Session state saved to {self.state_file}")
    
    @classmethod
    def load(cls, session_dir: Path) -> "SessionState":
        """Load state from session_state.json, or create new if not exists.
        
        Parameters
        ----------
        session_dir : Path
            Directory containing (or that will contain) session_state.json
            
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
            
            state = cls(
                session_dir=session_dir,
                scale_factor=data.get('scale_factor', 10.0),
                hit_counters=data.get('hit_counters', [0, 0]),
                circle_radii=data.get('circle_radii', [0.15, 0.15]),
                completed_runs=[
                    RunInfo.from_dict(r) for r in data.get('completed_runs', [])
                ],
                max_feedback_runs=data.get('max_feedback_runs', 5),
            )
            logger.info(f"Loaded session state from {state_file}")
            logger.info(f"  Completed runs: {len(state.completed_runs)}")
            logger.info(f"  Next run number: {state.next_run_number}")
            logger.info(f"  Scale factor: {state.scale_factor}")
            logger.info(f"  Hit counters: {state.hit_counters}")
            logger.info(f"  Feedback runs: {state.feedback_run_count}/{state.max_feedback_runs}")
            return state
        else:
            logger.info(f"No existing session state at {state_file}, creating fresh state")
            return cls(session_dir=session_dir)
    
    @classmethod
    def reset(cls, session_dir: Path) -> "SessionState":
        """Reset session state by deleting existing state file.
        
        Parameters
        ----------
        session_dir : Path
            Directory containing session_state.json
            
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
        
        return cls(session_dir=session_dir)
    
    def record_run(
        self,
        run_type: str,
        hit_counters: list[int],
        scale_factor: float,
        circle_radii: list[float],
        volumes_processed: int,
        min_hits_threshold: int = 3,
        max_hits_threshold: int = 5,
        scale_adjust_factor: float = 0.25,
    ) -> None:
        """Record a completed run and update state.
        
        Also applies adaptive difficulty adjustment to scale_factor for next run:
        - If total hits < min_hits_threshold: increase scale_factor
        - If either network > max_hits_threshold: decrease scale_factor
        - Otherwise: keep scale_factor unchanged
        
        Parameters
        ----------
        run_type : str
            'transfer' or 'feedback'
        hit_counters : list[int]
            Hit counts from this run [top, bottom]
        scale_factor : float
            Final scale factor at end of run
        circle_radii : list[float]
            Final circle radii at end of run
        volumes_processed : int
            Number of volumes processed in this run
        min_hits_threshold : int, default=3
            If total hits < this, increase scale_factor (ball too slow)
        max_hits_threshold : int, default=5
            If either network > this, decrease scale_factor (ball too fast)
        scale_adjust_factor : float, default=0.25
            Fraction to adjust scale_factor by (0.25 = 25%)
        """
        run_info = RunInfo(
            run_number=self.next_run_number,
            run_type=run_type,
            hit_counters=hit_counters.copy(),
            final_scale_factor=scale_factor,
            final_circle_radii=circle_radii.copy(),
            volumes_processed=volumes_processed,
        )
        
        self.completed_runs.append(run_info)
        
        # Update cumulative state
        self.hit_counters = [
            self.hit_counters[i] + hit_counters[i] 
            for i in range(len(hit_counters))
        ]
        self.circle_radii = circle_radii.copy()
        
        # Apply adaptive difficulty adjustment for next run
        # Only applies after FEEDBACK runs (not transfer runs where ball doesn't move)
        # Based on MIND-BPD logic from rt-network_feedback_mgh.py
        if run_type == 'feedback':
            total_hits = sum(hit_counters)
            max_single_network_hits = max(hit_counters)
            
            old_scale = scale_factor
            if max_single_network_hits > max_hits_threshold:
                # Too many hits in one direction - decrease scale factor (slower ball)
                new_scale = scale_factor * (1 - scale_adjust_factor)
                logger.info(f"  Adaptive difficulty: {max_single_network_hits} hits > {max_hits_threshold} max → "
                           f"decreasing scale_factor {old_scale:.1f} → {new_scale:.1f}")
            elif total_hits < min_hits_threshold:
                # Not enough hits overall - increase scale factor (faster ball)
                new_scale = scale_factor * (1 + scale_adjust_factor)
                logger.info(f"  Adaptive difficulty: {total_hits} total hits < {min_hits_threshold} min → "
                           f"increasing scale_factor {old_scale:.1f} → {new_scale:.1f}")
            else:
                # Hits in acceptable range - keep scale factor
                new_scale = scale_factor
                logger.info(f"  Adaptive difficulty: hits in range → keeping scale_factor at {new_scale:.1f}")
            
            self.scale_factor = new_scale
        else:
            # Transfer run - no adaptive adjustment
            self.scale_factor = scale_factor
            logger.info(f"  Transfer run - no adaptive difficulty adjustment")
        
        # Save immediately
        self.save()
        
        logger.info(f"Recorded run {run_info.run_number} ({run_type})")
        logger.info(f"  Hits this run: {hit_counters}")
        logger.info(f"  Cumulative hits: {self.hit_counters}")
