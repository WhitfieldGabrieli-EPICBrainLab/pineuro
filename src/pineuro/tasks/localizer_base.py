"""
Base class for localizer tasks (stimulus presentation without feedback).

Summary
-------
Provides a scaffold for PsychoPy tasks that present stimuli on a fixed time
schedule and record BIDS-compatible events. Unlike BaseTask, localizer tasks
do not require a NeurofeedbackCoordinator — they are time-driven, not
data-driven.

Components
----------
- BaseLocalizerTask: Abstract base with setup/run_paradigm/cleanup lifecycle

Workflow
--------
- setup(): create window/stimuli, load stimulus files
- run(): setup → instructions → trigger wait → run_paradigm → save events → cleanup
- cleanup(): close window, release resources

Dependencies
------------
PsychoPy (for concrete tasks), pandas (for events TSV output)
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Union

from .task_base import TaskBase
from .utils import make_bids_run_prefix

logger = logging.getLogger(__name__)


class BaseLocalizerTask(TaskBase):
    """Abstract base class for localizer tasks.

    Localizer tasks present stimuli in a fixed schedule (block design,
    event-related, etc.) and record onset/duration/trial_type events for
    downstream GLM analysis via mask_extraction_task().

    Unlike BaseTask (for real-time feedback), localizer tasks:
    - Do NOT require a NeurofeedbackCoordinator
    - Are time-driven (present stimulus at t=X seconds)
    - Output BIDS events TSV as their primary product

    Subclasses must implement setup() and run_paradigm().

    Parameters
    ----------
    output_dir : str or Path
        Directory for events TSV and any other output files.
    participant_id : str
        Participant identifier (e.g., "sub-01").
    session_id : str
        Session identifier (e.g., "ses-01").
    run : int
        Run number (1-based).
    trigger_keys : list of str, optional
        Keys that trigger acquisition start.
    monitor_position : str
        Monitor position: "left", "right", or "primary".
    simulator : VolumeSimulator, optional
        Volume simulator instance for offline testing. If provided, the task
        will automatically start the simulator thread, which injects a
        synthetic trigger key and then produces DICOMs at TR intervals.
    """

    task_id: str = "base_localizer"

    def __init__(
        self,
        output_dir: Union[str, Path],
        participant_id: str,
        session_id: str,
        attempt: int,
        run: int = 1,
        trigger_keys: list[str] | None = None,
        monitor_position: str = "right",
        simulator=None,
        **kwargs,
    ):
        super().__init__(
            trigger_keys=trigger_keys,
            simulator=simulator,
            **kwargs,
        )

        self.output_dir = Path(output_dir)
        self.participant_id = participant_id
        self.session_id = session_id
        self.run_number = run
        self.attempt_number = attempt
        self.monitor_position = monitor_position

        self._events: list[dict[str, Any]] = []
        self._experiment_clock: Any = None

    # =========================================================================
    # Stimulus presentation utilities
    # =========================================================================

    def show_fixation(self, duration: float, text: str = "+", height: float = 0.3) -> None:
        """Display fixation cross for a specified duration.

        Parameters
        ----------
        duration : float
            Duration in seconds.
        text : str
            Fixation character(s).
        height : float
            Text height.
        """
        from psychopy import core, event, visual

        fixation = visual.TextStim(
            self.win,
            text=text,
            height=height,
            color="white",
        )
        fixation.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        timer = core.CountdownTimer(duration)
        while timer.getTime() > 0 and self._running:
            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escaped during fixation")
            core.wait(0.01)

    # =========================================================================
    # BIDS events recording
    # =========================================================================

    def record_event(
        self,
        onset: float,
        duration: float,
        trial_type: str,
        **extras,
    ) -> None:
        """Record a single event for the BIDS events TSV.

        Parameters
        ----------
        onset : float
            Event onset time in seconds (relative to first trigger).
        duration : float
            Event duration in seconds.
        trial_type : str
            Event type (e.g., "self", "other", "rest", "question").
        **extras
            Additional columns (e.g., response, response_time, stim_file).
        """
        ev = {"onset": round(onset, 3), "duration": round(duration, 3), "trial_type": trial_type}
        ev.update(extras)
        self._events.append(ev)

    def save_events(self, path: Path | None = None) -> Path:
        """Write recorded events to a BIDS-compatible TSV file.

        Parameters
        ----------
        path : Path, optional
            Output path. Defaults to
            {output_dir}/{participant}_{session}_task-{task_id}_run-{run}_attempt-{attempt}_events.tsv.

        Returns
        -------
        Path
            Path to the saved events TSV.
        """
        import pandas as pd

        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = make_bids_run_prefix(
                self.participant_id,
                self.session_id,
                self.task_id,
                self.run_number,
                attempt=self.attempt_number,
            )
            path = self.output_dir / f"{prefix}_events.tsv"

        if not self._events:
            logger.warning("No events recorded, saving empty events file")
            df = pd.DataFrame(columns=["onset", "duration", "trial_type"])
        else:
            df = pd.DataFrame(self._events)
            # Ensure standard BIDS columns come first
            bids_cols = ["onset", "duration", "trial_type"]
            extra_cols = [c for c in df.columns if c not in bids_cols]
            df = df[bids_cols + extra_cols]

        df.to_csv(path, sep="\t", index=False)
        logger.info(f"Saved {len(self._events)} events to {path}")
        return path

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @abstractmethod
    def setup(self) -> None:
        """Initialize PsychoPy window, load stimuli, create visual objects."""
        pass

    @abstractmethod
    def run_paradigm(self) -> None:
        """Execute the stimulus presentation paradigm.

        This is where subclasses implement their block/event schedule.
        Should call record_event() for each stimulus onset.
        """
        pass

    def cleanup(self) -> None:
        """Record run completion (if task state configured), then release resources.

        Mirrors BaseTask.cleanup: records into TaskState before delegating to
        TaskBase. A no-op for localizers that declare no task_state_fields
        (then _task_state is None).
        """
        self._record_run_completion()
        super().cleanup()

    def run(self) -> Path:
        """Full task lifecycle: setup → paradigm → save events → cleanup.

        Returns
        -------
        Path
            Path to the saved BIDS events TSV file.
        """
        self._running = True
        events_path = None

        try:
            self.setup()
            self.run_paradigm()
        except KeyboardInterrupt:
            logger.info("Task interrupted")
        finally:
            self._running = False
            events_path = self.save_events()
            self.cleanup()

        return events_path
