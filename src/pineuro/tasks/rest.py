"""
Rest task for resting-state localizer runs.

Summary
-------
Simple task that displays a fixation cross for the duration of a rest run.
Used for resting-state acquisition to ensure participant is ready before
volume collection begins.

Dependencies
------------
PsychoPy, pandas
"""

import logging
from pathlib import Path
from typing import Union

from .localizer_base import BaseLocalizerTask
from .task_registry import TaskMetadata, TaskParameter, pineuro_task

logger = logging.getLogger(__name__)


REST_METADATA = TaskMetadata(
    task_id="rest",
    display_name="Rest Localizer",
    description="Fixation cross display for resting-state acquisition",
    roles={"localizer"},
    n_networks=0,
    parameters=[
        TaskParameter(
            key="duration",
            label="Duration (s)",
            type="float",
            default=200.0,
            min_value=10.0,
            max_value=600.0,
            hint="Total duration of fixation display (matches scan duration)",
        ),
    ],
    # Duration comes from the scan parameters (n_volumes * tr), not a
    # per-task localizer_task config section.
    localizer_params_from_scan=True,
)


@pineuro_task(REST_METADATA)
class RestTask(BaseLocalizerTask):
    """Rest task for resting-state localizer runs.

    Displays a fixation cross for the duration of the scan. Used to ensure the
    participant is ready and looking at the screen before volume acquisition
    begins.

    Parameters
    ----------
    output_dir : str or Path
        Directory for events TSV output.
    participant_id : str
        Participant identifier.
    run : int
        Run number (1-based).
    duration : float
        Duration of fixation display in seconds (should match scan duration).
    simulator : VolumeSimulator, optional
        Volume simulator instance for offline testing. If provided, the task
        will automatically start the simulator thread before waiting for the
        first volume to arrive.
    """

    task_id = "rest"

    def __init__(
        self,
        output_dir: Union[str, Path],
        participant_id: str,
        session_id: str,
        run: int = 1,
        duration: float = 200.0,
        simulator=None,
        **kwargs,
    ):
        super().__init__(
            output_dir=output_dir,
            participant_id=participant_id,
            session_id=session_id,
            run=run,
            simulator=simulator,
            **kwargs,
        )
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        self.duration = duration

    def setup(self) -> None:
        """Create PsychoPy window."""
        self.create_window(
            monitor_position=self.monitor_position,
        )
        logger.info(f"RestTask setup: duration={self.duration}s")

    def run_paradigm(self) -> None:
        """Execute the rest paradigm."""
        from psychopy import core

        # Instructions
        self.show_instructions(
            [
                "You will see a fixation cross (+) on the screen.\n\n"
                "Please relax and keep your eyes on the cross.\n\n"
                "Try to keep your head still and avoid moving.\n\n"
                "Press any button to begin."
            ]
        )

        # Wait for scanner trigger
        self.wait_for_trigger()
        self._experiment_clock = core.Clock()
        logger.info("[TIMING] Experiment clock started at trigger (t=0.00s)")

        # Show fixation for entire duration
        onset = self._experiment_clock.getTime()
        logger.info(f"[TIMING] Rest START: onset={onset:.2f}s, duration={self.duration}s")
        self.show_fixation(self.duration)
        self.record_event(onset, self.duration, "rest")
        logger.info(f"[TIMING] Rest END: time={self._experiment_clock.getTime():.2f}s")

        logger.info(f"Rest paradigm complete: {self.duration:.1f}s")
