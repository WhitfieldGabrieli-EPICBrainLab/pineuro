"""
Ball neurofeedback task.

Displays a ball that moves based on activation level, adapted from the MIND-BPD
study (https://github.com/cccbauer/MIND-BPD/blob/main/psychopy/balltask/rt-network_feedback_mgh.py).

Simplified version focusing on core feedback mechanics integrated with pyrtnf coordinator.
"""

# Standard library imports
import csv
import logging
import time
from pathlib import Path
from typing import Optional

# Third-party imports
import numpy as np
from psychopy import visual, core, event, monitors

# Local imports
from .base import BaseTask

# Configure module-level logger
logger = logging.getLogger(__name__)


class BallTask(BaseTask):
    """Ball task for dual-network neurofeedback.

    Displays two target circles (representing different brain networks) and a
    white ball that moves based on brain activation. When the ball reaches a
    target, it resets to center and the target shrinks.

    Based on MIND-BPD ball task with adaptive difficulty scaling.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data (must have 2 masks for dual-network)
    mask_names : list of str, optional
        Names of two masks to display. First mask appears at TOP of screen,
        second at bottom. E.g., ['ContA', 'DMN'] places ContA at top.
        If None, uses first two masks from coordinator.
    scale_factor : float, default=10
        Scaling factor for ball movement speed (constant throughout session)
    internal_scaler : float, default=10
        Additional scaling to slow down movement
    pda_outlier_threshold : float, default=2.0
        Threshold for outlier detection (skip movement if |activation| > threshold)
    trigger_keys : list of str, optional
        Keys that trigger acquisition start. Default is ['t'] for scanner trigger.
        In simulation mode, trigger is sent automatically.
    fullscreen : bool, default=False
        Whether to run in fullscreen mode
    target_fps : float, optional
        Fixed frame rate to use instead of measuring monitor refresh rate.
        When set, provides reproducible frame timing for testing.
        Default None uses measured/adaptive frame rate.

    Attributes
    ----------
    hit_counters : list of int
        Number of hits for each target circle
    ball_pos : tuple of float
        Current ball position (x, y)
    circles : list of visual.Circle
        Target circle stimuli

    Examples
    --------
    >>> from pyrtnf import run_feedback_session, BallTask
    >>> session_info = run_feedback_session(..., run_in_background=True)
    >>> task = BallTask(session_info['coordinator'], mask_names=['CEN', 'DMN'])
    >>> task.run()

    References
    ----------
    Adapted from MIND-BPD study ball task:
    https://github.com/cccbauer/MIND-BPD/blob/main/psychopy/balltask/rt-network_feedback_mgh.py
    """

    def __init__(
        self,
        coordinator,
        mask_names: Optional[list[str]] = None,
        scale_factor: float = 10.0,
        internal_scaler: float = 10.0,
        pda_outlier_threshold: float = 2.0,
        trigger_keys: list[str] = None,
        fullscreen: bool = False,
        target_fps: Optional[float] = None,
    ):
        # Don't call super().__init__() yet - need to set mask_names first
        self.coordinator = coordinator

        # Validate dual-network requirement
        if len(coordinator.mask_names) < 2:
            raise ValueError(
                f"BallTask requires at least 2 masks, got {len(coordinator.mask_names)}"
            )

        if mask_names is None:
            mask_names = coordinator.mask_names[:2]
        elif len(mask_names) != 2:
            raise ValueError(f"BallTask requires exactly 2 mask names, got {len(mask_names)}")

        self.mask_names = mask_names

        # Now initialize parent (which will use first mask_name for compatibility)
        super().__init__(coordinator, mask_name=mask_names[0])

        # Ball movement parameters
        self.scale_factor = scale_factor
        self.internal_scaler = internal_scaler
        self.pda_outlier_threshold = pda_outlier_threshold

        # Trigger configuration
        self.trigger_keys = trigger_keys if trigger_keys is not None else ['t']

        # Window parameters
        self.fullscreen = fullscreen
        self.target_fps = target_fps

        # Task state
        self.ball_pos = [0.0, 0.0]  # Start at center
        self.hit_counters = [0, 0]
        self.circle_positions = [1.0 / 3, -1.0 / 3]  # Vertical positions (top, bottom)
        self.circle_radii = [0.15, 0.15]
        self.circle_colors = ['yellow', 'lightblue']

        # Network positions for movement calculation
        # Top circle (index 0), Bottom circle (index 1)
        self.network_directions = np.array([1, -1])  # +1 = up, -1 = down

        # Override parent's data tracking to handle dual networks
        self._latest_data = {
            'activations': {name: 0.0 for name in self.mask_names},
            'volume': 0,
            'time': 0.0,
            'is_baseline': True,
        }

        # PsychoPy log setup with BIDS-style naming
        run_str = f"{coordinator.run_number:02d}"
        output_dir = Path(coordinator.output_dir)
        self._psychopy_log_path = output_dir / f"{coordinator.subject}_task-{coordinator.task}_run-{run_str}_psychopylog.tsv"
        self._psychopy_log_file = None
        self._psychopy_log_writer = None
        self._frame_counter = 0

        # Movement state (persists across frames for smooth animation)
        self.current_activity = 0.0  # Magnitude of movement speed
        self.current_direction = 0   # +1 (up) or -1 (down)
        self.pause_movement = False  # True when ball crosses circle center
        self.is_baseline = True      # True during baseline period
        self.current_volume = 0      # Current volume number
        self._task_start_time: float = 0.0  # Task start time for frame timestamps
        self._last_frame_time: float = 0.0  # For time-based movement (dt calculation)

    def _update_data(
        self,
        activations: dict,
        volume: int,
        elapsed: float,
        is_baseline: bool
    ):
        """Internal callback to update latest data for dual networks."""
        with self._lock:
            self._latest_data = {
                'activations': {name: activations.get(name, 0.0) for name in self.mask_names},
                'volume': volume,
                'time': elapsed,
                'is_baseline': is_baseline,
            }

    def setup(self):
        """Initialize PsychoPy window and ball task stimuli."""
        # Initialize PsychoPy log file
        self._psychopy_log_file = open(self._psychopy_log_path, 'w', newline='')
        self._psychopy_log_writer = csv.writer(self._psychopy_log_file, delimiter='\t')
        header = [
            'frame',
            'time',
            'volume',
            'ball_y_pos',
            'scale_factor',
            f'{self.mask_names[0]}_activation',
            f'{self.mask_names[1]}_activation',
            f'{self.mask_names[0]}_hits',
            f'{self.mask_names[1]}_hits'
        ]
        self._psychopy_log_writer.writerow(header)
        self._psychopy_log_file.flush()

        # Create minimal monitor to avoid "Monitor specification not found" warning
        mon = monitors.Monitor('neurofeedback')

        self.win = visual.Window(
            fullscr=self.fullscreen,
            screen=1,  # Display on secondary monitor
            monitor=mon,
            units='height',  # Use height units for consistent scaling
            color='black',
            allowGUI=False,
        )

        # Set frame rate - use fixed target_fps if specified, otherwise measure
        if self.target_fps is not None:
            self.frame_duration = 1.0 / self.target_fps
            logger.info(f"Using fixed framerate: {self.target_fps:.2f} Hz")
        else:
            measured_rate = self.win.getActualFrameRate()
            if measured_rate is not None:
                self.frame_duration = 1.0 / measured_rate
                logger.info(f"Measured framerate: {measured_rate:.2f} Hz")
            else:
                self.frame_duration = 1.0 / 60.0
                logger.warning("Could not measure framerate, assuming 60 Hz")

        # Calculate TR-to-frame ratio for smooth movement distribution
        # This distributes TR-based movement across monitor refresh frames
        self.tr_to_frame_ratio = self.coordinator.tr / self.frame_duration
        logger.info(f"TR-to-frame ratio: {self.tr_to_frame_ratio:.2f} frames per TR")

        # Create target circles
        self.circles = []
        for i in range(2):
            circle = visual.Circle(
                win=self.win,
                pos=(0, self.circle_positions[i]),
                radius=self.circle_radii[i],
                fillColor=None,
                lineColor=self.circle_colors[i],
                lineWidth=3,
            )
            self.circles.append(circle)

        # Create ball
        self.ball = visual.Circle(
            win=self.win,
            pos=self.ball_pos,
            radius=0.03,
            fillColor='white',
            lineColor='white',
        )

        # Text displays
        self.volume_text = visual.TextStim(
            win=self.win,
            text='',
            pos=(0, -0.45),
            height=0.03,
            color='white',
        )

        self.hits_text = visual.TextStim(
            win=self.win,
            text='',
            pos=(0, 0.45),
            height=0.03,
            color='white',
        )

        self.status_text = visual.TextStim(
            win=self.win,
            text='',
            pos=(0, 0.40),
            height=0.025,
            color='yellow',
        )

        # Fixation cross for baseline period
        self.fixation_cross = visual.TextStim(
            win=self.win,
            text='+',
            pos=(0, 0),
            height=0.3,
            color='white',
            font='Arial',
        )

        # Waiting for scanner text (displayed before trigger)
        self.waiting_text = visual.TextStim(
            win=self.win,
            text='waiting for scanner',
            pos=(0, 0),
            height=0.1,
            color='white',
            font='Arial',
        )

    def wait_for_trigger(self):
        """Display waiting screen and block until trigger received.

        Shows 'waiting for scanner' text and monitors for trigger key press.
        Sends trigger signal to coordinator when received.

        In simulation mode, this automatically proceeds after a brief delay.
        In real-time mode, this waits for the user to press a trigger key.
        """
        from ..feedback import logger

        self.waiting_text.draw()
        self.win.flip()

        # Check if we're in simulation mode (coordinator has a simulator)
        if self.coordinator._simulator is not None:
            logger.info("Simulation mode: auto-triggering")
            core.wait(0.1)  # Brief pause to show waiting screen
            self.coordinator.send_trigger()
        else:
            # Real-time mode: wait for trigger key
            logger.info(f"Waiting for trigger (keys: {self.trigger_keys})...")
            continueRoutine = True
            while continueRoutine:
                keys = event.getKeys(keyList=self.trigger_keys + ['escape'])
                if 'escape' in keys:
                    raise KeyboardInterrupt("User cancelled during trigger wait")
                if any(k in keys for k in self.trigger_keys):
                    logger.info(f"Trigger received: {keys}")
                    self.coordinator.send_trigger()
                    continueRoutine = False

        # Brief pause before starting task
        core.wait(0.1)

    def calculate_ball_movement(self, activations: dict) -> None:
        """Update movement parameters based on new activations.

        This should be called when NEW data arrives from the GLM (once per TR).
        Does NOT update position - that happens in apply_movement().

        Parameters
        ----------
        activations : dict
            Dictionary with activation values for both networks
        """
        acts = [activations[name] for name in self.mask_names]

        # Check for outlier - skip update but preserve existing activity
        if np.max(np.abs(acts)) > self.pda_outlier_threshold:
            return

        # Determine which network has higher activation
        max_idx = np.argmax(acts)

        # If mean of activations is zero, keep existing activity
        # Handles edge cases like [0.5, -0.5] where mean=0
        if np.nanmean(acts) == 0:
            return

        # Update activity magnitude (always positive)
        self.current_activity = np.abs(acts[0] - acts[1]) / 10.0

        # Update direction based on which network has higher activation
        self.current_direction = self.network_directions[max_idx]

    def apply_movement(self) -> None:
        """Apply movement to ball position based on elapsed time.

        Called every frame to create smooth movement. Movement is time-based
        (proportional to dt) for frame-rate independence and reproducibility.
        Uses current_activity and current_direction set by calculate_ball_movement().
        """
        # Calculate time delta since last frame
        current_time = core.getTime()
        dt = current_time - self._last_frame_time
        self._last_frame_time = current_time

        # Don't move if paused (waiting for hit to be processed)
        if self.pause_movement:
            return

        # Don't move during baseline
        if self.is_baseline:
            return

        # Calculate movement based on elapsed time (time-based, not frame-based)
        # Movement per second = activity * scale_factor / internal_scaler / TR
        # This ensures same total movement per TR regardless of frame rate
        movement_per_second = (
            self.current_activity
            * (self.scale_factor / self.internal_scaler)
            / self.coordinator.tr
        )

        dy = self.current_direction * movement_per_second * dt

        # Update position cumulatively
        self.ball_pos[1] += dy
        self.ball.pos = self.ball_pos

    def check_hits(self) -> bool:
        """Check if ball has hit either target circle.

        Returns
        -------
        bool
            True if hit occurred (ball was reset)
        """
        for i, circle in enumerate(self.circles):
            circle_y = self.circle_positions[i]

            # Check if ball has passed the circle center
            if self.network_directions[i] > 0:  # Top circle (upward)
                hit = self.ball_pos[1] >= circle_y
            else:  # Bottom circle (downward)
                hit = self.ball_pos[1] <= circle_y

            if hit:
                # Register hit
                self.hit_counters[i] += 1

                # Flash circle
                self.circles[i].fillColor = 'white'

                # Reset ball to center (preserve velocity for immediate continued movement)
                self.ball_pos = [0.0, 0.0]
                self.pause_movement = False  # Resume movement after reset

                # Reset frame time to prevent large dt jump after hit
                self._last_frame_time = core.getTime()

                # Shrink circle
                self.circle_radii[i] = max(self.circle_radii[i] * 0.9, 0.03)
                self.circles[i].radius = self.circle_radii[i]

                return True

        return False

    def check_pause_movement(self) -> None:
        """Check if ball should pause (has crossed circle center).

        Prevents ball from overshooting after crossing the threshold.
        """
        for i, circle in enumerate(self.circles):
            circle_y = self.circle_positions[i]

            # Check if ball has passed circle center
            if self.network_directions[i] > 0:  # Top circle
                if self.ball_pos[1] >= circle_y:
                    self.pause_movement = True
                    return
            else:  # Bottom circle
                if self.ball_pos[1] <= circle_y:
                    self.pause_movement = True
                    return

    def update(self, data: dict):
        """Update movement parameters based on latest activations.

        This is called when NEW GLM data arrives (once per TR).

        Parameters
        ----------
        data : dict
            Dictionary containing 'activations', 'volume', 'time', 'is_baseline'
        """
        self.is_baseline = data.get('is_baseline', False)

        # Update movement parameters with new data
        if not self.is_baseline:
            activations = data.get('activations', {})
            if activations:
                self.calculate_ball_movement(activations)

        # Store current volume for display
        self.current_volume = data.get('volume', 0)

    def _update_text(self):
        """Update text displays with current state."""
        self.volume_text.text = (
            f'Volume {self.current_volume} / {self.coordinator.n_volumes}  '
            f'Scale: {self.scale_factor:.1f}'
        )
        self.hits_text.text = (
            f'{self.mask_names[0]}: {self.hit_counters[0]}  '
            f'{self.mask_names[1]}: {self.hit_counters[1]}'
        )

    def _log_frame(self):
        """Log current frame data to TSV file."""
        # Get latest activations in thread-safe manner
        with self._lock:
            activations = self._latest_data['activations']

        # Calculate frame timestamp relative to task start
        frame_time = time.time() - self._task_start_time

        row = [
            self._frame_counter,
            frame_time,
            self.current_volume,
            self.ball_pos[1],
            self.scale_factor,
            activations[self.mask_names[0]],
            activations[self.mask_names[1]],
            self.hit_counters[0],
            self.hit_counters[1]
        ]
        self._psychopy_log_writer.writerow(row)
        self._psychopy_log_file.flush()

    def update_frame(self):
        """Update display for current frame.

        This should be called EVERY FRAME by the main loop (60Hz).
        Handles smooth ball movement between TR acquisitions.
        """
        # Apply movement (uses persisted activity/direction)
        self.apply_movement()

        # Check for pause condition (before checking hits)
        if not self.pause_movement:
            self.check_pause_movement()

        # Check for hits
        self.check_hits()

        # Update text displays
        self._update_text()

        # Draw scene based on baseline status
        if self.is_baseline:
            # During baseline: show only fixation cross
            self.fixation_cross.draw()
        else:
            # During active task: show ball, circles, and text
            for circle in self.circles:
                circle.draw()

            self.ball.draw()
            self.volume_text.draw()
            self.hits_text.draw()

        # Flip window
        self.win.flip()

        # Increment frame counter
        self._frame_counter += 1

        # Log frame data if not baseline
        if not self.is_baseline:
            self._log_frame()

        # Reset circle fill color after flash
        for circle in self.circles:
            circle.fillColor = None

    def cleanup(self):
        """Close window and quit."""
        # Close PsychoPy log file
        if self._psychopy_log_file is not None:
            self._psychopy_log_file.close()
            self._psychopy_log_file = None
            self._psychopy_log_writer = None

        self.win.close()
        core.quit()

    def run(self):
        """Main task loop - runs in main thread for PsychoPy.

        Overrides BaseTask.run() to call update_frame() at monitor refresh rate
        instead of update() once per TR.
        """
        from psychopy import event

        self._running = True
        self.setup()  # Creates window, measures framerate

        # Use PsychoPy clock for precise timing
        frame_clock = core.Clock()

        try:
            # Wait for coordinator to be ready
            self.coordinator.wait_until_ready(timeout=10.0)

            # Show waiting screen and wait for trigger
            self.wait_for_trigger()

            # Record task start time for frame timestamps
            self._task_start_time = core.getTime()
            self._last_frame_time = core.getTime()  # Initialize for time-based movement

            # Now start main task loop
            while self._running and not self.coordinator.is_complete():
                frame_clock.reset()

                # Check for errors from coordinator
                self.coordinator.check_error()

                # Get latest data and update state (refreshes is_baseline flag)
                data = self.get_latest_data()
                self.update(data)

                # Update display at target frame rate
                self.update_frame()

                # Check for quit
                if event.getKeys(['escape']):
                    break

                # Frame timing - use core.wait for more precise timing
                remaining = self.frame_duration - frame_clock.getTime()
                if remaining > 0:
                    core.wait(remaining, hogCPUperiod=0.001)

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.cleanup()
