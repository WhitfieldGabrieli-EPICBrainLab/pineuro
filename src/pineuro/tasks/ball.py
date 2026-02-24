"""
Ball neurofeedback task.

Summary
-------
Dual-network visual feedback task that moves a ball between two targets based
on real-time activations, with optional adaptive difficulty.

Features
--------
- Two target circles (top and bottom) with a central ball.
- Movement driven by two mask activations streamed from the coordinator.
- Adaptive scaling that speeds up or slows down movement based on hits.
- Supports feedback mode (movement) and transfer mode (stationary for practice).

Dependencies
------------
psychopy, numpy
"""

# Standard library imports
import csv
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Third-party imports
import numpy as np

# Note: psychopy imports are lazy (inside methods) to avoid initialising
# pyglet's OpenGL shadow window on the main thread, which breaks
# background-thread window creation on native Windows.
# Local imports
from .base import BaseTask

if TYPE_CHECKING:
    from .session import SessionState

# Configure module-level logger
logger = logging.getLogger(__name__)


# ==============================================================================
# HRF and Task Regressor Computation
# ==============================================================================

# Re-export for backward compatibility with existing imports from this module.
# compute_hrf and compute_boxcar_task_regressor are imported lazily via __getattr__
# to avoid pulling in pineuro.utils (and its heavy deps) at import time.
__all__ = ["BallTask", "compute_hrf", "compute_boxcar_task_regressor"]  # noqa: F822


def __getattr__(name):
    if name in ("compute_hrf", "compute_boxcar_task_regressor"):
        from pineuro.utils import compute_boxcar_task_regressor, compute_hrf

        globals()["compute_hrf"] = compute_hrf
        globals()["compute_boxcar_task_regressor"] = compute_boxcar_task_regressor
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Instruction text constants
INSTRUCTIONS_FIRST_TRANSFER = (
    "Next, you will continue the practice you just learned.\n\n"
    "You will see 2 circles with a white ball in the middle, "
    "but they won't move for now."
)

INSTRUCTIONS_LATER_TRANSFER = (
    "Great job! Next, you'll practice for another 2.5 min.\n\n"
    "This time the ball and circles will not move, "
    "so you don't need to check them."
)

INSTRUCTIONS_FIRST_FEEDBACK = (
    "Great job! Now, you'll get to try moving the ball with your practice!\n\n"
    "You will see the 2 circles and white ball again. "
    "When the white ball moves up towards the top yellow circle, "
    "this means you are in a mindful brain state with your practice.\n\n"
    "If the ball reaches either of the circles, it will move back to the center."
)

INSTRUCTIONS_LATER_FEEDBACK = (
    "Great job! Now, you're going to practice for another 2.5 min "
    "with more brain feedback from the ball.\n\n"
    "When the ball moves upwards, that corresponds to your practice."
)

INSTRUCTIONS_READY = (
    "When you see the plus sign (+), just relax.\n\n"
    "Once the circles appear, please start the practice. "
    "This scan will last 2.5 min."
)


class BallTask(BaseTask):
    """Ball task for dual-network neurofeedback.

    Displays two target circles (representing different brain networks) and a
    white ball that moves based on brain activation. When the ball reaches a
    target, it resets to center and the target shrinks.

    This is the reference implementation for a dual-network neurofeedback task.
    See ThermometerTask for a simpler single-network example.

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
    left_button : str, default='1'
        Key for moving slider left during post-run questions.
    right_button : str, default='2'
        Key for moving slider right during post-run questions.
    enter_button : str, default='0'
        Key for confirming slider selection during post-run questions.
    feedback_mode : bool, default=True
        If True, ball moves based on activation (feedback mode).
        If False, ball stays stationary but all computations still occur (non-feedback/transfer mode).
    run_number : int, default=1
        Overall run number in the session (used for BIDS naming).
    is_first_of_type : bool, default=True
        Whether this is the first run of this type (transfer or feedback).
        Controls instruction text: first runs show detailed explanations,
        later runs show brief reminders.
    circle_shrink_factor : float, default=0.9
        Multiplier applied to circle radius after each hit (0.9 = 10% shrink per hit,
        1.0 = no shrink). Lower values increase difficulty faster.

    Attributes
    ----------
    hit_counters : list of int
        Number of hits for each target circle
    ball_pos : tuple of float
        Current ball position (x, y)
    circles : list of visual.Circle
        Target circle stimuli

    Workflow
    --------
    - setup(): open window, create circles/ball, start log
    - update(): read latest activations, move ball, detect hits/shrink targets
    - cleanup(): close window and persist per-run results to session state

    Examples
    --------
    >>> from pineuro import run_feedback_session, BallTask
    >>> session_info = run_feedback_session(..., run_in_background=True)
    >>> task = BallTask(session_info['coordinator'], mask_names=['CEN', 'DMN'])
    >>> task.run()

    """

    # Task identifier for registry lookup
    task_id = "ball"

    def __init__(
        self,
        coordinator,
        mask_names: list[str] | None = None,
        scale_factor: float = 10.0,
        internal_scaler: float = 10.0,
        pda_outlier_threshold: float = 2.0,
        trigger_keys: list[str] | None = None,
        left_button: str = "1",
        right_button: str = "2",
        enter_button: str = "0",
        feedback_mode: bool = True,
        run_number: int = 1,
        is_first_of_type: bool = True,
        session_state: Optional["SessionState"] = None,
        # Adaptive difficulty parameters (passed to SessionState.record_run)
        min_hits_threshold: int = 3,
        max_hits_threshold: int = 5,
        scale_adjust_factor: float = 0.25,
        circle_shrink_factor: float = 0.9,
        presenter_monitor: str = "right",
        signal_mode: str = "activation",
        connectivity_target: str = "increase",
        **kwargs,  # Accept additional kwargs for factory compatibility
    ):
        self.coordinator = coordinator
        self._session_state = session_state
        self.run_number = run_number
        self.is_first_of_type = is_first_of_type
        self._signal_mode = signal_mode
        self._connectivity_target = connectivity_target
        self._last_connectivity = 0.0

        self._min_hits_threshold = min_hits_threshold
        self._max_hits_threshold = max_hits_threshold
        self._scale_adjust_factor = scale_adjust_factor
        self.circle_shrink_factor = circle_shrink_factor

        # Dual-network or connectivity mode: require 2 masks
        if len(coordinator.mask_names) < 2:
            raise ValueError(f"BallTask requires at least 2 masks, got {len(coordinator.mask_names)}")
        if mask_names is None:
            mask_names = coordinator.mask_names[:2]
        elif len(mask_names) != 2:
            raise ValueError(f"BallTask requires exactly 2 mask names, got {len(mask_names)}")

        self.mask_names = mask_names

        super().__init__(
            coordinator,
            mask_name=mask_names[0],
            feedback_mode=feedback_mode,
            session_state=session_state,
            **kwargs,
        )

        self.scale_factor = scale_factor
        self.internal_scaler = internal_scaler
        self.pda_outlier_threshold = pda_outlier_threshold

        self.trigger_keys = trigger_keys if trigger_keys is not None else ["t"]
        self.instruction_advance_keys = ["space"]

        self.left_button = left_button
        self.right_button = right_button
        self.enter_button = enter_button

        self.presenter_monitor = presenter_monitor

        self.ball_pos = [0.0, 0.0]
        self.hit_counters = [0, 0]
        self.circle_positions = [1.0 / 3, -1.0 / 3]
        self.circle_radii = [0.15, 0.15]
        self.circle_colors = ["yellow", "#1E90FF"]

        # Store slot info from coordinator for session state tracking
        self._slot_run_number = getattr(coordinator, "run_number", None)
        self._acq = getattr(coordinator, "acq", None)

        if session_state is not None:
            # Use slot-aware state lookup for retries
            if self._slot_run_number is not None and self._acq is not None:
                slot_state = session_state.get_state_for_slot(self._slot_run_number, self._acq)
                self.scale_factor = slot_state.get("scale_factor", self.scale_factor)
                logger.info(
                    f"Restored scale_factor from session: {self.scale_factor} (slot={self._slot_run_number}, acq={self._acq})"
                )
            else:
                self.scale_factor = session_state.get("scale_factor", self.scale_factor)
                logger.info(f"Restored scale_factor from session: {self.scale_factor}")

        self.network_directions = np.array([1, -1])  # +1 = up, -1 = down

        self._latest_data = {
            "activations": {name: 0.0 for name in self.mask_names},
            "volume": 0,
            "time": 0.0,
            "is_baseline": True,
        }

        output_dir = Path(coordinator.output_dir)
        # Use acq-based naming if coordinator has acq attribute
        if hasattr(coordinator, "acq") and coordinator.acq is not None:
            # Format acq as 2-digit number if int
            acq_str = f"{coordinator.acq:02d}" if isinstance(coordinator.acq, int) else coordinator.acq
            run_acq_str = f"acq-{acq_str}_run-{coordinator.run_number:02d}"
        else:
            run_acq_str = f"run-{coordinator.run_number:02d}"
        self._psychopy_log_path = (
            output_dir / f"{coordinator.subject}_task-{coordinator.task}_{run_acq_str}_psychopylog.tsv"
        )
        self._psychopy_log_file = None
        self._psychopy_log_writer = None
        self._frame_counter = 0

        self.current_activity = 0.0
        self.current_direction: float = 0  # +1 (up) or -1 (down)
        self.pause_movement = False
        self.is_baseline = True
        self.current_volume = 0
        self._task_start_time: float = 0.0
        self._last_frame_time: float = 0.0

        # Outlier tracking
        self.outlier_count = 0
        self._current_is_outlier = False

    def _update_data(self, activations: dict, volume: int, elapsed: float, is_baseline: bool, is_outlier: bool):
        """Internal callback with outlier detection for dual networks."""
        # Check for outliers based on activation threshold
        acts = [activations.get(name, 0.0) for name in self.mask_names]
        if np.max(np.abs(acts)) > self.pda_outlier_threshold:
            self.coordinator.report_outlier(volume)

        # Call base class to store data
        super()._update_data(activations, volume, elapsed, is_baseline, is_outlier)

        # Note: Don't set current_volume here - update() tracks it to detect new data

    def setup(self):
        """Initialize PsychoPy window and ball task stimuli."""
        from psychopy import logging as psychopy_logging
        from psychopy import visual

        psychopy_logging.console.setLevel(psychopy_logging.ERROR)

        # Initialize PsychoPy log file
        self._psychopy_log_file = open(self._psychopy_log_path, "w", newline="")  # noqa: SIM115
        self._psychopy_log_writer = csv.writer(self._psychopy_log_file, delimiter="\t")
        header = ["frame", "time", "volume", "ball_y_pos", "scale_factor"]
        for name in self.mask_names:
            header.append(f"{name}_activation")
        if self._signal_mode == "connectivity":
            header.extend(["connectivity", "target_hits", "nontarget_hits"])
        else:
            header.extend([f"{self.mask_names[0]}_hits", f"{self.mask_names[1]}_hits"])
        self._psychopy_log_writer.writerow(header)
        self._psychopy_log_file.flush()

        # Create window with cross-platform monitor targeting (from BaseTask)
        self.create_window(
            monitor_position=self.presenter_monitor,
            color="black",
            units="height",
        )

        # Use default frame duration for display loop timing (prevents CPU spinning)
        # Movement is time-based (dt) so actual frame rate doesn't affect physics
        self.frame_duration = 1.0 / 60.0  # ~60 Hz target for smooth display

        # Create target circles
        # Use thicker line width for visibility on full-screen display
        self.circles = []
        for i in range(2):
            circle = visual.Circle(
                win=self.win,
                pos=(0, self.circle_positions[i]),
                radius=self.circle_radii[i],
                fillColor=None,
                lineColor=self.circle_colors[i],
                lineWidth=8,
            )
            self.circles.append(circle)

        # Create ball
        self.ball = visual.Circle(
            win=self.win,
            pos=self.ball_pos,
            radius=0.03,
            fillColor="white",
            lineColor="white",
        )

        # Fixation cross for baseline period
        self.fixation_cross = visual.TextStim(
            win=self.win,
            text="+",
            pos=(0, 0),
            height=0.3,
            color="white",
            font=None,
        )

        # Question text stimulus (for post-run slider questions)
        self.question_text = visual.TextStim(
            win=self.win,
            text="",
            pos=(0, 0.2),
            height=0.06,
            wrapWidth=1.2,
            color="white",
            font=None,
        )

        # Slider instruction text
        self.slider_instruction_text = visual.TextStim(
            win=self.win,
            text=(
                "You'll see a few slider questions next\n"
                "Press the left and right buttons to move the slider\n"
                "And the top button to enter your response\n"
                "Press any button to continue"
            ),
            pos=(0, 0),
            height=0.06,
            wrapWidth=1.2,
            color="white",
            font=None,
            anchorVert="center",
        )

        # Setup questions CSV file with BIDS-style naming (use acq if available)
        output_dir = Path(self.coordinator.output_dir)
        if hasattr(self.coordinator, "acq") and self.coordinator.acq is not None:
            # Format acq as 2-digit number if int
            acq_str = f"{self.coordinator.acq:02d}" if isinstance(self.coordinator.acq, int) else self.coordinator.acq
            run_acq_str = f"acq-{acq_str}_run-{self.coordinator.run_number:02d}"
        else:
            run_acq_str = f"run-{self.coordinator.run_number:02d}"
        self._questions_file_path = (
            output_dir / f"{self.coordinator.subject}_task-{self.coordinator.task}_{run_acq_str}_questions.tsv"
        )
        self._questions_file = open(self._questions_file_path, "w", newline="")  # noqa: SIM115
        self._questions_writer = csv.writer(self._questions_file, delimiter="\t")
        self._questions_writer.writerow(["subject", "run", "feedback_mode", "question_text", "response", "rt"])
        self._questions_file.flush()

    def show_instructions(self):
        """Display instruction slides before task begins.

        Instructions vary based on feedback_mode and is_first_of_type:
        - First transfer run: explains task with no movement
        - Later transfer runs: brief reminder about no movement
        - First feedback run: explains ball movement mechanism
        - Later feedback runs: brief reminder
        """
        # Determine which instruction slides to show
        if not self.feedback_mode:
            if self.is_first_of_type:
                slides = [INSTRUCTIONS_FIRST_TRANSFER, INSTRUCTIONS_READY]
            else:
                slides = [INSTRUCTIONS_LATER_TRANSFER, INSTRUCTIONS_READY]
        else:  # feedback_mode
            if self.is_first_of_type:
                slides = [INSTRUCTIONS_FIRST_FEEDBACK, INSTRUCTIONS_READY]
            else:
                slides = [INSTRUCTIONS_LATER_FEEDBACK, INSTRUCTIONS_READY]

        # Use base class method
        super().show_instructions(
            slides=slides,
            advance_keys=self.instruction_advance_keys,
            text_height=0.06,
            wrap_width=1.2,
        )

    def _show_post_run_questions(
        self,
        left_button: str = "1",
        right_button: str = "2",
        enter_button: str = "3",
    ) -> list[dict[str, Any]]:
        """Display post-run slider questions from metadata.

        Shows BallTask-specific instruction screen first, then delegates
        to base class for question display using BallTask's button mapping.
        """
        from psychopy import core, event

        # Guard against setup() not completing (window/stimuli may not exist)
        if not hasattr(self, "win") or self.win is None:
            logger.info("No window available - skipping post-run questions")
            return []

        # Get questions from metadata
        metadata = self.get_task_metadata()
        if not metadata or not metadata.post_run_questions:
            logger.info("No post-run questions defined in metadata")
            return []

        # Show instruction screen (BallTask-specific)
        if not hasattr(self, "slider_instruction_text"):
            logger.warning("Slider instruction text not initialized; skipping post-run questions")
            return []
        self.slider_instruction_text.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        # Wait for any button press to continue
        event.clearEvents("keyboard")
        self._reset_poll_state()
        while True:
            # Check for stop signal from GUI
            if self.coordinator.was_stopped():
                logger.info("Stop signal received during slider instructions")
                raise KeyboardInterrupt("Stopped by user")

            self._inject_keys_if_unfocused([self.left_button, self.right_button, self.enter_button])
            keys = event.getKeys(keyList=[self.left_button, self.right_button, self.enter_button])
            if keys:
                break

            # Small sleep to avoid busy-waiting
            core.wait(0.01)

        # Delegate to base class for actual questions (reads from metadata)
        responses = super()._show_post_run_questions(
            left_button=self.left_button,
            right_button=self.right_button,
            enter_button=self.enter_button,
        )

        logger.info(f"Post-run questions complete: {len(responses)} responses")
        return responses

    def calculate_ball_movement(self, activations: dict) -> None:
        """Update movement parameters based on new activations.

        This should be called when NEW data arrives from the GLM (once per TR).
        Does NOT update position - that happens in apply_movement().

        Movement logic:
        - Ball moves toward the network with higher activation
        - Speed is proportional to the difference between network activations
        - Outliers are rejected to prevent spurious movements

        Parameters
        ----------
        activations : dict
            Dictionary with activation values for both networks
        """
        # Connectivity mode: use sliding-window correlation value directly
        if self._signal_mode == "connectivity":
            key = f"corr_{self.mask_names[0]}_{self.mask_names[1]}"
            val = activations.get(key, 0.0)
            self._last_connectivity = val
            if abs(val) > self.pda_outlier_threshold:
                self.outlier_count += 1
                self._current_is_outlier = True
                logger.info(f"Outlier detected: {key}={val:.4f} (threshold: {self.pda_outlier_threshold})")
                return
            self._current_is_outlier = False
            self.current_activity = abs(val) / 10.0
            # For "decrease" target, negate so negative correlation → ball UP
            directed_val = -val if self._connectivity_target == "decrease" else val
            self.current_direction = 1.0 if directed_val >= 0 else -1.0
            direction_label = "UP" if self.current_direction > 0 else "DOWN"
            logger.debug(
                f"Connectivity: r={val:.4f}, target={self._connectivity_target}, "
                f"directed={directed_val:.4f} → {direction_label}"
            )
            return

        # Dual-network activation mode
        acts = [activations[name] for name in self.mask_names]

        # Reject outliers to prevent extreme activations from dominating movement
        # Preserves existing activity direction/magnitude if outlier detected
        if np.max(np.abs(acts)) > self.pda_outlier_threshold:
            self.outlier_count += 1
            self._current_is_outlier = True
            logger.info(f"Outlier detected: {acts[0]:.2f}, {acts[1]:.2f} (threshold: {self.pda_outlier_threshold})")
            return

        # Not an outlier - clear flag
        self._current_is_outlier = False

        # Determine which network has higher activation (direction of movement)
        max_idx = np.argmax(acts)

        # Handle edge case: if activations balance out to zero mean, preserve current activity
        # Example: [0.5, -0.5] would give mean=0, indicating no clear winner
        if np.nanmean(acts) == 0:
            return

        # Compute movement magnitude from activation difference
        # Division by 10 scales to reasonable movement speed
        # Always positive - direction is determined separately
        self.current_activity = np.abs(acts[0] - acts[1]) / 10.0

        # Update direction based on which network has higher activation
        # network_directions[i] is +1 (upward) or -1 (downward) for each network
        self.current_direction = self.network_directions[max_idx]

    def apply_movement(self) -> None:
        """Apply movement to ball position based on elapsed time.

        Called every frame (~60 Hz) to create smooth movement. Movement is time-based
        (proportional to dt) for frame-rate independence and reproducibility.
        Uses current_activity and current_direction set by calculate_ball_movement().

        Movement formula:
        - dy = direction × (activity × scale_factor / internal_scaler / TR) × dt
        - Scales movement to be consistent per TR interval
        - Higher scale_factor = faster movement (adjusted adaptively between runs)
        - internal_scaler normalizes across different TR values
        """
        from psychopy import core

        # Calculate time delta since last frame
        current_time = core.getTime()
        dt = current_time - self._last_frame_time
        self._last_frame_time = current_time

        # Don't move if paused (ball passed circle center, waiting for hit detection)
        if self.pause_movement:
            return

        # Don't move during baseline period (collecting normalization data)
        if self.is_baseline:
            return

        # Don't move in non-feedback mode (transfer runs - ball frozen at center)
        if not self.feedback_mode:
            return

        # Calculate movement for this frame based on elapsed time
        # Movement per second = activity × scale_factor / internal_scaler / TR
        # This ensures consistent total movement per TR interval regardless of frame rate
        movement_per_second = self.current_activity * (self.scale_factor / self.internal_scaler) / self.coordinator.tr

        # Apply directional movement scaled by time delta
        dy = self.current_direction * movement_per_second * dt

        # Update position cumulatively
        self.ball_pos[1] += dy
        self.ball.pos = self.ball_pos

    def check_hits(self) -> bool:
        """Check if ball has hit either target circle.

        Hit detection:
        - Ball "hits" when it reaches the circle's center position
        - Top circle: hit when ball_y >= circle_y
        - Bottom circle: hit when ball_y <= circle_y

        On hit:
        - Increment hit counter for that network
        - Flash circle white briefly
        - Reset ball to center (0, 0)
        - Shrink circle by 10% (min radius 0.03) - increases difficulty
        - Resume movement immediately

        Returns
        -------
        bool
            True if hit occurred (ball was reset)
        """
        from psychopy import core

        for i, circle in enumerate(self.circles):
            circle_y = self.circle_positions[i]

            # Check if ball has reached the circle center
            if self.network_directions[i] > 0:  # Top circle (upward movement)
                hit = self.ball_pos[1] >= circle_y
            else:  # Bottom circle (downward movement)
                hit = self.ball_pos[1] <= circle_y

            if hit:
                # Register hit in counter
                self.hit_counters[i] += 1

                # Log hit event
                if self._signal_mode == "connectivity":
                    label = "target" if self.network_directions[i] > 0 else "nontarget"
                else:
                    label = self.mask_names[i]
                logger.info(f"Hit! {label} target (total: {self.hit_counters[i]})")

                # Flash circle white for visual feedback
                self.circles[i].fillColor = "white"

                # Reset ball to center position (game continues immediately)
                self.ball_pos = [0.0, 0.0]
                self.pause_movement = False  # Resume movement after reset

                # Reset frame time to prevent large dt jump after hit processing
                self._last_frame_time = core.getTime()

                # Shrink circle to increase difficulty (configurable reduction per hit, min 0.03)
                # Makes it progressively harder to hit the same target repeatedly
                # Move circle further from ball so inner edge recedes (harder to reach)
                old_radius = self.circle_radii[i]
                new_radius = max(old_radius * self.circle_shrink_factor, 0.03)
                radius_reduction = old_radius - new_radius

                # Move circle center further from ball (away from screen center)
                # This makes hit detection require more ball travel distance
                self.circle_positions[i] += self.network_directions[i] * radius_reduction
                self.circle_radii[i] = new_radius

                self.circles[i].pos = (0, self.circle_positions[i])
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

        Called every frame, but only processes new data when volume changes.

        Parameters
        ----------
        data : dict
            Dictionary containing 'activations', 'volume', 'time', 'is_baseline'
        """
        self.is_baseline = data.get("is_baseline", False)
        new_volume = data.get("volume", 0)

        # Only process when we have new data (volume changed)
        if new_volume != self.current_volume:
            self.current_volume = new_volume

            # Update movement parameters with new data
            if not self.is_baseline:
                activations = data.get("activations", {})
                if activations:
                    self.calculate_ball_movement(activations)

    def _log_frame(self):
        """Log current frame data to TSV file."""
        # Get latest activations in thread-safe manner
        with self._lock:
            activations = self._latest_data["activations"]

        # Calculate frame timestamp relative to task start
        frame_time = time.time() - self._task_start_time

        row = [self._frame_counter, frame_time, self.current_volume, self.ball_pos[1], self.scale_factor]
        for name in self.mask_names:
            row.append(activations.get(name, 0.0))
        if self._signal_mode == "connectivity":
            row.extend([self._last_connectivity, self.hit_counters[0], self.hit_counters[1]])
        else:
            row.extend([self.hit_counters[0], self.hit_counters[1]])
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

        # Draw scene based on baseline status
        if self.is_baseline:
            # During baseline: show only fixation cross
            self.fixation_cross.draw()
        else:
            # During active task: show ball and circles only
            for circle in self.circles:
                circle.draw()

            self.ball.draw()

        # Capture screenshot (back buffer) before flip for GUI mirror
        self._capture_and_emit_screenshot()

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

    def save_state(self, session_state: Optional["SessionState"] = None) -> None:
        """Save current task state to session state for multi-run continuity.

        Parameters
        ----------
        session_state : SessionState, optional
            Session state to save to. If None, uses the session_state passed
            to __init__. If neither is available, does nothing.
        """
        state = session_state or self._session_state
        if state is None:
            logger.debug("No session state provided, skipping save")
            return

        run_type = "feedback" if self.feedback_mode else "transfer"
        state.record_run(
            run_type=run_type,
            volumes_processed=self.current_volume,
            task_state_snapshot={
                "hit_counters": self.hit_counters,
                "scale_factor": self.scale_factor,
                "circle_radii": self.circle_radii,
                "min_hits_threshold": self._min_hits_threshold,
                "max_hits_threshold": self._max_hits_threshold,
                "scale_adjust_factor": self._scale_adjust_factor,
            },
            # Slot-based tracking
            slot_run_number=self._slot_run_number,
            acq=self._acq,
        )

    def get_run_results(self) -> dict:
        """Return BallTask-specific results from the completed run."""
        results = super().get_run_results()
        results.update(
            {
                "hit_counters": self.hit_counters.copy(),
                "scale_factor": self.scale_factor,
                "circle_radii": self.circle_radii.copy(),
            }
        )
        return results

    def _apply_adaptive_difficulty(self) -> None:
        """Apply adaptive difficulty adjustment to scale_factor.

        Called before recording run completion. Adjusts scale_factor based on
        hit counts for the run that just completed:
        - If max single-network hits > max_hits_threshold: decrease scale_factor
        - If total hits < min_hits_threshold: increase scale_factor
        - Only adjusts for feedback runs, not transfer runs
        - Only adjusts for new run slots, not retries (acq > 1)
        """
        if self._session_state is None:
            return

        run_type = "feedback" if self.feedback_mode else "transfer"

        # Check if this is a retry
        is_retry = False
        if self._slot_run_number is not None:
            for r in self._session_state.completed_runs:
                if r.slot_run_number == self._slot_run_number:
                    is_retry = True
                    break

        if is_retry:
            logger.info(f"  Retry (acq={self._acq}): skipping scale_factor adjustment")
            return

        if run_type != "feedback":
            logger.info("  Transfer run - no adaptive difficulty adjustment")
            return

        total_hits = sum(self.hit_counters)
        max_single = max(self.hit_counters)
        old_scale = self.scale_factor

        if max_single > self._max_hits_threshold:
            self.scale_factor = old_scale * (1 - self._scale_adjust_factor)
            logger.info(
                f"  Adaptive difficulty: {max_single} hits > {self._max_hits_threshold} max -> "
                f"decreasing scale_factor {old_scale:.1f} -> {self.scale_factor:.1f}"
            )
        elif total_hits < self._min_hits_threshold:
            self.scale_factor = old_scale * (1 + self._scale_adjust_factor)
            logger.info(
                f"  Adaptive difficulty: {total_hits} total hits < {self._min_hits_threshold} min -> "
                f"increasing scale_factor {old_scale:.1f} -> {self.scale_factor:.1f}"
            )
        else:
            logger.info(f"  Adaptive difficulty: hits in range -> keeping scale_factor at {old_scale:.1f}")

    def _record_run_completion(self) -> None:
        """Record run completion with BallTask-specific state.

        Overrides BaseTask to apply adaptive difficulty and include
        BallTask state in the task_state_snapshot.
        """
        if self._session_state is None:
            return

        # Apply adaptive difficulty before recording
        self._apply_adaptive_difficulty()

        # Build task state snapshot with all BallTask-specific state
        snapshot = {
            "hit_counters": self.hit_counters.copy(),
            "scale_factor": self.scale_factor,
            "circle_radii": self.circle_radii.copy(),
        }

        run_type = "feedback" if self.feedback_mode else "transfer"
        self._session_state.record_run(
            run_type=run_type,
            volumes_processed=self.current_volume,
            task_state_snapshot=snapshot,
            slot_run_number=self._slot_run_number,
            acq=self._acq,
        )

    def get_mirror_state(self) -> dict:
        """Return current visual state for GUI mirroring.

        Returns a dictionary describing the current visual display using
        normalized coordinates (0-1) that can be rendered by a generic
        GUI mirror component.

        Returns
        -------
        dict
            Dictionary with 'background' color and 'shapes' list.
            Coordinates are normalized: x (0=left, 1=right), y (0=top, 1=bottom).
        """
        shapes = []

        # Use the instance variable that's updated in update() from the main loop
        # This is guaranteed to be in sync with what PsychoPy is displaying
        is_baseline = self.is_baseline

        if is_baseline:
            # Fixation cross during baseline
            cross_size = 0.1  # Size in normalized units
            shapes.extend(
                [
                    {
                        "type": "line",
                        "x1": 0.5 - cross_size,
                        "y1": 0.5,
                        "x2": 0.5 + cross_size,
                        "y2": 0.5,
                        "stroke": "white",
                        "stroke_width": 3,
                    },
                    {
                        "type": "line",
                        "x1": 0.5,
                        "y1": 0.5 - cross_size,
                        "x2": 0.5,
                        "y2": 0.5 + cross_size,
                        "stroke": "white",
                        "stroke_width": 3,
                    },
                ]
            )
        else:
            # Active task: show circles and ball
            # Convert PsychoPy height units to normalized 0-1 coordinates
            # PsychoPy: y in [-0.5, 0.5], we need [0, 1] with y inverted
            for pos, color, radius in zip(self.circle_positions, self.circle_colors, self.circle_radii, strict=False):
                # +0.33 -> 0.17 (top), -0.33 -> 0.83 (bottom)
                y = 0.5 - pos
                shapes.append(
                    {"type": "circle", "x": 0.5, "y": y, "radius": radius, "stroke": color, "stroke_width": 3}
                )

            # Ball
            ball_y = 0.5 - self.ball_pos[1]
            shapes.append({"type": "circle", "x": 0.5, "y": ball_y, "radius": 0.03, "fill": "white"})

            # Hit counter text
            if self._signal_mode == "connectivity":
                hit_text = f"target: {self.hit_counters[0]}  nontarget: {self.hit_counters[1]}"
            else:
                hit_text = f"{self.mask_names[0]}: {self.hit_counters[0]}  {self.mask_names[1]}: {self.hit_counters[1]}"
            shapes.append(
                {
                    "type": "text",
                    "content": hit_text,
                    "x": 0.5,
                    "y": 0.95,
                    "size": 10,
                    "color": "gray",
                }
            )

        return {"background": "black", "shapes": shapes}

    def cleanup(self):
        """Close window and quit."""
        # Log outlier summary statistics
        if self.current_volume > 0:
            outlier_pct = (self.outlier_count / self.current_volume) * 100
            logger.info(f"Outlier summary: {self.outlier_count}/{self.current_volume} volumes ({outlier_pct:.1f}%)")

        # Note: State is saved in _record_run_completion() via super().cleanup()

        # Close PsychoPy log file
        if self._psychopy_log_file is not None:
            self._psychopy_log_file.close()
            self._psychopy_log_file = None
            self._psychopy_log_writer = None

        # Show post-run questions BEFORE deleting stimuli (they need GL context)
        # BaseTask.cleanup() would normally show these, but it's too late after
        # stimulus deletion. The guard in _show_post_run_questions() prevents
        # double-showing when super().cleanup() runs.
        should_show = True
        if hasattr(self, "coordinator") and hasattr(self.coordinator, "should_show_questions"):
            should_show = self.coordinator.should_show_questions()
        if should_show and hasattr(self, "win") and self.win is not None:
            # Temporarily clear the stop event so the question wait-loops
            # don't immediately raise KeyboardInterrupt (we're in cleanup
            # and the user explicitly asked for questions via show_questions).
            stop_event = getattr(self.coordinator, "_stop_event", None)
            was_set = stop_event is not None and stop_event.is_set()
            if was_set:
                stop_event.clear()
            try:
                self._show_post_run_questions()
            except KeyboardInterrupt:
                logger.info("Questions interrupted during cleanup")
            finally:
                if was_set and stop_event is not None:
                    stop_event.set()

        # Delete visual stimuli while GL context is still alive
        # (prevents PsychoPy TextStim.__del__ access violations during GC)
        for attr in ["ball", "fixation_cross", "question_text", "slider_instruction_text"]:
            if hasattr(self, attr):
                delattr(self, attr)

        # Call base class cleanup (closes window, clears events, resets pyglet)
        super().cleanup()

        # Close questions file after base class has written to it
        if self._questions_file is not None:
            self._questions_file.close()
            self._questions_file = None
            self._questions_writer = None
            logger.info(f"Questions saved: {self._questions_file_path}")

    def run(self):
        """Main task loop - runs in main thread for PsychoPy.

        Overrides BaseTask.run() to call update_frame() at monitor refresh rate
        instead of update() once per TR.
        """
        from psychopy import core

        self._running = True

        try:
            self.setup()  # Creates window and initializes task stimuli

            # Use PsychoPy clock for precise timing
            frame_clock = core.Clock()
            # Wait for coordinator to be ready
            self.coordinator.wait_until_ready(timeout=10.0)

            # Show instructions
            self.show_instructions()

            # Show waiting screen and wait for trigger (uses base class method)
            self.wait_for_trigger()

            # Record task start time for frame timestamps
            self._task_start_time = core.getTime()
            self._last_frame_time = core.getTime()  # Initialize for time-based movement

            logger.info("=== BallTask Session Started ===")
            logger.info(f"Mode: {'FEEDBACK' if self.feedback_mode else 'NON-FEEDBACK (transfer)'}")
            logger.info(f"Signal mode: {self._signal_mode}")
            if self._signal_mode == "connectivity":
                logger.info(f"Connectivity target: {self._connectivity_target}")
                logger.info(f"Connectivity key: corr_{self.mask_names[0]}_{self.mask_names[1]}")
                logger.info(f"Networks: {self.mask_names[0]} (top) vs {self.mask_names[1]} (bottom)")
            else:
                logger.info(f"Networks: {self.mask_names[0]} (top) vs {self.mask_names[1]} (bottom)")
            logger.info(f"Scale factor: {self.scale_factor}")
            logger.info(f"Target volumes: {self.coordinator.n_volumes}")
            logger.info(f"Baseline volumes: {self.coordinator.n_baseline}")
            logger.info(f"PsychoPy log: {self._psychopy_log_path.name}")

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

                # Frame timing - use core.wait for more precise timing
                remaining = self.frame_duration - frame_clock.getTime()
                if remaining > 0:
                    core.wait(remaining, hogCPUperiod=0.001)

            # Process final volume if not yet seen (race condition fix)
            # The loop exits when is_complete() returns True, which happens
            # immediately when the coordinator finishes the last volume.
            # This ensures we don't miss the final volume's data.
            data = self.get_latest_data()
            if data.get("volume", 0) > self.current_volume:
                self.update(data)
                self.update_frame()

        except KeyboardInterrupt:
            logger.info("Task interrupted by user")
            pass
        finally:
            self._running = False

            # Log session summary
            logger.info("")
            logger.info("=== BallTask Session Complete ===")
            logger.info(f"Total volumes: {self.current_volume}/{self.coordinator.n_volumes}")
            if self._signal_mode == "connectivity":
                logger.info(f"Hits - target: {self.hit_counters[0]}, nontarget: {self.hit_counters[1]}")
            else:
                logger.info(
                    f"Hits - {self.mask_names[0]}: {self.hit_counters[0]}, {self.mask_names[1]}: {self.hit_counters[1]}"
                )
            logger.info(f"Total hits: {sum(self.hit_counters)}")
            logger.info(f"Final scale factor: {self.scale_factor:.1f}")
            logger.info(f"Total frames rendered: {self._frame_counter}")
            if self.current_volume > 0:
                avg_frames_per_vol = self._frame_counter / self.current_volume
                logger.info(f"Average frames per volume: {avg_frames_per_vol:.1f}")
            logger.info(f"PsychoPy log saved: {self._psychopy_log_path}")

            # cleanup() handles post-run questions via super().cleanup()
            self.cleanup()


# ==============================================================================
# Task Metadata for GUI Configuration
# ==============================================================================

from typing import Any

from .task_registry import (
    NetworkRole,
    PostRunQuestion,
    SessionStateField,
    TaskMetadata,
    TaskParameter,
    register_task,
)


def _validate_ball_hits(config: dict[str, Any]) -> None:
    """Cross-parameter validator for ball task hit thresholds."""
    min_hits = config.get("min_hits_threshold")
    max_hits = config.get("max_hits_threshold")

    if min_hits is not None and max_hits is not None and min_hits >= max_hits:
        raise ValueError("Min hits threshold must be < max hits threshold")


BALL_TASK_METADATA = TaskMetadata(
    task_id="ball",
    display_name="Ball Task",
    description="Neurofeedback with ball control (single or dual network)",
    parameters=[
        TaskParameter(
            key="use_transfer_runs",
            label="Use Transfer Runs",
            type="bool",
            default=True,
            hint="Include transfer runs (1 before feedback, 1 after)",
        ),
        TaskParameter(
            key="circle_shrink_factor",
            label="Circle Shrink Factor",
            type="float",
            default=0.9,
            min_value=0.5,
            max_value=1.0,
            hint="Circle size multiplier after hit (0.9 = 10% shrink, 1.0 = no shrink)",
        ),
        TaskParameter(
            key="initial_scale_factor",
            label="Initial Scale Factor",
            type="float",
            default=10.0,
            min_value=1.0,
            max_value=50.0,
            hint="Starting difficulty (larger = easier)",
        ),
        TaskParameter(
            key="scale_adjust_factor",
            label="Scale Adjustment Factor",
            type="float",
            default=0.25,
            min_value=0.01,
            max_value=1.0,
            hint="How much to adjust scale between runs (0.25 = 25% change)",
        ),
        TaskParameter(
            key="min_hits_threshold",
            label="Min Hits (Too Slow)",
            type="int",
            default=3,
            min_value=0,
            max_value=10,
            hint="If hits < this, increase scale factor 25%",
        ),
        TaskParameter(
            key="max_hits_threshold",
            label="Max Hits (Too Fast)",
            type="int",
            default=5,
            min_value=0,
            max_value=10,
            hint="If hits > this, decrease scale factor 25%",
        ),
        TaskParameter(
            key="trigger_keys",
            label="Scanner Trigger Keys",
            type="list[str]",
            default=["t", "+", "5", "equal"],
            hint="Keys that trigger volume acquisition (comma-separated)",
        ),
        TaskParameter(
            key="left_button",
            label="Left Control Button",
            type="str",
            default="1",
            hint="Key for moving slider left",
        ),
        TaskParameter(
            key="right_button",
            label="Right Control Button",
            type="str",
            default="2",
            hint="Key for moving slider right",
        ),
        TaskParameter(
            key="enter_button",
            label="Enter/Confirm Button",
            type="str",
            default="0",
            hint="Key for confirming selections",
        ),
    ],
    cross_parameter_validator=_validate_ball_hits,
    n_networks=2,  # Requires 2 networks (dual-network activation or connectivity)
    network_roles=[
        NetworkRole("top_network", "Top Network (upward)", "Ball moves up when this network is more active"),
        NetworkRole("bottom_network", "Bottom Network (downward)", "Ball moves down when this network is more active"),
    ],
    session_state_fields=[
        SessionStateField(
            key="scale_factor",
            type="float",
            default=10.0,
            description="Adaptive difficulty scale factor for ball movement",
        ),
        SessionStateField(
            key="hit_counters",
            type="list[int]",
            default=[0, 0],
            description="Cumulative hit counts per network [top, bottom]",
        ),
        SessionStateField(
            key="circle_radii",
            type="list[float]",
            default=[0.15, 0.15],
            description="Current circle radii (shrink on hits)",
        ),
    ],
    post_run_questions=[
        # Core questions (always asked)
        PostRunQuestion(
            text="How often were you using the Mindful Describing practice?",
            left_label="Never",
            right_label="Always",
            condition="always",
        ),
        PostRunQuestion(
            text="How often did you check the position of the ball?",
            left_label="Never",
            right_label="All the time",
            condition="always",
        ),
        PostRunQuestion(
            text="How difficult was it to apply Mindful Describing?",
            left_label="Not at all",
            right_label="Very Difficult",
            condition="always",
        ),
        PostRunQuestion(
            text="How calm do you feel right now?",
            left_label="Not at all",
            right_label="Very calm",
            condition="always",
        ),
        # Conditional questions (last feedback run only)
        PostRunQuestion(
            text="The ball seemed to move up while I was practicing Mindful Describing",
            left_label="False",
            right_label="True",
            condition="last_feedback_only",
        ),
        PostRunQuestion(
            text="The ball seemed to move down while my mind was wandering.",
            left_label="False",
            right_label="True",
            condition="last_feedback_only",
        ),
        PostRunQuestion(
            text="The ball seemed to move down while I was trying to control it.",
            left_label="False",
            right_label="True",
            condition="last_feedback_only",
        ),
    ],
    design_type="block",
)

# Auto-register on import
register_task(BALL_TASK_METADATA)

# Register with factory for create_task() support
from .factory import register_task_class

register_task_class("ball", BallTask)
