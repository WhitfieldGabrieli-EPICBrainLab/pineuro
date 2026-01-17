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
from typing import Optional, TYPE_CHECKING

# Third-party imports
import numpy as np
from numpy.typing import NDArray
from psychopy import visual, core, event, monitors, prefs, logging as psychopy_logging
from scipy.stats import gamma as gamma_dist

# Silence PsychoPy warnings about monitor specifications and fonts
psychopy_logging.console.setLevel(psychopy_logging.ERROR)

# Local imports
from .base import BaseTask

if TYPE_CHECKING:
    from .session import SessionState

# Configure module-level logger
logger = logging.getLogger(__name__)


# ==============================================================================
# HRF and Task Regressor Computation (Ball Task Specific)
# ==============================================================================


def compute_hrf(tr: float, duration: float = 32.0) -> NDArray[np.float64]:
    """Compute canonical double-gamma HRF (SPM-style).

    Parameters
    ----------
    tr : float
        Repetition time in seconds.
    duration : float, optional
        Duration of HRF in seconds. Default is 32s.

    Returns
    -------
    hrf : ndarray
        HRF sampled at TR intervals, normalized to peak at 1.0.

    Notes
    -----
    Uses SPM canonical HRF parameters:
    - Positive gamma: shape=6, scale=1 (peak ~5s)
    - Negative gamma: shape=16, scale=1 (peak ~16s)
    - Undershoot ratio: 1/6
    """
    # Time points at TR intervals
    t = np.arange(0, duration, tr)

    # SPM canonical HRF parameters
    # Positive response: gamma(6, 1) peaks at ~5s
    # Undershoot: gamma(16, 1) peaks at ~16s
    a1, b1 = 6.0, 1.0  # shape, scale for positive response
    a2, b2 = 16.0, 1.0  # shape, scale for undershoot
    c = 1.0 / 6.0  # undershoot ratio

    # Double-gamma HRF
    hrf = gamma_dist.pdf(t, a1, scale=b1) - c * gamma_dist.pdf(t, a2, scale=b2)

    # Normalize to peak at 1.0
    hrf = hrf / np.max(hrf)

    return hrf.astype(np.float64)


def compute_boxcar_task_regressor(
    n_baseline: int,
    n_volumes: int,
    tr: float,
) -> NDArray[np.float64]:
    """Compute HRF-convolved boxcar task regressor for ball task.

    Creates a task regressor where:
    - Baseline period (0 to n_baseline-1): task = 0
    - Feedback period (n_baseline onwards): task = 1 convolved with HRF

    This is specific to the ball task design with a single baseline/active
    block structure.

    Parameters
    ----------
    n_baseline : int
        Number of baseline volumes (task = 0).
    n_volumes : int
        Total number of volumes.
    tr : float
        Repetition time in seconds.

    Returns
    -------
    regressor : ndarray, shape (n_volumes,)
        Task regressor values for each volume.

    Notes
    -----
    The convolution produces the characteristic HRF response shape:
    - Stays at 0 during baseline
    - Ramps up after baseline onset following HRF shape
    - Reaches steady state (~1.0) after ~25-30s
    """
    # Create boxcar: 0 during baseline, 1 during feedback
    boxcar = np.zeros(n_volumes, dtype=np.float64)
    boxcar[n_baseline:] = 1.0

    # Compute HRF kernel
    hrf = compute_hrf(tr)

    # Convolve boxcar with HRF
    convolved = np.convolve(boxcar, hrf, mode='full')

    # Trim to original length
    regressor = convolved[:n_volumes]

    # Normalize so steady-state is ~1.0 (divide by sum of HRF)
    # This ensures the regressor reaches 1.0 after the HRF response settles
    hrf_sum = np.sum(hrf)
    if hrf_sum > 0:
        regressor = regressor / hrf_sum

    return regressor.astype(np.float64)


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
    fullscreen : bool, default=False
        Whether to run in fullscreen mode
    feedback_mode : bool, default=True
        If True, ball moves based on activation (feedback mode).
        If False, ball stays stationary but all computations still occur (non-feedback/transfer mode).
    run_number : int, default=1
        Overall run number in the session (used for BIDS naming).
    is_first_of_type : bool, default=True
        Whether this is the first run of this type (transfer or feedback).
        Controls instruction text: first runs show detailed explanations,
        later runs show brief reminders.

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
    >>> from pyrtnf import run_feedback_session, BallTask
    >>> session_info = run_feedback_session(..., run_in_background=True)
    >>> task = BallTask(session_info['coordinator'], mask_names=['CEN', 'DMN'])
    >>> task.run()

    """

    def __init__(
        self,
        coordinator,
        mask_names: Optional[list[str]] = None,
        scale_factor: float = 10.0,
        internal_scaler: float = 10.0,
        pda_outlier_threshold: float = 2.0,
        trigger_keys: list[str] = None,
        left_button: str = '1',
        right_button: str = '2',
        enter_button: str = '0',
        fullscreen: bool = False,
        feedback_mode: bool = True,
        run_number: int = 1,
        is_first_of_type: bool = True,
        session_state: Optional["SessionState"] = None,
        # Adaptive difficulty parameters (passed to SessionState.record_run)
        min_hits_threshold: int = 3,
        max_hits_threshold: int = 5,
        scale_adjust_factor: float = 0.25,
    ):
        self.coordinator = coordinator
        self._session_state = session_state
        self.run_number = run_number
        self.is_first_of_type = is_first_of_type
        
        self._min_hits_threshold = min_hits_threshold
        self._max_hits_threshold = max_hits_threshold
        self._scale_adjust_factor = scale_adjust_factor

        if len(coordinator.mask_names) < 2:
            raise ValueError(
                f"BallTask requires at least 2 masks, got {len(coordinator.mask_names)}"
            )

        if mask_names is None:
            mask_names = coordinator.mask_names[:2]
        elif len(mask_names) != 2:
            raise ValueError(f"BallTask requires exactly 2 mask names, got {len(mask_names)}")

        self.mask_names = mask_names

        super().__init__(coordinator, mask_name=mask_names[0])

        self.scale_factor = scale_factor
        self.internal_scaler = internal_scaler
        self.pda_outlier_threshold = pda_outlier_threshold

        self.trigger_keys = trigger_keys if trigger_keys is not None else ['t']
        self.instruction_advance_keys = ['space']
        
        self.left_button = left_button
        self.right_button = right_button
        self.enter_button = enter_button

        self.fullscreen = fullscreen

        self.feedback_mode = feedback_mode

        self.ball_pos = [0.0, 0.0]
        self.hit_counters = [0, 0]
        self.circle_positions = [1.0 / 3, -1.0 / 3]
        self.circle_radii = [0.15, 0.15]
        self.circle_colors = ['yellow', 'lightblue']
        
        if session_state is not None:
            self.scale_factor = session_state.scale_factor
            logger.info(f"Restored scale_factor from session: {self.scale_factor}")

        self.network_directions = np.array([1, -1])  # +1 = up, -1 = down

        self._latest_data = {
            'activations': {name: 0.0 for name in self.mask_names},
            'volume': 0,
            'time': 0.0,
            'is_baseline': True,
        }

        output_dir = Path(coordinator.output_dir)
        # Use acq-based naming if coordinator has acq attribute
        if hasattr(coordinator, 'acq') and coordinator.acq is not None:
            # Format acq as 2-digit number if int
            acq_str = f"{coordinator.acq:02d}" if isinstance(coordinator.acq, int) else coordinator.acq
            run_acq_str = f"acq-{acq_str}_run-{coordinator.run_number:02d}"
        else:
            run_acq_str = f"run-{coordinator.run_number:02d}"
        self._psychopy_log_path = output_dir / f"{coordinator.subject}_task-{coordinator.task}_{run_acq_str}_psychopylog.tsv"
        self._psychopy_log_file = None
        self._psychopy_log_writer = None
        self._frame_counter = 0

        self.current_activity = 0.0
        self.current_direction = 0   # +1 (up) or -1 (down)
        self.pause_movement = False
        self.is_baseline = True
        self.current_volume = 0
        self._task_start_time: float = 0.0
        self._last_frame_time: float = 0.0

        # Outlier tracking
        self.outlier_count = 0
        self._current_is_outlier = False

    def _update_data(
        self,
        activations: dict,
        volume: int,
        elapsed: float,
        is_baseline: bool,
        is_outlier: bool
    ):
        """Internal callback to update latest data for dual networks.

        Performs outlier detection and reports to coordinator if detected.
        """
        # Check for outliers based on activation threshold
        acts = [activations.get(name, 0.0) for name in self.mask_names]
        if np.max(np.abs(acts)) > self.pda_outlier_threshold:
            self.coordinator.report_outlier(volume)

        with self._lock:
            self._latest_data = {
                'activations': {name: activations.get(name, 0.0) for name in self.mask_names},
                'volume': volume,
                'time': elapsed,
                'is_baseline': is_baseline,
            }
            # Keep current_volume in sync even if the main loop exits right after the last callback
            self.current_volume = volume

    def _move_window_to_secondary(self):
        """Move PsychoPy window to secondary monitor using Xlib.

        This is a workaround for Flet/XWayland ignoring PsychoPy's screen parameter.
        Only works on Linux/X11 systems.
        """
        try:
            # Get target screen position
            screens = self.win.winHandle.display.get_screens()
            target = screens[1] if len(screens) > 1 else screens[0]

            # Move window using Xlib
            from Xlib import display as xlib_display
            d = xlib_display.Display()
            x_win = d.create_resource_object('window', self.win.winHandle._window)
            x_win.configure(x=target.x, y=target.y)
            d.sync()
            d.close()
        except Exception as e:
            logger.warning(f"Could not move window to secondary: {e}")

    def setup(self):
        """Initialize PsychoPy window and ball task stimuli."""
        from psychopy import visual

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

        # Platform-specific window creation for secondary monitor
        import sys
        if sys.platform == 'darwin':
            # macOS: screen parameter works directly
            self.win = visual.Window(
                fullscr=self.fullscreen,
                monitor=mon,
                screen=1,
                units='height',
                color='black',
                allowGUI=False,
                waitBlanking=False,
            )
        else:
            # Linux (including WSL2): Need Xlib workaround for Flet/XWayland
            import pyglet
            screens = pyglet.canvas.Display().get_screens()
            target = screens[1] if len(screens) > 1 else screens[0]

            self.win = visual.Window(
                fullscr=False,
                size=(target.width, target.height),
                monitor=mon,
                screen=1,
                units='height',
                color='black',
                allowGUI=False,
                waitBlanking=False,
            )
            self._move_window_to_secondary()

        # Use default frame duration for display loop timing (prevents CPU spinning)
        # Movement is time-based (dt) so actual frame rate doesn't affect physics
        self.frame_duration = 1.0 / 60.0  # ~60 Hz target for smooth display

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

        # Fixation cross for baseline period
        self.fixation_cross = visual.TextStim(
            win=self.win,
            text='+',
            pos=(0, 0),
            height=0.3,
            color='white',
            font=None,
        )

        # Waiting for scanner text (displayed before trigger)
        self.waiting_text = visual.TextStim(
            win=self.win,
            text='waiting for scanner',
            pos=(0, 0),
            height=0.1,
            color='white',
            font=None,
        )

        # Instruction text stimulus
        self.instruction_text = visual.TextStim(
            win=self.win,
            text='',
            pos=(0, 0),
            height=0.06,
            wrapWidth=1.2,
            color='white',
            font=None,
            alignText='center',
            anchorVert='center',
        )
        
        # Question text stimulus (for post-run slider questions)
        self.question_text = visual.TextStim(
            win=self.win,
            text='',
            pos=(0, 0.2),
            height=0.06,
            wrapWidth=1.2,
            color='white',
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
            color='white',
            font=None,
            anchorVert='center',
        )
        
        # Setup questions CSV file with BIDS-style naming (use acq if available)
        output_dir = Path(self.coordinator.output_dir)
        if hasattr(self.coordinator, 'acq') and self.coordinator.acq is not None:
            # Format acq as 2-digit number if int
            acq_str = f"{self.coordinator.acq:02d}" if isinstance(self.coordinator.acq, int) else self.coordinator.acq
            run_acq_str = f"acq-{acq_str}_run-{self.coordinator.run_number:02d}"
        else:
            run_acq_str = f"run-{self.coordinator.run_number:02d}"
        self._questions_file_path = output_dir / f"{self.coordinator.subject}_task-{self.coordinator.task}_{run_acq_str}_questions.tsv"
        self._questions_file = open(self._questions_file_path, 'w', newline='')
        self._questions_writer = csv.writer(self._questions_file, delimiter='\t')
        self._questions_writer.writerow(['subject', 'run', 'feedback_mode', 'question_text', 'response', 'rt'])
        self._questions_file.flush()

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
            logger.info("▶ Simulation mode: auto-triggering")
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
                    logger.info(f"▶ Trigger received: starting task")
                    self.coordinator.send_trigger()
                    continueRoutine = False

        # Brief pause before starting task
        core.wait(0.1)

    def show_instructions(self):
        """Display instruction slides before task begins.
        
        Instructions vary based on feedback_mode and is_first_of_type:
        - First transfer run: explains task with no movement
        - Later transfer runs: brief reminder about no movement
        - First feedback run: explains ball movement mechanism
        - Later feedback runs: brief reminder
        
        Each slide is advanced by pressing spacebar (or configured advance key).
        Escape key can exit during instructions.
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
        
        # Display each instruction slide
        for slide_text in slides:
            self.instruction_text.setText(slide_text)
            self.instruction_text.draw()
            self.win.flip()
            
            # Wait for key press to advance
            continueRoutine = True
            while continueRoutine:
                keys = event.getKeys(keyList=self.instruction_advance_keys + ['escape'])
                
                if 'escape' in keys:
                    raise KeyboardInterrupt("User cancelled during instructions")
                
                if any(k in keys for k in self.instruction_advance_keys):
                    continueRoutine = False
            
            # Brief pause between slides
            core.wait(0.1)
        
        logger.info("Instructions complete")

    def _run_slider(
        self,
        question_text: str,
        left_label: str = 'left',
        right_label: str = 'right',
    ) -> int:
        """Display a slider question and return the user's response.
        
        Parameters
        ----------
        question_text : str
            The question to display above the slider
        left_label : str
            Label for the low end (1) of the scale
        right_label : str
            Label for the high end (9) of the scale
            
        Returns
        -------
        int
            The user's rating (1-9)
        """
        # Create slider
        slider = visual.Slider(
            win=self.win,
            size=(0.85, 0.1),
            ticks=(1, 9),
            labels=(left_label, right_label),
            granularity=1,
            color='white',
            fillColor='white',
            labelHeight=0.06,
        )
        
        # Set question text
        self.question_text.setText(question_text)
        
        # Initialize slider position
        event.clearEvents('keyboard')
        slider.markerPos = 5  # Start at center
        
        # Draw initial state
        slider.draw()
        self.question_text.draw()
        self.win.flip()
        
        # Track timing
        response_clock = core.Clock()
        
        # Wait for response
        continueRoutine = True
        while continueRoutine:
            keys = event.getKeys(keyList=[
                self.left_button, self.right_button, self.enter_button, 'escape'
            ])
            
            if 'escape' in keys:
                raise KeyboardInterrupt("User cancelled during slider question")
            
            if len(keys):
                if self.left_button in keys:
                    slider.markerPos = max(1, slider.markerPos - 1)
                elif self.right_button in keys:
                    slider.markerPos = min(9, slider.markerPos + 1)
                elif self.enter_button in keys:
                    slider.rating = slider.markerPos
                    continueRoutine = False
                
                # Redraw
                slider.draw()
                self.question_text.draw()
                self.win.flip()
        
        # Get response time
        rt = response_clock.getTime()
        rating = int(slider.rating)
        
        # Log to questions file
        self._questions_writer.writerow([
            self.coordinator.subject,
            self.coordinator.run_number,
            'feedback' if self.feedback_mode else 'transfer',
            question_text,
            rating,
            f'{rt:.3f}',
        ])
        self._questions_file.flush()
        
        logger.info(f"Question response: '{question_text[:40]}...' → {rating} (RT: {rt:.2f}s)")
        
        return rating

    def _show_post_run_questions(self) -> None:
        """Display post-run slider questions.
        
        Always asks 4 core questions. If this is the last feedback run
        (determined by session_state.is_last_feedback_run), asks 3 additional
        questions about the relationship between practice and ball movement.
        """
        # Show instruction screen
        self.slider_instruction_text.draw()
        self.win.flip()
        
        # Wait for any button press to continue
        event.clearEvents('keyboard')
        continueRoutine = True
        while continueRoutine:
            keys = event.getKeys(keyList=[
                self.left_button, self.right_button, self.enter_button, 'escape'
            ])
            if 'escape' in keys:
                raise KeyboardInterrupt("User cancelled during slider instructions")
            if len(keys):
                continueRoutine = False
        
        # Core questions (always asked)
        self._run_slider(
            question_text='How often were you using the Mindful Describing practice?',
            left_label='Never',
            right_label='Always',
        )
        
        self._run_slider(
            question_text='How often did you check the position of the ball?',
            left_label='Never',
            right_label='All the time',
        )
        
        self._run_slider(
            question_text='How difficult was it to apply Mindful Describing?',
            left_label='Not at all',
            right_label='Very Difficult',
        )
        
        self._run_slider(
            question_text='How calm do you feel right now?',
            left_label='Not at all',
            right_label='Very calm',
        )
        
        # Conditional questions (last feedback run only)
        is_last_feedback = False
        if self._session_state is not None:
            is_last_feedback = self._session_state.is_last_feedback_run
        
        if self.feedback_mode and is_last_feedback:
            logger.info("Last feedback run - asking additional questions")
            
            self._run_slider(
                question_text='The ball seemed to move up while I was practicing Mindful Describing',
                left_label='False',
                right_label='True',
            )
            
            self._run_slider(
                question_text='The ball seemed to move down while my mind was wandering.',
                left_label='False',
                right_label='True',
            )
            
            self._run_slider(
                question_text='The ball seemed to move down while I was trying to control it.',
                left_label='False',
                right_label='True',
            )
        
        logger.info("Post-run questions complete")

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
        movement_per_second = (
            self.current_activity
            * (self.scale_factor / self.internal_scaler)
            / self.coordinator.tr
        )

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
                network_name = self.mask_names[i]
                logger.info(f"✓ Hit! {network_name} target (total: {self.hit_counters[i]})")

                # Flash circle white for visual feedback
                self.circles[i].fillColor = 'white'

                # Reset ball to center position (game continues immediately)
                self.ball_pos = [0.0, 0.0]
                self.pause_movement = False  # Resume movement after reset

                # Reset frame time to prevent large dt jump after hit processing
                self._last_frame_time = core.getTime()

                # Shrink circle to increase difficulty (10% reduction per hit, min 0.03)
                # Makes it progressively harder to hit the same target repeatedly
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

        # Draw scene based on baseline status
        if self.is_baseline:
            # During baseline: show only fixation cross
            self.fixation_cross.draw()
        else:
            # During active task: show ball and circles only
            for circle in self.circles:
                circle.draw()

            self.ball.draw()

        # Flip window
        self.win.flip()

        # Emit visual state to GUI mirror if callback is set
        self._emit_mirror_state()

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
        
        run_type = 'feedback' if self.feedback_mode else 'transfer'
        state.record_run(
            run_type=run_type,
            hit_counters=self.hit_counters,
            scale_factor=self.scale_factor,
            circle_radii=self.circle_radii,
            volumes_processed=self.current_volume,
            min_hits_threshold=self._min_hits_threshold,
            max_hits_threshold=self._max_hits_threshold,
            scale_adjust_factor=self._scale_adjust_factor,
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
            shapes.extend([
                {
                    'type': 'line',
                    'x1': 0.5 - cross_size, 'y1': 0.5,
                    'x2': 0.5 + cross_size, 'y2': 0.5,
                    'stroke': 'white', 'stroke_width': 3
                },
                {
                    'type': 'line',
                    'x1': 0.5, 'y1': 0.5 - cross_size,
                    'x2': 0.5, 'y2': 0.5 + cross_size,
                    'stroke': 'white', 'stroke_width': 3
                },
            ])
        else:
            # Active task: show circles and ball
            # Convert PsychoPy height units to normalized 0-1 coordinates
            # PsychoPy: y in [-0.5, 0.5], we need [0, 1] with y inverted
            for pos, color, radius in zip(
                self.circle_positions, self.circle_colors, self.circle_radii
            ):
                # +0.33 -> 0.17 (top), -0.33 -> 0.83 (bottom)
                y = 0.5 - pos
                shapes.append({
                    'type': 'circle',
                    'x': 0.5, 'y': y,
                    'radius': radius,
                    'stroke': color, 'stroke_width': 3
                })

            # Ball
            ball_y = 0.5 - self.ball_pos[1]
            shapes.append({
                'type': 'circle',
                'x': 0.5, 'y': ball_y,
                'radius': 0.03,
                'fill': 'white'
            })

            # Hit counter text
            shapes.append({
                'type': 'text',
                'content': f"{self.mask_names[0]}: {self.hit_counters[0]}  "
                           f"{self.mask_names[1]}: {self.hit_counters[1]}",
                'x': 0.5, 'y': 0.95,
                'size': 10, 'color': 'gray'
            })

        return {'background': 'black', 'shapes': shapes}

    def cleanup(self):
        """Close window and quit."""
        # Log outlier summary statistics
        if self.current_volume > 0:
            outlier_pct = (self.outlier_count / self.current_volume) * 100
            logger.info(f"Outlier summary: {self.outlier_count}/{self.current_volume} volumes ({outlier_pct:.1f}%)")

        # Save state before cleanup if session_state was provided
        self.save_state()

        # Close PsychoPy log file
        if self._psychopy_log_file is not None:
            self._psychopy_log_file.close()
            self._psychopy_log_file = None
            self._psychopy_log_writer = None

        # Close questions file
        if self._questions_file is not None:
            self._questions_file.close()
            self._questions_file = None
            self._questions_writer = None
            logger.info(f"Questions saved: {self._questions_file_path}")

        # Close window without quitting PsychoPy runtime
        # (core.quit() prevents window recreation in the same process)
        if hasattr(self, 'win') and self.win is not None:
            self.win.close()
            self.win = None

        # Clear pending events for clean state
        from psychopy import event
        event.clearEvents()

        # Reset pyglet's gl_info state to allow clean window recreation
        # Without this, gl_info retains stale OpenGL version info that causes
        # "string index out of range" errors when creating a new window
        try:
            import pyglet.gl.gl_info as gl_info
            # Reset the global GLInfo instance to initial state
            gl_info.remove_active_context()
            # Also reset the cached version/renderer info so it re-queries on next window
            gl_info._have_info = False
            gl_info.version = '0.0.0'
            gl_info.vendor = ''
            gl_info.renderer = ''
            gl_info.extensions = set()
        except Exception as e:
            logger.debug(f"Could not reset pyglet gl_info state: {e}")

    def run(self):
        """Main task loop - runs in main thread for PsychoPy.

        Overrides BaseTask.run() to call update_frame() at monitor refresh rate
        instead of update() once per TR.
        """
        from psychopy import event

        self._running = True
        self.setup()  # Creates window and initializes task stimuli

        # Use PsychoPy clock for precise timing
        frame_clock = core.Clock()

        try:
            # Wait for coordinator to be ready
            self.coordinator.wait_until_ready(timeout=10.0)

            # Show instructions
            self.show_instructions()

            # Show waiting screen and wait for trigger
            self.wait_for_trigger()

            # Record task start time for frame timestamps
            self._task_start_time = core.getTime()
            self._last_frame_time = core.getTime()  # Initialize for time-based movement
            
            logger.info("=== BallTask Session Started ===")
            logger.info(f"Mode: {'FEEDBACK' if self.feedback_mode else 'NON-FEEDBACK (transfer)'}")
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

        except KeyboardInterrupt:
            logger.info("Task interrupted by user")
            pass
        finally:
            self._running = False
            
            # Log session summary
            logger.info("")
            logger.info("=== BallTask Session Complete ===")
            logger.info(f"Total volumes: {self.current_volume}/{self.coordinator.n_volumes}")
            logger.info(f"Hits - {self.mask_names[0]}: {self.hit_counters[0]}, {self.mask_names[1]}: {self.hit_counters[1]}")
            logger.info(f"Total hits: {sum(self.hit_counters)}")
            logger.info(f"Final scale factor: {self.scale_factor:.1f}")
            logger.info(f"Total frames rendered: {self._frame_counter}")
            if self.current_volume > 0:
                avg_frames_per_vol = self._frame_counter / self.current_volume
                logger.info(f"Average frames per volume: {avg_frames_per_vol:.1f}")
            logger.info(f"PsychoPy log saved: {self._psychopy_log_path}")

            # Show post-run slider questions if requested
            # (either session completed naturally, or stop was called with show_questions=True)
            if self.coordinator.should_show_questions():
                try:
                    self._show_post_run_questions()
                except KeyboardInterrupt:
                    logger.info("Questions interrupted by user")
            else:
                logger.info("Session was stopped - skipping post-run questions")

            self.cleanup()


# ==============================================================================
# Task Metadata for GUI Configuration
# ==============================================================================

from .task_registry import TaskMetadata, TaskParameter, register_task
from typing import Dict, Any


def _validate_ball_hits(config: Dict[str, Any]) -> None:
    """Cross-parameter validator for ball task hit thresholds."""
    min_hits = config.get("min_hits_threshold")
    max_hits = config.get("max_hits_threshold")

    if min_hits is not None and max_hits is not None:
        if min_hits >= max_hits:
            raise ValueError("Min hits threshold must be < max hits threshold")


BALL_TASK_METADATA = TaskMetadata(
    task_id="ball",
    display_name="Ball Task",
    description="Dual-network neurofeedback with ball control",
    parameters=[
        TaskParameter(
            key="use_transfer_runs",
            label="Use Transfer Runs",
            type="bool",
            default=True,
            hint="Include transfer runs (1 before feedback, 1 after)",
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
            key="fullscreen",
            label="Fullscreen Mode",
            type="bool",
            default=True,
            hint="Run task in fullscreen",
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
            hint="Key for moving ball left",
        ),
        TaskParameter(
            key="right_button",
            label="Right Control Button",
            type="str",
            default="2",
            hint="Key for moving ball right",
        ),
        TaskParameter(
            key="enter_button",
            label="Enter/Confirm Button",
            type="str",
            default="0",
            hint="Key for confirming selections",
        ),
        TaskParameter(
            key="scale_adjust_factor",
            label="Scale Adjustment Factor",
            type="float",
            default=0.25,
            min_value=0.01,
            max_value=1.0,
            hint="How much to adjust scale (0.25 = 25% change)",
        ),
    ],
    cross_parameter_validator=_validate_ball_hits,
)

# Auto-register on import
register_task(BALL_TASK_METADATA)
