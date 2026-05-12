"""
Gauge neurofeedback task (brain-o-meter).

Summary
-------
Dual-network visual feedback task that displays an analog gauge with a
rotating needle reflecting the differential signal between two brain
networks (e.g., ContA - DefaultA). Participants alternate between Focus
and Wander conditions across 12 blocks.

Based on the NKI-Rockland real-time fMRI neurofeedback paradigm
(Craddock et al., PMC5322045; OpenCogLabRepository/RTfMRI-neurofeedback).

Features
--------
- Tachometer-style gauge with 19 tick marks in a semicircular arc
- Cumulative score: needle drifts based on sustained differential activation
- 12 alternating Focus/Wander blocks with configurable durations
- Label swap halfway through session (counterbalancing)
- Button press catch trials between blocks
- BIDS-compatible events TSV logging

Dependencies
------------
psychopy, numpy
"""

# Standard library imports
import csv
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# Third-party imports
# Note: psychopy imports are lazy (inside methods) to avoid initialising
# pyglet's OpenGL shadow window on the main thread, which breaks
# background-thread window creation on native Windows.
# Local imports
from .base import BaseTask

if TYPE_CHECKING:
    from .session import SessionState

# Configure module-level logger
logger = logging.getLogger(__name__)

# Default block durations from the Rockland paradigm (seconds)
# Derived from create_stim_cfg.py: [[15,45,30],[15,45,30],[30,45,15],[30,45,15]] trials × 2s ISI
ROCKLAND_BLOCK_DURATIONS = [30, 90, 60, 30, 90, 60, 60, 90, 30, 60, 90, 30]

# Gauge geometry constants
N_TICK_MARKS = 19
GAUGE_RADIUS = 0.30  # In height units
GAUGE_CENTER_Y = -0.05  # Slightly below center
TICK_LENGTH = 0.03
CENTER_TICK_LENGTH = 0.05  # Thicker indicator at 90°
NEEDLE_LENGTH = GAUGE_RADIUS - 0.02


class GaugeTask(BaseTask):
    """Gauge (brain-o-meter) task for dual-network neurofeedback.

    Displays an analog gauge where the needle position reflects the
    differential activation between two brain networks. The score is
    cumulative — the needle drifts based on sustained activation rather
    than jumping to each new value.

    Participants alternate between Focus and Wander conditions. Labels
    swap position halfway through the session for counterbalancing.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data (must have 2 masks)
    mask_names : list of str, optional
        Names of two masks. First = focus network, second = wander network.
        If None, uses first two masks from coordinator.
    block_durations : str, default="30,90,60,30,90,60,60,90,30,60,90,30"
        Comma-separated block durations in seconds.
    conditions : str, default="wander,focus"
        Comma-separated condition labels (alternating). Rockland starts with Wander.
    swap_labels : bool, default=False
        If True, start with labels in swapped position (Wandering on left).
    score_factor : float, default=3.0
        Degrees of needle movement per unit of differential activation.
    label_swap_after_block : int, default=3
        Swap label positions every N blocks (Rockland: 3). 0 to disable.
    button_press_duration : float, default=2.0
        Duration of button press prompt between blocks (seconds).
    warmup_duration : float, default=2.0
        Duration of warm-up trial at start of each block (feedback off).
    trigger_keys : list of str, optional
        Keys that trigger acquisition start.
    button_key : str, default="1"
        Key for button press responses.
    presenter_monitor : str, default="right"
        Which monitor to display on.
    """

    task_id = "gauge"

    def __init__(
        self,
        coordinator,
        mask_names: list[str] | None = None,
        block_durations: str = "30,90,60,30,90,60,60,90,30,60,90,30",
        conditions: str = "wander,focus",
        swap_labels: bool = False,
        score_factor: float = 3.0,
        label_swap_after_block: int = 3,
        button_press_duration: float = 2.0,
        warmup_duration: float = 2.0,
        trigger_keys: list[str] | None = None,
        button_key: str = "1",
        presenter_monitor: str = "right",
        feedback_mode: bool = True,
        session_state: Optional["SessionState"] = None,
        **kwargs,
    ):
        if len(coordinator.mask_names) < 2:
            raise ValueError(f"GaugeTask requires at least 2 masks, got {len(coordinator.mask_names)}")

        if mask_names is None:
            mask_names = coordinator.mask_names[:2]
        elif len(mask_names) != 2:
            raise ValueError(f"GaugeTask requires exactly 2 mask names, got {len(mask_names)}")

        self.mask_names = mask_names

        super().__init__(
            coordinator,
            mask_name=mask_names[0],
            feedback_mode=feedback_mode,
            session_state=session_state,
            **kwargs,
        )

        # Parse block configuration
        self.block_durations = [float(d.strip()) for d in block_durations.split(",")]
        condition_list = [c.strip().lower() for c in conditions.split(",")]
        self.conditions = condition_list
        self.swap_labels = swap_labels
        self.score_factor = score_factor
        self.label_swap_after_block = label_swap_after_block
        self.button_press_duration = button_press_duration
        self.warmup_duration = warmup_duration
        self.trigger_keys = trigger_keys if trigger_keys is not None else ["t", "+", "5", "equal"]
        self.button_key = button_key
        self.presenter_monitor = presenter_monitor

        # State
        self.score = 90.0  # Degrees, starts at center (0=left, 90=center, 180=right)
        self._labels_swapped = swap_labels
        self._direction_sign = -1.0 if swap_labels else 1.0
        self._current_condition = "rest"
        self._feedback_active = False
        self._current_block = 0
        self.current_volume = 0
        self.is_baseline = True
        self._task_start_time: float = 0.0
        self._last_frame_time: float = 0.0

        # Events log
        self._events: list[dict] = []

        # Output paths
        output_dir = Path(coordinator.output_dir)
        if hasattr(coordinator, "acq") and coordinator.acq is not None:
            acq_str = f"{coordinator.acq:02d}" if isinstance(coordinator.acq, int) else coordinator.acq
            run_acq_str = f"acq-{acq_str}_run-{coordinator.run_number:02d}"
        else:
            run_acq_str = f"run-{coordinator.run_number:02d}"
        self._events_path = output_dir / f"{coordinator.subject}_task-{coordinator.task}_{run_acq_str}_events.tsv"
        self._psychopy_log_path = (
            output_dir / f"{coordinator.subject}_task-{coordinator.task}_{run_acq_str}_psychopylog.tsv"
        )
        self._psychopy_log_file = None
        self._psychopy_log_writer = None
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Initialize PsychoPy window and gauge stimuli."""
        from psychopy import logging as psychopy_logging
        from psychopy import visual

        psychopy_logging.console.setLevel(psychopy_logging.ERROR)

        # Open per-frame log
        self._psychopy_log_file = open(self._psychopy_log_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._psychopy_log_writer = csv.writer(self._psychopy_log_file, delimiter="\t")
        self._psychopy_log_writer.writerow(
            [
                "frame",
                "time",
                "volume",
                "score",
                "condition",
                "feedback_active",
                "labels_swapped",
                f"{self.mask_names[0]}_activation",
                f"{self.mask_names[1]}_activation",
            ]
        )
        self._psychopy_log_file.flush()

        # Create window
        self.create_window(
            monitor_position=self.presenter_monitor,
            color="black",
            units="height",
        )
        self.frame_duration = 1.0 / 60.0

        # -- Build gauge visual elements --

        # Tick marks: 19 lines evenly spaced from 0° to 180°
        self._tick_marks = []
        for i in range(N_TICK_MARKS):
            angle_deg = i * (180.0 / (N_TICK_MARKS - 1))
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            is_center = i == N_TICK_MARKS // 2
            length = CENTER_TICK_LENGTH if is_center else TICK_LENGTH
            width = 4.0 if is_center else 2.0

            # Inner and outer points on the arc
            r_inner = GAUGE_RADIUS - length
            r_outer = GAUGE_RADIUS
            start = (-cos_a * r_inner, sin_a * r_inner + GAUGE_CENTER_Y)
            end = (-cos_a * r_outer, sin_a * r_outer + GAUGE_CENTER_Y)

            tick = visual.Line(
                self.win,
                start=start,
                end=end,
                lineWidth=width,
                lineColor="white",
            )
            self._tick_marks.append(tick)

        # Needle (triangle arrow positioned on the arc)
        self._needle = visual.ShapeStim(
            self.win,
            vertices=[(-0.008, 0), (0.008, 0), (0, NEEDLE_LENGTH)],
            fillColor="white",
            lineColor="white",
            lineWidth=1,
            pos=(0, GAUGE_CENTER_Y),
            ori=0,  # Will be updated per frame
        )

        # Labels: "Focused" and "Wandering"
        label_y = GAUGE_CENTER_Y - 0.02
        self._left_label = visual.TextStim(
            self.win,
            text="Focused" if not self._labels_swapped else "Wandering",
            pos=(-GAUGE_RADIUS - 0.05, label_y),
            height=0.04,
            color="white",
            anchorHoriz="center",
        )
        self._right_label = visual.TextStim(
            self.win,
            text="Wandering" if not self._labels_swapped else "Focused",
            pos=(GAUGE_RADIUS + 0.05, label_y),
            height=0.04,
            color="white",
            anchorHoriz="center",
        )

        # Condition instruction text (large, below gauge)
        self._instruction_text = visual.TextStim(
            self.win,
            text="",
            pos=(0, GAUGE_CENTER_Y - 0.18),
            height=0.08,
            color="white",
            anchorHoriz="center",
        )

        # Fixation cross
        self._fixation = visual.TextStim(
            self.win,
            text="+",
            pos=(0, 0),
            height=0.15,
            color="white",
        )

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _update_needle_position(self):
        """Update needle orientation based on current score."""
        # Score: 0° = left (Wander end when normal), 180° = right (Focus end when normal)
        # PsychoPy ori: 0 = up, positive = clockwise
        # Map score 0-180 to ori: score 0 → ori 90 (pointing left), score 90 → ori 0 (up),
        # score 180 → ori -90 (pointing right)
        self._needle.ori = 90.0 - self.score

    def _draw_gauge(self):
        """Draw the complete gauge display."""
        for tick in self._tick_marks:
            tick.draw()
        self._update_needle_position()
        self._needle.draw()
        self._left_label.draw()
        self._right_label.draw()
        self._instruction_text.draw()

    def _swap_label_positions(self):
        """Swap the Focused/Wandering labels and flip direction sign."""
        self._labels_swapped = not self._labels_swapped
        if self._labels_swapped:
            self._left_label.text = "Wandering"
            self._right_label.text = "Focused"
        else:
            self._left_label.text = "Focused"
            self._right_label.text = "Wandering"
        self._direction_sign = -self._direction_sign

    def _set_needle_color(self, active: bool):
        """Set needle color: red when feedback active, white when inactive."""
        color = "red" if active else "white"
        self._needle.fillColor = color
        self._needle.lineColor = color

    # ------------------------------------------------------------------
    # Events logging
    # ------------------------------------------------------------------

    def _record_event(self, onset: float, duration: float, trial_type: str, **kwargs):
        """Record a BIDS-compatible event."""
        event = {"onset": onset, "duration": duration, "trial_type": trial_type}
        event.update(kwargs)
        self._events.append(event)

    def _save_events_tsv(self):
        """Write events to BIDS TSV file."""
        if not self._events:
            return
        columns = ["onset", "duration", "trial_type", "response", "response_time", "score", "labels_swapped"]
        with open(self._events_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for event in self._events:
                writer.writerow({col: event.get(col, "n/a") for col in columns})
        logger.info(f"Events saved: {self._events_path}")

    def _log_frame(self):
        """Log current frame data to TSV file."""
        with self._lock:
            activations = self._latest_data["activations"]

        frame_time = time.time() - self._task_start_time
        self._psychopy_log_writer.writerow(
            [
                self._frame_counter,
                f"{frame_time:.4f}",
                self.current_volume,
                f"{self.score:.2f}",
                self._current_condition,
                int(self._feedback_active),
                int(self._labels_swapped),
                f"{activations.get(self.mask_names[0], 0.0):.6f}",
                f"{activations.get(self.mask_names[1], 0.0):.6f}",
            ]
        )
        self._psychopy_log_file.flush()

    # ------------------------------------------------------------------
    # Score update
    # ------------------------------------------------------------------

    def _update_score(self, activations: dict):
        """Update cumulative score from differential activation.

        Score dynamics (matching original):
        score += direction_sign × (focus_act - wander_act) × score_factor
        Bounded to [0, 180].
        """
        focus_act = activations.get(self.mask_names[0], 0.0)
        wander_act = activations.get(self.mask_names[1], 0.0)
        differential = focus_act - wander_act

        self.score += self._direction_sign * differential * self.score_factor
        self.score = max(0.0, min(180.0, self.score))

    # ------------------------------------------------------------------
    # Block execution
    # ------------------------------------------------------------------

    def _run_baseline(self, label: str = "rest"):
        """Run baseline rest showing fixation cross until coordinator signals done.

        For opening baseline, loops until ``is_baseline`` flips to False.
        For closing baseline, loops until ``coordinator.is_complete()``.
        """
        from psychopy import core, event

        onset = core.getTime() - self._task_start_time
        self._current_condition = label
        self._feedback_active = False
        self._set_needle_color(False)

        while self._running:
            if self.coordinator.is_complete():
                break
            self.coordinator.check_error()

            # Poll new data
            data = self.get_latest_data()
            new_vol = data.get("volume", 0)
            if new_vol != self.current_volume:
                self.current_volume = new_vol
            is_baseline = data.get("is_baseline", True)

            # Exit once coordinator says baseline is over
            if not is_baseline:
                break

            self._fixation.draw()
            self._capture_and_emit_screenshot()
            self.win.flip()
            self._frame_counter += 1
            self._log_frame()

            remaining = self.frame_duration - core.getTime() % self.frame_duration
            if remaining > 0.001:
                core.wait(remaining, hogCPUperiod=0.001)

            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escape pressed")

        duration = core.getTime() - self._task_start_time - onset
        self._record_event(onset, duration, label)

    def _run_closing_rest(self):
        """Show fixation cross until coordinator signals all volumes received."""
        from psychopy import core, event

        onset = core.getTime() - self._task_start_time
        self._current_condition = "rest"
        self._feedback_active = False
        self._set_needle_color(False)

        while self._running and not self.coordinator.is_complete():
            self.coordinator.check_error()

            data = self.get_latest_data()
            new_vol = data.get("volume", 0)
            if new_vol != self.current_volume:
                self.current_volume = new_vol

            self._fixation.draw()
            self._capture_and_emit_screenshot()
            self.win.flip()
            self._frame_counter += 1
            self._log_frame()

            remaining = self.frame_duration - core.getTime() % self.frame_duration
            if remaining > 0.001:
                core.wait(remaining, hogCPUperiod=0.001)

            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escape pressed")

        duration = core.getTime() - self._task_start_time - onset
        self._record_event(onset, duration, "rest")

    def _run_block(self, block_idx: int, condition: str, duration: float):
        """Run a single feedback block.

        Includes warm-up (feedback off) then active feedback period.
        """
        from psychopy import core, event

        # Check for label swap — original paradigm swaps every N sub-blocks
        # (every major block boundary). label_swap_after_block is the interval.
        if self.label_swap_after_block > 0 and block_idx > 0 and block_idx % self.label_swap_after_block == 0:
            self._swap_label_positions()
            logger.info(f"Labels swapped at block {block_idx}")

        self._current_condition = condition
        self._current_block = block_idx + 1
        condition_display = condition.upper()

        # -- Warm-up phase (feedback off) --
        warmup_onset = core.getTime() - self._task_start_time
        self._feedback_active = False
        self._set_needle_color(False)
        self._instruction_text.text = condition_display

        warmup_clock = core.Clock()
        while warmup_clock.getTime() < self.warmup_duration and self._running:
            if self.coordinator.is_complete():
                break
            self.coordinator.check_error()

            data = self.get_latest_data()
            new_vol = data.get("volume", 0)
            if new_vol != self.current_volume:
                self.current_volume = new_vol

            self._draw_gauge()
            self._capture_and_emit_screenshot()
            self.win.flip()
            self._frame_counter += 1
            self._log_frame()

            remaining = self.frame_duration - warmup_clock.getTime() % self.frame_duration
            if remaining > 0.001:
                core.wait(remaining, hogCPUperiod=0.001)

            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escape pressed")

        self._record_event(
            warmup_onset,
            self.warmup_duration,
            f"{condition}_warmup",
            score=f"{self.score:.2f}",
            labels_swapped=int(self._labels_swapped),
        )

        # -- Active feedback phase --
        active_duration = duration - self.warmup_duration
        if active_duration <= 0:
            return

        active_onset = core.getTime() - self._task_start_time
        self._feedback_active = True
        self._set_needle_color(True)

        active_clock = core.Clock()
        while active_clock.getTime() < active_duration and self._running:
            if self.coordinator.is_complete():
                break
            self.coordinator.check_error()

            data = self.get_latest_data()
            new_vol = data.get("volume", 0)
            if new_vol != self.current_volume:
                self.current_volume = new_vol
                activations = data.get("activations", {})
                if activations and not data.get("is_baseline", False) and self.feedback_mode:
                    self._update_score(activations)

            self._draw_gauge()
            self._capture_and_emit_screenshot()
            self.win.flip()
            self._frame_counter += 1
            self._log_frame()

            remaining = self.frame_duration - active_clock.getTime() % self.frame_duration
            if remaining > 0.001:
                core.wait(remaining, hogCPUperiod=0.001)

            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escape pressed")

        self._record_event(
            active_onset,
            active_duration,
            condition,
            score=f"{self.score:.2f}",
            labels_swapped=int(self._labels_swapped),
        )

    def _run_button_press(self):
        """Run a button press catch trial between blocks."""
        from psychopy import core, event

        onset = core.getTime() - self._task_start_time
        self._current_condition = "button_press"
        self._feedback_active = False
        self._set_needle_color(False)
        self._instruction_text.text = "Push Button"

        response = "n/a"
        response_time = "n/a"

        clock = core.Clock()
        while clock.getTime() < self.button_press_duration and self._running:
            if self.coordinator.is_complete():
                break
            self.coordinator.check_error()

            data = self.get_latest_data()
            new_vol = data.get("volume", 0)
            if new_vol != self.current_volume:
                self.current_volume = new_vol

            self._draw_gauge()
            self._capture_and_emit_screenshot()
            self.win.flip()
            self._frame_counter += 1
            self._log_frame()

            # Check for button press
            keys = event.getKeys(keyList=[self.button_key], timeStamped=clock)
            if keys and response == "n/a":
                response = keys[0][0]
                response_time = f"{keys[0][1]:.4f}"

            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escape pressed")

            remaining = self.frame_duration - clock.getTime() % self.frame_duration
            if remaining > 0.001:
                core.wait(remaining, hogCPUperiod=0.001)

        self._record_event(
            onset, self.button_press_duration, "button_press", response=response, response_time=response_time
        )

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def update(self, data: dict):
        """Called by base class polling loop (not used — we override run())."""
        pass

    # ------------------------------------------------------------------
    # Mirror support
    # ------------------------------------------------------------------

    def get_mirror_state(self) -> dict:
        """Return current visual state for GUI mirroring."""
        shapes = []

        if self._current_condition == "rest":
            # Fixation cross
            cross_size = 0.1
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
            # Gauge needle indicator (simplified for mirror)
            needle_color = "red" if self._feedback_active else "white"
            angle_rad = math.radians(self.score)
            nx = 0.5 - 0.3 * math.cos(angle_rad)
            ny = 0.3 + 0.3 * math.sin(angle_rad)
            shapes.append(
                {"type": "line", "x1": 0.5, "y1": 0.6, "x2": nx, "y2": ny, "stroke": needle_color, "stroke_width": 3}
            )

            # Condition text
            shapes.append(
                {
                    "type": "text",
                    "content": self._current_condition.upper(),
                    "x": 0.5,
                    "y": 0.85,
                    "size": 14,
                    "color": "white",
                }
            )

            # Score
            shapes.append(
                {
                    "type": "text",
                    "content": f"Score: {self.score:.0f}°",
                    "x": 0.5,
                    "y": 0.95,
                    "size": 10,
                    "color": "gray",
                }
            )

        return {"background": "black", "shapes": shapes}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Main task loop — time-driven block execution."""
        from psychopy import core

        self._running = True

        try:
            self.setup()

            # Wait for coordinator
            self.coordinator.wait_until_ready(timeout=10.0)

            # Show instructions (vary by feedback vs transfer mode)
            if self.feedback_mode:
                slides = [
                    "Next is the feedback task.\n\n"
                    "During this scan, direct your gaze to the instruction\n"
                    "below the gauge.\n\n"
                    "Depending on the instruction, modulate your thoughts\n"
                    "to move the needle toward either the 'Focused' or\n"
                    "'Wandering' label.\n\n"
                    "You will be periodically asked to push a button.\n\n"
                    "Press any key to continue.",
                ]
            else:
                slides = [
                    "Next is a transfer run.\n\n"
                    "The gauge will be visible but the needle will\n"
                    "NOT move during this scan.\n\n"
                    "Follow the same instructions as before and\n"
                    "try to modulate your thoughts accordingly.\n\n"
                    "Press any key to continue.",
                ]
            super().show_instructions(
                slides=slides,
                advance_keys=["space", "1", "2", "3"],
                text_height=0.05,
                wrap_width=1.4,
            )

            # Wait for scanner trigger
            self.wait_for_trigger()

            self._task_start_time = core.getTime()
            self._last_frame_time = core.getTime()

            run_type = "feedback" if self.feedback_mode else "transfer"
            logger.info(f"=== GaugeTask Session Started ({run_type}) ===")
            logger.info(f"Networks: {self.mask_names[0]} (focus) vs {self.mask_names[1]} (wander)")
            logger.info(f"Blocks: {len(self.block_durations)}, durations: {self.block_durations}")
            logger.info(f"Score factor: {self.score_factor}")
            logger.info(f"Label swap after block: {self.label_swap_after_block}")

            # -- Opening baseline (driven by coordinator's n_baseline) --
            logger.info(f"Baseline volumes: {self.coordinator.n_baseline}")
            self._run_baseline(label="rest")

            # -- Main block sequence --
            for i, duration in enumerate(self.block_durations):
                if not self._running or self.coordinator.is_complete():
                    break

                # Determine condition (alternating)
                condition = self.conditions[i % len(self.conditions)]

                logger.info(
                    f"Block {i + 1}/{len(self.block_durations)}: {condition} ({duration}s), score={self.score:.1f}°"
                )

                self._run_block(i, condition, duration)

                # Button press between blocks
                if not self._running or self.coordinator.is_complete():
                    break
                self._run_button_press()

            # -- Closing rest (runs until coordinator has all volumes) --
            if self._running and not self.coordinator.is_complete():
                self._run_closing_rest()

        except KeyboardInterrupt:
            logger.info("Task interrupted by user")
        finally:
            self._running = False

            logger.info("")
            logger.info("=== GaugeTask Session Complete ===")
            logger.info(f"Total volumes: {self.current_volume}/{self.coordinator.n_volumes}")
            logger.info(f"Final score: {self.score:.1f}°")
            logger.info(f"Total frames: {self._frame_counter}")
            logger.info(f"Events logged: {len(self._events)}")

            # Save outputs
            self._save_events_tsv()
            self.cleanup()

    def cleanup(self):
        """Close resources."""
        # Close log file
        if self._psychopy_log_file is not None:
            self._psychopy_log_file.close()
            self._psychopy_log_file = None
            self._psychopy_log_writer = None

        # Delete visual stimuli while GL context is alive
        for attr in ["_fixation", "_instruction_text", "_left_label", "_right_label", "_needle"]:
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "_tick_marks"):
            del self._tick_marks

        super().cleanup()


# ==============================================================================
# Task Metadata for GUI Configuration
# ==============================================================================

from .task_registry import (
    LiveMetric,
    NetworkRole,
    TaskMetadata,
    TaskParameter,
)

GAUGE_METADATA = TaskMetadata(
    task_id="gauge",
    display_name="Gauge Task",
    description="Analog gauge neurofeedback with differential feedback (Focus vs Wander)",
    n_networks=2,
    network_roles=[
        NetworkRole(
            "focus_network",
            "Focus Network",
            "Active during focused attention (e.g., ContA)",
        ),
        NetworkRole(
            "wander_network",
            "Wander Network",
            "Active during mind-wandering (e.g., DefaultA)",
        ),
    ],
    parameters=[
        TaskParameter(
            key="use_transfer_runs",
            label="Use Transfer Runs",
            type="bool",
            default=False,
            hint="Include transfer runs (1 before feedback, 1 after)",
        ),
        TaskParameter(
            key="block_durations",
            label="Block Durations",
            type="str",
            default="30,90,60,30,90,60,60,90,30,60,90,30",
            hint="Comma-separated block durations in seconds. Default matches Rockland paradigm.",
        ),
        TaskParameter(
            key="conditions",
            label="Conditions",
            type="str",
            default="wander,focus",
            hint="Comma-separated condition labels (alternating). Rockland starts with Wander.",
        ),
        TaskParameter(
            key="swap_labels",
            label="Swap Labels",
            type="bool",
            default=False,
            hint="Start with Wandering on left, Focused on right (counterbalancing)",
        ),
        TaskParameter(
            key="score_factor",
            label="Score Factor",
            type="float",
            default=3.0,
            min_value=0.1,
            max_value=20.0,
            hint="Degrees per unit of differential activation. Original: 3.0",
        ),
        TaskParameter(
            key="label_swap_after_block",
            label="Label Swap Interval",
            type="int",
            default=3,
            min_value=0,
            max_value=24,
            hint="Swap Focused/Wandering labels every N blocks (Rockland: 3). 0 to disable.",
        ),
        TaskParameter(
            key="button_press_duration",
            label="Button Press Duration",
            type="float",
            default=2.0,
            min_value=0.5,
            max_value=10.0,
            hint="Duration of button press prompt between blocks (seconds)",
        ),
        TaskParameter(
            key="warmup_duration",
            label="Warm-up Duration",
            type="float",
            default=2.0,
            min_value=0.0,
            max_value=10.0,
            hint="Duration of warm-up at start of each block (feedback off)",
        ),
        TaskParameter(
            key="button_key",
            label="Button Key",
            type="str",
            default="1",
            hint="Key for button press responses",
        ),
        TaskParameter(
            key="trigger_keys",
            label="Scanner Trigger Keys",
            type="list[str]",
            default=["t", "+", "5", "equal"],
            hint="Keys that trigger volume acquisition",
        ),
    ],
    design_type="nuisance_only",
    live_metrics=[
        LiveMetric(
            key="score",
            label="Score",
            column_pattern="score",
            value_format="{value:.1f}",
        ),
    ],
)

# Register task
from .factory import register_task_class
from .task_registry import register_task

register_task(GAUGE_METADATA)
register_task_class("gauge", GaugeTask)
