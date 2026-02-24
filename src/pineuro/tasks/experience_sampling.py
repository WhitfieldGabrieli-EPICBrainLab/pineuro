"""Triggered experience sampling neurofeedback task.

Summary
-------
Monitors BOLD activation in real time and triggers experience sampling
prompts (sadness and focus ratings) when configurable threshold, trend,
and cooldown conditions are met.  Timeout prompts fire if no trigger
occurs within a configurable window.

Task Flow (per run)
-------------------
1. Instructions screen
2. Scanner trigger wait
3. Baseline fixation (until coordinator exits baseline)
4. Monitoring loop:
   a. Show fixation cross, poll coordinator for new volumes
   b. On each new volume: check trigger conditions
   c. If triggered or timed out: show two rating scales (sadness, focus)
   d. Return to monitoring
5. Save BIDS events TSV (includes per-TR activation rows)

Trigger Conditions (all must be met simultaneously)
---------------------------------------------------
- Cooldown elapsed: >= min_prompt_interval since last prompt
- Threshold: current activation above (or below) baseline
- Trend: N consecutive volumes of increasing (or decreasing) signal

Timeout prompts bypass threshold/trend checks but still respect cooldown.

Dependencies
------------
PsychoPy (visual, event, core)
"""

from __future__ import annotations

import csv
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseTask
from .factory import register_task_class
from .task_registry import (
    NetworkRole,
    TaskMetadata,
    TaskParameter,
    register_task,
)

if TYPE_CHECKING:
    from .session import SessionState

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class TriggerState:
    """Tracks state for trigger detection."""

    last_prompt_time: float = -float("inf")
    activation_history: deque = field(default_factory=lambda: deque(maxlen=50))
    volume_history: deque = field(default_factory=lambda: deque(maxlen=50))
    n_prompts_triggered: int = 0
    n_prompts_timeout: int = 0
    last_processed_volume: int = 0
    monitoring_start_time: float = 0.0


# ============================================================================
# Task implementation
# ============================================================================


class ExperienceSamplingTask(BaseTask):
    """Triggered experience sampling neurofeedback task.

    Monitors BOLD activation and triggers sadness/focus rating prompts
    when threshold + trend conditions are met, with cooldown and timeout
    logic for controlled sampling.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data.
    mask_names : list of str, optional
        Names of masks. Only the first mask is used.
    trigger_direction : str
        ``"above"`` triggers when activation > 0 (above baseline),
        ``"below"`` triggers when activation < 0 (below baseline).
    consecutive_volumes : int
        Number of consecutive volumes with trending signal needed to trigger.
    signal_trend : str
        ``"increasing"`` or ``"decreasing"`` — direction of consecutive trend.
    min_prompt_interval : float
        Minimum seconds between prompts (cooldown).
    timeout_duration : float
        Seconds of no trigger before a timeout prompt is shown.
    rating_timeout : float
        Maximum seconds for each rating response.
    left_key, right_key, confirm_key : str
        Scanner button mappings for the rating scale.
    feedback_mode : bool
        True for feedback runs, False for transfer runs.
    session_state : SessionState, optional
        Session state for persistence across runs.
    trigger_keys : list of str
        Keys that trigger volume acquisition.
    presenter_monitor : str
        Monitor position for task window.
    """

    task_id = "experience_sampling"

    def __init__(
        self,
        coordinator,
        mask_names: list[str] | None = None,
        trigger_direction: str = "above",
        consecutive_volumes: int = 3,
        signal_trend: str = "increasing",
        min_prompt_interval: float = 30.0,
        timeout_duration: float = 60.0,
        rating_timeout: float = 15.0,
        left_key: str = "1",
        right_key: str = "2",
        confirm_key: str = "3",
        feedback_mode: bool = True,
        session_state: SessionState | None = None,
        trigger_keys: list[str] | None = None,
        presenter_monitor: str = "right",
        **kwargs,
    ):
        mask_names = mask_names or coordinator.mask_names[:1]

        super().__init__(
            coordinator=coordinator,
            mask_name=mask_names[0],
            feedback_mode=feedback_mode,
            session_state=session_state,
            trigger_keys=trigger_keys or ["t", "+", "5", "equal"],
            **kwargs,
        )

        self.mask_names = mask_names
        self.presenter_monitor = presenter_monitor

        # Trigger parameters
        self.trigger_direction = trigger_direction
        self.consecutive_volumes = consecutive_volumes
        self.signal_trend = signal_trend
        self.min_prompt_interval = min_prompt_interval
        self.timeout_duration = timeout_duration
        self.rating_timeout = rating_timeout

        # Response button mapping
        self.left_key = left_key
        self.right_key = right_key
        self.confirm_key = confirm_key

        # Trigger state
        self._trigger_state = TriggerState()

        # All per-volume activations (for end-of-run per-TR logging)
        self._all_activations: list[dict] = []

        # BIDS events log
        self._events: list[dict] = []

        # Prompt counter
        self._prompt_number = 0

        # Task clock (set in run)
        self._task_clock: Any = None

        # Date string for TSV
        self._date = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        logger.info(
            f"ExperienceSamplingTask initialized: mask={self.mask_name}, "
            f"direction={trigger_direction}, consecutive={consecutive_volumes}, "
            f"trend={signal_trend}, cooldown={min_prompt_interval}s, "
            f"timeout={timeout_duration}s"
        )

    # =========================================================================
    # Activation accumulation (thread-safe, from coordinator)
    # =========================================================================

    def _update_data(self, activations, volume, elapsed, is_baseline, is_outlier):
        """Override to accumulate per-TR activations for end-of-run log."""
        super()._update_data(activations, volume, elapsed, is_baseline, is_outlier)

        self._all_activations.append(
            {
                "volume": volume,
                "elapsed": elapsed,
                "activation": activations.get(self.mask_name, float("nan")),
                "is_baseline": is_baseline,
                "is_outlier": is_outlier,
            }
        )

    # =========================================================================
    # Trigger detection
    # =========================================================================

    def _check_trigger(self, current_time: float, activation: float) -> str | None:
        """Check if a prompt should be triggered.

        Parameters
        ----------
        current_time : float
            Current task clock time in seconds.
        activation : float
            Current normalized activation value (baseline-relative).

        Returns
        -------
        str or None
            ``"triggered"`` if threshold + trend conditions met,
            ``"timeout"`` if timeout elapsed, ``None`` otherwise.
        """
        time_since_last = current_time - self._trigger_state.last_prompt_time

        # Cooldown: minimum time since last prompt
        if time_since_last < self.min_prompt_interval:
            return None

        # Timeout: too long since last prompt
        if time_since_last >= self.timeout_duration:
            return "timeout"

        # Threshold: activation above or below baseline
        if self.trigger_direction == "above":
            threshold_met = activation > 0
        else:
            threshold_met = activation < 0

        if not threshold_met:
            return None

        # Trend: N consecutive volumes of increasing/decreasing signal
        history = list(self._trigger_state.activation_history)
        if len(history) < self.consecutive_volumes:
            return None

        recent = history[-self.consecutive_volumes:]
        if self.signal_trend == "increasing":
            trending = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
        else:
            trending = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

        if trending:
            return "triggered"

        return None

    # =========================================================================
    # Rating display (RNT-style 1-5 scale)
    # =========================================================================

    def _create_stimuli(self) -> None:
        """Pre-create reusable PsychoPy stimuli to avoid GPU memory leaks."""
        from psychopy import visual

        # Fixation cross
        self._fixation = visual.TextStim(
            self.win, text="+", pos=(0, 0), height=0.3, color="white",
        )

        # Text stimulus for instructions / prompts
        self._text_stim = visual.TextStim(
            self.win, text="", height=0.06, color="white", wrapWidth=1.2,
        )

        # Rating scale components (1-5 scale, matching RNT pattern)
        scale_y = -0.15
        scale_width = 0.5
        n_ticks = 5
        tick_spacing = scale_width / (n_ticks - 1)
        scale_left_x = -scale_width / 2
        tick_x = [scale_left_x + i * tick_spacing for i in range(n_ticks)]

        self._rating_scale_line = visual.Line(
            self.win,
            start=(-scale_width / 2, scale_y),
            end=(scale_width / 2, scale_y),
            lineColor="white", lineWidth=2,
        )
        self._rating_tick_marks = []
        self._rating_tick_labels = []
        for i, val in enumerate(range(1, 6)):
            self._rating_tick_marks.append(visual.Line(
                self.win,
                start=(tick_x[i], scale_y - 0.015),
                end=(tick_x[i], scale_y + 0.015),
                lineColor="white", lineWidth=2,
            ))
            self._rating_tick_labels.append(visual.TextStim(
                self.win,
                text=str(val),
                pos=(tick_x[i], scale_y - 0.05),
                height=0.035, color="white",
            ))

        self._rating_left_label = visual.TextStim(
            self.win, text="", pos=(tick_x[0], scale_y - 0.1),
            height=0.035, color="white",
        )
        self._rating_right_label = visual.TextStim(
            self.win, text="", pos=(tick_x[-1], scale_y - 0.1),
            height=0.035, color="white",
        )
        self._rating_question = visual.TextStim(
            self.win, text="", pos=(0, 0.15),
            height=0.06, color="white", wrapWidth=1.2,
        )

        # Triangle marker — vertices relative to (0,0), moved via .pos
        marker_size = 0.02
        marker_gap = 0.02
        self._rating_marker = visual.ShapeStim(
            self.win,
            vertices=[
                (-marker_size, scale_y + marker_gap + marker_size * 2),
                (marker_size, scale_y + marker_gap + marker_size * 2),
                (0, scale_y + marker_gap),
            ],
            fillColor="white", lineColor="white", closeShape=True,
        )
        self._scale_left_x = scale_left_x
        self._tick_spacing = tick_spacing

    def _update_marker(self, x_pos: float) -> None:
        """Move the triangle marker to a new x position."""
        self._rating_marker.pos = (x_pos, 0)

    def _pos_to_x(self, pos: float) -> float:
        """Convert a rating position (1-5) to screen x coordinate."""
        return self._scale_left_x + (pos - 1) * self._tick_spacing

    def _draw_rating_frame(self) -> None:
        """Draw all rating scale components."""
        self._rating_question.draw()
        self._rating_scale_line.draw()
        for tm in self._rating_tick_marks:
            tm.draw()
        for tl in self._rating_tick_labels:
            tl.draw()
        self._rating_left_label.draw()
        self._rating_right_label.draw()
        self._rating_marker.draw()

    def _run_rating(
        self,
        question_text: str,
        left_label_text: str,
        right_label_text: str,
    ) -> tuple[int | None, float | None]:
        """Display a 1-5 rating scale with sliding marker and timeout.

        Parameters
        ----------
        question_text : str
            Question displayed above the scale.
        left_label_text : str
            Label below the left endpoint (value 1).
        right_label_text : str
            Label below the right endpoint (value 5).

        Returns
        -------
        tuple of (int or None, float or None)
            (rating_value, response_time_seconds). Both None if timed out.
        """
        from psychopy import core, event

        scale_low = 1
        scale_high = 5

        # Update text for this specific rating call
        self._rating_question.text = question_text
        self._rating_left_label.text = left_label_text
        self._rating_right_label.text = right_label_text

        # Start at midpoint (3.0 for 1-5 scale)
        marker_pos = (scale_low + scale_high) / 2.0
        self._update_marker(self._pos_to_x(marker_pos))

        clock = core.Clock()
        timer = core.CountdownTimer(self.rating_timeout)
        confirmed = False
        rt = None

        event.clearEvents("keyboard")
        self._ensure_focus()

        while timer.getTime() > 0 and self._running:
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")

            keys = event.getKeys(
                keyList=[
                    self.left_key,
                    self.right_key,
                    self.confirm_key,
                    "return",
                    "escape",
                ],
                timeStamped=clock,
            )
            for key, t in keys:
                if key == "escape":
                    raise KeyboardInterrupt("Escaped during rating")
                if key == self.left_key:
                    if marker_pos != int(marker_pos):
                        marker_pos = math.floor(marker_pos)
                    elif marker_pos > scale_low:
                        marker_pos -= 1
                elif key == self.right_key:
                    if marker_pos != int(marker_pos):
                        marker_pos = math.ceil(marker_pos)
                    elif marker_pos < scale_high:
                        marker_pos += 1
                elif key in (self.confirm_key, "return"):
                    confirmed = True
                    rt = t
                    break

            if confirmed:
                break

            self._update_marker(self._pos_to_x(marker_pos))
            self._draw_rating_frame()
            self._capture_and_emit_screenshot()
            self.win.flip()

        response = int(round(marker_pos)) if confirmed else None
        rt_value = round(rt, 3) if rt is not None else None

        # Wait out remaining time with fixation
        remaining = timer.getTime()
        if remaining > 0:
            self._show_fixation(remaining)

        return response, rt_value

    # =========================================================================
    # Prompt execution
    # =========================================================================

    def _run_prompt(
        self,
        prompt_type: str,
        volume: int,
        activation: float,
        current_time: float,
    ) -> None:
        """Execute a prompt: two sequential rating scales.

        Parameters
        ----------
        prompt_type : str
            ``"triggered"`` or ``"timeout"``.
        volume : int
            Volume number at time of trigger.
        activation : float
            Activation value at time of trigger.
        current_time : float
            Task clock time at trigger.
        """
        self._prompt_number += 1
        onset = self._task_clock.getTime()

        # Rating 1: Sadness
        sadness_rating, sadness_rt = self._run_rating(
            "How sad do you feel right now?",
            "Not sad",
            "Very sad",
        )

        # Rating 2: Focus on feelings
        focus_rating, focus_rt = self._run_rating(
            "How focused are you on your feelings?",
            "Not focused",
            "Very focused",
        )

        duration = self._task_clock.getTime() - onset

        # Record BIDS event
        trial_type = f"prompt_{prompt_type}"
        self._record_event(
            onset,
            duration,
            trial_type,
            volume=volume,
            activation=round(activation, 4),
            sadness_rating=sadness_rating if sadness_rating is not None else "n/a",
            sadness_rt=round(sadness_rt, 3) if sadness_rt is not None else "n/a",
            focus_rating=focus_rating if focus_rating is not None else "n/a",
            focus_rt=round(focus_rt, 3) if focus_rt is not None else "n/a",
            prompt_number=self._prompt_number,
        )

        # Update trigger state
        self._trigger_state.last_prompt_time = self._task_clock.getTime()
        if prompt_type == "triggered":
            self._trigger_state.n_prompts_triggered += 1
        else:
            self._trigger_state.n_prompts_timeout += 1

        # Clear activation history (fresh trend detection after prompt)
        self._trigger_state.activation_history.clear()
        self._trigger_state.volume_history.clear()

        logger.info(
            f"Prompt #{self._prompt_number} ({prompt_type}): "
            f"sadness={sadness_rating}, focus={focus_rating}, "
            f"vol={volume}, act={activation:.3f}"
        )

    # =========================================================================
    # Display helpers
    # =========================================================================

    def _wait_with_escape(self, duration: float, draw_func=None) -> None:
        """Wait for a duration while checking for escape/stop."""
        from psychopy import core, event

        clock = core.Clock()
        while clock.getTime() < duration and self._running:
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")
            keys = event.getKeys(keyList=["escape"])
            if keys:
                raise KeyboardInterrupt("Escaped")
            if draw_func:
                draw_func()
                self._capture_and_emit_screenshot()
                self.win.flip()
            else:
                core.wait(0.01)

    def _wait_for_baseline_end(self) -> None:
        """Show fixation cross until coordinator's is_baseline flips to False."""
        from psychopy import core, event

        while self._running:
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")
            if hasattr(self, "coordinator") and self.coordinator.is_complete():
                break

            data = self.get_latest_data()
            if not data.get("is_baseline", True):
                break

            self._fixation.draw()
            self._capture_and_emit_screenshot()
            self.win.flip()

            keys = event.getKeys(keyList=["escape"])
            if keys:
                raise KeyboardInterrupt("Escaped")

            core.wait(0.01)

    def _show_fixation(self, duration: float) -> None:
        """Display fixation cross for a given duration."""

        def draw():
            self._fixation.draw()

        self._wait_with_escape(duration, draw_func=draw)

    # =========================================================================
    # BIDS events
    # =========================================================================

    _TSV_COLUMNS = [
        "onset",
        "duration",
        "trial_type",
        "volume",
        "activation",
        "sadness_rating",
        "sadness_rt",
        "focus_rating",
        "focus_rt",
        "prompt_number",
        "date",
    ]

    def _record_event(self, onset: float, duration: float, trial_type: str, **kwargs: Any) -> None:
        """Record an event for the BIDS events TSV."""
        event_row: dict[str, Any] = {col: "n/a" for col in self._TSV_COLUMNS}
        event_row["onset"] = round(onset, 2)
        event_row["duration"] = round(duration, 2) if not isinstance(duration, str) else duration
        event_row["trial_type"] = trial_type
        event_row["date"] = self._date
        event_row.update(kwargs)
        self._events.append(event_row)

    def _append_activation_rows(self) -> None:
        """Append per-TR activation rows to events at end of run."""
        if not self._all_activations:
            return

        tr = getattr(self.coordinator, "tr", 1.0)

        for act_data in self._all_activations:
            if act_data["is_baseline"]:
                continue

            vol = act_data["volume"]
            onset = round(act_data["elapsed"], 2)

            self._record_event(
                onset,
                tr,
                "activation",
                volume=vol,
                activation=round(act_data["activation"], 4) if not math.isnan(act_data["activation"]) else "n/a",
            )

    def _save_events_tsv(self) -> None:
        """Write accumulated events to BIDS-compatible TSV."""
        if not self._events:
            return

        output_dir = getattr(self.coordinator, "output_dir", None)
        if output_dir is None:
            logger.warning("No output_dir on coordinator, cannot save events TSV")
            return

        output_dir = Path(output_dir)
        subject = getattr(self.coordinator, "subject", "unknown")
        task = getattr(self.coordinator, "task", "experience_sampling")
        run_number = getattr(self.coordinator, "run_number", 1)
        run_str = f"run-{run_number:02d}"

        path = output_dir / f"{subject}_task-{task}_{run_str}_events.tsv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._TSV_COLUMNS,
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            for event_row in self._events:
                writer.writerow(event_row)

        logger.info(f"Saved {len(self._events)} events to {path}")

    # =========================================================================
    # Monitoring loop
    # =========================================================================

    def _run_monitoring_loop(self) -> None:
        """Main monitoring loop: fixation + trigger detection + prompts."""
        from psychopy import core, event

        while self._running and not self.coordinator.is_complete():
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")

            keys = event.getKeys(keyList=["escape"])
            if keys:
                raise KeyboardInterrupt("Escaped")

            # Get latest data
            data = self.get_latest_data()
            volume = data["volume"]
            activation = data["activation"]
            current_time = self._task_clock.getTime()

            # Process new volume (only non-baseline, non-duplicate)
            if volume > self._trigger_state.last_processed_volume and not data["is_baseline"]:
                self._trigger_state.last_processed_volume = volume
                self._trigger_state.activation_history.append(activation)
                self._trigger_state.volume_history.append(volume)

                # Check trigger
                result = self._check_trigger(current_time, activation)
                if result is not None:
                    self._run_prompt(result, volume, activation, current_time)

            # Draw fixation cross
            self._fixation.draw()
            self._capture_and_emit_screenshot()
            self.win.flip()

            core.wait(0.01)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def setup(self) -> None:
        """Initialize PsychoPy window and stimuli."""
        self.create_window(
            monitor_position=self.presenter_monitor,
            color="black",
            units="height",
        )
        self._create_stimuli()

        # Show task instructions
        task_keys = [self.left_key, self.right_key, self.confirm_key]
        advance_keys = list(set(task_keys + ["space", "return"]))
        self.show_instructions(
            [
                "In this task, you will see a fixation cross (+).\n\n"
                "From time to time, you will be asked to rate\n"
                "how you are feeling.\n\n"
                f"Use button {self.left_key} to move left, "
                f"button {self.right_key} to move right,\n"
                f"and button {self.confirm_key} to confirm.\n\n"
                "Press any button to begin."
            ],
            advance_keys=advance_keys,
        )

        # Wait for scanner trigger
        self.wait_for_trigger()

        logger.info("ExperienceSamplingTask setup complete")

    def update(self, data: dict) -> None:
        """No-op — custom run() loop handles display updates."""
        pass

    def run(self) -> None:
        """Main task loop — monitoring with triggered prompts."""
        from psychopy import core

        self._running = True

        try:
            self.setup()

            self._task_clock = core.Clock()

            # Phase 1: Wait for baseline to end
            onset = self._task_clock.getTime()
            self._wait_for_baseline_end()
            baseline_dur = self._task_clock.getTime() - onset
            self._record_event(onset, baseline_dur, "baseline")

            # Initialize trigger state timing
            self._trigger_state.last_prompt_time = self._task_clock.getTime()
            self._trigger_state.monitoring_start_time = self._task_clock.getTime()

            # Phase 2: Monitoring loop
            self._run_monitoring_loop()

            # Phase 3: Append per-TR activation rows
            self._append_activation_rows()

        except KeyboardInterrupt:
            logger.info("ExperienceSamplingTask interrupted")
        finally:
            self._running = False
            self._save_events_tsv()
            self.cleanup()

    def get_run_results(self) -> dict:
        """Return task-specific results from the completed run."""
        results = super().get_run_results()
        results["n_prompts_triggered"] = self._trigger_state.n_prompts_triggered
        results["n_prompts_timeout"] = self._trigger_state.n_prompts_timeout
        results["n_prompts_total"] = self._prompt_number
        return results

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info(
            f"ExperienceSamplingTask cleanup: "
            f"{self._trigger_state.n_prompts_triggered} triggered, "
            f"{self._trigger_state.n_prompts_timeout} timeout, "
            f"{self._prompt_number} total prompts"
        )

        # Track volumes processed for base class reporting
        self._volumes_processed = len(self._all_activations)

        # Set snapshot for base class run recording
        self._task_state_snapshot = {
            "n_prompts_triggered": self._trigger_state.n_prompts_triggered,
            "n_prompts_timeout": self._trigger_state.n_prompts_timeout,
            "n_prompts_total": self._prompt_number,
        }

        # Delete visual stimuli while GL context is still alive
        # (prevents PsychoPy TextStim.__del__ access violations during GC)
        for attr in [
            "_fixation",
            "_text_stim",
            "_rating_scale_line",
            "_rating_left_label",
            "_rating_right_label",
            "_rating_question",
            "_rating_marker",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)
        for attr in ["_rating_tick_marks", "_rating_tick_labels"]:
            if hasattr(self, attr):
                for item in getattr(self, attr):
                    del item
                delattr(self, attr)

        super().cleanup()

    def stop(self) -> None:
        """Signal task to stop."""
        self._running = False


# ============================================================================
# Task metadata and registration
# ============================================================================

EXPERIENCE_SAMPLING_METADATA = TaskMetadata(
    task_id="experience_sampling",
    display_name="Experience Sampling",
    description=(
        "Triggered experience sampling: monitors BOLD activation and prompts "
        "sadness/focus ratings when threshold + trend conditions are met"
    ),
    n_networks=1,
    network_roles=[
        NetworkRole(
            "target_network",
            "Target Network",
            "Brain network monitored for trigger conditions (e.g. rumination network)",
        ),
    ],
    design_type="nuisance_only",
    parameters=[
        TaskParameter(
            key="trigger_direction",
            label="Trigger Direction",
            type="str",
            default="above",
            hint='Compare activation to baseline: "above" (activation > 0) or "below" (activation < 0)',
        ),
        TaskParameter(
            key="consecutive_volumes",
            label="Consecutive Volumes",
            type="int",
            default=3,
            min_value=1,
            max_value=20,
            hint="Number of consecutive trending volumes required to trigger",
        ),
        TaskParameter(
            key="signal_trend",
            label="Signal Trend",
            type="str",
            default="increasing",
            hint='Direction of consecutive volume trend: "increasing" or "decreasing"',
        ),
        TaskParameter(
            key="min_prompt_interval",
            label="Min Prompt Interval (s)",
            type="float",
            default=30.0,
            min_value=5.0,
            max_value=300.0,
            hint="Minimum seconds between prompts (cooldown)",
        ),
        TaskParameter(
            key="timeout_duration",
            label="Timeout Duration (s)",
            type="float",
            default=60.0,
            min_value=10.0,
            max_value=600.0,
            hint="Seconds of no trigger before showing a timeout prompt",
        ),
        TaskParameter(
            key="rating_timeout",
            label="Rating Timeout (s)",
            type="float",
            default=15.0,
            min_value=5.0,
            max_value=60.0,
            hint="Maximum seconds for each rating response",
        ),
        TaskParameter(
            key="left_key",
            label="Left Key",
            type="str",
            default="1",
            hint="Scanner button for moving rating left",
        ),
        TaskParameter(
            key="right_key",
            label="Right Key",
            type="str",
            default="2",
            hint="Scanner button for moving rating right",
        ),
        TaskParameter(
            key="confirm_key",
            label="Confirm Key",
            type="str",
            default="3",
            hint="Scanner button for confirming rating",
        ),
        TaskParameter(
            key="trigger_keys",
            label="Scanner Trigger Keys",
            type="list[str]",
            default=["t", "+", "5", "equal"],
            hint="Keys that trigger volume acquisition",
        ),
    ],
    session_state_fields=[],
    post_run_questions=[],
)

register_task(EXPERIENCE_SAMPLING_METADATA)
register_task_class("experience_sampling", ExperienceSamplingTask)
