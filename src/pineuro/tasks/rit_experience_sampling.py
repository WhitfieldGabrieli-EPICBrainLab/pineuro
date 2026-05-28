"""RIT experience sampling neurofeedback task.

Summary
-------
Monitors BOLD activation in a rumination network and triggers experience
sampling prompts (sadness and focus ratings) when configurable threshold,
trend, and cooldown conditions are met.  Timeout prompts fire if no
trigger occurs within a configurable window.

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
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseTask
from .factory import register_task_class
from .rit_triggered_ball import _least_squares_slope
from .task_registry import (
    LiveMetric,
    NetworkRole,
    TaskMetadata,
    TaskParameter,
    register_task,
)
from .utils import make_bids_run_prefix

if TYPE_CHECKING:
    from .state import TaskState

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

# Phase constants
_PHASE_BASELINE = "baseline"
_PHASE_MONITORING = "monitoring"
_PHASE_PROMPT = "prompt"

# Rating definitions: (question, left_label, right_label)
_RATINGS = [
    ("How sad do you feel right now?", "Not sad", "Very sad"),
    ("How focused are you on your feelings?", "Not focused", "Very focused"),
]


class RITExperienceSamplingTask(BaseTask):
    """RIT experience sampling neurofeedback task.

    Uses the base class run()/update() loop.  All display logic lives in
    update(), which is called every frame by the base class at refresh_rate Hz.
    A simple state machine (_phase) drives the three task phases: baseline,
    monitoring, and prompt (rating scales).

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data.
    mask_names : list of str, optional
        Names of masks. Only the first mask is used.
    activation_contrast : str
        ``"positive"`` triggers when activation > 0 (above baseline),
        ``"negative"`` triggers when activation < 0 (below baseline).
    consecutive_volumes : int
        Number of consecutive volumes with trending signal needed to trigger.
    activation_trend : str
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
    task_state : TaskState, optional
        Task state for persistence across runs.
    trigger_keys : list of str
        Keys that trigger volume acquisition.
    presenter_monitor : str
        Monitor position for task window.
    """

    task_id = "rit_experience_sampling"

    def __init__(
        self,
        coordinator,
        mask_names: list[str] | None = None,
        activation_contrast: str = "positive",
        consecutive_volumes: int = 3,
        activation_trend: str = "increasing",
        trend_method: str = "strict",
        min_prompt_interval: float = 30.0,
        timeout_duration: float = 60.0,
        rating_timeout: float = 15.0,
        left_key: str = "1",
        right_key: str = "2",
        confirm_key: str = "3",
        feedback_mode: bool = True,
        task_state: TaskState | None = None,
        trigger_keys: list[str] | None = None,
        presenter_monitor: str = "right",
        **kwargs,
    ):
        mask_names = mask_names or coordinator.mask_names[:1]
        if trend_method not in ("strict", "slope"):
            raise ValueError(f"trend_method must be 'strict' or 'slope', got {trend_method!r}")

        super().__init__(
            coordinator=coordinator,
            mask_name=mask_names[0],
            feedback_mode=feedback_mode,
            task_state=task_state,
            trigger_keys=trigger_keys or ["t", "+", "5", "equal"],
            **kwargs,
        )

        self.mask_names = mask_names
        self.presenter_monitor = presenter_monitor

        # Trigger parameters
        self.activation_contrast = activation_contrast
        self.consecutive_volumes = consecutive_volumes
        self.activation_trend = activation_trend
        self.trend_method = trend_method
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

        # Task clock (set in setup)
        self._task_clock: Any = None

        # Date string for TSV
        self._date = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Phase state machine
        self._phase = _PHASE_BASELINE
        self._baseline_onset = 0.0

        # Prompt state (active during _PHASE_PROMPT)
        self._prompt_onset = 0.0
        self._prompt_type = ""
        self._prompt_volume = 0
        self._prompt_activation = 0.0
        self._current_rating_index = 0
        self._rating_responses: list[tuple[int | None, float | None]] = []

        # Per-rating frame state
        self._rating_marker_pos = 3.0
        self._rating_clock: Any = None
        self._rating_timer: Any = None

        # Volume log suffix (computed in _update_data, read by coordinator log line)
        self._volume_log_suffix: str = ""

        # Separate dedup counter for trigger checks (history managed in _update_data)
        self._last_trigger_checked_volume: int = 0

        # Per-volume psychopylog for live monitoring
        prefix = make_bids_run_prefix(
            coordinator.subject,
            coordinator.session,
            coordinator.task,
            coordinator.run_number,
            attempt=coordinator.attempt,
        )
        self._psychopy_log_path = Path(coordinator.output_dir) / f"{prefix}_psychopylog.tsv"
        self._psychopy_log_file: Any = None
        self._psychopy_log_writer: Any = None
        self._frame_counter: int = 0

        logger.info(
            f"RITExperienceSamplingTask initialized: mask={self.mask_name}, "
            f"contrast={activation_contrast}, consecutive={consecutive_volumes}, "
            f"trend={activation_trend}, method={trend_method}, "
            f"cooldown={min_prompt_interval}s, timeout={timeout_duration}s"
        )

    # =========================================================================
    # Activation accumulation (thread-safe, from coordinator)
    # =========================================================================

    def _update_data(self, activations, volume, elapsed, is_baseline, is_outlier):
        """Override to accumulate activations and update trigger history.

        Called from the coordinator's background thread. Also computes the
        consecutive trend count so it can be appended inline to the
        coordinator's per-volume log line via get_volume_log_suffix().
        """
        super()._update_data(activations, volume, elapsed, is_baseline, is_outlier)

        activation_val = activations.get(self.mask_name, float("nan"))

        self._all_activations.append(
            {
                "volume": volume,
                "elapsed": elapsed,
                "activation": activation_val,
                "is_baseline": is_baseline,
                "is_outlier": is_outlier,
            }
        )

        # Update activation history and compute consecutive trend suffix
        if not is_baseline and volume > self._trigger_state.last_processed_volume:
            self._trigger_state.last_processed_volume = volume
            self._trigger_state.activation_history.append(activation_val)
            self._trigger_state.volume_history.append(volume)

            history = list(self._trigger_state.activation_history)
            if self.trend_method == "strict":
                # Same window algorithm as _check_trigger: largest window
                # (up to consecutive_volumes) ending at the current value
                # that is monotonically increasing/decreasing.
                consecutive = 0
                for w in range(min(len(history), self.consecutive_volumes), 0, -1):
                    window = history[-w:]
                    if self.activation_trend == "increasing":
                        if all(window[i] < window[i + 1] for i in range(len(window) - 1)):
                            consecutive = w
                            break
                    else:
                        if all(window[i] > window[i + 1] for i in range(len(window) - 1)):
                            consecutive = w
                            break

                trend_str = f"{consecutive}/{self.consecutive_volumes}"
                self._volume_log_suffix = f" | {self.activation_trend} {trend_str}"
            else:  # "slope"
                window = history[-self.consecutive_volumes :] if len(history) >= self.consecutive_volumes else history
                slope = _least_squares_slope(window)
                self._volume_log_suffix = f" | slope {slope:+.3f} (n={len(window)})"
        else:
            self._volume_log_suffix = ""

        if self._psychopy_log_writer is not None:
            self._frame_counter += 1
            self._psychopy_log_writer.writerow(
                [
                    self._frame_counter,
                    f"{elapsed:.4f}",
                    volume,
                    self._trigger_state.n_prompts_triggered,
                    self._trigger_state.n_prompts_timeout,
                ]
            )
            self._psychopy_log_file.flush()

    def get_volume_log_suffix(self) -> str:
        return self._volume_log_suffix

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

        # Threshold: activation positive or negative relative to baseline
        if self.activation_contrast == "positive":
            threshold_met = activation > 0
        else:
            threshold_met = activation < 0

        if not threshold_met:
            return None

        # Trend: N consecutive volumes of increasing/decreasing signal
        history = list(self._trigger_state.activation_history)
        if len(history) < self.consecutive_volumes:
            return None

        recent = history[-self.consecutive_volumes :]
        if self.trend_method == "strict":
            if self.activation_trend == "increasing":
                trending = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
            else:
                trending = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
        else:  # "slope"
            slope = _least_squares_slope(recent)
            if self.activation_trend == "increasing":
                trending = slope > 0
            else:
                trending = slope < 0

        if trending:
            return "triggered"

        return None

    # =========================================================================
    # Rating display (RIT-style 1-5 scale)
    # =========================================================================

    def _create_stimuli(self) -> None:
        """Pre-create reusable PsychoPy stimuli to avoid GPU memory leaks."""
        from psychopy import visual

        # Fixation cross
        self._fixation = visual.TextStim(
            self.win,
            text="+",
            pos=(0, 0),
            height=0.3,
            color="white",
        )

        # Rating scale components (1-5 scale, matching RIT pattern)
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
            lineColor="white",
            lineWidth=2,
        )
        self._rating_tick_marks = []
        self._rating_tick_labels = []
        for i, val in enumerate(range(1, 6)):
            self._rating_tick_marks.append(
                visual.Line(
                    self.win,
                    start=(tick_x[i], scale_y - 0.015),
                    end=(tick_x[i], scale_y + 0.015),
                    lineColor="white",
                    lineWidth=2,
                )
            )
            self._rating_tick_labels.append(
                visual.TextStim(
                    self.win,
                    text=str(val),
                    pos=(tick_x[i], scale_y - 0.05),
                    height=0.035,
                    color="white",
                )
            )

        self._rating_left_label = visual.TextStim(
            self.win,
            text="",
            pos=(tick_x[0], scale_y - 0.1),
            height=0.035,
            color="white",
        )
        self._rating_right_label = visual.TextStim(
            self.win,
            text="",
            pos=(tick_x[-1], scale_y - 0.1),
            height=0.035,
            color="white",
        )
        self._rating_question = visual.TextStim(
            self.win,
            text="",
            pos=(0, 0.15),
            height=0.06,
            color="white",
            wrapWidth=1.2,
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
            fillColor="white",
            lineColor="white",
            closeShape=True,
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
        session = getattr(self.coordinator, "session", "ses-01")
        run_number = getattr(self.coordinator, "run_number", 1)
        run_str = f"run-{run_number:02d}"

        path = output_dir / f"{subject}_{session}_task-{self.task_id}_{run_str}_events.tsv"

        with open(path, "w", newline="", encoding="utf-8") as f:
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
    # Lifecycle — uses base class run()
    # =========================================================================

    def setup(self) -> None:
        """Initialize PsychoPy window, stimuli, instructions, and trigger."""
        from psychopy import core

        # Open per-volume psychopylog so the GUI live-metrics panel can poll prompt counts.
        self._psychopy_log_file = open(  # noqa: SIM115
            self._psychopy_log_path, "w", newline="", encoding="utf-8"
        )
        self._psychopy_log_writer = csv.writer(self._psychopy_log_file, delimiter="\t")
        self._psychopy_log_writer.writerow(["frame", "time", "volume", "triggered_prompts", "timeout_prompts"])
        self._psychopy_log_file.flush()

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

        # Task clock starts after trigger
        self._task_clock = core.Clock()
        self._baseline_onset = self._task_clock.getTime()
        self._phase = _PHASE_BASELINE

    def run(self):
        """Main task loop — lets participant finish current prompt before stopping.

        Overrides BaseTask.run() so that when the coordinator signals
        completion mid-prompt, the rating scales keep running until the
        participant finishes (or escape is pressed).
        """
        self._running = True

        try:
            self.setup()

            while self._running and not self.coordinator.is_complete():
                frame_start = time.time()
                self.coordinator.check_error()
                data = self.get_latest_data()
                self.update(data)
                self._emit_mirror_state()
                elapsed = time.time() - frame_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)

            # Process final data
            data = self.get_latest_data()
            self.update(data)
            self._emit_mirror_state()

            # Let participant finish current prompt before stopping
            if self._phase == _PHASE_PROMPT:
                logger.info("Run complete — waiting for participant to finish current prompt")
                while self._running and self._phase == _PHASE_PROMPT:
                    frame_start = time.time()
                    data = self.get_latest_data()
                    self.update(data)
                    self._emit_mirror_state()
                    elapsed = time.time() - frame_start
                    if elapsed < self.frame_duration:
                        time.sleep(self.frame_duration - elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.cleanup()

    def update(self, data: dict) -> None:
        """Frame-by-frame update — state machine drives display.

        Called every frame by the base class run() loop.
        """
        from psychopy import event

        # Check for escape
        keys = event.getKeys(keyList=["escape"])
        if keys:
            raise KeyboardInterrupt("Escaped")

        if self._phase == _PHASE_BASELINE:
            self._update_baseline(data)
        elif self._phase == _PHASE_MONITORING:
            self._update_monitoring(data)
        elif self._phase == _PHASE_PROMPT:
            self._update_prompt(data)

    # =========================================================================
    # Phase handlers
    # =========================================================================

    def _update_baseline(self, data: dict) -> None:
        """Show fixation until coordinator's baseline period ends."""
        if not data.get("is_baseline", True):
            # Baseline ended — record event and transition
            baseline_dur = self._task_clock.getTime() - self._baseline_onset
            self._record_event(self._baseline_onset, baseline_dur, "baseline")
            self._trigger_state.last_prompt_time = self._task_clock.getTime() - self.min_prompt_interval
            self._trigger_state.monitoring_start_time = self._task_clock.getTime()
            self._phase = _PHASE_MONITORING
            logger.debug("Baseline complete, entering monitoring phase")

        # Draw fixation every frame (like ball/gauge tasks)
        self._fixation.draw()
        self._capture_and_emit_screenshot()
        self.win.flip()

    def _update_monitoring(self, data: dict) -> None:
        """Show fixation, check trigger on each new volume."""
        volume = data["volume"]
        activation = data["activation"]
        current_time = self._task_clock.getTime()

        # Check trigger for each new non-baseline volume
        # (history is already updated by _update_data in the background thread)
        if not data["is_baseline"] and volume > self._last_trigger_checked_volume:
            self._last_trigger_checked_volume = volume
            result = self._check_trigger(current_time, activation)
            if result is not None:
                self._begin_prompt(result, volume, activation)
                return

        # Draw fixation every frame (like ball/gauge tasks)
        self._fixation.draw()
        self._capture_and_emit_screenshot()
        self.win.flip()

    def _update_prompt(self, data: dict) -> None:
        """Frame-by-frame rating scale handler."""
        from psychopy import event

        # Timer expired — record no response and immediately advance
        if self._rating_timer.getTime() <= 0:
            self._rating_responses.append((None, None))
            q = _RATINGS[self._current_rating_index][0]
            logger.info(f'  Timed out (no response) — "{q}"')
            self._advance_rating()
            return

        # Handle key presses
        self._inject_keys_if_unfocused([self.left_key, self.right_key, self.confirm_key])
        keys = event.getKeys(
            keyList=[self.left_key, self.right_key, self.confirm_key, "return"],
            timeStamped=self._rating_clock,
        )
        for key, t in keys:
            if key == self.left_key:
                if self._rating_marker_pos != int(self._rating_marker_pos):
                    self._rating_marker_pos = math.floor(self._rating_marker_pos)
                elif self._rating_marker_pos > 1:
                    self._rating_marker_pos -= 1
            elif key == self.right_key:
                if self._rating_marker_pos != int(self._rating_marker_pos):
                    self._rating_marker_pos = math.ceil(self._rating_marker_pos)
                elif self._rating_marker_pos < 5:
                    self._rating_marker_pos += 1
            elif key in (self.confirm_key, "return"):
                response = int(round(self._rating_marker_pos))
                self._rating_responses.append((response, round(t, 3)))
                q = _RATINGS[self._current_rating_index][0]
                logger.info(f'  Response: {response}/5 (RT={t:.2f}s) — "{q}"')
                self._advance_rating()
                return

        # Draw rating frame (flips every frame for responsive input)
        self._update_marker(self._pos_to_x(self._rating_marker_pos))
        self._draw_rating_frame()
        self._capture_and_emit_screenshot()
        self.win.flip()

    # =========================================================================
    # Prompt flow
    # =========================================================================

    def _begin_prompt(self, prompt_type: str, volume: int, activation: float) -> None:
        """Start a new prompt sequence (sadness + focus ratings)."""
        logger.info(f"Prompt #{self._prompt_number + 1} ({prompt_type}) at vol={volume}")
        self._prompt_number += 1
        self._prompt_onset = self._task_clock.getTime()
        self._prompt_type = prompt_type
        self._prompt_volume = volume
        self._prompt_activation = activation
        self._current_rating_index = 0
        self._rating_responses = []
        self._phase = _PHASE_PROMPT
        self._init_current_rating()

    def _init_current_rating(self) -> None:
        """Set up stimuli and timers for the current rating scale."""
        from psychopy import core, event

        q, left, right = _RATINGS[self._current_rating_index]
        logger.info(f'  Question {self._current_rating_index + 1}/{len(_RATINGS)}: "{q}"')
        self._rating_question.text = q
        self._rating_left_label.text = left
        self._rating_right_label.text = right

        self._rating_marker_pos = 3.0  # Midpoint of 1-5
        self._update_marker(self._pos_to_x(self._rating_marker_pos))
        self._rating_clock = core.Clock()
        self._rating_timer = core.CountdownTimer(self.rating_timeout)
        event.clearEvents("keyboard")

    def _advance_rating(self) -> None:
        """Move to the next rating scale, or finish the prompt."""
        if self._current_rating_index < len(_RATINGS) - 1:
            self._current_rating_index += 1
            self._init_current_rating()
        else:
            self._finish_prompt()

    def _finish_prompt(self) -> None:
        """Record completed prompt and return to monitoring."""
        duration = self._task_clock.getTime() - self._prompt_onset

        sadness = self._rating_responses[0] if len(self._rating_responses) > 0 else (None, None)
        focus = self._rating_responses[1] if len(self._rating_responses) > 1 else (None, None)

        self._record_event(
            self._prompt_onset,
            duration,
            f"prompt_{self._prompt_type}",
            volume=self._prompt_volume,
            activation=round(self._prompt_activation, 4),
            sadness_rating=sadness[0] if sadness[0] is not None else "n/a",
            sadness_rt=sadness[1] if sadness[1] is not None else "n/a",
            focus_rating=focus[0] if focus[0] is not None else "n/a",
            focus_rt=focus[1] if focus[1] is not None else "n/a",
            prompt_number=self._prompt_number,
        )

        # Update trigger state
        self._trigger_state.last_prompt_time = self._task_clock.getTime()
        if self._prompt_type == "triggered":
            self._trigger_state.n_prompts_triggered += 1
        else:
            self._trigger_state.n_prompts_timeout += 1

        # Clear activation history (fresh trend detection after prompt)
        self._trigger_state.activation_history.clear()
        self._trigger_state.volume_history.clear()

        logger.info(f"Prompt #{self._prompt_number} complete: sadness={sadness[0]}, focus={focus[0]}")

        self._phase = _PHASE_MONITORING

    # =========================================================================
    # Cleanup and results
    # =========================================================================

    def get_run_results(self) -> dict:
        """Return task-specific results from the completed run."""
        results = super().get_run_results()
        results["n_prompts_triggered"] = self._trigger_state.n_prompts_triggered
        results["n_prompts_timeout"] = self._trigger_state.n_prompts_timeout
        results["n_prompts_total"] = self._prompt_number
        return results

    def cleanup(self) -> None:
        """Save events and clean up resources."""
        # Record partial prompt if interrupted mid-prompt (escape/error).
        # Normal run completion waits for the prompt to finish via run().
        if self._phase == _PHASE_PROMPT:
            duration = self._task_clock.getTime() - self._prompt_onset if self._task_clock else 0.0
            sadness = self._rating_responses[0] if len(self._rating_responses) > 0 else (None, None)
            focus = self._rating_responses[1] if len(self._rating_responses) > 1 else (None, None)

            self._record_event(
                self._prompt_onset,
                duration,
                f"prompt_{self._prompt_type}_incomplete",
                volume=self._prompt_volume,
                activation=round(self._prompt_activation, 4),
                sadness_rating=sadness[0] if sadness[0] is not None else "n/a",
                sadness_rt=sadness[1] if sadness[1] is not None else "n/a",
                focus_rating=focus[0] if focus[0] is not None else "n/a",
                focus_rt=focus[1] if focus[1] is not None else "n/a",
                prompt_number=self._prompt_number,
            )
            logger.info(f"Prompt #{self._prompt_number} incomplete (run ended): sadness={sadness[0]}, focus={focus[0]}")

        logger.info(
            f"RITExperienceSamplingTask cleanup: "
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

        # Append per-TR rows and save events
        try:
            self._append_activation_rows()
            self._save_events_tsv()
        except Exception:
            logger.exception("Failed to save events TSV during cleanup")

        if self._psychopy_log_file is not None:
            try:
                self._psychopy_log_file.close()
            except Exception:
                logger.exception("Failed to close psychopylog during cleanup")
            self._psychopy_log_file = None
            self._psychopy_log_writer = None

        # Delete visual stimuli while GL context is still alive
        for attr in [
            "_fixation",
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


# ============================================================================
# Task metadata and registration
# ============================================================================

RIT_EXPERIENCE_SAMPLING_METADATA = TaskMetadata(
    task_id="rit_experience_sampling",
    display_name="RIT Experience Sampling",
    description=(
        "Triggered experience sampling: monitors rumination network activation "
        "and prompts sadness/focus ratings when threshold + trend conditions are met"
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
            key="activation_contrast",
            label="Activation Contrast",
            type="str",
            default="positive",
            choices=["positive", "negative"],
            hint="Positive = activation > baseline, Negative = activation < baseline",
        ),
        TaskParameter(
            key="trend_method",
            label="Trend Method",
            type="str",
            default="strict",
            choices=["strict", "slope"],
            hint=(
                "strict = require monotonic change across all Consecutive Volumes "
                "(current behavior). slope = require least-squares slope sign to "
                "match Activation Trend (tolerant of single blips; increase "
                "Consecutive Volumes to ~5 for stronger blip-tolerance)."
            ),
        ),
        TaskParameter(
            key="consecutive_volumes",
            label="Consecutive Volumes",
            type="int",
            default=3,
            min_value=1,
            max_value=20,
            hint=(
                "With trend_method=strict: number of consecutive trending volumes "
                "required. With trend_method=slope: regression window length."
            ),
        ),
        TaskParameter(
            key="activation_trend",
            label="Activation Trend",
            type="str",
            default="increasing",
            choices=["increasing", "decreasing"],
            hint="Consecutive volumes increasing or decreasing in activation",
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
    ],
    task_state_fields=[],
    post_run_questions=[],
    live_metrics=[
        LiveMetric(key="triggered", label="Triggered", column_pattern="triggered_prompts"),
        LiveMetric(key="timeout", label="Timeout", column_pattern="timeout_prompts"),
    ],
)

register_task(RIT_EXPERIENCE_SAMPLING_METADATA)
register_task_class("rit_experience_sampling", RITExperienceSamplingTask)
