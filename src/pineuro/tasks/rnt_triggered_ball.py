"""RNT triggered ball neurofeedback task.

Summary
-------
Monitors BOLD activation in a rumination network and triggers ball feedback
episodes when configurable threshold, trend, and cooldown conditions are met.
Timeout episodes fire if no trigger occurs within a configurable window.

Task Flow (per run)
-------------------
1. Instructions screen
2. Scanner trigger wait
3. Baseline fixation (until coordinator exits baseline)
4. Monitoring loop:
   a. Show fixation cross, poll coordinator for new volumes
   b. On each new volume: check trigger conditions (on trigger network)
   c. If triggered or timed out: show ball task for episode_duration seconds
   d. Return to monitoring
5. Save BIDS events TSV (includes per-TR activation rows)

Network Setup (3 networks)
--------------------------
- mask_names[0]: Trigger network (e.g. RNT) — monitored for threshold/trend
- mask_names[1]: Top ball network (e.g. ContA) — ball moves up when more active
- mask_names[2]: Bottom ball network (e.g. DefaultA) — ball moves down when more active

Trigger Conditions (all must be met simultaneously)
---------------------------------------------------
- Cooldown elapsed: >= min_prompt_interval since last episode
- Threshold: current activation above (or below) baseline
- Trend: N consecutive volumes of increasing (or decreasing) signal

Timeout episodes bypass threshold/trend checks but still respect cooldown.

Dependencies
------------
PsychoPy (visual, event, core), numpy
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

import numpy as np

from .base import BaseTask
from .factory import register_task_class
from .task_registry import (
    NetworkRole,
    TaskMetadata,
    TaskParameter,
    register_task,
)
from .utils import is_activation_outlier

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
    n_episodes_triggered: int = 0
    n_episodes_timeout: int = 0
    last_processed_volume: int = 0
    monitoring_start_time: float = 0.0


# ============================================================================
# Task implementation
# ============================================================================

# Phase constants
_PHASE_BASELINE = "baseline"
_PHASE_MONITORING = "monitoring"
_PHASE_BALL_EPISODE = "ball_episode"


class RNTTriggeredBallTask(BaseTask):
    """RNT triggered ball neurofeedback task.

    Monitors a trigger network and launches timed ball feedback episodes
    when activation conditions are met. The ball moves between two target
    circles driven by two separate networks.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data.
    mask_names : list of str, optional
        Names of masks. Must have 3 entries: [trigger, top_ball, bottom_ball].
    activation_contrast : str
        ``"positive"`` triggers when activation > 0 (above baseline),
        ``"negative"`` triggers when activation < 0 (below baseline).
    consecutive_volumes : int
        Number of consecutive volumes with trending signal needed to trigger.
    activation_trend : str
        ``"increasing"`` or ``"decreasing"`` — direction of consecutive trend.
    min_prompt_interval : float
        Minimum seconds between episodes (cooldown).
    timeout_duration : float
        Seconds of no trigger before a timeout episode is shown.
    episode_duration : float
        Duration of each ball feedback episode in seconds.
    scale_factor : float
        Ball movement speed scaling.
    internal_scaler : float
        Additional movement normalization.
    pda_outlier_threshold : float
        Skip ball movement if |activation| exceeds this.
    circle_shrink_factor : float
        Circle radius multiplier after each hit.
    reset_ball_each_episode : bool
        If True, reset ball position and circles at each episode start.
    feedback_mode : bool
        True for feedback runs, False for transfer runs.
    session_state : SessionState, optional
        Session state for persistence across runs.
    trigger_keys : list of str
        Keys that trigger volume acquisition.
    presenter_monitor : str
        Monitor position for task window.
    """

    task_id = "rnt_triggered_ball"

    def __init__(
        self,
        coordinator,
        mask_names: list[str] | None = None,
        activation_contrast: str = "positive",
        consecutive_volumes: int = 3,
        activation_trend: str = "increasing",
        min_prompt_interval: float = 30.0,
        timeout_duration: float = 60.0,
        episode_duration: float = 30.0,
        scale_factor: float = 10.0,
        internal_scaler: float = 10.0,
        pda_outlier_threshold: float = 2.0,
        circle_shrink_factor: float = 0.9,
        reset_ball_each_episode: bool = True,
        feedback_mode: bool = True,
        session_state: SessionState | None = None,
        trigger_keys: list[str] | None = None,
        presenter_monitor: str = "right",
        **kwargs,
    ):
        if mask_names is None:
            mask_names = coordinator.mask_names[:3]
        if len(mask_names) < 3:
            raise ValueError(
                f"RNTTriggeredBallTask requires 3 mask names [trigger, top_ball, bottom_ball], got {len(mask_names)}"
            )

        super().__init__(
            coordinator=coordinator,
            mask_name=mask_names[0],  # Trigger network
            feedback_mode=feedback_mode,
            session_state=session_state,
            trigger_keys=trigger_keys or ["t", "+", "5", "equal"],
            **kwargs,
        )

        self.mask_names = mask_names
        self.presenter_monitor = presenter_monitor

        # Trigger parameters
        self.activation_contrast = activation_contrast
        self.consecutive_volumes = consecutive_volumes
        self.activation_trend = activation_trend
        self.min_prompt_interval = min_prompt_interval
        self.timeout_duration = timeout_duration

        # Ball episode parameters
        self.episode_duration = episode_duration
        self.scale_factor = scale_factor
        self.internal_scaler = internal_scaler
        self.pda_outlier_threshold = pda_outlier_threshold
        self.circle_shrink_factor = circle_shrink_factor
        self.reset_ball_each_episode = reset_ball_each_episode

        # Ball network names (for dual-network movement)
        self._ball_mask_names = [mask_names[1], mask_names[2]]

        # Trigger state
        self._trigger_state = TriggerState()

        # All per-volume activations (for end-of-run per-TR logging)
        self._all_activations: list[dict] = []

        # BIDS events log
        self._events: list[dict] = []

        # Episode counter
        self._episode_number = 0
        self._total_hits = [0, 0]  # Cumulative across all episodes

        # Task clock (set in setup)
        self._task_clock: Any = None

        # Date string for TSV
        self._date = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Phase state machine
        self._phase = _PHASE_BASELINE
        self._baseline_onset = 0.0

        # Ball episode state
        self._episode_onset = 0.0
        self._episode_type = ""
        self._episode_volume = 0
        self._episode_activation = 0.0
        self._episode_hits = [0, 0]

        # Ball movement state
        self._ball_pos = [0.0, 0.0]
        self._circle_positions = [1.0 / 3, -1.0 / 3]
        self._circle_radii = [0.15, 0.15]
        self._circle_colors = ["yellow", "#1E90FF"]
        self._network_directions = np.array([1, -1])  # +1 = up, -1 = down
        self._current_activity = 0.0
        self._current_direction: float = 0
        self._last_frame_time: float = 0.0
        self._current_is_outlier = False
        self._last_ball_volume = 0

        # Volume log suffix
        self._volume_log_suffix: str = ""

        # Separate dedup counter for trigger checks
        self._last_trigger_checked_volume: int = 0

        logger.info(
            f"RNTTriggeredBallTask initialized: trigger={self.mask_name}, "
            f"ball=[{self._ball_mask_names[0]}, {self._ball_mask_names[1]}], "
            f"contrast={activation_contrast}, consecutive={consecutive_volumes}, "
            f"trend={activation_trend}, cooldown={min_prompt_interval}s, "
            f"timeout={timeout_duration}s, episode={episode_duration}s"
        )

    # =========================================================================
    # Activation accumulation (thread-safe, from coordinator)
    # =========================================================================

    def _update_data(self, activations, volume, elapsed, is_baseline, is_outlier):
        """Override to accumulate activations and update trigger history.

        Called from the coordinator's background thread.
        """
        super()._update_data(activations, volume, elapsed, is_baseline, is_outlier)

        trigger_val = activations.get(self.mask_name, float("nan"))
        ball_val_1 = activations.get(self._ball_mask_names[0], float("nan"))
        ball_val_2 = activations.get(self._ball_mask_names[1], float("nan"))

        self._all_activations.append(
            {
                "volume": volume,
                "elapsed": elapsed,
                "trigger_activation": trigger_val,
                "ball_activation_1": ball_val_1,
                "ball_activation_2": ball_val_2,
                "is_baseline": is_baseline,
                "is_outlier": is_outlier,
            }
        )

        # Update trigger history for monitoring phase
        if not is_baseline and volume > self._trigger_state.last_processed_volume:
            self._trigger_state.last_processed_volume = volume
            self._trigger_state.activation_history.append(trigger_val)
            self._trigger_state.volume_history.append(volume)

            # Compute consecutive trend for log suffix
            history = list(self._trigger_state.activation_history)
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
        else:
            self._volume_log_suffix = ""

    def get_volume_log_suffix(self) -> str:
        return self._volume_log_suffix

    # =========================================================================
    # Trigger detection
    # =========================================================================

    def _check_trigger(self, current_time: float, activation: float) -> str | None:
        """Check if a ball episode should be triggered.

        Returns ``"triggered"`` if threshold + trend conditions met,
        ``"timeout"`` if timeout elapsed, ``None`` otherwise.
        """
        time_since_last = current_time - self._trigger_state.last_prompt_time

        # Cooldown
        if time_since_last < self.min_prompt_interval:
            return None

        # Timeout
        if time_since_last >= self.timeout_duration:
            return "timeout"

        # Threshold
        if self.activation_contrast == "positive":
            threshold_met = activation > 0
        else:
            threshold_met = activation < 0

        if not threshold_met:
            return None

        # Trend
        history = list(self._trigger_state.activation_history)
        if len(history) < self.consecutive_volumes:
            return None

        recent = history[-self.consecutive_volumes :]
        if self.activation_trend == "increasing":
            trending = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
        else:
            trending = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

        if trending:
            return "triggered"

        return None

    # =========================================================================
    # Ball movement (simplified from BallTask)
    # =========================================================================

    def _calculate_ball_movement(self, activations: dict) -> None:
        """Update movement parameters from dual-network activations.

        Called once per new volume during ball episodes.
        """
        acts = [activations.get(name, 0.0) for name in self._ball_mask_names]

        # Reject outliers
        if is_activation_outlier(acts, self.pda_outlier_threshold):
            self._current_is_outlier = True
            return

        self._current_is_outlier = False

        # Handle zero-mean edge case
        if np.nanmean(acts) == 0:
            return

        # Magnitude from activation difference
        self._current_activity = np.abs(acts[0] - acts[1]) / 10.0

        # Direction from dominant network
        max_idx = np.argmax(acts)
        self._current_direction = self._network_directions[max_idx]

    def _apply_movement(self) -> None:
        """Apply time-based ball movement. Called every frame during episodes."""
        from psychopy import core  # type: ignore[import-untyped]

        current_time = core.getTime()
        dt = current_time - self._last_frame_time
        self._last_frame_time = current_time

        if not self.feedback_mode:
            return

        movement_per_second = self._current_activity * (self.scale_factor / self.internal_scaler) / self.coordinator.tr
        dy = self._current_direction * movement_per_second * dt
        self._ball_pos[1] += dy
        self._ball.pos = self._ball_pos

    def _check_hits(self) -> bool:
        """Check if ball hit either target circle."""
        from psychopy import core

        for i, circle in enumerate(self._circles):
            circle_y = self._circle_positions[i]

            if self._network_directions[i] > 0:  # Top circle
                hit = self._ball_pos[1] >= circle_y
            else:  # Bottom circle
                hit = self._ball_pos[1] <= circle_y

            if hit:
                self._episode_hits[i] += 1
                self._total_hits[i] += 1

                label = self._ball_mask_names[i]
                logger.info(f"Hit! {label} target (episode: {self._episode_hits[i]}, total: {self._total_hits[i]})")

                # Flash circle white
                self._circles[i].fillColor = "white"

                # Reset ball to center
                self._ball_pos = [0.0, 0.0]

                # Reset frame time to prevent dt jump
                self._last_frame_time = core.getTime()

                # Shrink circle
                old_radius = self._circle_radii[i]
                new_radius = max(old_radius * self.circle_shrink_factor, 0.03)
                radius_reduction = old_radius - new_radius

                self._circle_positions[i] += self._network_directions[i] * radius_reduction
                self._circle_radii[i] = new_radius

                self._circles[i].pos = (0, self._circle_positions[i])
                self._circles[i].radius = self._circle_radii[i]

                return True

        return False

    # =========================================================================
    # Stimuli creation
    # =========================================================================

    def _create_stimuli(self) -> None:
        """Pre-create reusable PsychoPy stimuli."""
        from psychopy import visual

        # Fixation cross
        self._fixation = visual.TextStim(
            self.win,
            text="+",
            pos=(0, 0),
            height=0.3,
            color="white",
        )

        # Ball target circles
        self._circles = []
        for i in range(2):
            circle = visual.Circle(
                win=self.win,
                pos=(0, self._circle_positions[i]),
                radius=self._circle_radii[i],
                fillColor=None,
                lineColor=self._circle_colors[i],
                lineWidth=8,
            )
            self._circles.append(circle)

        # Ball
        self._ball = visual.Circle(
            win=self.win,
            pos=self._ball_pos,
            radius=0.03,
            fillColor="white",
            lineColor="white",
        )

    # =========================================================================
    # BIDS events
    # =========================================================================

    _TSV_COLUMNS = [
        "onset",
        "duration",
        "trial_type",
        "volume",
        "trigger_activation",
        "episode_number",
        "episode_type",
        "hits_top",
        "hits_bottom",
        "hits_total",
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
                trigger_activation=(
                    round(act_data["trigger_activation"], 4)
                    if not math.isnan(act_data["trigger_activation"])
                    else "n/a"
                ),
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
        run_number = getattr(self.coordinator, "run_number", 1)
        run_str = f"run-{run_number:02d}"

        path = output_dir / f"{subject}_task-{self.task_id}_{run_str}_events.tsv"

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
    # Lifecycle
    # =========================================================================

    def setup(self) -> None:
        """Initialize PsychoPy window, stimuli, instructions, and trigger."""
        from psychopy import core

        self.create_window(
            monitor_position=self.presenter_monitor,
            color="black",
            units="height",
        )
        self._create_stimuli()

        # Show task instructions
        self.show_instructions(
            [
                "In this task, you will see a fixation cross (+).\n\n"
                "From time to time, you will see a ball and two circles.\n"
                "The ball moves based on your brain activity.\n\n"
                "Press any button to begin."
            ],
            advance_keys=["space", "return", "1", "2", "3"],
        )

        # Wait for scanner trigger
        self.wait_for_trigger()

        # Task clock starts after trigger
        self._task_clock = core.Clock()
        self._baseline_onset = self._task_clock.getTime()
        self._phase = _PHASE_BASELINE

    def run(self):
        """Main task loop — lets participant finish ball episode before stopping."""
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

            # Let ball episode finish if active
            if self._phase == _PHASE_BALL_EPISODE:
                logger.info("Run complete — waiting for ball episode to finish")
                while self._running and self._phase == _PHASE_BALL_EPISODE:
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
        """Frame-by-frame update — state machine drives display."""
        from psychopy import event

        # Check for escape
        keys = event.getKeys(keyList=["escape"])
        if keys:
            raise KeyboardInterrupt("Escaped")

        if self._phase == _PHASE_BASELINE:
            self._update_baseline(data)
        elif self._phase == _PHASE_MONITORING:
            self._update_monitoring(data)
        elif self._phase == _PHASE_BALL_EPISODE:
            self._update_ball_episode(data)

    # =========================================================================
    # Phase handlers
    # =========================================================================

    def _update_baseline(self, data: dict) -> None:
        """Show fixation until coordinator's baseline period ends."""
        if not data.get("is_baseline", True):
            baseline_dur = self._task_clock.getTime() - self._baseline_onset
            self._record_event(self._baseline_onset, baseline_dur, "baseline")
            self._trigger_state.last_prompt_time = self._task_clock.getTime() - self.min_prompt_interval
            self._trigger_state.monitoring_start_time = self._task_clock.getTime()
            self._phase = _PHASE_MONITORING
            logger.debug("Baseline complete, entering monitoring phase")

        self._fixation.draw()
        self._capture_and_emit_screenshot()
        self.win.flip()

    def _update_monitoring(self, data: dict) -> None:
        """Show fixation, check trigger on each new volume."""
        volume = data["volume"]
        activation = data["activation"]
        current_time = self._task_clock.getTime()

        if not data["is_baseline"] and volume > self._last_trigger_checked_volume:
            self._last_trigger_checked_volume = volume
            result = self._check_trigger(current_time, activation)
            if result is not None:
                self._begin_ball_episode(result, volume, activation)
                return

        self._fixation.draw()
        self._capture_and_emit_screenshot()
        self.win.flip()

    def _update_ball_episode(self, data: dict) -> None:
        """Frame-by-frame ball episode handler."""
        current_time = self._task_clock.getTime()

        # Check if episode duration elapsed
        if current_time - self._episode_onset >= self.episode_duration:
            self._finish_ball_episode()
            return

        # Process new volume for ball movement
        volume = data["volume"]
        if not data["is_baseline"] and volume > self._last_ball_volume:
            self._last_ball_volume = volume
            activations = data.get("activations", {})
            if activations:
                self._calculate_ball_movement(activations)

        # Apply movement and check hits
        self._apply_movement()
        self._check_hits()

        # Draw ball scene
        for circle in self._circles:
            circle.draw()
        self._ball.draw()
        self._capture_and_emit_screenshot()
        self.win.flip()

        # Reset circle fill after flash
        for circle in self._circles:
            circle.fillColor = None

    # =========================================================================
    # Episode flow
    # =========================================================================

    def _begin_ball_episode(self, episode_type: str, volume: int, activation: float) -> None:
        """Start a new ball feedback episode."""
        from psychopy import core

        logger.info(f"Episode #{self._episode_number + 1} ({episode_type}) at vol={volume}")
        self._episode_number += 1
        self._episode_onset = self._task_clock.getTime()
        self._episode_type = episode_type
        self._episode_volume = volume
        self._episode_activation = activation
        self._episode_hits = [0, 0]
        self._last_ball_volume = 0

        # Reset ball state if configured
        if self.reset_ball_each_episode:
            self._ball_pos = [0.0, 0.0]
            self._circle_positions = [1.0 / 3, -1.0 / 3]
            self._circle_radii = [0.15, 0.15]
            for i, circle in enumerate(self._circles):
                circle.pos = (0, self._circle_positions[i])
                circle.radius = self._circle_radii[i]

        # Reset movement state
        self._current_activity = 0.0
        self._current_direction = 0
        self._last_frame_time = core.getTime()

        self._phase = _PHASE_BALL_EPISODE

    def _finish_ball_episode(self) -> None:
        """Record completed episode and return to monitoring."""
        duration = self._task_clock.getTime() - self._episode_onset

        self._record_event(
            self._episode_onset,
            duration,
            f"ball_{self._episode_type}",
            volume=self._episode_volume,
            trigger_activation=round(self._episode_activation, 4),
            episode_number=self._episode_number,
            episode_type=self._episode_type,
            hits_top=self._episode_hits[0],
            hits_bottom=self._episode_hits[1],
            hits_total=sum(self._episode_hits),
        )

        # Update trigger state
        self._trigger_state.last_prompt_time = self._task_clock.getTime()
        if self._episode_type == "triggered":
            self._trigger_state.n_episodes_triggered += 1
        else:
            self._trigger_state.n_episodes_timeout += 1

        # Clear activation history (fresh trend detection after episode)
        self._trigger_state.activation_history.clear()
        self._trigger_state.volume_history.clear()

        logger.info(
            f"Episode #{self._episode_number} complete: hits=[{self._episode_hits[0]}, {self._episode_hits[1]}]"
        )

        self._phase = _PHASE_MONITORING

    # =========================================================================
    # Mirror state (for GUI)
    # =========================================================================

    def get_mirror_state(self) -> dict[str, Any]:
        """Return current visual state for GUI mirroring."""
        shapes: list[dict] = []

        if self._phase == _PHASE_BALL_EPISODE:
            # Circles
            for pos, color, radius in zip(
                self._circle_positions,
                self._circle_colors,
                self._circle_radii,
                strict=False,
            ):
                y = 0.5 - pos
                shapes.append(
                    {
                        "type": "circle",
                        "x": 0.5,
                        "y": y,
                        "radius": radius,
                        "stroke": color,
                        "stroke_width": 3,
                    }
                )

            # Ball
            ball_y = 0.5 - self._ball_pos[1]
            shapes.append({"type": "circle", "x": 0.5, "y": ball_y, "radius": 0.03, "fill": "white"})

            # Hit counter
            hit_text = (
                f"{self._ball_mask_names[0]}: {self._episode_hits[0]}  "
                f"{self._ball_mask_names[1]}: {self._episode_hits[1]}"
            )
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
        else:
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

        return {"background": "black", "shapes": shapes}

    # =========================================================================
    # Cleanup and results
    # =========================================================================

    def get_run_results(self) -> dict:
        """Return task-specific results from the completed run."""
        results = super().get_run_results()
        results["n_episodes_triggered"] = self._trigger_state.n_episodes_triggered
        results["n_episodes_timeout"] = self._trigger_state.n_episodes_timeout
        results["n_episodes_total"] = self._episode_number
        results["total_hits"] = list(self._total_hits)
        return results

    def cleanup(self) -> None:
        """Save events and clean up resources."""
        # Record partial episode if interrupted
        if self._phase == _PHASE_BALL_EPISODE:
            duration = self._task_clock.getTime() - self._episode_onset if self._task_clock else 0.0

            self._record_event(
                self._episode_onset,
                duration,
                f"ball_{self._episode_type}_incomplete",
                volume=self._episode_volume,
                trigger_activation=round(self._episode_activation, 4),
                episode_number=self._episode_number,
                episode_type=self._episode_type,
                hits_top=self._episode_hits[0],
                hits_bottom=self._episode_hits[1],
                hits_total=sum(self._episode_hits),
            )
            logger.info(
                f"Episode #{self._episode_number} incomplete (run ended): "
                f"hits=[{self._episode_hits[0]}, {self._episode_hits[1]}]"
            )

        logger.info(
            f"RNTTriggeredBallTask cleanup: "
            f"{self._trigger_state.n_episodes_triggered} triggered, "
            f"{self._trigger_state.n_episodes_timeout} timeout, "
            f"{self._episode_number} total episodes, "
            f"hits=[{self._total_hits[0]}, {self._total_hits[1]}]"
        )

        # Track volumes processed for base class reporting
        self._volumes_processed = len(self._all_activations)

        # Set snapshot for base class run recording
        self._task_state_snapshot = {
            "n_episodes_triggered": self._trigger_state.n_episodes_triggered,
            "n_episodes_timeout": self._trigger_state.n_episodes_timeout,
            "n_episodes_total": self._episode_number,
            "total_hits": list(self._total_hits),
        }

        # Append per-TR rows and save events
        try:
            self._append_activation_rows()
            self._save_events_tsv()
        except Exception:
            logger.exception("Failed to save events TSV during cleanup")

        # Delete visual stimuli while GL context is still alive
        for attr in ["_fixation", "_ball"]:
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "_circles"):
            for item in self._circles:
                del item
            delattr(self, "_circles")

        super().cleanup()


# ============================================================================
# Task metadata and registration
# ============================================================================

RNT_TRIGGERED_BALL_METADATA = TaskMetadata(
    task_id="rnt_triggered_ball",
    display_name="RNT Triggered Ball",
    description=(
        "Monitors rumination network activation and triggers ball feedback "
        "episodes when threshold + trend conditions are met. Ball movement "
        "is driven by two separate networks."
    ),
    n_networks=3,
    network_roles=[
        NetworkRole(
            "trigger_network",
            "Trigger Network",
            "Brain network monitored for trigger conditions (e.g. rumination network)",
        ),
        NetworkRole(
            "top_network",
            "Top Ball Network (upward)",
            "Ball moves up when this network is more active",
        ),
        NetworkRole(
            "bottom_network",
            "Bottom Ball Network (downward)",
            "Ball moves down when this network is more active",
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
            hint="Positive = trigger activation > baseline, Negative = trigger activation < baseline",
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
            key="activation_trend",
            label="Activation Trend",
            type="str",
            default="increasing",
            choices=["increasing", "decreasing"],
            hint="Consecutive volumes increasing or decreasing in trigger activation",
        ),
        TaskParameter(
            key="min_prompt_interval",
            label="Min Episode Interval (s)",
            type="float",
            default=30.0,
            min_value=5.0,
            max_value=300.0,
            hint="Minimum seconds between ball episodes (cooldown)",
        ),
        TaskParameter(
            key="timeout_duration",
            label="Timeout Duration (s)",
            type="float",
            default=60.0,
            min_value=10.0,
            max_value=600.0,
            hint="Seconds of no trigger before showing a timeout ball episode",
        ),
        TaskParameter(
            key="episode_duration",
            label="Episode Duration (s)",
            type="float",
            default=30.0,
            min_value=5.0,
            max_value=120.0,
            hint="Duration of each ball feedback episode",
        ),
        TaskParameter(
            key="scale_factor",
            label="Scale Factor",
            type="float",
            default=10.0,
            min_value=1.0,
            max_value=50.0,
            hint="Ball movement speed (larger = faster)",
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
            key="reset_ball_each_episode",
            label="Reset Ball Each Episode",
            type="bool",
            default=True,
            hint="If true, ball position and circles reset at each episode start",
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

register_task(RNT_TRIGGERED_BALL_METADATA)
register_task_class("rnt_triggered_ball", RNTTriggeredBallTask)
