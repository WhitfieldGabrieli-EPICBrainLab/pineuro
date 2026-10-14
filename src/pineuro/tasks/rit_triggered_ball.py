"""RIT triggered ball neurofeedback task.

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

Network Setup
-------------
mask_names[0] is always the trigger network (e.g. the rumination network)
monitored for threshold/trend. Ball movement depends on ``movement_mode``:

- ``single`` (default): one network's signed activation drives the ball.
  It defaults to the trigger network (so the main use case needs only one
  mask) but can be overridden by assigning the ``movement_network`` role.
  ``movement_direction`` chooses the sign mapping: ``positive_down``
  (default) sends the ball down when activation is positive (more
  rumination) and up when negative (less rumination); ``positive_up`` flips
  it.
- ``dual``: two networks drive the ball by their activation difference —
  mask_names[1] (top, ball up when more active) minus mask_names[2]
  (bottom, ball down when more active). Requires 3 masks.

Trigger Conditions
------------------
Cooldown must always be elapsed (>= min_prompt_interval since last episode).
Beyond that, up to four neural condition gates can be independently enabled and
combined via ``trigger_logic`` (``"all"`` = AND, ``"any"`` = OR):

- Activation level: current activation past ``activation_threshold``
- Trend: slope/strict trend over ``consecutive_volumes`` (vs ``slope_threshold``)
- Mean: mean activation over the window past ``mean_threshold``
- Count: at least ``count_min`` of the window past ``activation_threshold``

The activation contrast (``positive``/``negative``) sets the comparison
direction for the level, mean, and count gates; ``activation_trend`` sets it for
the slope gate. Defaults reproduce the original behavior: activation > 0 AND
slope > 0. Timeout episodes bypass the neural gates but still respect cooldown.

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
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .base import BaseTask
from .factory import register_task_class
from .task_registry import (
    LiveMetric,
    NetworkRole,
    TaskMetadata,
    TaskParameter,
    TriggeredEventReport,
    register_task,
)
from .utils import is_activation_outlier, make_bids_run_prefix

if TYPE_CHECKING:
    from .state import TaskState

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _least_squares_slope(values: list[float]) -> float:
    """Slope of least-squares line through ``values`` with x = 0..N-1.

    Returns 0.0 for windows shorter than 2.
    """
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


class _TriggerEval(NamedTuple):
    """Result of evaluating trigger conditions for one volume.

    ``decision`` is authoritative (it drives the actual trigger). The remaining
    fields are diagnostics consumed by the per-volume log suffix, computed even
    when ``decision`` short-circuits so the log can explain why a volume did or
    did not fire.
    """

    decision: str | None
    in_cooldown: bool
    cooldown_remaining: float
    timeout_fired: bool
    threshold_met: bool
    warming_up: bool
    history_len: int
    trend_met: bool
    slope: float
    consecutive: int
    mean_met: bool
    mean_value: float
    count_met: bool
    count_above: int


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


class RITTriggeredBallTask(BaseTask):
    """RIT triggered ball neurofeedback task.

    Monitors a trigger network and launches timed ball feedback episodes
    when activation conditions are met. The ball moves between two target
    circles driven by two separate networks.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data.
    mask_names : list of str, optional
        Ordered mask names by role. ``mask_names[0]`` is the trigger network
        (always required). In ``single`` movement mode an optional
        ``mask_names[1]`` overrides the movement network (otherwise the
        trigger network drives the ball). In ``dual`` mode entries 1 and 2 are
        the top/bottom ball networks (3 masks required).
    movement_mode : str
        ``"single"`` (default) drives the ball from one network's signed
        activation; ``"dual"`` drives it from the difference of two networks.
    movement_direction : str
        Single mode only. ``"positive_down"`` (default) moves the ball down
        when activation is positive (more rumination) and up when negative;
        ``"positive_up"`` flips it. Ignored in dual mode.
    activation_contrast : str
        ``"positive"`` triggers when activation > 0 (above baseline),
        ``"negative"`` triggers when activation < 0 (below baseline).
    consecutive_volumes : int
        Number of consecutive volumes with trending signal needed to trigger.
    activation_trend : str
        ``"increasing"`` or ``"decreasing"`` — direction of consecutive trend.
    use_activation_condition : bool
        Enable the activation-level gate (current activation vs
        ``activation_threshold``; direction from ``activation_contrast``).
    activation_threshold : float
        Activation level used by the activation-level and count gates. Positive
        contrast fires when activation > this value; negative when < this value.
    use_trend_condition : bool
        Enable the slope/strict trend gate over ``consecutive_volumes``.
    slope_threshold : float
        Slope-method comparison value: increasing fires when slope >
        ``slope_threshold``, decreasing when slope < it. 0 = any rise/fall.
    use_mean_condition : bool
        Enable the mean-over-window gate (mean of the last
        ``consecutive_volumes`` activations vs ``mean_threshold``).
    mean_threshold : float
        Mean-activation threshold over the window (direction from
        ``activation_contrast``).
    use_count_condition : bool
        Enable the sustained-count gate (how many of the last
        ``consecutive_volumes`` pass ``activation_threshold``).
    count_min : int
        Minimum number of in-window volumes that must pass
        ``activation_threshold`` for the count gate.
    trigger_logic : str
        How enabled gates combine: ``"all"`` (AND) or ``"any"`` (OR).
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
    task_state : TaskState, optional
        Task state for persistence across runs.
    presenter_monitor : str
        Monitor position for task window.
    """

    task_id = "rittriggeredball"

    def __init__(
        self,
        coordinator,
        mask_names: list[str] | None = None,
        movement_mode: str = "single",
        movement_direction: str = "positive_down",
        activation_contrast: str = "positive",
        consecutive_volumes: int = 3,
        activation_trend: str = "increasing",
        trend_method: str = "slope",
        use_activation_condition: bool = True,
        activation_threshold: float = 0.0,
        use_trend_condition: bool = True,
        slope_threshold: float = 0.0,
        use_mean_condition: bool = False,
        mean_threshold: float = 0.0,
        use_count_condition: bool = False,
        count_min: int = 3,
        trigger_logic: str = "all",
        min_prompt_interval: float = 30.0,
        timeout_duration: float = 60.0,
        episode_duration: float = 30.0,
        scale_factor: float = 10.0,
        internal_scaler: float = 10.0,
        pda_outlier_threshold: float = 2.0,
        circle_shrink_factor: float = 0.9,
        reset_ball_each_episode: bool = True,
        feedback_mode: bool = True,
        task_state: TaskState | None = None,
        presenter_monitor: str = "right",
        **kwargs,
    ):
        if mask_names is None:
            mask_names = coordinator.mask_names[:1]
        if len(mask_names) < 1:
            raise ValueError("RITTriggeredBallTask requires at least 1 mask name (trigger network)")
        if trend_method not in ("strict", "slope"):
            raise ValueError(f"trend_method must be 'strict' or 'slope', got {trend_method!r}")
        if trigger_logic not in ("all", "any"):
            raise ValueError(f"trigger_logic must be 'all' or 'any', got {trigger_logic!r}")
        if not (use_activation_condition or use_trend_condition or use_mean_condition or use_count_condition):
            raise ValueError("At least one trigger condition must be enabled")
        if movement_mode not in ("single", "dual"):
            raise ValueError(f"movement_mode must be 'single' or 'dual', got {movement_mode!r}")
        if movement_direction not in ("positive_down", "positive_up"):
            raise ValueError(f"movement_direction must be 'positive_down' or 'positive_up', got {movement_direction!r}")
        if movement_mode == "dual" and len(mask_names) < 3:
            raise ValueError(
                f"dual movement mode requires 3 mask names [trigger, top_ball, bottom_ball], got {len(mask_names)}"
            )

        super().__init__(
            coordinator=coordinator,
            mask_name=mask_names[0],  # Trigger network
            feedback_mode=feedback_mode,
            task_state=task_state,
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

        # Trigger condition gates (each independently toggleable, combined by trigger_logic)
        self.use_activation_condition = use_activation_condition
        self.activation_threshold = activation_threshold
        self.use_trend_condition = use_trend_condition
        self.slope_threshold = slope_threshold
        self.use_mean_condition = use_mean_condition
        self.mean_threshold = mean_threshold
        self.use_count_condition = use_count_condition
        self.count_min = count_min
        self.trigger_logic = trigger_logic

        # Ball episode parameters
        self.episode_duration = episode_duration
        self.scale_factor = scale_factor
        self.internal_scaler = internal_scaler
        self.pda_outlier_threshold = pda_outlier_threshold
        self.circle_shrink_factor = circle_shrink_factor
        self.reset_ball_each_episode = reset_ball_each_episode

        # Ball movement configuration
        self.movement_mode = movement_mode
        self.movement_direction = movement_direction
        # positive_down: positive activation -> ball down (negative -> up).
        self._direction_sign: float = -1.0 if movement_direction == "positive_down" else 1.0

        # Movement network(s). Single mode uses one signed network (defaults to
        # the trigger network unless a movement override was assigned via the
        # optional movement_network role). Dual mode uses two networks driven by
        # their activation difference.
        if movement_mode == "single":
            self._movement_mask: str | None = mask_names[1] if len(mask_names) >= 2 else mask_names[0]
            self._ball_mask_names = [self._movement_mask]
        else:
            self._movement_mask = None
            self._ball_mask_names = [mask_names[1], mask_names[2]]

        # Trigger state
        self._trigger_state = TriggerState()

        # All per-volume activations (for end-of-run per-TR logging)
        self._all_activations: list[dict] = []

        # BIDS events log
        self._events: list[dict] = []

        # Episode counter
        self._episode_number = 0
        self._n_episodes_incomplete = 0  # 0 or 1: an episode cut short by the end of the run
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

        if movement_mode == "single":
            movement_desc = f"single({self._movement_mask}, {movement_direction})"
        else:
            movement_desc = f"dual[{self._ball_mask_names[0]}, {self._ball_mask_names[1]}]"
        logger.info(
            f"RITTriggeredBallTask initialized: trigger={self.mask_name}, "
            f"movement={movement_desc}, "
            f"contrast={activation_contrast}, consecutive={consecutive_volumes}, "
            f"trend={activation_trend}, method={trend_method}, "
            f"cooldown={min_prompt_interval}s, "
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
        ball_val_1 = activations.get(self._ball_mask_names[0], float("nan")) if self._ball_mask_names else float("nan")
        ball_val_2 = (
            activations.get(self._ball_mask_names[1], float("nan")) if len(self._ball_mask_names) > 1 else float("nan")
        )

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

            history = list(self._trigger_state.activation_history)
            time_since_last = elapsed - self._trigger_state.last_prompt_time
            ev = self._evaluate_trigger(time_since_last, trigger_val, history)
            self._volume_log_suffix = self._format_trigger_suffix(ev)
        else:
            self._volume_log_suffix = ""

        if self._psychopy_log_writer is not None:
            self._frame_counter += 1
            self._psychopy_log_writer.writerow(
                [
                    self._frame_counter,
                    f"{elapsed:.4f}",
                    volume,
                    self._trigger_state.n_episodes_triggered,
                    self._trigger_state.n_episodes_timeout,
                ]
            )
            self._psychopy_log_file.flush()

    def get_volume_log_suffix(self) -> str:
        return self._volume_log_suffix

    # =========================================================================
    # Trigger detection
    # =========================================================================

    def _check_trigger(self, current_time: float, activation: float) -> str | None:
        """Check if a ball episode should be triggered.

        Returns ``"triggered"`` if threshold + trend conditions met,
        ``"timeout"`` if timeout elapsed, ``None`` otherwise. Delegates to
        :meth:`_evaluate_trigger` so the decision and the per-volume log suffix
        can never disagree.
        """
        time_since_last = current_time - self._trigger_state.last_prompt_time
        history = list(self._trigger_state.activation_history)
        return self._evaluate_trigger(time_since_last, activation, history).decision

    def _evaluate_trigger(self, time_since_last: float, activation: float, history: list[float]) -> _TriggerEval:
        """Evaluate every trigger gate for one volume (single source of truth).

        Cooldown and timeout are timing gates evaluated first (cooldown blocks
        everything; timeout forces an episode but still respects cooldown).
        Beyond those, up to four neural condition gates -- activation level,
        trend, mean over the window, and sustained count -- are each evaluated
        only when enabled, then combined by ``trigger_logic`` (``"all"`` = AND,
        ``"any"`` = OR). The remaining fields are diagnostics consumed by the
        per-volume log suffix. Shared by :meth:`_check_trigger` (uses
        ``.decision``) and the suffix (uses the diagnostics).
        """
        in_cooldown = time_since_last < self.min_prompt_interval
        cooldown_remaining = max(0.0, self.min_prompt_interval - time_since_last)
        timeout_fired = time_since_last >= self.timeout_duration

        contrast_positive = self.activation_contrast == "positive"
        history_len = len(history)
        window = history[-self.consecutive_volumes :]

        # Window-based gates (trend / mean / count) need a full window first.
        needs_window = self.use_trend_condition or self.use_mean_condition or self.use_count_condition
        warming_up = needs_window and history_len < self.consecutive_volumes

        # --- Activation-level gate: current activation vs activation_threshold ---
        if contrast_positive:
            threshold_met = activation > self.activation_threshold
        else:
            threshold_met = activation < self.activation_threshold

        # --- Trend gate: slope sign (vs slope_threshold) or strict monotonic ---
        slope = 0.0
        trend_met = False
        if self.use_trend_condition and not warming_up:
            if self.trend_method == "strict":
                if self.activation_trend == "increasing":
                    trend_met = all(window[i] < window[i + 1] for i in range(len(window) - 1))
                else:
                    trend_met = all(window[i] > window[i + 1] for i in range(len(window) - 1))
            else:  # "slope"
                slope = _least_squares_slope(window)
                if self.activation_trend == "increasing":
                    trend_met = slope > self.slope_threshold
                else:
                    trend_met = slope < self.slope_threshold

        # --- Mean-over-window gate: mean activation vs mean_threshold ---
        mean_value = 0.0
        mean_met = False
        if self.use_mean_condition and not warming_up:
            mean_value = float(np.mean(window))
            mean_met = mean_value > self.mean_threshold if contrast_positive else mean_value < self.mean_threshold

        # --- Sustained-count gate: K of N window volumes past activation_threshold ---
        count_above = 0
        count_met = False
        if self.use_count_condition and not warming_up:
            if contrast_positive:
                count_above = sum(1 for v in window if v > self.activation_threshold)
            else:
                count_above = sum(1 for v in window if v < self.activation_threshold)
            count_met = count_above >= self.count_min

        # Combine the enabled gates by trigger_logic.
        gate_results: list[bool] = []
        if self.use_activation_condition:
            gate_results.append(threshold_met)
        if self.use_trend_condition:
            gate_results.append(trend_met)
        if self.use_mean_condition:
            gate_results.append(mean_met)
        if self.use_count_condition:
            gate_results.append(count_met)
        if not gate_results:
            conditions_met = False
        elif self.trigger_logic == "any":
            conditions_met = any(gate_results)
        else:  # "all"
            conditions_met = all(gate_results)

        # Strict display fraction: largest monotonic run (up to consecutive_volumes)
        # ending at the current value. This is the display metric, distinct from
        # trend_met (whole-window rule that actually gates the trigger).
        consecutive = 0
        if self.use_trend_condition and self.trend_method == "strict":
            for w in range(min(history_len, self.consecutive_volumes), 0, -1):
                run = history[-w:]
                if self.activation_trend == "increasing":
                    if all(run[i] < run[i + 1] for i in range(len(run) - 1)):
                        consecutive = w
                        break
                else:
                    if all(run[i] > run[i + 1] for i in range(len(run) - 1)):
                        consecutive = w
                        break

        if in_cooldown:
            decision: str | None = None
        elif timeout_fired:
            decision = "timeout"
        elif warming_up:
            decision = None
        elif conditions_met:
            decision = "triggered"
        else:
            decision = None

        return _TriggerEval(
            decision=decision,
            in_cooldown=in_cooldown,
            cooldown_remaining=cooldown_remaining,
            timeout_fired=timeout_fired,
            threshold_met=threshold_met,
            warming_up=warming_up,
            history_len=history_len,
            trend_met=trend_met,
            slope=slope,
            consecutive=consecutive,
            mean_met=mean_met,
            mean_value=mean_value,
            count_met=count_met,
            count_above=count_above,
        )

    def _format_trigger_suffix(self, ev: _TriggerEval) -> str:
        """Build the ASCII per-volume log suffix from a trigger evaluation.

        Shows one part per *enabled* condition gate plus the combine mode and,
        only when relevant, the verdict (TRIGGER / TIMEOUT / cooldown wait).
        Strictly ASCII: the session log FileHandler has no explicit encoding, so
        non-ASCII would raise UnicodeEncodeError on Windows and drop the line.
        """
        if ev.warming_up:
            return f" | warmup {ev.history_len}/{self.consecutive_volumes}"

        parts: list[str] = []
        if self.use_activation_condition:
            op = ">" if self.activation_contrast == "positive" else "<"
            parts.append(f"act{op}{self.activation_threshold:g} {'ok' if ev.threshold_met else 'no'}")
        if self.use_trend_condition:
            if self.trend_method == "strict":
                parts.append(
                    f"trend {self.activation_trend} {ev.consecutive}/{self.consecutive_volumes} "
                    f"{'ok' if ev.trend_met else 'no'}"
                )
            else:
                op = ">" if self.activation_trend == "increasing" else "<"
                parts.append(f"slope {ev.slope:+.3f}{op}{self.slope_threshold:g} {'ok' if ev.trend_met else 'no'}")
        if self.use_mean_condition:
            op = ">" if self.activation_contrast == "positive" else "<"
            parts.append(f"mean {ev.mean_value:+.3f}{op}{self.mean_threshold:g} {'ok' if ev.mean_met else 'no'}")
        if self.use_count_condition:
            parts.append(
                f"count {ev.count_above}/{self.consecutive_volumes}>={self.count_min} {'ok' if ev.count_met else 'no'}"
            )

        if ev.decision == "triggered":
            trailer = " => TRIGGER"
        elif ev.decision == "timeout":
            trailer = f" => TIMEOUT (no trigger in {self.timeout_duration:.0f}s)"
        elif ev.in_cooldown:
            trailer = f" => waiting (cooldown {math.ceil(ev.cooldown_remaining)}s)"
        else:
            trailer = ""

        joined = " | ".join(parts) if parts else "no-conditions"
        return f" | [{self.trigger_logic}] {joined}{trailer}"

    # =========================================================================
    # Ball movement (simplified from BallTask)
    # =========================================================================

    def _calculate_ball_movement(self, activations: dict) -> None:
        """Update movement parameters from the configured movement signal.

        Called once per new volume during ball episodes. In ``single`` mode
        the ball is driven by one network's signed activation (mirrors the
        connectivity-mode mapping in ``BallTask``); in ``dual`` mode by the
        difference between two networks.
        """
        if self.movement_mode == "single":
            self._calculate_single_movement(activations)
        else:
            self._calculate_dual_movement(activations)

    def _calculate_single_movement(self, activations: dict) -> None:
        """Single-network movement: magnitude from |z|, direction from sign."""
        val = activations.get(self._movement_mask, 0.0)

        # Reject outliers
        if abs(val) > self.pda_outlier_threshold:
            self._current_is_outlier = True
            return

        self._current_is_outlier = False
        self._current_activity = abs(val) / 10.0

        # Direction from signed activation. positive_down (_direction_sign=-1)
        # sends a positive activation (more rumination) downward; negative up.
        directed = self._direction_sign * val
        self._current_direction = 1.0 if directed >= 0 else -1.0

    def _calculate_dual_movement(self, activations: dict) -> None:
        """Dual-network movement: magnitude from the activation difference."""
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
        max_idx = int(np.argmax(acts))
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

                label = (
                    self._ball_mask_names[i]
                    if i < len(self._ball_mask_names)
                    else ("up" if self._network_directions[i] > 0 else "down")
                )
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
    # Lifecycle
    # =========================================================================

    def setup(self) -> None:
        """Initialize PsychoPy window, stimuli, instructions, and trigger."""
        from psychopy import core

        # Open per-volume psychopylog so the GUI live-metrics panel can poll triggers.
        self._psychopy_log_file = open(  # noqa: SIM115
            self._psychopy_log_path, "w", newline="", encoding="utf-8"
        )
        self._psychopy_log_writer = csv.writer(self._psychopy_log_file, delimiter="\t")
        self._psychopy_log_writer.writerow(["frame", "time", "volume", "triggers", "timeouts"])
        self._psychopy_log_file.flush()

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
        """Main task loop; the task ends with the scan.

        The loop exits as soon as the coordinator has processed every expected
        volume. An episode still in progress is truncated there and recorded as
        ``ball_<type>_incomplete`` by :meth:`cleanup`: after the last volume no
        brain data drives the ball, so carrying on would only show movement, and
        score hits, that the participant's activity did not produce.
        """
        self._running = True

        try:
            self.setup()

            while self._running and not self.coordinator.is_complete():
                frame_start = time.time()
                self.coordinator.check_error()
                data = self.get_latest_data()
                self.update(data)
                elapsed = time.time() - frame_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)

            self._await_final_activation()

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.cleanup()

    def _await_final_activation(self, timeout: float = 0.5) -> None:
        """Give the coordinator's last activation callback a moment to land.

        ``coordinator.is_complete()`` turns True as soon as the final volume is
        counted, a moment before that volume's callback appends it to
        ``_all_activations``; without this wait, cleanup could write the events
        file one row short. Bounded, because in intermittent mode the final
        volumes may never be delivered.
        """
        target = getattr(self.coordinator, "n_processed", 0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._all_activations and self._all_activations[-1]["volume"] >= target:
                return
            time.sleep(0.01)

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
            logger.info("Baseline complete, entering monitoring phase")

        self._fixation.draw()
        self._capture_and_emit_screenshot()
        self.win.flip()

    def _update_monitoring(self, data: dict) -> None:
        """Show fixation, check trigger on each new volume."""
        volume = data["volume"]
        activation = data["activations"].get(self.mask_name, 0.0)
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

    # =========================================================================
    # Cleanup and results
    # =========================================================================

    def get_run_results(self) -> dict:
        """Return task-specific results from the completed run."""
        results = super().get_run_results()
        results["n_episodes_triggered"] = self._trigger_state.n_episodes_triggered
        results["n_episodes_timeout"] = self._trigger_state.n_episodes_timeout
        results["n_episodes_incomplete"] = self._n_episodes_incomplete
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
            self._n_episodes_incomplete = 1
            n_processed = getattr(self.coordinator, "n_processed", "?")
            n_volumes = getattr(self.coordinator, "n_volumes", "?")
            logger.info(
                f"Episode #{self._episode_number} incomplete: run ended after {n_processed}/{n_volumes} volumes, "
                f"{duration:.1f}s into the {self.episode_duration:g}s episode "
                f"(hits=[{self._episode_hits[0]}, {self._episode_hits[1]}])"
            )

        logger.info(
            f"RITTriggeredBallTask cleanup: "
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

        if self._psychopy_log_file is not None:
            try:
                self._psychopy_log_file.close()
            except Exception:
                logger.exception("Failed to close psychopylog during cleanup")
            self._psychopy_log_file = None
            self._psychopy_log_writer = None

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


def _validate_rit_trigger_conditions(config: dict[str, Any]) -> None:
    """Cross-parameter validator for the configurable trigger condition gates.

    Ensures at least one neural condition is enabled (otherwise only timeout
    episodes would ever fire) and that ``count_min`` fits the window when the
    count gate is enabled.
    """
    enabled = any(
        config.get(key, default)
        for key, default in (
            ("use_activation_condition", True),
            ("use_trend_condition", True),
            ("use_mean_condition", False),
            ("use_count_condition", False),
        )
    )
    if not enabled:
        raise ValueError("At least one trigger condition must be enabled")
    if config.get("use_count_condition", False):
        count_min = config.get("count_min", 3)
        window = config.get("consecutive_volumes", 3)
        if count_min is not None and window is not None and count_min > window:
            raise ValueError("Count Min must be <= Consecutive Volumes when the count condition is enabled")


RIT_TRIGGERED_BALL_METADATA = TaskMetadata(
    task_id="rittriggeredball",
    display_name="RIT Triggered Ball",
    description=(
        "Monitors rumination network activation and triggers ball feedback "
        "episodes when threshold + trend conditions are met. By default the "
        "ball is driven by the rumination network's signed activation "
        "(single mode); it can also be driven by the difference between two "
        "networks (dual mode)."
    ),
    n_networks=1,
    network_roles=[
        NetworkRole(
            "trigger_network",
            "Trigger / Movement Network",
            "Network monitored for trigger conditions (e.g. rumination network). "
            "In single movement mode it also drives the ball unless a movement override is assigned.",
        ),
        NetworkRole(
            "movement_network",
            "Movement Network (single-mode override)",
            "Optional: drives the ball in single movement mode instead of the trigger network. "
            "Leave unassigned to use the trigger network.",
            optional=True,
            default_from="trigger_network",
            visible_when=("movement_mode", "single"),
        ),
        NetworkRole(
            "top_network",
            "Top Ball Network (dual mode, upward)",
            "Dual movement mode: ball moves up when this network is more active.",
            optional=True,
            visible_when=("movement_mode", "dual"),
        ),
        NetworkRole(
            "bottom_network",
            "Bottom Ball Network (dual mode, downward)",
            "Dual movement mode: ball moves down when this network is more active.",
            optional=True,
            visible_when=("movement_mode", "dual"),
        ),
    ],
    design_type="nuisance_only",
    parameters=[
        TaskParameter(
            key="movement_mode",
            label="Ball Movement Mode",
            type="str",
            default="single",
            choices=["single", "dual"],
            hint=(
                "single = one network's signed activation drives the ball "
                "(defaults to the trigger network unless a Movement Network is assigned). "
                "dual = activation difference between the Top and Bottom networks."
            ),
        ),
        TaskParameter(
            key="movement_direction",
            label="Movement Direction (single mode)",
            type="str",
            default="positive_down",
            choices=["positive_down", "positive_up"],
            hint=(
                "Single mode only. positive_down (default) = positive activation "
                "(more rumination) moves the ball down, negative (less rumination) "
                "moves it up. positive_up flips this. Ignored in dual mode."
            ),
        ),
        TaskParameter(
            key="activation_contrast",
            label="Activation Contrast",
            type="str",
            default="positive",
            choices=["positive", "negative"],
            hint="Positive = trigger activation > baseline, Negative = trigger activation < baseline",
        ),
        TaskParameter(
            key="trend_method",
            label="Trend Method",
            type="str",
            default="slope",
            choices=["strict", "slope"],
            hint=(
                "slope = require least-squares slope sign to match Activation Trend "
                "(default; tolerant of single blips; increase Consecutive Volumes to "
                "~5 for stronger blip-tolerance). strict = require monotonic change "
                "across all Consecutive Volumes."
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
            hint="Consecutive volumes increasing or decreasing in trigger activation",
        ),
        TaskParameter(
            key="use_activation_condition",
            label="Use Activation-Level Condition",
            type="bool",
            default=True,
            hint="Gate on current activation vs Activation Threshold (direction from Activation Contrast).",
        ),
        TaskParameter(
            key="activation_threshold",
            label="Activation Threshold",
            type="float",
            default=0.0,
            hint=(
                "Level used by the activation and count gates. Positive contrast fires when "
                "activation > this; negative when < this. 0 = at baseline."
            ),
        ),
        TaskParameter(
            key="use_trend_condition",
            label="Use Trend Condition",
            type="bool",
            default=True,
            hint="Gate on the slope/strict trend over Consecutive Volumes (uses Trend Method + Activation Trend).",
        ),
        TaskParameter(
            key="slope_threshold",
            label="Slope Threshold",
            type="float",
            default=0.0,
            hint="Slope method only: increasing fires when slope > this, decreasing when slope < this. 0 = any rise/fall.",
        ),
        TaskParameter(
            key="use_mean_condition",
            label="Use Mean-Activation Condition",
            type="bool",
            default=False,
            hint="Gate on the mean activation across Consecutive Volumes vs Mean Threshold (direction from Activation Contrast).",
        ),
        TaskParameter(
            key="mean_threshold",
            label="Mean Threshold",
            type="float",
            default=0.0,
            hint="Mean-activation threshold over the window. Positive contrast fires when mean > this; negative when < this.",
        ),
        TaskParameter(
            key="use_count_condition",
            label="Use Sustained-Count Condition",
            type="bool",
            default=False,
            hint="Gate on how many of the last Consecutive Volumes pass Activation Threshold (>= Count Min).",
        ),
        TaskParameter(
            key="count_min",
            label="Count Min (K of window)",
            type="int",
            default=3,
            min_value=1,
            max_value=20,
            hint="Minimum in-window volumes that must pass Activation Threshold for the count condition.",
        ),
        TaskParameter(
            key="trigger_logic",
            label="Combine Conditions",
            type="str",
            default="all",
            choices=["all", "any"],
            hint="all = every enabled condition must hold (AND). any = at least one enabled condition holds (OR).",
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
    ],
    cross_parameter_validator=_validate_rit_trigger_conditions,
    task_state_fields=[],
    post_run_questions=[],
    live_metrics=[
        LiveMetric(key="triggers", label="Triggers", column_pattern="triggers"),
    ],
    triggered_event_report=TriggeredEventReport(label="Triggered Episode Durations", episode_noun="Episode"),
)

register_task(RIT_TRIGGERED_BALL_METADATA)
register_task_class("rittriggeredball", RITTriggeredBallTask)
