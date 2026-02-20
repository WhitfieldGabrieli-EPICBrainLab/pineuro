"""STG block-design neurofeedback task with thermometer feedback.

This module implements the STG neurofeedback paradigm where participants
hear audio sentences and are instructed to either "listen" (upregulate STG)
or "ignore" (downregulate STG). After each block, a thermometer displays
the median activation level with green (success) or red (failure) coloring.

Task Flow (per run)
-------------------
1. Instructions screen
2. Scanner trigger wait
3. Baseline fixation (30s default)
4. Block loop (from condition CSV, typically 4 blocks):
   a. Instruction prompt (2s): "listen to self" or "ignore all sounds"
   b. Stimulus block (16s): 4x [ISI(0.5s) + voice(3.5s)]
   c. Rating (8s): "How well could you attend/ignore?" (1-6 slider)
   d. Feedback thermometer (4s, NFB runs only)
   e. Inter-block fixation (11s NFB / 15s transfer)
5. Save BIDS events TSV (includes per-TR activation rows)

Reference
---------
Based on the STG rt-fMRI neurofeedback paradigm (Run_STG_Task.py).
Transfer runs (runs 1 & 6) show fixation instead of the thermometer.
Feedback runs (runs 2-5) show post-block thermometer feedback from
the participant-specific STG mask.
"""

import csv
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

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


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class BlockInfo:
    """Condition information for a single block."""

    instruction_text: str
    audio_files: list[str]
    question_text: str
    block_type: str  # "listen" or "ignore"


# =============================================================================
# Task implementation
# =============================================================================


class ThermometerTask(BaseTask):
    """Block-design audio neurofeedback task with thermometer feedback.

    Each run consists of a baseline period followed by blocks of audio
    stimuli (listen/ignore), per-block ratings, and post-block thermometer
    feedback showing the median activation during that block.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data.
    mask_names : list of str, optional
        Names of masks. Only the first mask is used.
    stim_dir : str
        Directory containing condition CSV and audio files. The task
        auto-discovers the condition file by matching participant ID,
        session, and run number (e.g., sub-{id}_ses-{session}_task-stg_run-{run}.csv).
        Audio file paths in the CSV are resolved relative to this directory.
    feedback_mode : bool
        True for NFB runs (show thermometer), False for transfer runs.
    instruction_duration : float
        Duration of instruction prompt before each block.
    voice_duration : float
        Fixed duration for each voice file playback (seconds).
    isi_duration : float
        ISI/transition fixation before each voice file (seconds).
    rating_duration : float
        Timeout for the per-block rating scale.
    feedback_duration : float
        Duration of thermometer feedback display.
    fixation_nfb_duration : float
        Inter-block fixation duration for NFB runs.
    fixation_transfer_duration : float
        Inter-block fixation duration for transfer runs.
    activation_max : float
        Upper clamp for thermometer bar height.
    activation_min : float
        Lower clamp for thermometer bar height.
    session_state : SessionState, optional
        Session state for persistence across runs.
    trigger_keys : list of str
        Keys that trigger acquisition start.
    presenter_monitor : str
        Monitor position for task window.
    """

    task_id = "thermometer"

    def __init__(
        self,
        coordinator,
        mask_names: list[str] | None = None,
        stim_dir: str = "",
        feedback_mode: bool = True,
        instruction_duration: float = 2.0,
        voice_duration: float = 3.5,
        isi_duration: float = 0.5,
        rating_duration: float = 8.0,
        feedback_duration: float = 8.0,
        fixation_nfb_duration: float = 11.0,
        fixation_transfer_duration: float = 15.0,
        activation_max: float = 1.0,
        activation_min: float = -1.0,
        left_button: str = "1",
        right_button: str = "2",
        confirm_button: str = "3",
        session_state: Optional["SessionState"] = None,
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
            **kwargs,
        )

        self.mask_names = mask_names
        self.stim_dir = Path(stim_dir) if stim_dir else None
        self.trigger_keys = trigger_keys or ["t", "+", "5", "equal"]
        self.presenter_monitor = presenter_monitor

        # Response button mapping
        self.left_button = left_button
        self.right_button = right_button
        self.confirm_button = confirm_button

        # Timing parameters
        self.instruction_duration = instruction_duration
        self.voice_duration = voice_duration
        self.isi_duration = isi_duration
        self.rating_duration = rating_duration
        self.feedback_duration = feedback_duration
        self.fixation_nfb_duration = fixation_nfb_duration
        self.fixation_transfer_duration = fixation_transfer_duration
        self.activation_max = activation_max
        self.activation_min = activation_min

        # Activation accumulation state
        self._accumulating = False
        self._block_activations: list[float] = []

        # All per-volume activations (for end-of-run per-TR logging)
        self._all_activations: list[dict] = []

        # BIDS events log
        self._events: list[dict] = []

        # Block conditions (loaded in setup)
        self._blocks: list[BlockInfo] = []

        # Pre-loaded audio (populated in setup)
        self._preloaded_sounds: dict[str, object] = {}

        # Task clock (set in run)
        self._task_clock = None

        # Date string for TSV
        self._date = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        logger.info(
            f"ThermometerTask initialized: mask={self.mask_name}, "
            f"feedback_mode={feedback_mode}, stim_dir={stim_dir or 'None'}"
        )

    # =========================================================================
    # Condition file auto-discovery
    # =========================================================================

    def _resolve_condition_file(self) -> Path:
        """Auto-discover condition CSV file for this participant and run.

        Searches stim_dir for files matching:
        - sub-{participant}_ses-{session}_task-stg_run-{run}.csv
        - sub-{participant}_*stg*run-{run}.csv
        - sub-{participant}_*thermometer*run-{run}.csv

        Returns
        -------
        Path
            Resolved condition file path.

        Raises
        ------
        FileNotFoundError
            If no matching condition file is found.
        """
        if self.stim_dir is None or not self.stim_dir.is_dir():
            raise FileNotFoundError(f"Stimulus directory not found or not set: {self.stim_dir}")

        # Get participant/session/run info from coordinator
        participant = getattr(self.coordinator, "subject", "unknown")
        run_number = getattr(self.coordinator, "run_number", 1)
        session = getattr(self.coordinator, "session", "feedback")

        run_str = f"run-{run_number:02d}"

        # Try multiple patterns
        patterns = [
            f"*{participant}*ses-{session}*task-stg*{run_str}*.csv",
            f"*{participant}*stg*{run_str}*.csv",
            f"*{participant}*thermometer*{run_str}*.csv",
            f"*{participant}*{run_str}*.csv",
        ]

        for pattern in patterns:
            matches = list(self.stim_dir.glob(pattern))
            if matches:
                condition_file = matches[0]
                logger.info(f"Auto-discovered condition file: {condition_file}")
                return condition_file

        raise FileNotFoundError(
            f"No condition CSV found in {self.stim_dir} for "
            f"participant='{participant}', session='{session}', run={run_number}. "
            f"Tried patterns: {patterns}"
        )

    # =========================================================================
    # Condition file loading
    # =========================================================================

    def _load_conditions(self) -> list[BlockInfo]:
        """Load block conditions from auto-discovered CSV file.

        Expected columns: instruction_text, voice_file1, voice_file2,
        voice_file3, voice_file4, question.

        Audio file paths are resolved relative to stim_dir.
        """
        blocks = []
        condition_path = self._resolve_condition_file()

        with open(condition_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                instruction = row["instruction_text"].strip()
                raw_files = [
                    row["voice_file1"].strip(),
                    row["voice_file2"].strip(),
                    row["voice_file3"].strip(),
                    row["voice_file4"].strip(),
                ]

                # Resolve audio file paths relative to stim_dir if provided
                audio_files = []
                for f_path in raw_files:
                    f_path = Path(f_path)
                    if self.stim_dir and not f_path.is_absolute():
                        audio_files.append(str(self.stim_dir / f_path))
                    else:
                        audio_files.append(str(f_path))

                question = row["question"].strip()

                if "listen" in instruction.lower():
                    block_type = "listen"
                else:
                    block_type = "ignore"

                blocks.append(
                    BlockInfo(
                        instruction_text=instruction,
                        audio_files=audio_files,
                        question_text=question,
                        block_type=block_type,
                    )
                )

        logger.info(f"Loaded {len(blocks)} blocks from {condition_path}")
        if self.stim_dir:
            logger.info(f"Audio paths resolved relative to {self.stim_dir}")
        return blocks

    # =========================================================================
    # Activation accumulation
    # =========================================================================

    def _update_data(self, activations, volume, elapsed, is_baseline, is_outlier):
        """Override to accumulate activations during stimulus blocks."""
        super()._update_data(activations, volume, elapsed, is_baseline, is_outlier)

        # Store all activations for end-of-run per-TR logging
        self._all_activations.append(
            {
                "volume": volume,
                "elapsed": elapsed,
                "stg": activations.get(self.mask_name, float("nan")),
                "is_baseline": is_baseline,
                "is_outlier": is_outlier,
            }
        )

        if self._accumulating and not is_outlier:
            activation = activations.get(self.mask_name, 0.0)
            self._block_activations.append(activation)

    def _start_accumulating(self):
        """Begin collecting per-volume activations for current block."""
        self._block_activations = []
        self._accumulating = True

    def _stop_accumulating(self) -> list[float]:
        """Stop collecting and return accumulated activations."""
        self._accumulating = False
        return self._block_activations.copy()

    # =========================================================================
    # BIDS events
    # =========================================================================

    # Fixed column order matching the original Run_STG_Task.py
    _TSV_COLUMNS = [
        "onset",
        "duration",
        "trial_type",
        "block_count",
        "block_type",
        "response",
        "response_time",
        "stg_activity",
        "smc_activity",
        "feedback",
        "sound_content",
        "visual_content",
        "number",
        "stim_file",
        "stim_duration",
        "date",
    ]

    def _record_event(self, onset: float, duration: float, trial_type: str, **kwargs):
        """Record an event for the BIDS events TSV."""
        event = {col: "n/a" for col in self._TSV_COLUMNS}
        event["onset"] = round(onset, 1)
        event["duration"] = round(duration, 1) if not isinstance(duration, str) else duration
        event["trial_type"] = trial_type
        event["date"] = self._date
        event.update(kwargs)
        self._events.append(event)

    def _save_events_tsv(self):
        """Write accumulated events to BIDS-compatible TSV."""
        if not self._events:
            return

        output_dir = getattr(self.coordinator, "output_dir", None)
        if output_dir is None:
            logger.warning("No output_dir on coordinator, cannot save events TSV")
            return

        output_dir = Path(output_dir)
        subject = getattr(self.coordinator, "subject", "unknown")
        task = getattr(self.coordinator, "task", "thermometer")
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
            for event in self._events:
                writer.writerow(event)

        logger.info(f"Saved {len(self._events)} events to {path}")

    # =========================================================================
    # Display helpers
    # =========================================================================

    def _wait_with_escape(self, duration: float, draw_func=None):
        """Wait for a duration while redrawing each frame for the GUI mirror.

        Parameters
        ----------
        duration : float
            How long to wait in seconds.
        draw_func : callable, optional
            Function to call each frame to draw stimuli. If provided, the
            display is redrawn, captured, and flipped every frame (matching
            the ball task pattern). If None, the screen is not updated.
        """
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

    def _wait_for_baseline_end(self):
        """Show fixation cross until coordinator's ``is_baseline`` flips to False."""
        from psychopy import core, event

        while self._running:
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")
            if hasattr(self, "coordinator") and self.coordinator.is_complete():
                break

            data = self.get_latest_data()
            is_baseline = data.get("is_baseline", True)
            if not is_baseline:
                break

            self._fixation.draw()
            self._capture_and_emit_screenshot()
            self.win.flip()

            keys = event.getKeys(keyList=["escape"])
            if keys:
                raise KeyboardInterrupt("Escaped")

            core.wait(0.01)

    def _show_fixation(self, duration: float):
        """Display fixation cross for a given duration, updating GUI each frame."""

        def draw():
            self._fixation.draw()

        self._wait_with_escape(duration, draw_func=draw)

    def _show_instruction_prompt(self, text: str, duration: float):
        """Display block instruction for a given duration, updating GUI each frame."""
        self._instruction_stim.text = text

        def draw():
            self._instruction_stim.draw()

        self._wait_with_escape(duration, draw_func=draw)

    def _play_audio_block(self, audio_files: list[str], block_idx: int, block_type: str):
        """Play audio sentences with fixed timing: ISI(0.5s) + voice(3.5s) each.

        Matches the original Run_STG_Task.py timing structure where each
        voice file is preceded by a 0.5s ISI/transition fixation and plays
        for exactly 3.5s (force-stopped if longer).

        Returns
        -------
        list[dict]
            Per-voice event info dicts with onset, file, duration, number.
        """

        voice_events = []

        for voice_idx, audio_path in enumerate(audio_files):
            if not self._running:
                break

            # --- ISI transition (0.5s fixation) ---
            isi_onset = self._task_clock.getTime()

            def draw_fix():
                self._fixation.draw()

            self._wait_with_escape(self.isi_duration, draw_func=draw_fix)

            self._record_event(
                isi_onset,
                self.isi_duration,
                "ISI",
                block_count=block_idx + 1,
                block_type=block_type,
                visual_content="+",
            )

            # --- Voice file (3.5s fixed) ---
            voice_onset = self._task_clock.getTime()

            # Use pre-loaded sound or load on demand
            snd = self._preloaded_sounds.get(audio_path)
            if snd is None:
                from psychopy.sound.backend_pygame import SoundPygame

                snd = SoundPygame(audio_path)

            # Play audio over fixation cross for fixed duration
            snd.play()
            self._wait_with_escape(self.voice_duration, draw_func=draw_fix)
            snd.stop()

            # Extract file number from path for logging
            file_stem = Path(audio_path).stem
            # Try to extract numeric portion (e.g., "036_103" -> "103", "JO_S347" -> "347")
            number = "n/a"
            parts = file_stem.split("_")
            for part in reversed(parts):
                if part.isdigit():
                    number = part
                    break
                # Handle e.g. "S347" -> "347"
                stripped = part.lstrip("SsQq")
                if stripped.isdigit():
                    number = stripped
                    break

            # Get actual duration from wav for logging
            actual_duration = snd.getDuration() or self.voice_duration

            self._record_event(
                voice_onset,
                self.voice_duration,
                "voice_file",
                block_count=block_idx + 1,
                block_type=block_type,
                visual_content="+",
                number=number,
                stim_file=audio_path,
                stim_duration=round(actual_duration, 1),
            )

            voice_events.append(
                {
                    "onset": voice_onset,
                    "file": audio_path,
                    "number": number,
                    "actual_duration": actual_duration,
                }
            )

        return voice_events

    def _run_block_rating(self, question_text: str, duration: float):
        """Display 1-6 rating scale with sliding marker and timeout.

        Replicates the original Run_STG_Task.py RatingScale behavior:
        - Triangle marker starting at position 3.5 (midpoint)
        - Key '1' moves marker one tick to the left
        - Key '2' moves marker one tick to the right
        - Key '3' or 'return' confirms the current position
        - Labels: 'Completely' (left/1) and 'Not at all' (right/6)
        - 8s timeout

        Returns
        -------
        tuple of (rating_value, response_time)
            (int or None, float or None). None if timed out.
        """
        from psychopy import core, event, visual

        # Scale layout parameters
        scale_low = 1
        scale_high = 6
        n_ticks = scale_high - scale_low + 1
        scale_y = -0.15
        scale_width = 0.6
        tick_spacing = scale_width / (n_ticks - 1)
        scale_left_x = -scale_width / 2

        # Tick positions (x coords for values 1-6)
        tick_x = [scale_left_x + i * tick_spacing for i in range(n_ticks)]

        # Current marker position (start at 3.5, between tick indices 2 and 3)
        marker_pos = 3.5

        def pos_to_x(pos):
            """Convert scale position (1-6) to x coordinate."""
            return scale_left_x + (pos - scale_low) * tick_spacing

        # Create scale line
        scale_line = visual.Line(
            self.win,
            start=(-scale_width / 2, scale_y),
            end=(scale_width / 2, scale_y),
            lineColor="white",
            lineWidth=2,
        )

        # Create tick marks and number labels
        tick_marks = []
        tick_labels_stim = []
        for i, val in enumerate(range(scale_low, scale_high + 1)):
            tick = visual.Line(
                self.win,
                start=(tick_x[i], scale_y - 0.015),
                end=(tick_x[i], scale_y + 0.015),
                lineColor="white",
                lineWidth=2,
            )
            tick_marks.append(tick)
            label = visual.TextStim(
                self.win,
                text=str(val),
                pos=(tick_x[i], scale_y - 0.05),
                height=0.035,
                color="white",
            )
            tick_labels_stim.append(label)

        # Endpoint labels below number labels
        left_label = visual.TextStim(
            self.win,
            text="Completely",
            pos=(tick_x[0], scale_y - 0.1),
            height=0.035,
            color="white",
        )
        right_label = visual.TextStim(
            self.win,
            text="Not at all",
            pos=(tick_x[-1], scale_y - 0.1),
            height=0.035,
            color="white",
        )

        # Triangle marker (pointing down toward the scale line)
        marker_size = 0.02
        marker_gap = 0.02  # gap between scale line and marker tip

        def make_marker(x_pos):
            return visual.ShapeStim(
                self.win,
                vertices=[
                    (x_pos - marker_size, scale_y + marker_gap + marker_size * 2),
                    (x_pos + marker_size, scale_y + marker_gap + marker_size * 2),
                    (x_pos, scale_y + marker_gap),
                ],
                fillColor="white",
                lineColor="white",
                closeShape=True,
            )

        # Question text - positioned above the scale with enough clearance
        self._rating_question.text = question_text
        self._rating_question.pos = (0, 0.15)
        self._rating_question.height = 0.06
        self._rating_question.wrapWidth = 1.2

        clock = core.Clock()
        timer = core.CountdownTimer(duration)
        confirmed = False
        rt = None

        event.clearEvents("keyboard")

        while timer.getTime() > 0 and self._running:
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")

            keys = event.getKeys(
                keyList=[self.left_button, self.right_button, self.confirm_button, "return", "escape"],
                timeStamped=clock,
            )
            for key, t in keys:
                if key == "escape":
                    raise KeyboardInterrupt("Escaped during rating")
                if key == self.left_button:
                    if marker_pos != int(marker_pos):
                        # First move from 3.5: snap to nearest integer left
                        marker_pos = math.floor(marker_pos)
                    elif marker_pos > scale_low:
                        marker_pos -= 1
                elif key == self.right_button:
                    if marker_pos != int(marker_pos):
                        # First move from 3.5: snap to nearest integer right
                        marker_pos = math.ceil(marker_pos)
                    elif marker_pos < scale_high:
                        marker_pos += 1
                elif key in (self.confirm_button, "return"):
                    confirmed = True
                    rt = t
                    break

            if confirmed:
                break

            # Draw all components each frame (for GUI mirror updates)
            self._rating_question.draw()
            scale_line.draw()
            for tm in tick_marks:
                tm.draw()
            for tl in tick_labels_stim:
                tl.draw()
            left_label.draw()
            right_label.draw()
            make_marker(pos_to_x(marker_pos)).draw()
            self._capture_and_emit_screenshot()
            self.win.flip()

        response = int(round(marker_pos)) if confirmed else None
        last_rating = (response, rt)

        # Wait out remaining time showing fixation
        remaining = timer.getTime()
        if remaining > 0:

            def draw_fix():
                self._fixation.draw()

            self._wait_with_escape(remaining, draw_func=draw_fix)

        return last_rating

    def _show_feedback_thermometer(self, block_median: float, block_type: str, duration: float):
        """Display animated thermometer feedback for a block.

        Creates 30 rectangle segments and reveals them progressively
        frame-by-frame, matching the original Run_STG_Task.py animation.

        Parameters
        ----------
        block_median : float
            Median activation during the block.
        block_type : str
            "listen" or "ignore".
        duration : float
            How long to display the thermometer.
        """
        from psychopy import core, event, visual

        if not self.feedback_mode:
            self._show_fixation(duration)
            return

        # Determine color based on instruction and direction
        act_max = self.activation_max
        act_min = self.activation_min
        zero_val = 0.0

        fb = float(np.clip(block_median, act_min, act_max))

        # Build 30 rectangle segments matching original makeFBrecs()
        recs = []
        n_segments = 30

        if block_type == "ignore":
            if fb >= zero_val:
                # Positive when should be negative -> Red (bad)
                rec_height = (fb - zero_val) / (n_segments * (act_max - zero_val)) * 0.5
                for idx in range(n_segments):
                    rec = visual.ShapeStim(
                        self.win,
                        closeShape=True,
                        vertices=(
                            (-0.125, idx * rec_height),
                            (0.125, idx * rec_height),
                            (0.125, (idx + 1) * rec_height),
                            (-0.125, (idx + 1) * rec_height),
                        ),
                        depth=-3,
                        opacity=1,
                        fillColor="Red",
                        lineColor="Red",
                    )
                    recs.append(rec)
            else:
                # Negative when should be negative -> Green (good)
                rec_height = (fb - zero_val) / (n_segments * (act_min - zero_val)) * 0.5
                for idx in range(n_segments):
                    rec = visual.ShapeStim(
                        self.win,
                        closeShape=True,
                        vertices=(
                            (-0.125, idx * -1 * rec_height),
                            (0.125, idx * -1 * rec_height),
                            (0.125, (idx + 1) * -1 * rec_height),
                            (-0.125, (idx + 1) * -1 * rec_height),
                        ),
                        depth=-3,
                        opacity=1,
                        fillColor="Green",
                        lineColor="Green",
                    )
                    recs.append(rec)
        else:  # "listen"
            if fb >= zero_val:
                # Positive when should be positive -> Green (good)
                rec_height = (fb - zero_val) / (n_segments * (act_max - zero_val)) * 0.5
                for idx in range(n_segments):
                    rec = visual.ShapeStim(
                        self.win,
                        closeShape=True,
                        vertices=(
                            (-0.125, idx * rec_height),
                            (0.125, idx * rec_height),
                            (0.125, (idx + 1) * rec_height),
                            (-0.125, (idx + 1) * rec_height),
                        ),
                        depth=-3,
                        opacity=1,
                        fillColor="Green",
                        lineColor="Green",
                    )
                    recs.append(rec)
            else:
                # Negative when should be positive -> Red (bad)
                rec_height = (fb - zero_val) / (n_segments * (act_min - zero_val)) * 0.5
                for idx in range(n_segments):
                    rec = visual.ShapeStim(
                        self.win,
                        closeShape=True,
                        vertices=(
                            (-0.125, idx * -1 * rec_height),
                            (0.125, idx * -1 * rec_height),
                            (0.125, (idx + 1) * -1 * rec_height),
                            (-0.125, (idx + 1) * -1 * rec_height),
                        ),
                        depth=-3,
                        opacity=1,
                        fillColor="Red",
                        lineColor="Red",
                    )
                    recs.append(rec)

        # Set top/bottom labels
        self._thermo_top_label.text = "listen to self"
        self._thermo_bottom_label.text = "ignore all sounds"

        # Animated display loop
        timer = core.CountdownTimer(duration)
        frame_n = 0

        while timer.getTime() > 0 and self._running:
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")

            keys = event.getKeys(keyList=["escape"])
            if keys:
                raise KeyboardInterrupt("Escaped during feedback")

            # Draw static elements
            self._thermo_outline.draw()
            self._thermo_zero_line.draw()
            self._thermo_zero_label.draw()
            self._thermo_top_label.draw()
            self._thermo_bottom_label.draw()

            # Progressive reveal: show up to frame_n segments (max 30)
            max_idx = min(frame_n, n_segments)
            if max_idx > 1:
                for idx in range(max_idx):
                    recs[idx].draw()

            self._capture_and_emit_screenshot(force=(frame_n == 0))
            self.win.flip()
            frame_n += 1

        # Determine feedback color for TSV logging
        if recs:
            feedback_color = "red" if recs[0].fillColor[0] == 1.0 else "green"
        else:
            feedback_color = "n/a"

        return feedback_color

    # =========================================================================
    # Per-TR activation logging (end of run)
    # =========================================================================

    def _append_activation_rows(self):
        """Append per-TR activation rows to events, matching original task.

        The original task appends per-TR activation values covering all TRs
        from the first block to the last, tagged with block numbers and
        'intertrial' labels.
        """
        if not self._all_activations:
            return

        tr = getattr(self.coordinator, "tr", 1.2)

        # Find block stimulus events to determine TR ranges
        block_events = [e for e in self._events if e["trial_type"] == "voice_file"]
        if not block_events:
            return

        # Build block TR ranges from voice_file events
        # Group voice events by block_count
        blocks_by_count: dict[int, list[dict]] = {}
        for e in block_events:
            bc = e["block_count"]
            if bc not in blocks_by_count:
                blocks_by_count[bc] = []
            blocks_by_count[bc].append(e)

        # Calculate TR index ranges for each block
        # Each block's activation window = first voice onset to last voice onset + voice_duration
        block_tr_ranges = []
        block_types = []
        for bc in sorted(blocks_by_count.keys()):
            events = blocks_by_count[bc]
            first_onset = min(float(e["onset"]) for e in events)
            last_onset = max(float(e["onset"]) for e in events)
            block_end = last_onset + self.voice_duration

            # Convert to TR indices (1-indexed, offset by baseline)
            start_tr = int(round(first_onset / tr)) + 1
            end_tr = int(round(block_end / tr)) + 1
            block_tr_ranges.append((start_tr, end_tr))
            block_types.append(events[0]["block_type"])

        # Determine overall range: first block start to last block end
        if not block_tr_ranges:
            return

        overall_start_tr = block_tr_ranges[0][0]
        overall_end_tr = block_tr_ranges[-1][1]

        # Filter activations to the relevant range
        intertrial_num = 1
        intertrial_count = 0

        for act_data in self._all_activations:
            vol = act_data["volume"]
            if vol < overall_start_tr or vol > overall_end_tr:
                continue

            onset = round(vol * tr, 1)
            stg_val = act_data["stg"]

            # Determine which block this TR belongs to
            in_block = False
            for block_idx, (start, end) in enumerate(block_tr_ranges):
                if start <= vol <= end:
                    self._record_event(
                        onset,
                        tr,
                        "activation",
                        block_count=block_idx + 1,
                        block_type=block_types[block_idx],
                        stg_activity=stg_val,
                        stim_file="murfi",
                    )
                    in_block = True
                    break

            if not in_block:
                intertrial_count += 1
                if intertrial_count % 25 == 0:
                    intertrial_num += 1

                self._record_event(
                    onset,
                    tr,
                    "activation",
                    block_type=f"intertrial{intertrial_num}",
                    stg_activity=stg_val,
                    stim_file="murfi",
                )

    # =========================================================================
    # Block execution
    # =========================================================================

    def _run_block(self, block_idx: int, block_info: BlockInfo):
        """Execute a single block: instruction -> stimulus -> rating -> feedback -> fixation."""
        # 1. Instruction prompt
        onset = self._task_clock.getTime()
        self._show_instruction_prompt(block_info.instruction_text, self.instruction_duration)
        self._record_event(
            onset,
            self.instruction_duration,
            block_info.instruction_text,
            block_count=block_idx + 1,
            block_type="instruction",
            visual_content=block_info.instruction_text,
        )

        # 2. Stimulus block with activation accumulation
        self._start_accumulating()
        _voice_events = self._play_audio_block(
            block_info.audio_files,
            block_idx,
            block_info.block_type,
        )
        block_activations = self._stop_accumulating()
        block_median = float(np.median(block_activations)) if block_activations else 0.0

        # 3. Rating
        onset = self._task_clock.getTime()
        last_rating = self._run_block_rating(block_info.question_text, self.rating_duration)
        rating_response = last_rating[0] if last_rating[0] is not None else "n/a"
        rating_rt = round(last_rating[1], 2) if last_rating[1] is not None else "n/a"
        self._record_event(
            onset,
            self.rating_duration,
            "rating",
            block_count=block_idx + 1,
            block_type=block_info.instruction_text,
            response=rating_response,
            response_time=rating_rt,
            visual_content=block_info.question_text,
        )

        # 4. Feedback thermometer (or fixation for transfer)
        onset = self._task_clock.getTime()
        if self.feedback_mode:
            feedback_color = self._show_feedback_thermometer(
                block_median,
                block_info.block_type,
                self.feedback_duration,
            )
            self._record_event(
                onset,
                self.feedback_duration,
                "feedback",
                block_count=block_idx + 1,
                block_type=block_info.instruction_text,
                feedback=round(block_median, 3),
                visual_content=feedback_color,
            )
            fixation_dur = self.fixation_nfb_duration
        else:
            self._show_fixation(self.feedback_duration)
            self._record_event(
                onset,
                self.feedback_duration,
                "nofeedback",
                block_count=block_idx + 1,
                block_type=block_info.instruction_text,
                visual_content="+",
            )
            fixation_dur = self.fixation_transfer_duration

        # 5. Inter-block fixation
        onset = self._task_clock.getTime()
        self._show_fixation(fixation_dur)
        self._record_event(
            onset,
            fixation_dur,
            "fixation",
            block_count=block_idx + 1,
            block_type=block_info.instruction_text,
            visual_content="+",
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def setup(self):
        """Initialize PsychoPy window, load conditions, pre-load audio, create stimuli."""
        from psychopy import visual
        from psychopy.sound.backend_pygame import SoundPygame

        # Load conditions
        self._blocks = self._load_conditions()

        # Create window
        self.create_window(
            monitor_position=self.presenter_monitor,
            color="black",
            units="height",
        )

        # Pre-load all audio files using pygame backend.
        # We use SoundPygame directly because PsychoPy's default PTB backend
        # does not respect the OS default audio device on Windows.
        for block in self._blocks:
            for audio_path in block.audio_files:
                if audio_path not in self._preloaded_sounds:
                    try:
                        self._preloaded_sounds[audio_path] = SoundPygame(audio_path)
                    except Exception as e:
                        logger.warning(f"Could not pre-load audio {audio_path}: {e}")

        # Create fixation cross
        self._fixation = visual.TextStim(
            self.win,
            text="+",
            pos=(0, 0),
            height=0.3,
            color="white",
        )

        # Create instruction text stimulus
        self._instruction_stim = visual.TextStim(
            self.win,
            text="",
            pos=(0, 0),
            height=0.1,
            color="white",
        )

        # Create rating question stimulus
        self._rating_question = visual.TextStim(
            self.win,
            text="",
            pos=(0, 0),
            height=0.1,
            wrapWidth=None,
            color="white",
        )

        # Create thermometer visual components
        self._thermo_outline = visual.ShapeStim(
            self.win,
            lineWidth=2.0,
            lineColor="white",
            vertices=((-0.125, -0.5), (0.125, -0.5), (0.125, 0.5), (-0.125, 0.5)),
            closeShape=True,
            fillColor=None,
        )
        self._thermo_zero_line = visual.ShapeStim(
            self.win,
            lineWidth=2.0,
            lineColor="white",
            vertices=((-0.125, 0), (0.125, 0)),
            closeShape=False,
        )
        self._thermo_zero_label = visual.TextStim(
            self.win,
            text="0",
            pos=(0.155, 0),
            height=0.04,
            color="white",
        )
        self._thermo_top_label = visual.TextStim(
            self.win,
            text="listen to self",
            pos=(0, 0.6),
            height=0.06,
            color="white",
        )
        self._thermo_bottom_label = visual.TextStim(
            self.win,
            text="ignore all sounds",
            pos=(0, -0.6),
            height=0.06,
            color="white",
        )
        self._thermo_top_star = visual.TextStim(
            self.win,
            text="*",
            pos=(0.5, 0.5),
            height=0.5,
            color="white",
        )
        self._thermo_bottom_star = visual.TextStim(
            self.win,
            text="*",
            pos=(0.5, -0.5),
            height=0.5,
            color="white",
        )

        # Show task instructions (matching original wording)
        # Only confirm_button advances (original accepted any button box key)
        self.show_instructions(
            [
                "In this task you will be asked to either "
                "attend to your own voice or ignore all sounds "
                "including the voices and any other noise "
                "in the environment.\n\n"
                "When ignoring sounds, you may use "
                '"noting practice" strategy suggested '
                "by the experimenter.\n\n"
                'When you are ready please press the "YES" button.',
            ],
            advance_keys=[self.confirm_button, "space"],
        )

        # Wait for scanner trigger
        self.wait_for_trigger()

        logger.info("ThermometerTask setup complete")

    def update(self, data: dict):
        """Minimal implementation for BaseTask contract. Not used in main loop."""
        pass

    def run(self):
        """Main task loop -- time-driven block structure."""
        from psychopy import core

        self._running = True

        try:
            self.setup()

            self._task_clock = core.Clock()

            # Initial baseline (driven by coordinator's n_baseline)
            onset = self._task_clock.getTime()
            self._wait_for_baseline_end()
            baseline_dur = self._task_clock.getTime() - onset
            self._record_event(onset, baseline_dur, "baseline", block_type="baseline", visual_content="+")

            # Block loop
            for block_idx, block_info in enumerate(self._blocks):
                if not self._running:
                    break
                self._run_block(block_idx, block_info)

            # Append per-TR activation rows at end of run
            self._append_activation_rows()

        except KeyboardInterrupt:
            logger.info("ThermometerTask interrupted")
        finally:
            self._running = False
            self._save_events_tsv()
            self.cleanup()

    def get_run_results(self) -> dict:
        """Return ThermometerTask-specific results from the completed run."""
        results = super().get_run_results()
        results["n_blocks_completed"] = (
            len([e for e in self._events if e["trial_type"] == "voice_file"]) // 4
        )  # 4 voice files per block
        return results

    def cleanup(self):
        """Clean up resources."""
        logger.info("ThermometerTask cleanup")

        # Track volumes processed for base class reporting
        self._volumes_processed = len(self._all_activations)

        # Set snapshot for base class run recording
        self._task_state_snapshot = {
            "n_blocks_completed": len([e for e in self._events if e["trial_type"] == "voice_file"])
            // 4,  # 4 voice files per block
        }

        # Delete visual stimuli while GL context is still alive
        # (prevents PsychoPy TextStim.__del__ access violations during GC)
        for attr in [
            "_fixation",
            "_instruction_stim",
            "_rating_question",
            "_thermo_outline",
            "_thermo_zero_line",
            "_thermo_zero_label",
            "_thermo_top_label",
            "_thermo_bottom_label",
            "_thermo_top_star",
            "_thermo_bottom_star",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

        # Let base class handle window close and gl_info reset
        super().cleanup()

    def stop(self):
        """Signal task to stop."""
        self._running = False


# =============================================================================
# Task Metadata and Registration
# =============================================================================

THERMOMETER_METADATA = TaskMetadata(
    task_id="thermometer",
    display_name="STG Neurofeedback Task",
    description="Block-design audio neurofeedback with thermometer feedback",
    n_networks=1,
    network_roles=[
        NetworkRole("target_mask", "Target Region", "Brain region for thermometer feedback"),
    ],
    design_type="nuisance_only",
    parameters=[
        TaskParameter(
            key="use_transfer_runs",
            label="Use Transfer Runs",
            type="bool",
            default=True,
            hint="Include transfer runs (1 before feedback, 1 after)",
        ),
        TaskParameter(
            key="stim_dir",
            label="Stimulus Directory",
            type="str",
            default="",
            hint="Directory containing condition CSV files (auto-discovered by participant/run) and audio files referenced in the CSVs.",
        ),
        TaskParameter(
            key="instruction_duration",
            label="Instruction Duration (s)",
            type="float",
            default=2.0,
            min_value=0.5,
            max_value=10.0,
            hint="Duration of block instruction prompt",
        ),
        TaskParameter(
            key="voice_duration",
            label="Voice Duration (s)",
            type="float",
            default=3.5,
            min_value=1.0,
            max_value=10.0,
            hint="Fixed duration for each voice file playback",
        ),
        TaskParameter(
            key="isi_duration",
            label="ISI Duration (s)",
            type="float",
            default=0.5,
            min_value=0.0,
            max_value=5.0,
            hint="Inter-stimulus interval (fixation) before each voice file",
        ),
        TaskParameter(
            key="rating_duration",
            label="Rating Duration (s)",
            type="float",
            default=8.0,
            min_value=2.0,
            max_value=30.0,
            hint="Timeout for per-block rating scale",
        ),
        TaskParameter(
            key="feedback_duration",
            label="Feedback Duration (s)",
            type="float",
            default=8.0,
            min_value=1.0,
            max_value=15.0,
            hint="Duration of thermometer feedback display",
        ),
        TaskParameter(
            key="fixation_nfb_duration",
            label="NFB Fixation Duration (s)",
            type="float",
            default=11.0,
            min_value=0.0,
            max_value=30.0,
            hint="Inter-block fixation for neurofeedback runs",
        ),
        TaskParameter(
            key="fixation_transfer_duration",
            label="Transfer Fixation Duration (s)",
            type="float",
            default=15.0,
            min_value=0.0,
            max_value=30.0,
            hint="Inter-block fixation for transfer runs",
        ),
        TaskParameter(
            key="activation_max",
            label="Activation Max",
            type="float",
            default=1.0,
            min_value=0.1,
            max_value=10.0,
            hint="Upper clamp for thermometer bar",
        ),
        TaskParameter(
            key="activation_min",
            label="Activation Min",
            type="float",
            default=-1.0,
            min_value=-10.0,
            max_value=-0.1,
            hint="Lower clamp for thermometer bar",
        ),
        TaskParameter(
            key="left_button",
            label="Left Button",
            type="str",
            default="1",
            hint="Key for moving rating scale left (index finger)",
        ),
        TaskParameter(
            key="right_button",
            label="Right Button",
            type="str",
            default="2",
            hint="Key for moving rating scale right (middle finger)",
        ),
        TaskParameter(
            key="confirm_button",
            label="Confirm/YES Button",
            type="str",
            default="3",
            hint="Key for confirming rating and advancing instructions (ring finger)",
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

# Auto-register on import
register_task(THERMOMETER_METADATA)
register_task_class("thermometer", ThermometerTask)
