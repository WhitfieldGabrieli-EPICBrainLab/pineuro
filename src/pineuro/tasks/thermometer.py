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
   b. Stimulus block (16s): 4 audio sentences with fixation cross
   c. Rating (8s): "How well could you attend/ignore?" (1-6 scale)
   d. Feedback thermometer (4s, NFB runs only)
   e. Inter-block fixation (11s NFB / 15s transfer)
5. Save BIDS events TSV

Reference
---------
Based on the STG rt-fMRI neurofeedback paradigm. Transfer runs (runs 1 & 6)
show fixation instead of the thermometer. Feedback runs (runs 2-5) show
post-block thermometer feedback from the participant-specific STG mask.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from .base import BaseTask
from .task_registry import (
    TaskMetadata,
    TaskParameter,
    register_task,
)
from .factory import register_task_class

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
    baseline_duration : float
        Initial baseline fixation duration in seconds.
    instruction_duration : float
        Duration of instruction prompt before each block.
    stimulus_duration : float
        Duration of the audio stimulus block.
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
    fullscreen : bool
        Whether to run in fullscreen mode.
    trigger_keys : list of str
        Keys that trigger acquisition start.
    response_keys : list of str
        Keys for rating scale responses (1-6).
    presenter_monitor : str
        Monitor position for task window.
    """

    task_id = "thermometer"

    def __init__(
        self,
        coordinator,
        mask_names: Optional[list[str]] = None,
        stim_dir: str = "",
        feedback_mode: bool = True,
        baseline_duration: float = 30.0,
        instruction_duration: float = 2.0,
        stimulus_duration: float = 16.0,
        rating_duration: float = 8.0,
        feedback_duration: float = 4.0,
        fixation_nfb_duration: float = 11.0,
        fixation_transfer_duration: float = 15.0,
        activation_max: float = 1.0,
        activation_min: float = -1.0,
        session_state: Optional["SessionState"] = None,
        fullscreen: bool = True,
        trigger_keys: Optional[list[str]] = None,
        response_keys: Optional[list[str]] = None,
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
        self.fullscreen = fullscreen
        self.trigger_keys = trigger_keys or ["t", "+", "5", "equal"]
        self.response_keys = response_keys or ["1", "2", "3", "4", "5", "6"]
        self.presenter_monitor = presenter_monitor

        # Timing parameters
        self.baseline_duration = baseline_duration
        self.instruction_duration = instruction_duration
        self.stimulus_duration = stimulus_duration
        self.rating_duration = rating_duration
        self.feedback_duration = feedback_duration
        self.fixation_nfb_duration = fixation_nfb_duration
        self.fixation_transfer_duration = fixation_transfer_duration
        self.activation_max = activation_max
        self.activation_min = activation_min

        # Activation accumulation state
        self._accumulating = False
        self._block_activations: list[float] = []

        # BIDS events log
        self._events: list[dict] = []

        # Block conditions (loaded in setup)
        self._blocks: list[BlockInfo] = []

        # Pre-loaded audio (populated in setup)
        self._preloaded_sounds: dict[str, object] = {}

        # Task clock (set in run)
        self._task_clock = None

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
            raise FileNotFoundError(
                f"Stimulus directory not found or not set: {self.stim_dir}"
            )

        # Get participant/session/run info from coordinator
        participant = getattr(self.coordinator, 'subject', 'unknown')
        run_number = getattr(self.coordinator, 'run_number', 1)
        session = getattr(self.coordinator, 'session', 'feedback')

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
                instruction = row['instruction_text'].strip()
                raw_files = [
                    row['voice_file1'].strip(),
                    row['voice_file2'].strip(),
                    row['voice_file3'].strip(),
                    row['voice_file4'].strip(),
                ]

                # Resolve audio file paths relative to stim_dir if provided
                audio_files = []
                for f_path in raw_files:
                    f_path = Path(f_path)
                    if self.stim_dir and not f_path.is_absolute():
                        audio_files.append(str(self.stim_dir / f_path))
                    else:
                        audio_files.append(str(f_path))

                question = row['question'].strip()

                if 'listen' in instruction.lower():
                    block_type = 'listen'
                else:
                    block_type = 'ignore'

                blocks.append(BlockInfo(
                    instruction_text=instruction,
                    audio_files=audio_files,
                    question_text=question,
                    block_type=block_type,
                ))

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

    def _record_event(self, onset: float, duration: float, trial_type: str, **kwargs):
        """Record an event for the BIDS events TSV."""
        event = {'onset': round(onset, 3), 'duration': round(duration, 3), 'trial_type': trial_type}
        event.update(kwargs)
        self._events.append(event)

    def _save_events_tsv(self):
        """Write accumulated events to BIDS-compatible TSV."""
        if not self._events:
            return

        output_dir = getattr(self.coordinator, 'output_dir', None)
        if output_dir is None:
            logger.warning("No output_dir on coordinator, cannot save events TSV")
            return

        output_dir = Path(output_dir)
        subject = getattr(self.coordinator, 'subject', 'unknown')
        task = getattr(self.coordinator, 'task', 'thermometer')
        run_number = getattr(self.coordinator, 'run_number', 1)
        run_str = f"run-{run_number:02d}"

        path = output_dir / f"{subject}_task-{task}_{run_str}_events.tsv"

        # Collect all columns across events
        all_keys = []
        for event in self._events:
            for k in event:
                if k not in all_keys:
                    all_keys.append(k)

        # Ensure BIDS-required columns come first
        bids_cols = ['onset', 'duration', 'trial_type']
        extra_cols = [k for k in all_keys if k not in bids_cols]
        columns = bids_cols + extra_cols

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t',
                                    extrasaction='ignore')
            writer.writeheader()
            for event in self._events:
                writer.writerow(event)

        logger.info(f"Saved {len(self._events)} events to {path}")

    # =========================================================================
    # Display helpers
    # =========================================================================

    def _wait_with_escape(self, duration: float):
        """Wait for a duration while checking for escape/stop signals."""
        from psychopy import event, core

        clock = core.Clock()
        while clock.getTime() < duration and self._running:
            if hasattr(self, 'coordinator') and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")
            keys = event.getKeys(keyList=['escape'])
            if keys:
                raise KeyboardInterrupt("Escaped")
            core.wait(0.01)

    def _show_fixation(self, duration: float):
        """Display fixation cross for a given duration."""
        self._fixation.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()
        self._wait_with_escape(duration)

    def _show_instruction_prompt(self, text: str, duration: float):
        """Display block instruction for a given duration."""
        self._instruction_stim.text = text
        self._instruction_stim.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()
        self._wait_with_escape(duration)

    def _play_audio_block(self, audio_files: list[str], total_duration: float):
        """Play audio sentences during stimulus block with fixation cross."""
        from psychopy import core

        block_clock = core.Clock()

        for audio_path in audio_files:
            if not self._running or block_clock.getTime() >= total_duration:
                break

            # Use pre-loaded sound or load on demand
            snd = self._preloaded_sounds.get(audio_path)
            if snd is None:
                from psychopy import sound
                snd = sound.Sound(audio_path)

            # Show fixation during audio
            self._fixation.draw()
            self._capture_and_emit_screenshot()
            self.win.flip()

            snd.play()
            snd_duration = snd.getDuration() or 3.5
            core.wait(snd_duration)
            snd.stop()

            # Inter-sentence gap (0.5s)
            remaining_in_block = total_duration - block_clock.getTime()
            if remaining_in_block > 0.5:
                self._fixation.draw()
                self.win.flip()
                core.wait(0.5)

        # Fill any remaining time with fixation
        remaining = total_duration - block_clock.getTime()
        if remaining > 0:
            self._fixation.draw()
            self.win.flip()
            core.wait(remaining)

    def _run_block_rating(self, question_text: str, duration: float):
        """Display 1-6 rating scale with timeout.

        Returns
        -------
        tuple of (int or None, float or None)
            (response, response_time). None if timed out.
        """
        from psychopy import visual, event, core

        # Question text
        self._rating_question.text = question_text

        # Create scale display text
        scale_text = "1    2    3    4    5    6"
        scale_labels = visual.TextStim(
            self.win,
            text=scale_text,
            pos=(0, -0.1),
            height=0.06,
            color='white',
        )
        left_label = visual.TextStim(
            self.win,
            text="Completely",
            pos=(-0.4, -0.2),
            height=0.04,
            color='white',
        )
        right_label = visual.TextStim(
            self.win,
            text="Not at all",
            pos=(0.4, -0.2),
            height=0.04,
            color='white',
        )

        # Draw initial state
        self._rating_question.draw()
        scale_labels.draw()
        left_label.draw()
        right_label.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        clock = core.Clock()
        timer = core.CountdownTimer(duration)
        selected = None
        rt = None

        event.clearEvents('keyboard')

        while timer.getTime() > 0 and self._running:
            if hasattr(self, 'coordinator') and self.coordinator.was_stopped():
                raise KeyboardInterrupt("Stopped by user")

            keys = event.getKeys(keyList=self.response_keys + ['escape'])
            for key in keys:
                if key == 'escape':
                    raise KeyboardInterrupt("Escaped during rating")
                if selected is None:
                    selected = int(key)
                    rt = clock.getTime()
                    # Briefly highlight selection
                    self._rating_question.draw()
                    selected_text = visual.TextStim(
                        self.win,
                        text=f"Selected: {selected}",
                        pos=(0, -0.3),
                        height=0.05,
                        color='yellow',
                    )
                    scale_labels.draw()
                    left_label.draw()
                    right_label.draw()
                    selected_text.draw()
                    self._capture_and_emit_screenshot(force=True)
                    self.win.flip()

            core.wait(0.01)

        return selected, rt

    def _show_feedback_thermometer(self, block_median: float, block_type: str, duration: float):
        """Display thermometer feedback for a block.

        Parameters
        ----------
        block_median : float
            Median activation during the block.
        block_type : str
            "listen" or "ignore".
        duration : float
            How long to display the thermometer.
        """
        from psychopy import visual, core

        if not self.feedback_mode:
            self._show_fixation(duration)
            return

        # Determine success and color
        if block_type == "listen":
            is_success = block_median > 0
        else:  # "ignore"
            is_success = block_median < 0

        bar_color = 'green' if is_success else 'red'

        # Clamp and normalize bar height
        clamped = max(self.activation_min, min(self.activation_max, block_median))
        # Map to 0-0.5 range (half the thermometer height) based on magnitude
        max_range = max(abs(self.activation_max), abs(self.activation_min))
        if max_range > 0:
            bar_magnitude = abs(clamped) / max_range * 0.45
        else:
            bar_magnitude = 0.0

        bar_direction = 1.0 if clamped >= 0 else -1.0

        # Draw thermometer outline
        self._thermo_outline.draw()

        # Draw center zero line
        self._thermo_zero_line.draw()
        self._thermo_zero_label.draw()

        # Draw fill bar from center
        if bar_magnitude > 0.001:
            self._thermo_fill.height = bar_magnitude
            self._thermo_fill.pos = (0, bar_direction * bar_magnitude / 2)
            self._thermo_fill.fillColor = bar_color
            self._thermo_fill.lineColor = bar_color
            self._thermo_fill.draw()

        # Draw labels
        self._thermo_top_label.draw()
        self._thermo_bottom_label.draw()

        # Draw star next to target direction
        if block_type == "listen":
            self._thermo_top_star.draw()
        else:
            self._thermo_bottom_star.draw()

        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        self._wait_with_escape(duration)

    # =========================================================================
    # Block execution
    # =========================================================================

    def _run_block(self, block_idx: int, block_info: BlockInfo):
        """Execute a single block: instruction → stimulus → rating → feedback → fixation."""
        # 1. Instruction prompt
        onset = self._task_clock.getTime()
        self._show_instruction_prompt(block_info.instruction_text, self.instruction_duration)
        self._record_event(onset, self.instruction_duration, 'instruction',
                           block_count=block_idx + 1, block_type=block_info.block_type)

        # 2. Stimulus block with activation accumulation
        onset = self._task_clock.getTime()
        self._start_accumulating()
        self._play_audio_block(block_info.audio_files, self.stimulus_duration)
        block_activations = self._stop_accumulating()
        block_median = float(np.median(block_activations)) if block_activations else 0.0
        self._record_event(onset, self.stimulus_duration, 'stimulus',
                           block_count=block_idx + 1, block_type=block_info.block_type,
                           stg_activity=round(block_median, 4),
                           stim_file=';'.join(block_info.audio_files))

        # 3. Rating
        onset = self._task_clock.getTime()
        response, rt = self._run_block_rating(block_info.question_text, self.rating_duration)
        self._record_event(onset, self.rating_duration, 'rating',
                           block_count=block_idx + 1, block_type=block_info.block_type,
                           response=response, response_time=round(rt, 3) if rt else '')

        # 4. Feedback thermometer (or fixation for transfer)
        onset = self._task_clock.getTime()
        if self.feedback_mode:
            is_success = (block_info.block_type == "listen" and block_median > 0) or \
                         (block_info.block_type == "ignore" and block_median < 0)
            feedback_color = 'green' if is_success else 'red'
            self._show_feedback_thermometer(block_median, block_info.block_type, self.feedback_duration)
            self._record_event(onset, self.feedback_duration, 'feedback',
                               block_count=block_idx + 1, block_type=block_info.block_type,
                               stg_activity=round(block_median, 4), feedback=feedback_color)
            fixation_dur = self.fixation_nfb_duration
        else:
            self._show_fixation(self.feedback_duration)
            self._record_event(onset, self.feedback_duration, 'nofeedback',
                               block_count=block_idx + 1, block_type=block_info.block_type)
            fixation_dur = self.fixation_transfer_duration

        # 5. Inter-block fixation
        onset = self._task_clock.getTime()
        self._show_fixation(fixation_dur)
        self._record_event(onset, fixation_dur, 'fixation',
                           block_count=block_idx + 1)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def setup(self):
        """Initialize PsychoPy window, load conditions, pre-load audio, create stimuli."""
        from psychopy import visual, sound

        # Load conditions
        self._blocks = self._load_conditions()

        # Create window
        self.create_window(
            fullscreen=self.fullscreen,
            monitor_position=self.presenter_monitor,
            color='black',
            units='height',
        )

        # Pre-load all audio files
        for block in self._blocks:
            for audio_path in block.audio_files:
                if audio_path not in self._preloaded_sounds:
                    try:
                        self._preloaded_sounds[audio_path] = sound.Sound(audio_path)
                    except Exception as e:
                        logger.warning(f"Could not pre-load audio {audio_path}: {e}")

        # Create fixation cross
        self._fixation = visual.TextStim(
            self.win, text='+', pos=(0, 0), height=0.3, color='white',
        )

        # Create instruction text stimulus
        self._instruction_stim = visual.TextStim(
            self.win, text='', pos=(0, 0), height=0.1, color='white',
        )

        # Create rating question stimulus
        self._rating_question = visual.TextStim(
            self.win, text='', pos=(0, 0.15), height=0.06, wrapWidth=1.5, color='white',
        )

        # Create thermometer visual components
        self._thermo_outline = visual.ShapeStim(
            self.win, lineWidth=2.0, lineColor='white',
            vertices=((-0.125, -0.5), (0.125, -0.5), (0.125, 0.5), (-0.125, 0.5)),
            closeShape=True, fillColor=None,
        )
        self._thermo_zero_line = visual.ShapeStim(
            self.win, lineWidth=2.0, lineColor='white',
            vertices=((-0.125, 0), (0.125, 0)),
            closeShape=False,
        )
        self._thermo_zero_label = visual.TextStim(
            self.win, text='0', pos=(0.155, 0), height=0.04, color='white',
        )
        self._thermo_fill = visual.Rect(
            self.win, width=0.24, height=0.0, pos=(0, 0),
            fillColor='green', lineColor='green',
        )
        self._thermo_top_label = visual.TextStim(
            self.win, text='listen to self', pos=(0, 0.6), height=0.06, color='white',
        )
        self._thermo_bottom_label = visual.TextStim(
            self.win, text='ignore all sounds', pos=(0, -0.6), height=0.06, color='white',
        )
        self._thermo_top_star = visual.TextStim(
            self.win, text='*', pos=(0.5, 0.5), height=0.5, color='white',
        )
        self._thermo_bottom_star = visual.TextStim(
            self.win, text='*', pos=(0.5, -0.5), height=0.5, color='white',
        )

        # Show task instructions
        self.show_instructions([
            "In this task you will be asked to either\n"
            "attend to your own voice or ignore all sounds\n"
            "including the voices and any other noise\n"
            "in the environment.\n\n"
            "When ignoring sounds, you may use the\n"
            "focused attention method suggested\n"
            "by the experimenter.\n\n"
            "When you are ready please press a button.",
        ])

        # Wait for scanner trigger
        self.wait_for_trigger(trigger_keys=self.trigger_keys)

        logger.info("ThermometerTask setup complete")

    def update(self, data: dict):
        """Minimal implementation for BaseTask contract. Not used in main loop."""
        pass

    def run(self):
        """Main task loop — time-driven block structure."""
        from psychopy import core

        self._running = True

        try:
            self.setup()

            self._task_clock = core.Clock()

            # Initial baseline
            onset = self._task_clock.getTime()
            self._show_fixation(self.baseline_duration)
            self._record_event(onset, self.baseline_duration, 'baseline')

            # Block loop
            for block_idx, block_info in enumerate(self._blocks):
                if not self._running:
                    break
                self._run_block(block_idx, block_info)

        except KeyboardInterrupt:
            logger.info("ThermometerTask interrupted")
        finally:
            self._running = False
            self._save_events_tsv()
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("ThermometerTask cleanup")

        # Set snapshot for base class run recording
        self._task_state_snapshot = {
            "n_blocks_completed": len([e for e in self._events if e['trial_type'] == 'stimulus']),
        }

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
    design_type="nuisance_only",
    parameters=[
        TaskParameter(
            key="stim_dir",
            label="Stimulus Directory",
            type="str",
            default="",
            hint="Directory containing condition CSV files (auto-discovered by participant/run) and audio files referenced in the CSVs.",
        ),
        TaskParameter(
            key="baseline_duration",
            label="Baseline Duration (s)",
            type="float",
            default=30.0,
            min_value=0.0,
            max_value=120.0,
            hint="Initial baseline fixation period",
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
            key="stimulus_duration",
            label="Stimulus Duration (s)",
            type="float",
            default=16.0,
            min_value=5.0,
            max_value=60.0,
            hint="Duration of audio stimulus block",
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
            default=4.0,
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
            hint="Keys that trigger volume acquisition",
        ),
        TaskParameter(
            key="response_keys",
            label="Rating Response Keys",
            type="list[str]",
            default=["1", "2", "3", "4", "5", "6"],
            hint="Keys for 1-6 rating scale responses",
        ),
    ],
    session_state_fields=[],
    post_run_questions=[],
)

# Auto-register on import
register_task(THERMOMETER_METADATA)
register_task_class("thermometer", ThermometerTask)
