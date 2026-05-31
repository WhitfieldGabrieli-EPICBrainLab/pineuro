"""
Self/Other voice localizer task for STG ROI definition.

Summary
-------
Block design task where participants listen to sentences in their own voice
(self) and a stranger's voice (other), with silent rest blocks and
comprehension questions. Used to define participant-specific STG ROIs via
the contrast "self > other" in the GLM-based mask extraction pipeline.

Protocol (per paper)
--------------------
- 30s baseline (fixation cross)
- 12 blocks per run (4 self, 4 other, 4 rest), each 16s
- 4s comprehension question after each block
- Default block order: voice, voice, rest, voice, voice, rest, ...
- Self/other is inferred from the audio filename (contains "other" → other)
- Total run duration: ~4 min 26s (270s)
- Two runs per session (counterbalanced stimulus assignment)

Stimulus File Format
--------------------
Tab-delimited text file with 40 rows (8 voice blocks × 5 sentences).
Each participant has per-run files:
    {subject}_{session}_task-selfother_{run}.txt

Columns:
- 0: audio file path (paths containing "other" → other-voice block)
- 1: stimulus duration (seconds)
- 2: inter-stimulus interval (seconds)
- 3: question image path (first 12 rows only; rows 13+ may be empty)

Rows are grouped by 5 (each group = one voice block). Voice blocks are
played in stim file order. Block type (self/other) is inferred from the
first audio filename of each block.

Reference
---------
Raij et al. (2018). Prefrontal Cortex Stimulation Enhances Self-regulation
of Voice Perception. Psychiatry Research.

Dependencies
------------
PsychoPy (sound, visual, event, core), pandas
"""

import csv
import logging
from pathlib import Path
from typing import Union

from .localizer_base import BaseLocalizerTask
from .task_registry import TaskMetadata, TaskParameter, pineuro_task

logger = logging.getLogger(__name__)

# Default block order: voice/rest pattern (self/other inferred from stim file)
DEFAULT_BLOCK_ORDER = "voice,voice,rest,voice,voice,rest,voice,rest,voice,voice,voice,rest"


SELF_OTHER_METADATA = TaskMetadata(
    task_id="selfother",
    display_name="Self/Other Voice Localizer",
    description="Block design: self-voice vs other-voice sentences for STG ROI definition",
    roles={"localizer"},
    n_networks=0,
    default_contrasts={"STG_self": "self - other"},
    parameters=[
        TaskParameter(
            key="stim_dir",
            label="Stimulus Directory",
            type="dir",
            default="",
            hint="Directory containing per-participant stim files (auto-matched by participant ID and run)",
        ),
        TaskParameter(
            key="baseline_duration",
            label="Baseline Duration (s)",
            type="float",
            default=30.0,
            min_value=0.0,
            max_value=120.0,
            hint="Duration of initial fixation baseline",
        ),
        TaskParameter(
            key="block_duration",
            label="Block Duration (s)",
            type="float",
            default=16.0,
            min_value=5.0,
            max_value=60.0,
            hint="Duration of each voice/rest block",
        ),
        TaskParameter(
            key="question_duration",
            label="Question Duration (s)",
            type="float",
            default=4.0,
            min_value=1.0,
            max_value=30.0,
            hint="Duration to display comprehension question",
        ),
        TaskParameter(
            key="block_order",
            label="Block Order",
            type="str",
            default=DEFAULT_BLOCK_ORDER,
            hint="Comma-separated: voice, rest (self/other inferred from stim file)",
        ),
        TaskParameter(
            key="response_keys",
            label="Response Keys",
            type="list[str]",
            default=["1", "2"],
            hint="Keys for Yes/No responses to comprehension questions",
        ),
    ],
)


@pineuro_task(SELF_OTHER_METADATA)
class SelfOtherTask(BaseLocalizerTask):
    """Self/Other voice localizer task.

    Presents sentences in self-voice and other-voice blocks with rest blocks
    and comprehension questions. Outputs BIDS events TSV for GLM analysis.

    Block type (self/other) is inferred from the audio filename - if the path
    contains "other" (case-insensitive), it's an other-voice block.

    Parameters
    ----------
    output_dir : str or Path
        Directory for events TSV output.
    participant_id : str
        Participant identifier.
    stim_dir : str or Path
        Directory containing per-participant stimulus files. Files are
        matched by participant ID and run number. Alternatively, pass
        ``stim_file`` directly for standalone use.
    run : int
        Run number (1-based).
    baseline_duration : float
        Duration of initial fixation baseline in seconds.
    block_duration : float
        Duration of each voice/rest block in seconds.
    question_duration : float
        Duration of comprehension question display in seconds.
    block_order : str
        Comma-separated: "voice" or "rest" for each block position.
    response_keys : list of str
        Keys for yes/no responses during questions.
    stim_file : str or Path, optional
        Direct path to a specific stim file (overrides stim_dir lookup).
    simulator : VolumeSimulator, optional
        Volume simulator instance for offline testing. If provided, the task
        will automatically start the simulator thread before waiting for the
        first volume to arrive.
    """

    task_id = "selfother"

    def __init__(
        self,
        output_dir: Union[str, Path],
        participant_id: str,
        session_id: str,
        stim_dir: Union[str, Path] = "",
        run: int = 1,
        baseline_duration: float = 30.0,
        block_duration: float = 16.0,
        question_duration: float = 4.0,
        block_order: str = DEFAULT_BLOCK_ORDER,
        response_keys: list[str] | None = None,
        stim_file: Union[str, Path] | None = None,
        simulator=None,  # VolumeSimulator instance for simulation mode
        **kwargs,
    ):
        super().__init__(
            output_dir=output_dir,
            participant_id=participant_id,
            session_id=session_id,
            run=run,
            simulator=simulator,
            **kwargs,
        )
        self.stim_dir = Path(stim_dir) if stim_dir else None
        self._stim_file_override = Path(stim_file) if stim_file else None
        self.baseline_duration = baseline_duration
        self.block_duration = block_duration
        self.question_duration = question_duration
        self.block_order = [b.strip() for b in block_order.split(",")]
        self.response_keys = response_keys or ["1", "2"]

        # Populated in _load_stim_file()
        self._voice_blocks: list[list[dict]] = []
        self._voice_block_types: list[str] = []  # "self" or "other" per voice block
        self._questions: list[str] = []
        self._preloaded_sounds: dict = {}  # Pre-loaded PsychoPy Sound objects

    def _resolve_stim_file(self) -> Path:
        """Find the stimulus file for this participant and run.

        Checks (in order):
        1. Direct stim_file override
        2. Glob in stim_dir for matching file

        Returns
        -------
        Path
            Resolved stimulus file path.

        Raises
        ------
        FileNotFoundError
            If no matching stim file is found.
        """
        if self._stim_file_override is not None:
            if not self._stim_file_override.exists():
                raise FileNotFoundError(f"Stimulus file not found: {self._stim_file_override}")
            return self._stim_file_override

        if self.stim_dir is None or not self.stim_dir.is_dir():
            raise FileNotFoundError(
                f"Stimulus directory not found: {self.stim_dir}. "
                f"Set stim_dir to a directory containing per-participant stim files, "
                f"or pass stim_file directly."
            )

        # Extract participant number from ID (e.g., "sub-R33rtsz037" → "037")
        # Also try the full participant_id
        run_str = f"run-{self.run_number:02d}"
        patterns = [
            f"*{self.participant_id}*selfother*{run_str}*",
            f"*{self.participant_id}*self*other*{run_str}*",
        ]

        for pattern in patterns:
            matches = list(self.stim_dir.glob(pattern))
            if matches:
                stim_file = matches[0]
                logger.info(f"Found stim file: {stim_file}")
                return stim_file

        raise FileNotFoundError(
            f"No stim file found in {self.stim_dir} matching "
            f"participant='{self.participant_id}', run={self.run_number}. "
            f"Tried patterns: {patterns}"
        )

    def _load_stim_file(self) -> None:
        """Load and parse the stimulus file.

        Uses csv.reader for robust parsing (handles rows with missing trailing
        columns). Groups audio rows into blocks of 5. Infers self/other from
        the first audio filename of each voice block.
        """
        stim_path = self._resolve_stim_file()
        logger.info(f"Loading stimulus file: {stim_path}")

        rows = []
        with open(stim_path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Pad to 4 columns if short (handles missing question images)
                while len(row) < 4:
                    row.append("")
                rows.append(row)

        # Count voice blocks needed
        n_voice_blocks = sum(1 for b in self.block_order if b != "rest")
        n_audio_rows_needed = n_voice_blocks * 5

        if len(rows) < n_audio_rows_needed:
            raise ValueError(
                f"Stimulus file has {len(rows)} rows but {n_voice_blocks} voice blocks "
                f"require {n_audio_rows_needed} rows (5 sentences per block)"
            )

        # Group audio rows into blocks of 5, infer type from first audio filename
        for i in range(n_voice_blocks):
            start = i * 5
            block = []
            for j in range(5):
                r = rows[start + j]
                block.append(
                    {
                        "path": r[0],
                        "duration": float(r[1]),
                        "isi": float(r[2]),
                    }
                )
            self._voice_blocks.append(block)

            # Infer block type from first audio path
            first_path = rows[start][0]
            block_type = "other" if "other" in first_path.lower() else "self"
            self._voice_block_types.append(block_type)

        # Question images: one per voice block, taken from column 3 of that
        # block's first stimulus row. rows[] holds only voice-block rows
        # (5 per voice block), so index by voice-block position; rest blocks
        # have no stimulus rows and get an empty (fixation) question.
        n_blocks = len(self.block_order)
        voice_idx = 0
        for block_entry in self.block_order:
            if block_entry != "rest":
                first_row = voice_idx * 5
                q = rows[first_row][3].strip() if first_row < len(rows) else ""
                voice_idx += 1
            else:
                q = ""
            self._questions.append(q)

        # Log what we found
        type_summary = ", ".join(self._voice_block_types)
        logger.info(f"Loaded stimulus file: {n_voice_blocks} voice blocks [{type_summary}], {n_blocks} question images")

    def setup(self) -> None:
        """Load stimuli and create PsychoPy window."""
        from psychopy.sound.backend_pygame import SoundPygame

        self._load_stim_file()
        self.create_window(
            monitor_position=self.monitor_position,
        )

        # Pre-load all audio as pygame-backed Sound objects.
        # We use SoundPygame directly because PsychoPy's default PTB backend
        # does not respect the OS default audio device on Windows, routing
        # audio to the wrong output (e.g. a monitor's HDMI audio).
        logger.info("Pre-loading all audio as SoundPygame objects...")
        for block in self._voice_blocks:
            for item in block:
                audio_path = item["path"]
                duration = item["duration"]
                if audio_path not in self._preloaded_sounds:
                    snd = SoundPygame(audio_path, secs=duration)
                    snd.setVolume(1.0)
                    self._preloaded_sounds[audio_path] = snd

        logger.info(f"All {len(self._preloaded_sounds)} Sound objects pre-loaded!")

        logger.info(
            f"SelfOtherTask setup: {len(self.block_order)} blocks, "
            f"baseline={self.baseline_duration}s, block={self.block_duration}s, "
            f"question={self.question_duration}s"
        )

    def run_paradigm(self) -> None:
        """Execute the full Self/Other localizer paradigm."""
        from psychopy import core

        # Instructions
        self.show_instructions(
            [
                "Listen to the sentences and answer the question\n"
                "regarding the last sentence you hear in each block.\n\n"
                "In the silent blocks, you will be asked a\n"
                "real-world question.\n\n"
                "To answer, respond YES with your index finger\n"
                "or NO with your middle finger.\n\n"
                "Press any button to begin."
            ]
        )

        # Wait for scanner trigger
        self.wait_for_trigger()
        self._experiment_clock = core.Clock()
        logger.info("[TIMING] Experiment clock started at trigger (t=0.00s)")

        # Baseline
        onset = self._experiment_clock.getTime()
        logger.info(f"[TIMING] Baseline START: onset={onset:.2f}s, duration={self.baseline_duration}s")
        self.show_fixation(self.baseline_duration)
        self.record_event(onset, self.baseline_duration, "baseline")

        # Main block sequence
        voice_block_idx = 0
        for block_num, block_entry in enumerate(self.block_order):
            if not self._running:
                break

            block_onset = self._experiment_clock.getTime()

            if block_entry == "rest":
                self._run_rest_block()
                trial_type = "rest"
            else:
                # Get inferred type (self/other) from stim file
                trial_type = self._voice_block_types[voice_block_idx]
                self._run_voice_block(voice_block_idx, trial_type)
                voice_block_idx += 1

            self.record_event(block_onset, self.block_duration, trial_type)

            # Comprehension question after each block
            q_onset = self._experiment_clock.getTime()
            response, rt = self._run_question(block_num)
            self.record_event(
                q_onset,
                self.question_duration,
                "question",
                response=response,
                response_time=rt,
                question_type=trial_type,
            )

        total_time = self._experiment_clock.getTime()
        expected_time = self.baseline_duration + (self.block_duration + self.question_duration) * len(self.block_order)
        logger.info(
            f"[TIMING] Paradigm complete: {block_num + 1} blocks, "
            f"total time {total_time:.1f}s (expected ~{expected_time:.1f}s)"
        )

    def _run_voice_block(self, block_idx: int, block_type: str) -> None:
        """Play 5 sentences with ISIs, fixation cross displayed throughout.

        Parameters
        ----------
        block_idx : int
            Index into self._voice_blocks.
        block_type : str
            "self" or "other" (for logging).
        """
        from psychopy import core, visual

        block = self._voice_blocks[block_idx]

        # Show fixation cross throughout the block
        fixation = visual.TextStim(self.win, text="+", height=0.3, color="white")
        fixation.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        for item in block:
            if not self._running:
                break
            try:
                # Get pre-loaded PsychoPy Sound object
                snd = self._preloaded_sounds.get(item["path"])
                if snd is None:
                    logger.error(f"Sound not pre-loaded: {item['path']}")
                    continue

                # Simple play/wait/stop pattern (matches original code)
                snd.play()  # Start the sound (non-blocking)
                core.wait(item["duration"])  # Wait for duration
                snd.stop()  # Stop the sound (in case it's longer than duration)
            except Exception as e:
                logger.error(f"Audio playback error for {item['path']}: {e}", exc_info=True)

            # Inter-stimulus interval
            if item["isi"] > 0:
                core.wait(item["isi"])

    def _run_rest_block(self) -> None:
        """Display fixation cross for block_duration."""
        self.show_fixation(self.block_duration)

    def _run_question(self, block_num: int) -> tuple[str, str]:
        """Show comprehension question image and collect response.

        If the question image path is empty, shows a fixation cross instead.

        Parameters
        ----------
        block_num : int
            Block index to select the question image.

        Returns
        -------
        tuple of (str, str)
            (response_key, response_time), or ("n/a", "n/a") if no response.
        """
        from psychopy import core, event, visual

        question_path = self._questions[block_num]

        if question_path:
            # Show question image
            img = visual.ImageStim(
                self.win,
                image=question_path,
                size=(1, 1),
            )
            img.draw()
        else:
            # No question image - show fixation
            fixation = visual.TextStim(
                self.win,
                text="+",
                height=0.3,
                color="white",
            )
            fixation.draw()

        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        event.clearEvents("keyboard")
        response_clock = core.Clock()
        response = ""
        rt = ""

        timer = core.CountdownTimer(self.question_duration)
        while timer.getTime() > 0 and self._running:
            keys = event.getKeys(keyList=self.response_keys + ["escape"])
            if "escape" in keys:
                raise KeyboardInterrupt("Escaped during question")
            if keys and not response:
                response = keys[0]
                rt = f"{response_clock.getTime():.3f}"
            core.wait(0.01)

        # Record BIDS "n/a" when no key was pressed, so a timeout is
        # distinguishable from an empty value downstream.
        return (response or "n/a", rt or "n/a")

    def cleanup(self) -> None:
        """Release resources after task completes."""
        # PsychoPy Sound objects clean up automatically
        super().cleanup()
