"""
Repetitive Negative Thinking (RNT) localizer task.

Summary
-------
Block design task where participants alternate between rumination and
distraction phases, each preceded by a personalized memory prompt and
followed by sadness/focus ratings. Used to define participant-specific
rumination network ROIs via the contrast "rumination > distraction" in
the GLM-based mask extraction pipeline.

Protocol (per Shreiner et al., 2025)
-------------------------------------
- 2 runs per session, 4 trials each
- Run 1 uses memories from pass 1, run 2 from pass 2
- Each trial (~145-165s):
    1. Memory prompt + keyword (5s)
    2. Reflection: "Think about this memory in detail" (25s)
    3. Rumination prompt (30s)
    4. Sadness rating 1-5 (up to 15s)
    5. Focus-on-feelings rating 1-5 (up to 15s)
    6. Rest/fixation (10-15s jittered)
    7. Distraction prompt (30s)
    8. Sadness rating 1-5 (up to 15s)
    9. Focus-on-feelings rating 1-5 (up to 15s)
- Total run duration: ~10-11 minutes

Keywords File
-------------
Pre-created file in keywords_dir, discovered by participant ID.
Supported formats: JSON or TSV.

JSON: {"memories": [{"keyword": "...", "pass": 1}, ...]}
TSV: keyword<tab>pass header, 8 rows (4 per pass)

Reference
---------
Shreiner et al. (2025). Rumination induction task in fMRI: Effects of
rumination focused cognitive behavioral therapy and stability in youth.
Journal of Affective Disorders, 372, 608-615.

Dependencies
------------
PsychoPy (visual, event, core), pandas, json
"""

import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Union

from .localizer_base import BaseLocalizerTask
from .task_registry import TaskMetadata, TaskParameter, pineuro_task

logger = logging.getLogger(__name__)

# ============================================================================
# Constants — prompts from Shreiner et al. (2025)
# ============================================================================

MEMORY_PROMPTS = [
    "Remember a time when your feelings were badly hurt.",
    "Remember a time when you felt rejected by someone.",
    "Remember a time when you felt ashamed.",
    "Remember a time when you felt very lonely.",
]

RUMINATION_PROMPTS = [
    "Think about why this memory still bothers you.",
    "Think about what this memory says about you as a person.",
    "Think about how this memory affects how you see yourself.",
    "Think about what caused you to feel this way.",
    "Think about the possible consequences of the way you feel.",
    "Think about what you could have done differently.",
    "Think about how this memory relates to other times you felt similar.",
    "Think about why you react the way you do.",
]

DISTRACTION_PROMPTS = [
    "Imagine a boat slowly crossing the Atlantic.",
    "Think of a row of shampoo bottles on display.",
    "Think of the expression of Mona Lisa.",
    "Think of the lay-out of the local post office.",
    "Imagine raindrops sliding down a windowpane.",
    "Think of two birds sitting on a tree branch.",
    "Picture the Golden Gate Bridge.",
    "Think about the shape of California.",
]

# ============================================================================
# Task metadata
# ============================================================================

RNT_METADATA = TaskMetadata(
    task_id="rnt",
    display_name="RNT Localizer (Rumination)",
    description="Repetitive Negative Thinking: rumination vs distraction blocks for network ROI definition",
    roles={"localizer"},
    n_networks=0,
    default_contrasts={"rumination_network": "rumination - distraction"},
    design_type="block",
    parameters=[
        TaskParameter(
            key="keywords_dir",
            label="Keywords Directory",
            type="dir",
            default="",
            hint="Directory containing per-participant keyword files (auto-matched by participant ID)",
        ),
        TaskParameter(
            key="memory_display_duration",
            label="Memory Display (s)",
            type="float",
            default=5.0,
            min_value=2.0,
            max_value=15.0,
            hint="Duration to display memory prompt + keyword",
        ),
        TaskParameter(
            key="reflection_duration",
            label="Reflection Duration (s)",
            type="float",
            default=25.0,
            min_value=5.0,
            max_value=60.0,
            hint="Duration for detailed memory reflection",
        ),
        TaskParameter(
            key="rumination_duration",
            label="Rumination/Distraction Duration (s)",
            type="float",
            default=30.0,
            min_value=5.0,
            max_value=60.0,
            hint="Duration for rumination and distraction prompts",
        ),
        TaskParameter(
            key="rating_timeout",
            label="Rating Timeout (s)",
            type="float",
            default=15.0,
            min_value=5.0,
            max_value=30.0,
            hint="Maximum time for each rating response",
        ),
        TaskParameter(
            key="rest_min",
            label="Rest Min (s)",
            type="float",
            default=10.0,
            min_value=3.0,
            max_value=30.0,
            hint="Minimum rest/fixation jitter duration",
        ),
        TaskParameter(
            key="rest_max",
            label="Rest Max (s)",
            type="float",
            default=15.0,
            min_value=3.0,
            max_value=30.0,
            hint="Maximum rest/fixation jitter duration",
        ),
        TaskParameter(
            key="left_key",
            label="Left Key",
            type="str",
            default="1",
            hint="Key for moving rating left",
        ),
        TaskParameter(
            key="right_key",
            label="Right Key",
            type="str",
            default="3",
            hint="Key for moving rating right",
        ),
        TaskParameter(
            key="confirm_key",
            label="Confirm Key",
            type="str",
            default="2",
            hint="Key for confirming rating selection",
        ),
        TaskParameter(
            key="n_trials_per_run",
            label="Trials Per Run",
            type="int",
            default=4,
            min_value=1,
            max_value=8,
            hint="Number of trials per run (default: 4)",
        ),
    ],
)


# ============================================================================
# Task class
# ============================================================================


@pineuro_task(RNT_METADATA)
class RNTTask(BaseLocalizerTask):
    """Repetitive Negative Thinking (RNT) localizer task.

    Alternates rumination and distraction blocks around personalized
    memory prompts, with sadness/focus ratings after each block. Outputs
    BIDS events TSV for GLM analysis (rumination - distraction contrast).

    Parameters
    ----------
    output_dir : str or Path
        Directory for events TSV output.
    participant_id : str
        Participant identifier.
    keywords_dir : str or Path
        Directory containing per-participant keyword files. Files are
        matched by participant ID. Alternatively, pass ``keywords_file``
        directly for standalone use.
    run : int
        Run number (1 or 2). Run 1 uses pass-1 keywords, run 2 uses pass-2.
    memory_display_duration : float
        Duration to show memory prompt + keyword (default: 5.0s).
    reflection_duration : float
        Duration for "think about this memory" instruction (default: 25.0s).
    rumination_duration : float
        Duration for rumination and distraction prompts (default: 30.0s).
    rating_timeout : float
        Maximum time for each rating (default: 15.0s).
    rest_min : float
        Minimum rest jitter duration (default: 10.0s).
    rest_max : float
        Maximum rest jitter duration (default: 15.0s).
    left_key, right_key, confirm_key : str
        Keys for rating scale navigation.
    n_trials_per_run : int
        Number of trials per run (default: 4).
    keywords_file : str or Path, optional
        Direct path to a specific keywords file (overrides keywords_dir lookup).
    simulator : VolumeSimulator, optional
        Volume simulator instance for offline testing.
    """

    task_id = "rnt"

    def __init__(
        self,
        output_dir: Union[str, Path],
        participant_id: str,
        keywords_dir: Union[str, Path] = "",
        run: int = 1,
        memory_display_duration: float = 5.0,
        reflection_duration: float = 25.0,
        rumination_duration: float = 30.0,
        rating_timeout: float = 15.0,
        rest_min: float = 10.0,
        rest_max: float = 15.0,
        left_key: str = "1",
        right_key: str = "3",
        confirm_key: str = "2",
        n_trials_per_run: int = 4,
        keywords_file: Union[str, Path] | None = None,
        simulator=None,
        **kwargs,
    ):
        super().__init__(
            output_dir=output_dir,
            participant_id=participant_id,
            run=run,
            simulator=simulator,
            **kwargs,
        )
        self.keywords_dir = Path(keywords_dir) if keywords_dir else None
        self._keywords_file_override = Path(keywords_file) if keywords_file else None
        self.memory_display_duration = memory_display_duration
        self.reflection_duration = reflection_duration
        self.rumination_duration = rumination_duration
        self.rating_timeout = rating_timeout
        self.rest_min = rest_min
        self.rest_max = rest_max
        self.left_key = left_key
        self.right_key = right_key
        self.confirm_key = confirm_key
        self.n_trials_per_run = n_trials_per_run

        # Populated in setup()
        self._memories: list[dict[str, Any]] = []
        self._rumination_prompts: list[str] = []
        self._distraction_prompts: list[str] = []

    # =========================================================================
    # Keywords loading
    # =========================================================================

    def _resolve_keywords_file(self) -> Path:
        """Find the keywords file for this participant.

        Checks (in order):
        1. Direct keywords_file override
        2. Glob in keywords_dir for matching file

        Returns
        -------
        Path
            Resolved keywords file path.

        Raises
        ------
        FileNotFoundError
            If no matching keywords file is found.
        """
        if self._keywords_file_override is not None:
            if not self._keywords_file_override.exists():
                raise FileNotFoundError(
                    f"Keywords file not found: {self._keywords_file_override}"
                )
            return self._keywords_file_override

        if self.keywords_dir is None or not self.keywords_dir.is_dir():
            raise FileNotFoundError(
                f"Keywords directory not found: {self.keywords_dir}. "
                f"Set keywords_dir to a directory containing per-participant "
                f"keyword files, or pass keywords_file directly."
            )

        patterns = [
            f"*{self.participant_id}*rnt*keywords*",
            f"*{self.participant_id}*rnt*",
            f"*{self.participant_id}*keywords*",
        ]

        for pattern in patterns:
            matches = [
                p
                for p in self.keywords_dir.glob(pattern)
                if p.suffix in (".json", ".tsv", ".csv")
            ]
            if matches:
                keywords_file = matches[0]
                logger.info(f"Found keywords file: {keywords_file}")
                return keywords_file

        raise FileNotFoundError(
            f"No keywords file found in {self.keywords_dir} matching "
            f"participant='{self.participant_id}'. "
            f"Tried patterns: {patterns}"
        )

    def _load_keywords(self) -> None:
        """Load keywords file and assign memories to this run.

        Run 1 uses pass-1 keywords, run 2 uses pass-2 keywords. Each
        keyword is paired with the corresponding memory prompt from
        MEMORY_PROMPTS by position within the pass.

        Also shuffles rumination and distraction prompts for this run
        using a deterministic seed based on participant_id.
        """
        keywords_path = self._resolve_keywords_file()
        logger.info(f"Loading keywords file: {keywords_path}")

        all_memories = self._parse_keywords_file(keywords_path)

        # Assign memories to this run based on pass number
        target_pass = self.run_number  # run 1 → pass 1, run 2 → pass 2
        run_memories = [m for m in all_memories if m["pass"] == target_pass]

        if len(run_memories) < self.n_trials_per_run:
            raise ValueError(
                f"Keywords file has {len(run_memories)} memories for pass "
                f"{target_pass}, but n_trials_per_run={self.n_trials_per_run}. "
                f"Need at least {self.n_trials_per_run} memories per pass."
            )

        # Pair each keyword with corresponding memory prompt by position
        for i, memory in enumerate(run_memories[: self.n_trials_per_run]):
            prompt_idx = i % len(MEMORY_PROMPTS)
            memory["prompt"] = MEMORY_PROMPTS[prompt_idx]
            memory["prompt_index"] = prompt_idx

        self._memories = run_memories[: self.n_trials_per_run]

        # Shuffle rumination and distraction prompts deterministically
        seed = hash(self.participant_id) & 0xFFFFFFFF
        rng = random.Random(seed)

        rumination_order = list(range(len(RUMINATION_PROMPTS)))
        distraction_order = list(range(len(DISTRACTION_PROMPTS)))
        rng.shuffle(rumination_order)
        rng.shuffle(distraction_order)

        # Run 1 gets first half, run 2 gets second half
        start = (self.run_number - 1) * self.n_trials_per_run
        end = start + self.n_trials_per_run
        self._rumination_prompts = [
            RUMINATION_PROMPTS[rumination_order[i % len(RUMINATION_PROMPTS)]]
            for i in range(start, end)
        ]
        self._distraction_prompts = [
            DISTRACTION_PROMPTS[distraction_order[i % len(DISTRACTION_PROMPTS)]]
            for i in range(start, end)
        ]

        logger.info(
            f"Loaded {len(self._memories)} memories for run {self.run_number} "
            f"(pass {target_pass})"
        )

    def _parse_keywords_file(self, path: Path) -> list[dict[str, Any]]:
        """Parse a keywords file (JSON or TSV/CSV).

        Parameters
        ----------
        path : Path
            Path to the keywords file.

        Returns
        -------
        list of dict
            Each dict has at least "keyword" (str) and "pass" (int).
        """
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            memories = data.get("memories", data) if isinstance(data, dict) else data
            for m in memories:
                m["pass"] = int(m["pass"])
            return memories

        # TSV or CSV
        delimiter = "\t" if path.suffix == ".tsv" else ","
        memories = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                memories.append(
                    {
                        "keyword": row["keyword"].strip(),
                        "pass": int(row["pass"]),
                    }
                )
        return memories

    # =========================================================================
    # Stimulus presentation
    # =========================================================================

    def _show_text(self, text: str, duration: float) -> None:
        """Display text centered on screen for a given duration.

        Parameters
        ----------
        text : str
            Text to display.
        duration : float
            Duration in seconds.
        """
        from psychopy import core, event, visual

        text_stim = visual.TextStim(
            self.win,
            text=text,
            height=0.06,
            color="white",
            wrapWidth=1.2,
        )
        text_stim.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        timer = core.CountdownTimer(duration)
        while timer.getTime() > 0 and self._running:
            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escaped during text display")
            core.wait(0.01)

    def _run_rating(
        self,
        question_text: str,
        left_label_text: str,
        right_label_text: str,
    ) -> tuple[int | None, float | None]:
        """Display a 1-5 rating scale with sliding marker and timeout.

        Adapted from ThermometerTask._run_block_rating(). Horizontal line
        with tick marks, number labels, endpoint labels, and a triangle
        marker that starts at the midpoint.

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
        from psychopy import core, event, visual

        scale_low = 1
        scale_high = 5
        n_ticks = scale_high - scale_low + 1
        scale_y = -0.15
        scale_width = 0.5
        tick_spacing = scale_width / (n_ticks - 1)
        scale_left_x = -scale_width / 2

        tick_x = [scale_left_x + i * tick_spacing for i in range(n_ticks)]

        # Start at midpoint (3.0 for 1-5 scale)
        marker_pos = (scale_low + scale_high) / 2.0

        def pos_to_x(pos):
            return scale_left_x + (pos - scale_low) * tick_spacing

        # Scale line
        scale_line = visual.Line(
            self.win,
            start=(-scale_width / 2, scale_y),
            end=(scale_width / 2, scale_y),
            lineColor="white",
            lineWidth=2,
        )

        # Tick marks and number labels
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

        # Endpoint labels
        left_label = visual.TextStim(
            self.win,
            text=left_label_text,
            pos=(tick_x[0], scale_y - 0.1),
            height=0.035,
            color="white",
        )
        right_label = visual.TextStim(
            self.win,
            text=right_label_text,
            pos=(tick_x[-1], scale_y - 0.1),
            height=0.035,
            color="white",
        )

        # Triangle marker (pointing down toward scale line)
        marker_size = 0.02
        marker_gap = 0.02

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

        # Question text
        question_stim = visual.TextStim(
            self.win,
            text=question_text,
            pos=(0, 0.15),
            height=0.06,
            color="white",
            wrapWidth=1.2,
        )

        clock = core.Clock()
        timer = core.CountdownTimer(self.rating_timeout)
        confirmed = False
        rt = None

        event.clearEvents("keyboard")

        while timer.getTime() > 0 and self._running:
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

            # Redraw each frame
            question_stim.draw()
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
        rt_value = round(rt, 3) if rt is not None else None

        # Wait out remaining time with fixation
        remaining = timer.getTime()
        if remaining > 0:
            self.show_fixation(remaining)

        return response, rt_value

    # =========================================================================
    # Trial execution
    # =========================================================================

    def _run_trial(self, trial_idx: int) -> None:
        """Execute a single trial of the RNT paradigm.

        Parameters
        ----------
        trial_idx : int
            Trial index (0-based) within this run.
        """
        memory = self._memories[trial_idx]
        rumination_prompt = self._rumination_prompts[trial_idx]
        distraction_prompt = self._distraction_prompts[trial_idx]
        trial_num = trial_idx + 1

        logger.info(
            f"[TIMING] Trial {trial_num}/{self.n_trials_per_run} START: "
            f"keyword='{memory['keyword']}', t={self._experiment_clock.getTime():.2f}s"
        )

        # 1. Memory prompt + keyword
        onset = self._experiment_clock.getTime()
        self._show_text(
            f"{memory['prompt']}\n\nKeyword: {memory['keyword']}",
            self.memory_display_duration,
        )
        self.record_event(
            onset,
            self.memory_display_duration,
            "memory_prompt",
            trial=trial_num,
            keyword=memory["keyword"],
            prompt_index=memory["prompt_index"],
        )

        # 2. Reflection
        onset = self._experiment_clock.getTime()
        self._show_text(
            "Think about this memory in detail.",
            self.reflection_duration,
        )
        self.record_event(onset, self.reflection_duration, "reflection", trial=trial_num)

        # 3. Rumination prompt
        onset = self._experiment_clock.getTime()
        logger.info(
            f"[TIMING] Rumination START: onset={onset:.2f}s, "
            f"duration={self.rumination_duration}s"
        )
        self._show_text(rumination_prompt, self.rumination_duration)
        self.record_event(
            onset,
            self.rumination_duration,
            "rumination",
            trial=trial_num,
            prompt=rumination_prompt,
        )

        # 4. Sadness rating (post-rumination)
        onset = self._experiment_clock.getTime()
        rating, rt = self._run_rating(
            "How sad do you feel right now?",
            "Not sad",
            "Very sad",
        )
        actual_dur = self._experiment_clock.getTime() - onset
        self.record_event(
            onset,
            actual_dur,
            "rating_sadness_post_rumination",
            trial=trial_num,
            response=rating,
            response_time=rt,
        )

        # 5. Focus-on-feelings rating (post-rumination)
        onset = self._experiment_clock.getTime()
        rating, rt = self._run_rating(
            "How focused are you on your feelings?",
            "Not focused",
            "Very focused",
        )
        actual_dur = self._experiment_clock.getTime() - onset
        self.record_event(
            onset,
            actual_dur,
            "rating_focus_post_rumination",
            trial=trial_num,
            response=rating,
            response_time=rt,
        )

        # 6. Rest/fixation with jitter
        rest_duration = random.uniform(self.rest_min, self.rest_max)
        onset = self._experiment_clock.getTime()
        logger.info(
            f"[TIMING] Rest: onset={onset:.2f}s, duration={rest_duration:.2f}s"
        )
        self.show_fixation(rest_duration)
        self.record_event(onset, rest_duration, "rest", trial=trial_num)

        # 7. Distraction prompt
        onset = self._experiment_clock.getTime()
        logger.info(
            f"[TIMING] Distraction START: onset={onset:.2f}s, "
            f"duration={self.rumination_duration}s"
        )
        self._show_text(distraction_prompt, self.rumination_duration)
        self.record_event(
            onset,
            self.rumination_duration,
            "distraction",
            trial=trial_num,
            prompt=distraction_prompt,
        )

        # 8. Sadness rating (post-distraction)
        onset = self._experiment_clock.getTime()
        rating, rt = self._run_rating(
            "How sad do you feel right now?",
            "Not sad",
            "Very sad",
        )
        actual_dur = self._experiment_clock.getTime() - onset
        self.record_event(
            onset,
            actual_dur,
            "rating_sadness_post_distraction",
            trial=trial_num,
            response=rating,
            response_time=rt,
        )

        # 9. Focus-on-feelings rating (post-distraction)
        onset = self._experiment_clock.getTime()
        rating, rt = self._run_rating(
            "How focused are you on your feelings?",
            "Not focused",
            "Very focused",
        )
        actual_dur = self._experiment_clock.getTime() - onset
        self.record_event(
            onset,
            actual_dur,
            "rating_focus_post_distraction",
            trial=trial_num,
            response=rating,
            response_time=rt,
        )

        logger.info(
            f"[TIMING] Trial {trial_num} END: t={self._experiment_clock.getTime():.2f}s"
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def setup(self) -> None:
        """Load keywords and create PsychoPy window."""
        self._load_keywords()
        self.create_window(monitor_position=self.monitor_position)

        logger.info(
            f"RNTTask setup: run={self.run_number}, "
            f"{len(self._memories)} memories loaded"
        )

    def run_paradigm(self) -> None:
        """Execute the RNT localizer paradigm."""
        from psychopy import core

        self.show_instructions(
            [
                "In this task, you will be asked to think about\n"
                "memories from your past.\n\n"
                "You will also be asked to think about\n"
                "unrelated topics as a distraction.\n\n"
                "After each period, you will rate how sad you feel\n"
                "and how focused you are on your feelings.\n\n"
                "Use button 1 to move left, button 3 to move right,\n"
                "and button 2 to confirm your rating.\n\n"
                "Press any button to begin."
            ]
        )

        self.wait_for_trigger()
        self._experiment_clock = core.Clock()
        logger.info("[TIMING] Experiment clock started at trigger (t=0.00s)")

        for trial_idx in range(self.n_trials_per_run):
            if not self._running:
                break
            self._run_trial(trial_idx)

        total_time = self._experiment_clock.getTime()
        logger.info(
            f"[TIMING] Paradigm complete: {self.n_trials_per_run} trials, "
            f"total={total_time:.1f}s"
        )

    def cleanup(self) -> None:
        """Release resources after task completes."""
        super().cleanup()
