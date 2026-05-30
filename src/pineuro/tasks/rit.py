"""
Rumination Induction Task (RIT) localizer task.

Summary
-------
Block design task where participants alternate between rumination and
distraction phases, each preceded by a personalized memory prompt and
followed by sadness/focus ratings. Used to define participant-specific
rumination network ROIs via the contrast "rumination > distraction" in
the GLM-based mask extraction pipeline.

Protocol
--------
Each trial's structure (below) follows Shreiner et al. (2025). The paper
introduced the task but did not prescribe how it is packaged into a session,
so the number of runs and the number of trials per run are
study-configuration choices.

- Each run draws its trials from one of two memory passes; the pass → run
  assignment is shuffled deterministically per participant, and memories
  within the chosen pass are shuffled before presentation.
- Each trial (~120-125s):
    1. Memory prompt + keyword (5s)
    2. Reflection: "Think about this memory in detail" (25s)
    3. Rumination prompt (30s)
    4. Sadness rating 1-5 (up to 5s)
    5. Focus-on-feelings rating 1-5 (up to 5s)
    6. Rest/fixation (10-15s jittered)
    7. Distraction prompt (30s)
    8. Sadness rating 1-5 (up to 5s)
    9. Focus-on-feelings rating 1-5 (up to 5s)
- Total run duration: ~8 minutes

Keywords File
-------------
Pre-created JSON file per participant in ``keywords_dir``, discovered by
participant ID via glob matching. Each memory entry must include:

- ``keyword`` (str): shown on screen with the memory prompt as a recall anchor.
- ``pass`` (int, 1 or 2): which of the two passes this memory belongs to.
- ``memory_prompt`` (str): the prompt text shown to the participant.

Optional fields (recorded in the events TSV but not used to drive task logic):

- ``unpleasantness``: participant's pre-rated unpleasantness for this memory.
- ``trial``: the trial index from the source file.

JSON: ``{"memories": [{"keyword": "...", "pass": 1, "memory_prompt": "..."}, ...]}``

Reference
---------
Shreiner et al. (2025). Rumination induction task in fMRI: Effects of
rumination focused cognitive behavioral therapy and stability in youth.
Journal of Affective Disorders, 372, 608-615.

Dependencies
------------
PsychoPy (visual, event, core), pandas, json
"""

import hashlib
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

# Head/tail buffer for trigger wait + small end-of-run grace before the
# scanner cuts off. Added to the per-trial sum to compute n_volumes.
HEAD_TAIL_BUFFER_S = 10.0


def _rit_max_run_seconds(params: dict[str, Any]) -> float:
    """Worst-case duration of a single RIT run, in seconds.

    Assumes every rating phase runs to its full ``rating_timeout`` and the
    rest jitter takes its ``rest_max``. Defaults match ``RIT_METADATA``.
    """
    n_trials = int(params.get("n_trials_per_run", 4))
    memory = float(params.get("memory_display_duration", 5.0))
    reflection = float(params.get("reflection_duration", 25.0))
    block = float(params.get("rumination_duration", 30.0))  # rumination + distraction
    rating_to = float(params.get("rating_timeout", 5.0))
    rest_max = float(params.get("rest_max", 15.0))

    per_trial = memory + reflection + 2 * block + 4 * rating_to + rest_max
    return n_trials * per_trial + HEAD_TAIL_BUFFER_S


# ============================================================================
# Task metadata
# ============================================================================

RIT_METADATA = TaskMetadata(
    task_id="rit",
    display_name="RIT Localizer",
    description="Rumination Induction Task: rumination vs distraction blocks for network ROI definition",
    roles={"localizer"},
    n_networks=0,
    default_contrasts={"rumination_network": "rumination - distraction"},
    default_n_runs=1,
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
            default=5.0,
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
            default="2",
            hint="Key for moving rating right",
        ),
        TaskParameter(
            key="confirm_key",
            label="Confirm Key",
            type="str",
            default="0",
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
    compute_max_run_seconds=_rit_max_run_seconds,
)


# ============================================================================
# Task class
# ============================================================================


@pineuro_task(RIT_METADATA)
class RITTask(BaseLocalizerTask):
    """Rumination Induction Task (RIT) localizer task.

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
        Directory containing per-participant keyword JSON files. The file
        for this participant is discovered by glob match on ``participant_id``.
        See the module docstring for the required JSON schema. Alternatively,
        pass ``keywords_file`` directly for standalone use.
    run : int
        Run number (1 or 2). The pass each run draws from is randomized
        deterministically per participant.
    memory_display_duration : float
        Duration to show memory prompt + keyword (default: 5.0s).
    reflection_duration : float
        Duration for "think about this memory" instruction (default: 25.0s).
    rumination_duration : float
        Duration for rumination and distraction prompts (default: 30.0s).
    rating_timeout : float
        Maximum time for each rating (default: 5.0s).
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

    task_id = "rit"

    def __init__(
        self,
        output_dir: Union[str, Path],
        participant_id: str,
        session_id: str,
        keywords_dir: Union[str, Path] = "",
        run: int = 1,
        memory_display_duration: float = 5.0,
        reflection_duration: float = 25.0,
        rumination_duration: float = 30.0,
        rating_timeout: float = 5.0,
        rest_min: float = 10.0,
        rest_max: float = 15.0,
        left_key: str = "1",
        right_key: str = "2",
        confirm_key: str = "0",
        n_trials_per_run: int = 4,
        keywords_file: Union[str, Path] | None = None,
        simulator=None,
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
        self._target_pass: int = 0

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
                raise FileNotFoundError(f"Keywords file not found: {self._keywords_file_override}")
            return self._keywords_file_override

        if self.keywords_dir is None or not self.keywords_dir.is_dir():
            raise FileNotFoundError(
                f"Keywords directory not found: {self.keywords_dir}. "
                f"Set keywords_dir to a directory containing per-participant "
                f"keyword files, or pass keywords_file directly."
            )

        patterns = [
            f"*{self.participant_id}*rit*keywords*",
            f"*rit*keywords*{self.participant_id}*",
            f"*{self.participant_id}*keywords*",
            f"*keywords*{self.participant_id}*",
            f"*{self.participant_id}*rit*",
            f"*rit*{self.participant_id}*",
        ]

        for pattern in patterns:
            matches = [p for p in self.keywords_dir.glob(pattern) if p.suffix == ".json"]
            if matches:
                keywords_file = matches[0]
                logger.info(f"Found keywords file: {keywords_file}")
                return keywords_file

        raise FileNotFoundError(
            f"No .json keywords file found in {self.keywords_dir} matching "
            f"participant='{self.participant_id}'. "
            f"Tried patterns: {patterns}"
        )

    def _load_keywords(self) -> None:
        """Load keywords file and assign memories to this run.

        Randomizes deterministically per participant (seed derived from
        ``participant_id``):

        1. Pass → run mapping is shuffled, so for some participants run 1
           draws from pass 1 and for others from pass 2.
        2. The memories within the selected pass are shuffled before the
           first ``n_trials_per_run`` are taken.

        Also shuffles rumination and distraction prompts using the same
        RNG (existing behavior, preserved).
        """
        keywords_path = self._resolve_keywords_file()
        logger.info(f"Loading keywords file: {keywords_path}")

        all_memories = self._parse_keywords_file(keywords_path)

        # On a retry/resume of this run, re-use the exact selection recorded last
        # time so a mid-session keywords-file edit can't change what is shown.
        if self._restore_prior_selection(all_memories):
            self._stage_task_state_snapshot()
            return

        # Deterministic per-participant RNG. Use a stable hash (hashlib) rather
        # than the builtin hash(), which is salted per process (PYTHONHASHSEED)
        # and would give a different randomization on every run.
        seed = int(hashlib.sha256(str(self.participant_id).encode()).hexdigest(), 16) & 0xFFFFFFFF
        rng = random.Random(seed)

        # 1. Randomize which pass this run draws from.
        pass_order = [1, 2]
        rng.shuffle(pass_order)
        self._target_pass = pass_order[self.run_number - 1]

        # 2. Filter to this pass and shuffle within-pass order.
        run_memories = [m for m in all_memories if m["pass"] == self._target_pass]
        if len(run_memories) < self.n_trials_per_run:
            raise ValueError(
                f"Keywords file has {len(run_memories)} memories for pass "
                f"{self._target_pass}, but n_trials_per_run={self.n_trials_per_run}. "
                f"Need at least {self.n_trials_per_run} memories per pass."
            )
        rng.shuffle(run_memories)
        self._memories = run_memories[: self.n_trials_per_run]

        # 3. Shuffle rumination / distraction prompts (unchanged).
        rumination_order = list(range(len(RUMINATION_PROMPTS)))
        distraction_order = list(range(len(DISTRACTION_PROMPTS)))
        rng.shuffle(rumination_order)
        rng.shuffle(distraction_order)

        # Run 1 gets first half, run 2 gets second half
        start = (self.run_number - 1) * self.n_trials_per_run
        end = start + self.n_trials_per_run
        self._rumination_prompts = [
            RUMINATION_PROMPTS[rumination_order[i % len(RUMINATION_PROMPTS)]] for i in range(start, end)
        ]
        self._distraction_prompts = [
            DISTRACTION_PROMPTS[distraction_order[i % len(DISTRACTION_PROMPTS)]] for i in range(start, end)
        ]

        logger.info(f"Loaded {len(self._memories)} memories for run {self.run_number} (pass {self._target_pass})")
        self._stage_task_state_snapshot()

    def _restore_prior_selection(self, all_memories: list[dict[str, Any]]) -> bool:
        """Restore a previously-recorded selection for this run, if one exists.

        Returns True if the prior selection was fully restored. Returns False
        (caller computes a fresh selection) when there is no task state, no prior
        run for this slot, no recorded selection, or a recorded keyword is no
        longer present in the keywords file.
        """
        if self._task_state is None:
            return False
        prior = self._task_state.get_run_for_slot(self.run_number)
        if prior is None:
            return False
        snap = prior.task_state_snapshot
        if "target_pass" not in snap or "keywords" not in snap:
            return False

        by_keyword = {m["keyword"]: m for m in all_memories}
        restored = [by_keyword[k] for k in snap["keywords"] if k in by_keyword]
        if len(restored) != len(snap["keywords"]):
            logger.warning(
                "RIT run %d: prior keyword(s) missing from current file; recomputing selection fresh.",
                self.run_number,
            )
            return False

        self._target_pass = snap["target_pass"]
        self._memories = restored
        self._rumination_prompts = list(snap.get("rumination_prompts", []))
        self._distraction_prompts = list(snap.get("distraction_prompts", []))
        logger.info(
            "RIT run %d: restored prior selection (pass %d, %d memories) from task state.",
            self.run_number,
            self._target_pass,
            len(self._memories),
        )
        return True

    def _stage_task_state_snapshot(self) -> None:
        """Record this run's selection so it persists and can be restored on retry."""
        self._task_state_snapshot = {
            "target_pass": self._target_pass,
            "keywords": [m["keyword"] for m in self._memories],
            "rumination_prompts": list(self._rumination_prompts),
            "distraction_prompts": list(self._distraction_prompts),
        }

    def _parse_keywords_file(self, path: Path) -> list[dict[str, Any]]:
        """Parse a keywords JSON file.

        Each entry must include ``keyword`` (str), ``pass`` (int, 1 or 2),
        and ``memory_prompt`` (str). Optional ``unpleasantness`` and ``trial``
        fields are passed through for event recording but do not drive task
        logic.

        Parameters
        ----------
        path : Path
            Path to the keywords JSON file.

        Returns
        -------
        list of dict

        Raises
        ------
        ValueError
            If any entry is missing a required field.
        """
        with open(path) as f:
            data = json.load(f)
        memories: list[dict[str, Any]] = data.get("memories", data) if isinstance(data, dict) else data

        required = ("keyword", "pass", "memory_prompt")
        for i, m in enumerate(memories):
            missing = [k for k in required if k not in m]
            if missing:
                raise ValueError(
                    f"Keywords file {path} entry {i} is missing required field(s): {missing}. "
                    f"Each memory must include: {list(required)}."
                )
            m["pass"] = int(m["pass"])
            m["keyword"] = str(m["keyword"]).strip()
            m["memory_prompt"] = str(m["memory_prompt"])
        return memories

    # =========================================================================
    # Stimulus presentation
    # =========================================================================

    def _show_fixation(self, duration: float) -> None:
        """Display fixation cross using pre-created stimulus."""
        from psychopy import core, event

        self._fixation_stim.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        timer = core.CountdownTimer(duration)
        while timer.getTime() > 0 and self._running:
            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escaped during fixation")
            self._fixation_stim.draw()
            self.win.flip()

    def _show_text(self, text: str, duration: float) -> None:
        """Display text centered on screen for a given duration.

        Parameters
        ----------
        text : str
            Text to display.
        duration : float
            Duration in seconds.
        """
        from psychopy import core, event

        self._text_stim.text = text
        self._text_stim.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        timer = core.CountdownTimer(duration)
        while timer.getTime() > 0 and self._running:
            if event.getKeys(keyList=["escape"]):
                raise KeyboardInterrupt("Escaped during text display")
            # Redraw every frame to survive window occlusion/focus changes
            self._text_stim.draw()
            self.win.flip()

    def _update_marker(self, x_pos: float) -> None:
        """Move the triangle marker to a new x position."""
        self._rating_marker.pos = (x_pos, 0)

    def _pos_to_x(self, pos: float) -> float:
        """Convert a rating position (1-5) to screen x coordinate."""
        return self._scale_left_x + (pos - 1) * self._tick_spacing

    def _draw_rating_frame(self) -> None:
        """Draw all rating scale components (reusable stimuli)."""
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
    # Trial execution
    # =========================================================================

    def _run_trial(self, trial_idx: int) -> None:
        """Execute a single trial of the RIT paradigm.

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
            f"{memory['memory_prompt']}\n\nKeyword: {memory['keyword']}",
            self.memory_display_duration,
        )
        self.record_event(
            onset,
            self.memory_display_duration,
            "memory_prompt",
            trial=trial_num,
            keyword=memory["keyword"],
            memory_prompt=memory["memory_prompt"],
            pass_num=memory["pass"],
            unpleasantness=memory.get("unpleasantness"),
            trial_id=memory.get("trial"),
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
        self._show_fixation(rest_duration)
        self.record_event(onset, rest_duration, "rest", trial=trial_num)

        # 7. Distraction prompt
        onset = self._experiment_clock.getTime()
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

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def setup(self) -> None:
        """Load keywords and create PsychoPy window."""
        self._load_keywords()
        self.create_window(monitor_position=self.monitor_position)
        self._create_stimuli()

        logger.info(f"RITTask setup: run={self.run_number}, {len(self._memories)} memories loaded")

    def _create_stimuli(self) -> None:
        """Pre-create reusable PsychoPy stimuli to avoid GPU memory leaks."""
        from psychopy import visual

        # Reusable text stimulus for _show_text()
        self._text_stim = visual.TextStim(
            self.win,
            text="",
            height=0.06,
            color="white",
            wrapWidth=1.2,
        )

        # Reusable fixation stimulus
        self._fixation_stim = visual.TextStim(
            self.win,
            text="+",
            height=0.3,
            color="white",
        )

        # Rating scale components (static geometry reused across all ratings)
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

        # Reusable triangle marker — vertices defined relative to (0,0),
        # moved to correct position via .pos to avoid vertex buffer issues
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

    def run_paradigm(self) -> None:
        """Execute the RIT localizer paradigm."""
        from psychopy import core

        task_keys = [self.left_key, self.right_key, self.confirm_key]
        advance_keys = list(set(task_keys + ["space", "return"]))
        self.show_instructions(
            [
                "In this task, you will be asked to think about\n"
                "memories from your past.\n\n"
                "You will also be asked to think about\n"
                "unrelated topics as a distraction.\n\n"
                "After each period, you will rate how sad you feel\n"
                "and how focused you are on your feelings.\n\n"
                f"Use button {self.left_key} to move left, "
                f"button {self.right_key} to move right,\n"
                f"and button {self.confirm_key} to confirm your rating.\n\n"
                "Press any button to begin."
            ],
            advance_keys=advance_keys,
        )

        self.wait_for_trigger()
        self._experiment_clock = core.Clock()
        logger.info("[TIMING] Experiment clock started at trigger (t=0.00s)")

        for trial_idx in range(self.n_trials_per_run):
            if not self._running:
                break
            self._run_trial(trial_idx)

        total_time = self._experiment_clock.getTime()
        logger.info(f"[TIMING] Paradigm complete: {self.n_trials_per_run} trials, total={total_time:.1f}s")

    def cleanup(self) -> None:
        """Release resources after task completes."""
        super().cleanup()
