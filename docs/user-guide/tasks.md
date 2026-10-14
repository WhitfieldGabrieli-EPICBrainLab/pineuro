# Tasks

Tasks are the visual displays shown to participants. pineuro includes three built-in feedback tasks, three RIT (Rumination Induction Task) variants, and two localizer tasks, with a plugin system for custom tasks. (For how feedback values are *computed* — activation, differential, connectivity, intermittent — see [Feedback Tasks](feedback-tasks.md).)

## Built-In Feedback Tasks

### Gauge Task

A brain-o-meter (analog gauge dial) where needle position reflects differential activation between two networks. Designed for a mind-wandering neurofeedback paradigm with alternating wander/focus blocks.

**Display:** A semicircular gauge with "Focused" and "Wandering" labels. The needle moves based on the difference between focus and wander network activations.

**Block Design:** The session is divided into blocks of configurable duration. Each block has a condition (wander or focus) displayed as an instruction below the gauge. Between blocks, a button-press catch trial verifies the participant is awake.

**Label Swapping:** To control for spatial bias, the left/right position of "Focused" and "Wandering" labels periodically swaps (configurable interval).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_transfer_runs` | False | Include transfer runs (1 before feedback, 1 after) |
| `block_durations` | `30,90,60,...` | Comma-separated block durations in seconds |
| `conditions` | `wander,focus` | Alternating condition names |
| `swap_labels` | False | Start with Wandering on the left rather than the right (counterbalancing) |
| `label_swap_after_block` | 3 | Swap labels every N blocks (0 = disabled) |
| `score_factor` | 3.0 | Needle movement degrees per unit differential |
| `warmup_duration` | 2.0 | Feedback-off warmup at the start of each block |
| `button_press_duration` | 2.0 | Duration of button-press catch trials |
| `button_key` | `1` | Response key for catch trials |

**Networks required:** 2 — assigned to the `focus_network` role (drives the "Focused" label, e.g. ContA) and the `wander_network` role (drives "Wandering", e.g. DefaultA). Network roles are assigned in the Study Editor, not set as task parameters.

### Ball Task

A dual-network game where a white ball moves between two target circles. The ball's vertical position reflects which of two brain networks is more active. Participants try to hit the correct target circle.

**Display:** A ball moves up (top network active) or down (bottom network active). Two target circles appear at the top and bottom. Successful hits are counted; misses are tracked.

**Adaptive Difficulty:** After each run:
- If hits < `min_hits_threshold` → speed increases 25% (easier)
- If hits > `max_hits_threshold` → speed decreases 25% (harder)
- Circle size shrinks by `circle_shrink_factor` per hit

**Task State:** Scale factor, hit counts, and circle sizes persist across runs within a session, enabling progressive difficulty.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_transfer_runs` | True | Include transfer runs (1 before feedback, 1 after) |
| `initial_scale_factor` | 10.0 | Ball movement speed (larger = more movement) |
| `scale_adjust_factor` | 0.25 | Adaptive difficulty adjustment per run |
| `min_hits_threshold` | 3 | Below this → increase speed |
| `max_hits_threshold` | 5 | Above this → decrease speed |
| `circle_shrink_factor` | 0.9 | Circle radius multiplier per hit |
| `left_button` | `1` | Left control |
| `right_button` | `2` | Right control |
| `enter_button` | `0` | Confirm button |

**Post-run questions:** 7 questions about mindfulness practice, ball movement correspondence, and self-regulation ability.

**Networks required:** 2 — the `top_network` and `bottom_network` roles. The ball moves up when `top_network` is more active.

### Thermometer Task

A block-design audio task with thermometer feedback. Participants listen to voice recordings and rate their ability to attend or ignore, then see a thermometer indicating their median block activation.

**Display:** Animated thermometer bar that fills to reflect median activation for the block. Green = above target, red = below target.

**Block Structure:** Each block consists of:
1. Instruction ("Listen to self" or "Ignore all sounds")
2. Four voice playback segments with ISI
3. Rating scale (1-6: "How well could you attend/ignore?")
4. Thermometer feedback display
5. Inter-block fixation rest

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stim_dir` | — | Directory with per-participant stimulus CSV files |
| `instruction_duration` | 2.0 | Instruction display time (seconds) |
| `voice_duration` | 3.5 | Fixed playback duration per voice file |
| `isi_duration` | 0.5 | Inter-stimulus interval |
| `rating_duration` | 8.0 | Rating scale display time |
| `feedback_duration` | 8.0 | Thermometer display time |
| `fixation_nfb_duration` | 11.0 | Inter-block rest (feedback runs) |
| `fixation_transfer_duration` | 15.0 | Inter-block rest (transfer runs) |
| `activation_max` | 1.0 | Thermometer ceiling |
| `activation_min` | -1.0 | Thermometer floor |
| `left_button` | `1` | Rating scale left |
| `right_button` | `2` | Rating scale right |
| `confirm_button` | `3` | Rating scale confirm |

Also accepts `use_transfer_runs` (default False).

**Networks required:** 1 — the `target_mask` role.

## RIT (Rumination Induction Task) Tasks

Three specialized tasks for studying rumination.

### Trigger Conditions (shared)

RIT Experience Sampling and RIT Triggered Ball both monitor a network's activation and fire a discrete episode when trigger conditions are met. Four condition gates can be enabled independently and combined with `trigger_logic` (`all` = AND, `any` = OR). The defaults reproduce the original hardcoded behavior — activation above baseline AND a rising slope.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `activation_contrast` | `positive` | `positive` fires above the threshold, `negative` below |
| `trigger_logic` | `all` | `all` = every enabled gate must hold, `any` = at least one |
| `consecutive_volumes` | 3 | Window length: trending volumes required (strict) or regression window (slope) |
| `use_activation_condition` | True | Gate on current activation vs `activation_threshold` |
| `activation_threshold` | 0.0 | Level for the activation and count gates. 0 = at baseline |
| `use_trend_condition` | True | Gate on the trend over the window |
| `trend_method` | `slope` | `slope` = least-squares slope sign (tolerant of single blips), `strict` = monotonic across all volumes |
| `activation_trend` | `increasing` | Direction the trend gate requires |
| `slope_threshold` | 0.0 | Slope method only; 0 = any rise/fall |
| `use_mean_condition` | False | Gate on mean activation across the window |
| `mean_threshold` | 0.0 | Mean-activation threshold |
| `use_count_condition` | False | Gate on how many in-window volumes pass `activation_threshold` |
| `count_min` | 3 | Minimum passing volumes for the count gate |
| `min_prompt_interval` | 30.0 | Cooldown — minimum seconds between episodes |
| `timeout_duration` | 60.0 | Seconds without a trigger before a forced episode |

A cross-parameter validator rejects configurations with no gate enabled, and `count_min` greater than `consecutive_volumes`.

### RIT Localizer

A rumination localizer where participants recall personal memories cued by their own keywords, then alternate between ruminating on and being distracted from each one. Used to identify the rumination network.

**Paradigm:** Per trial — memory prompt with keyword, detailed reflection, then rumination and distraction prompts, with jittered rest between. Ratings use 1–5 scales. Produces a BIDS events TSV.

**Default contrast:** `rumination_network = rumination - distraction`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keywords_dir` | — | Directory of per-participant keyword files (auto-matched by participant ID) |
| `memory_display_duration` | 5.0 | Memory prompt + keyword display (seconds) |
| `reflection_duration` | 25.0 | Detailed memory reflection |
| `rumination_duration` | 30.0 | Rumination and distraction prompts |
| `rating_timeout` | 5.0 | Maximum time for each rating response |
| `rest_min` | 10.0 | Minimum rest/fixation jitter |
| `rest_max` | 15.0 | Maximum rest/fixation jitter |
| `n_trials_per_run` | 4 | Trials per run |
| `total_runs` | 2 | Total RIT runs across the whole study; the keywords file must contain exactly this many memory passes |
| `left_key` | `1` | Move rating left |
| `right_key` | `2` | Move rating right |
| `confirm_key` | `0` | Confirm rating |
| `left_key_label` | — | How the instructions name the left key (e.g. `index finger`); blank shows "button <left_key>" |
| `right_key_label` | — | How the instructions name the right key (e.g. `middle finger`); blank shows "button <right_key>" |
| `confirm_key_label` | — | How the instructions name the confirm key (e.g. `thumb`); blank shows "button <confirm_key>" |

**Networks required:** 0 (localizer only)

### RIT Experience Sampling

An activation-triggered experience sampling task. Monitors real-time brain activation and presents experience sampling probes (unpleasantness + focus ratings) when activation exceeds a threshold.

**Trigger logic:** See [Trigger Conditions](#trigger-conditions-shared) above. Phases: baseline → monitoring → prompt.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rating_timeout` | 15.0 | Maximum time for each rating response |
| `left_key` | `1` | Move rating left |
| `right_key` | `2` | Move rating right |
| `confirm_key` | `0` | Confirm rating |
| `left_key_label` | — | How the instructions name the left key (e.g. `index finger`); blank shows "button <left_key>" |
| `right_key_label` | — | How the instructions name the right key (e.g. `middle finger`); blank shows "button <right_key>" |
| `confirm_key_label` | — | How the instructions name the confirm key (e.g. `thumb`); blank shows "button <confirm_key>" |

**Networks required:** 1 — the `target_network` role.

### RIT Triggered Ball

A feedback task combining rumination detection with the ball game. When the trigger network meets the trigger conditions, a ball feedback episode begins.

**Display:** During an episode the ball moves in one of two modes, set by `movement_mode`:

- **`single`** (default) — one network's signed activation drives the ball. This is the trigger network itself unless a `movement_network` is assigned. `movement_direction` sets the polarity: `positive_down` (default) moves the ball down as rumination increases; `positive_up` flips it.
- **`dual`** — the ball moves on the difference between the `top_network` and `bottom_network` activations, like the Ball task.

**Trigger logic:** See [Trigger Conditions](#trigger-conditions-shared) above.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `movement_mode` | `single` | `single` = one network's signed activation, `dual` = difference between two |
| `movement_direction` | `positive_down` | Single mode only: which way positive activation moves the ball |
| `episode_duration` | 30.0 | Duration of each ball episode |
| `scale_factor` | 10.0 | Ball movement speed (larger = faster) |
| `circle_shrink_factor` | 0.9 | Circle radius multiplier after a hit (1.0 = no shrink) |
| `reset_ball_each_episode` | True | Reset ball position and circles at each episode start |

**Networks required:** 1 — the `trigger_network` role. Three further roles are **optional** and shown only when relevant: `movement_network` (single mode, overrides which network drives the ball) and `top_network` / `bottom_network` (dual mode).

## Built-In Localizer Tasks

### Rest Localizer

Fixation-cross display for resting-state data collection (`task_id: rest`, BIDS `task-rest`). Non-interactive — acquires the resting-state scan used for ICA-based mask extraction.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | 200.0 | Display duration in seconds |

**Networks required:** 0 (localizer only)

### Self/Other Task

Voice localizer for STG (superior temporal gyrus) ROI identification. Participants listen to sentences in their own voice vs. a stranger's voice, with comprehension questions.

**Block structure:** Alternating voice blocks (self/other, inferred from filename) and silent rest blocks. Each voice block contains 5 sentences with ISI and a comprehension question.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stim_dir` | — | Directory with per-participant stimulus TSV files |
| `baseline_duration` | 30.0 | Initial fixation rest |
| `block_duration` | 16.0 | Duration per voice/rest block |
| `question_duration` | 4.0 | Comprehension question display |
| `block_order` | `voice,voice,rest,...` | Comma-separated block sequence |
| `response_keys` | `1,2` | Yes/No response buttons |

**Default contrast:** STG_self = self - other

**Networks required:** 0 (localizer only)

## Transfer Runs

Transfer runs are practice runs where the task display is visible but feedback is not active (the needle/ball/thermometer does not move). This allows participants to practice the task without feedback before actual neurofeedback training.

Transfer runs are configured per-task via the `use_transfer_runs` parameter:
- When enabled, the workflow includes Transfer Run 1 (before feedback) and Transfer Run 2 (after feedback)
- When disabled, the workflow goes directly to feedback runs

During transfer runs, the task displays modified instructions informing the participant that the display will not move.

## Trigger Keys

All tasks support scanner trigger keys for synchronization. The scanner sends a trigger pulse at the start of each TR, which pineuro uses to time-lock the task display.

Default trigger keys: `t`, `+`, `5`, `=` (covers most scanner button boxes). Enter keys as the symbols they produce — use `=`, not the word `equal`.

Configure trigger keys in the Study Editor's Scanner settings; they apply to every task in the study.

## Task Output Files

Each task produces standardized output:

| File | Description |
|------|-------------|
| `*_psychopylog.tsv` | Per-frame (~60 Hz) task state: condition, score, feedback active, button presses |
| `*_events.tsv` | BIDS-format event file with onset, duration, trial_type for each block |
| `task_state.json` | Persistent state across runs (adaptive difficulty, hit counts, etc.) |

## Custom Tasks

pineuro uses a plugin system for task registration. To create a custom task:

1. Extend `BaseTask` from `pineuro.tasks.base`
2. Define `TaskMetadata` with parameters, network requirements, and descriptions — including `design_type`, which selects the real-time GLM task regressor (see [Supported Task Designs](feedback-tasks.md#supported-task-designs); event-related paradigms run under `nuisance_only`, since the online GLM cannot model event onsets)
3. Register with `@pineuro_task` decorator or `register_task()` function

Custom tasks automatically appear in the Study Editor's Feedback tab dropdown and receive real-time activations via the coordinator callback system. See `CONTRIBUTING.md` for details.
