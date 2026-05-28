# Feedback Tasks

Tasks are the visual displays shown to participants during neurofeedback. pineuro includes three built-in feedback tasks, three RNT (Repetitive Negative Thinking) task variants, and two localizer tasks, with a plugin system for custom tasks.

## Built-In Feedback Tasks

### Gauge Task

A brain-o-meter (analog gauge dial) where needle position reflects differential activation between two networks. Designed for the Rockland NFB paradigm with alternating wander/focus blocks.

**Display:** A semicircular gauge with "Focused" and "Wandering" labels. The needle moves based on the difference between focus and wander network activations.

**Block Design:** The session is divided into blocks of configurable duration. Each block has a condition (wander or focus) displayed as an instruction below the gauge. Between blocks, a button-press catch trial verifies the participant is awake.

**Label Swapping:** To control for spatial bias, the left/right position of "Focused" and "Wandering" labels periodically swaps (configurable interval).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_durations` | `30,90,60,...` | Comma-separated block durations in seconds |
| `conditions` | `wander,focus` | Alternating condition names |
| `swap_labels` | False | Enable label position swapping |
| `label_swap_after_block` | 3 | Swap every N blocks (0 = disabled) |
| `score_factor` | 3.0 | Needle movement degrees per unit differential |
| `warmup_duration` | 2.0 | Feedback-off warmup at the start of each block |
| `button_press_duration` | 2.0 | Duration of button-press catch trials |
| `button_key` | `1` | Response key for catch trials |
| `baseline_duration` | 30.0 | Pre-feedback baseline rest period (seconds) |
| `focus_network` | `Cont` | Network driving the "Focused" label |
| `wander_network` | `Default` | Network driving the "Wandering" label |

**Networks required:** 2

### Ball Task

A dual-network game where a white ball moves between two target circles. The ball's vertical position reflects which of two brain networks is more active. Participants try to hit the correct target circle.

**Display:** A ball moves up (top network active) or down (bottom network active). Two target circles appear at the top and bottom. Successful hits are counted; misses are tracked.

**Adaptive Difficulty:** After each run:
- If hits < `min_hits_threshold` → speed increases 25% (easier)
- If hits > `max_hits_threshold` → speed decreases 25% (harder)
- Circle size shrinks by `circle_shrink_factor` per hit

**Session State:** Scale factor, hit counts, and circle sizes persist across runs within a session, enabling progressive difficulty.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_scale_factor` | 10.0 | Ball movement speed (larger = more movement) |
| `scale_adjust_factor` | 0.25 | Adaptive difficulty adjustment per run |
| `min_hits_threshold` | 3 | Below this → increase speed |
| `max_hits_threshold` | 5 | Above this → decrease speed |
| `circle_shrink_factor` | 0.9 | Circle radius multiplier per hit |
| `left_button` | `1` | Left control |
| `right_button` | `2` | Right control |
| `enter_button` | `0` | Confirm button |

**Post-run questions:** 7 questions about mindfulness practice, ball movement correspondence, and self-regulation ability.

**Networks required:** 2

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

**Networks required:** 1

## RNT (Repetitive Negative Thinking) Tasks

Three specialized tasks for studying rumination and repetitive negative thinking.

### RNT Localizer

A rumination localizer task where participants view personal keywords and rate how much they ruminate about each one. Used to identify brain networks involved in repetitive negative thinking.

**Paradigm:** 9-stage trial protocol with keyword presentation, rumination induction, and 1–5 rating scales. Produces BIDS events TSV output.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keywords_file` | — | Path to participant-specific keywords file |
| `n_trials` | 10 | Number of keyword trials |
| `rating_duration` | 6.0 | Duration for each rating scale |

**Networks required:** 0 (localizer only)

### RNT Experience Sampling

An activation-triggered experience sampling task. Monitors real-time brain activation and presents experience sampling probes (sadness + focus ratings) when activation exceeds a threshold.

**Trigger logic:** Activation threshold with trend detection and cooldown period to avoid over-sampling. Phases: baseline → monitoring → prompt.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trigger_threshold` | 1.5 | Activation z-score to trigger a probe |
| `cooldown_duration` | 30.0 | Minimum seconds between probes |

**Networks required:** 1

### RNT Triggered Ball

A 3-network feedback task combining rumination detection with the ball game. When the trigger network exceeds a threshold, a ball feedback episode begins using the other two networks.

**Display:** Ball moves based on differential activation of top and bottom networks (like BallTask), but episodes are triggered by the third network crossing a threshold.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trigger_threshold` | 1.5 | Trigger network z-score to start an episode |
| `episode_duration` | 30.0 | Duration of each ball episode |
| `timeout_duration` | 60.0 | Max wait before forced episode |

**Networks required:** 3 (trigger, top, bottom)

## Built-In Localizer Tasks

### Fixation Task

Simple fixation cross for resting-state data collection. Non-interactive.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | 200.0 | Display duration in seconds |

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

## Transfer Runs

Transfer runs are practice runs where the task display is visible but feedback is not active (the needle/ball/thermometer does not move). This allows participants to practice the task without feedback before actual neurofeedback training.

Transfer runs are configured per-task via the `use_transfer_runs` parameter:
- When enabled, the workflow includes Transfer Run 1 (before feedback) and Transfer Run 2 (after feedback)
- When disabled, the workflow goes directly to feedback runs

During transfer runs, the task displays modified instructions informing the participant that the display will not move.

## Trigger Keys

All tasks support scanner trigger keys for synchronization. The scanner sends a trigger pulse at the start of each TR, which pineuro uses to time-lock the task display.

Default trigger keys: `t`, `+`, `5`, `equal` (covers most scanner button boxes).

Configure trigger keys per-task in the Study Editor Task tab.

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
2. Define `TaskMetadata` with parameters, network requirements, and descriptions
3. Register with `@pineuro_task` decorator or `register_task()` function

Custom tasks automatically appear in the Study Editor Task tab dropdown and receive real-time activations via the coordinator callback system. See `CONTRIBUTING.md` for details.
