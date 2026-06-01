# Changelog

All notable changes to pineuro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **Dead and legacy code swept out** (pre-rollout, no backwards-compat surface to preserve). This is a no-behavior-change cleanup of symbols with no live callers, defensive branches that could never run, and superseded compatibility shims:
  - **Public API symbols** with no internal or external consumers: `resolve_sham_masks` (sham "alternative masks" path is reimplemented inline in the pipeline runner), `SessionConfig.from_yaml` / `to_yaml` / `from_json` / `to_json` (config persists via the GUI config manager; the internal `_from_dict` / `_to_dict` remain), the `SimulatorSourceMapping` convenience accessors `get_localizer_sources` / `get_feedback_source` / `get_transfer_source` / `has_subject` / `subjects` (use `get_source`), and the `PathConfig.participant_session_dir` property.
  - **Test-only production symbols** (removed with their tests): `tasks.utils.make_bids_subject_prefix`, `tasks.protocol.TaskProtocol` (the `CoordinatorProtocol` contract stays), `IncrementalGLM.get_residual_mse`, `preprocessing.spatial_smoothing` / `temporal_filtering`, `utils.get_dicom_files`, and `display.get_monitor_by_index` / `get_monitor_count`.
  - **The `"activation"` primary-mask alias** in task data dicts — consumers now read `data["activations"][mask_name]` directly.
  - Numerous internal dead methods/constants and unreachable `ImportError` fallbacks for hard dependencies (`flet-charts`, `nilearn`).

## [0.10.0] - 2026-05-28

### Breaking

- **Terminology unification: `session` ↔ `task` ↔ `visit`.** Aligned the codebase on a single hierarchy — `study → participant → session (= scanner appointment, BIDS ses-XX) → task (one localizer or feedback paradigm) → run`. Renames:
  - **visit → session** (BIDS-aligned): `VisitConfig` → `SessionMetadata`, `visit_id` → `session_id`, `visit_number` → `session_number`, `mask_source_visit` → `mask_source_session`, `detect_existing_visits()` → `detect_existing_sessions()`, `next_visit_id()` → `next_session_id()`, `migrate_to_multi_visit()` → `migrate_to_multi_session()`, `is_single_visit_data()` → `is_single_session_data()`, `setup_visit_with_reused_masks()` → `setup_session_with_reused_masks()`, `VisitSelectionDialog` → `SessionSelectionDialog` (file `gui/components/visit_selection_dialog.py` → `session_selection_dialog.py`), study-config YAML key `multi_visit_enabled` → `multi_session_enabled` and per-session block `visit:` → `session:`, doc `docs/user-guide/multi-visit.md` → `multi-session.md`.
  - **session → task** (for things that were per-task, not per-session): `run_feedback_session()` → `run_feedback_task()`, `SessionState` → `TaskState` (file `pineuro/tasks/session.py` → `pineuro/tasks/state.py`), on-disk state file `session_state.json` → `task_state.json`, `reset_session_state()` → `reset_task_state()`, `PathConfig.session_dir` → `PathConfig.feedback_dir` (it was always the feedback workspace, never the session dir), the `session_dir=` parameter on `TaskState.load/reset/create_for_task` → `feedback_dir=`. GUI section headers "Localizer Session" / "Feedback Session" → "Localizer Tasks" / "Feedback Tasks"; doc `docs/user-guide/feedback-sessions.md` → `feedback-tasks.md`.
  - **Existing study configs in `~/.pineuro/studies/` are auto-migrated** in this commit (`multi_visit_enabled` → `multi_session_enabled` across active configs, version snapshots, and archives). Existing on-disk `session_state.json` files in per-participant data directories are **not** migrated (per the user's direction — they were test data). To carry old data forward, rename the file by hand.
  - No legacy aliases. Imports of the old names will fail with `ImportError`.
- **`ses-XX` is now always present in output paths and filenames.** Every participant's data is organized under `sub-XX/ses-XX/` and every output filename embeds `_ses-XX_` right after the subject id (e.g. `sub-01_ses-01_task-rest_run-01_preproc.nii.gz`), regardless of how many sessions the study has. The per-study `multi_session_enabled` flag is removed; the session-selection dialog now always runs (new participants default to `ses-01`). `PathConfig.session_id` is now required, and `StudyInfoConfig`, `is_single_session_data`, and `migrate_to_multi_session` are deleted. Existing `sub-XX/` directories without a `ses-XX/` level are unsupported.
- **The acquisition-retry counter is now the `attempt` entity, placed after `run`.** Output filenames use `run-NN_attempt-NN` (e.g. `run-01_attempt-02`) instead of the former `acq-NN_run-NN`. The retry counter had been labeled with the BIDS `acq` entity, which inverts BIDS semantics and put the inner counter before the outer run; `attempt` is a pineuro-specific entity that reads as `session → run → attempt`. `acq` is removed across construction, parsers, `TaskState`/`RunInfo` fields, GUI labels, and the selection dialog (`unified_acq_dialog` → `unified_attempt_dialog`). No on-disk migration.
- **Rest localizer renamed `FixationTask` → `RestTask`.** The class was named after its fixation-cross stimulus rather than its protocol role, leaving the YAML config (`type: rest`) and the BIDS task entity (`task-fixation`) inconsistent. The event-file task entity is now `task-rest`; `tasks/fixation.py` → `tasks/rest.py`, `FixationTask` → `RestTask`, `FIXATION_METADATA` → `REST_METADATA`, `task_id` `"fixation"` → `"rest"`. (The fixation-cross UI helpers are unchanged — they describe the stimulus, not the task.)
- **Rumination tasks renamed RNT → RIT (Rumination Induction Task).** Matches Shreiner et al. (2025) and the on-disk `RIT_keywords` / `task-rit` convention. Task IDs `rnt` → `rit`; covers file names, class names, display names, tests, and docs. Paradigm-content terms (rumination, distraction) are unchanged.
- **`network_atlas` is the single source of truth for the network count; `yeo_targets` → `networks`.** The `n_yeo` parameter is removed — the count is derived from `network_atlas` via `ATLAS_SOURCES[...]["n_networks"]`, so it works for Schaefer atlases too, not just Yeo. (This also fixes `validate_session_config`, which read the count from a missing key and so always failed.)
- **Scanner trigger keys moved to study-level `ScannerConfig`.** Trigger keys are determined by the scanner, not the task, so they are declared once on the study (in the renamed "Scanner & Simulator" study-tab section) rather than per feedback task. Previously only the 5 feedback tasks exposed the field; localizers silently fell back to the hardcoded `["t", "+", "5", "equal"]` default, so changing the keys in a feedback tab had no effect on localizers in the same session. Both the feedback orchestrator and the localizer launcher now read the study-wide value.

### Added

- **Discard-on-stop for partial acquisitions.** Stopping a localizer or feedback run mid-acquisition now prompts Keep (files stay; the next attempt increments) or Discard (delete the partial files so the next attempt reuses the same number). Keep is the recommended default (green button); Discard is de-emphasized.
- **Stop now interrupts long-running steps.** `preprocessing()` checks for a stop request at step boundaries and before each motion-correction volume, and mask extraction runs the non-interruptible CanICA fit in a killable subprocess (its process tree is terminated via `psutil`, a new dependency). A requested stop is reported as a clean stop, not an error. Completed preprocessing / mask-extraction steps now offer a "Re-run" affordance.
- **Per-task state is now universal.** Every localizer (not just feedback tasks) gets a `task_state.json` run ledger, lifted from `BaseTask` onto the shared `TaskBase`. RIT persists its per-run selection (target pass, keywords, prompts) and restores it exactly on a retry/resume of the same run, warning and recomputing if a keyword file was edited in the meantime.

### Changed

- **The generated `reproduce_masks.py` script is generalized to all extraction modes.** It was rest/ICA-only and had drifted from the core API (it dropped `network_atlas` and `sham_hemisphere_network`; its task path was dead code and it had no MNI support). A single mode-agnostic generator now serializes the resolved call and replays it through a shared `run_mask_extraction_from_config` dispatcher (rest/task/MNI); the pipeline runner routes all extraction call sites through that same dispatcher and rewrites the script after every extraction, so it can no longer drift from the `mask_extraction_*` signatures.

### Removed

- **Migration shims and legacy accommodations** (pineuro is pre-rollout, so config-schema changes need none): the hard-fail top-level-key validators, the silent migration shims (`use_efficiency_weights`, old `TaskState` format, `MaskSelection` secondary-field defaulting, boldref/Yeo-CSV fallbacks, GUI flat-key fallbacks), and the public-API aliases `DicomSimulator`, `collect_dicoms`, `FDData`, and the `ball.py` HRF re-export. Also gone: the dead `atlas_name` path, the single-localizer-else branch, and the vestigial `n_feedback_runs` parameter.

### Fixed

- **GUI launched the wrong task on migrated study configs.** The participant workflow and pipeline runner still read the removed legacy top-level YAML blocks (`scan`/`mask`/`feedback`/`task`/`localizer_task`), so on a migrated config every read fell through to defaults — most visibly launching a rest task instead of the configured RIT localizer. ~50 reads were swept to the new `localizers[*]` / `feedback_runs[*]` / top-level `simulator`/`scanner`/`reference` schema.
- **Participant-workflow paths dropped the session component.** Step-completion detection, QC-report lookup, mask-extraction inputs, and feedback-log paths were built from a bare `participant_id` without `ses-XX`, so preprocessing never registered COMPLETE (locking mask extraction), mask extraction could not find its input, and several status globs / QC / feedback-metric lookups silently missed their files. All now use the consistent `{participant}_{session}` BIDS prefix.
- **`preprocessing(overwrite=True)` left stale `01_nifti/` files.** `dcm2niix` appends a letter suffix on filename collision rather than overwriting, so the canonical path kept old data while fresh output sat beside it with a suffix and downstream steps silently consumed the stale file. Prior outputs for the current label (primary, suffixed, multi-echo/phase splits, sidecars) are now removed first.
- **Stopped runs stayed PENDING instead of INTERRUPTED.** The feedback-slot status glob still used the pre-`ses-XX` filename pattern, so stopped feedback/transfer runs never matched and missed their INTERRUPTED (caution) flag.

## [0.9.0] - 2026-05-22

### Added

- **Manual Mask Import localizer type.** Researchers can supply pre-built, participant-specific NIfTI masks (native/EPI, T1, or MNI space) for neurofeedback instead of running ICA (rest) or GLM (task) extraction. The study config holds only participant-agnostic data (mask space, network names); each participant's mask / reference-EPI / T1 files are selected in their workflow and persisted per-participant to `manual_masks.json`. `import_manual_masks` supports opt-in QC reporting, and the pipeline runner imports the masks into `03_masks/`, generates QC, and coregisters them.
- **Validation suite** (`tests/validation/`) — a scientific-correctness test layer, separate from the regression suite, that runs synthetic phantoms with known ground truth through the real pipeline (real ANTsPy/nilearn, no mocking) and asserts the outputs recover that ground truth: HRF/design math, the incremental-GLM solver, ANTs motion correction & brain extraction, nilearn ICA & GLM task-mask extraction, real-time feedback signal recovery, and an end-to-end chained pipeline test. Includes a self-contained synthetic-data phantom library and a markdown/HTML metrics report. Opt-in locally (`pytest -m validation`); runs as a hard correctness gate on every push in CI. See `tests/validation/README.md`.

### Changed

- **Motion correction is now deterministic.** `motion_correction()` seeds ANTs' rigid-registration metric sampling (new `random_seed` parameter, default `0`), so re-running preprocessing on identical input yields identical motion parameters. Previously the metric's random sampling was unseeded, giving small run-to-run variation.
- **The RIT localizer's per-participant randomization is now reproducible.** It seeded its RNG with the builtin `hash()`, which is salted per process (`PYTHONHASHSEED`) — the same participant got a different trial order on every run. It now derives the seed from a stable `hashlib` digest.
- **`pineuro.__version__` now reflects the installed package metadata** (via `importlib.metadata`) instead of a hardcoded string, which had drifted to `0.2.0`. It can no longer go stale.
- **`pipeline_runner` no longer special-cases task IDs.** The fixation localizer's parameter handling branched on `task_id == "fixation"`; that decision now reads a `TaskMetadata.localizer_params_from_scan` flag, so fixed-duration localizers work without editing shared GUI code.
- **FD/DVARS/Siemens-MoCo motion parsing deduplicated.** Framewise displacement, DVARS, and Siemens MoCo motion-string parsing were copy-pasted between `feedback.py` and `collection_qc.py`, so a fix to one copy would silently not reach the other. `compute_fd`, `compute_dvars`, and `parse_siemens_motion_comment` now live once in `pineuro.utils`; both modules delegate to them.

### Removed

- **`tests/integration/` (Rockland_NFB real-data suite) and the `integration` pytest marker.** The preprocessing-pipeline, ICA mask-extraction, and feedback-estimation paths it exercised are covered by the synthetic-phantom validation suite; the marker and the `sourcedata/Rockland_NFB/` data dependency are gone. Note: real DICOM→NIfTI (`dcm2niix`) conversion no longer has dedicated test coverage.

### Fixed

- **ROI feedback weighting was inverted.** `FeedbackEstimator` weighted ROI voxels by their residual sum of squares (proportional to variance), *up*-weighting noisy voxels — a faithful copy of MURFI's `RtRoiCombine`, but the opposite of the Hinds et al. (2011) 1/σ efficiency weighting it was meant to implement. The `use_efficiency_weights` boolean is replaced by a `roi_combine_method` choice — `inverse_sd` (the corrected 1/σ weighting, now the default) or `mean` (unweighted) — routed through a single `_combine_roi` helper, with a study-editor dropdown and a migration shim for the old config key.
- **`script_generator` produced invalid Python on Windows.** Generated mask-extraction scripts embedded file paths via raw f-string interpolation (`Path("{path}")`), so a Windows path like `C:\Users\...` became a string literal where `\U` raised a `SyntaxError` and `\t`/`\m` etc. silently corrupted the path. Paths and the participant ID are now embedded with `repr()`, so generated scripts are valid on every platform.
- **The Self/Other localizer showed the wrong comprehension-question image** when the block order contained `rest` blocks. Questions were indexed by absolute block position, but the stimulus file holds only voice-block rows (5 per block); they are now indexed per voice block, so each voice block shows its own question.
- **GUI study-editor edits could mutate the source configuration.** The study editor and the localizer/feedback tab builders copied configs shallowly (`dict(config)`), so edits to nested blocks — including edits to a *duplicated* study — leaked back into the original. All config copies are now deep copies.
- **The feedback QC report crashed on logs missing optional columns.** It raised a `KeyError` when a feedback log lacked `dvars`/`tsnr` columns, and the ICA component-grid plot raised an `AttributeError` for a single component. Both are guarded, and reports no longer print `nan` for all-missing motion metrics.
- **Task-mask extraction could select a network from a `NaN` correlation.** The ICA spatial-correlation step divided by zero for a zero-variance component or an empty network mask; those cases are now detected and handled explicitly.
- **Resource leaks in the participant-workflow GUI.** The per-session log file handler was never removed or closed (handlers accumulated on the `pineuro` logger across sessions), and the feedback-monitor thread was not stopped when a pipeline step failed. Both are now cleaned up.
- **Numerous robustness fixes.** Runtime `assert`s used for input validation (silently stripped under `python -O`) are replaced with explicit errors; guards were added for empty `feedback_runs`, malformed session IDs, 3-D inputs to temporal filtering, empty ROI masks, and non-finite activation values; the task factory now verifies a registered task class subclasses `BaseTask`. The collection-run-number helper now matches the actual `acq-NN_run-NN` directory naming.

## [0.8.0] - 2026-05-13

### Breaking

- **Study config schema restructured.** The legacy top-level blocks `mask`, `feedback`, `task`, `scan`, and `localizer_task` are removed. Per-run configuration now lives exclusively in `localizers[*]` and `feedback_runs[*]`; environment/setup config moves to dedicated top-level blocks (`simulator`, `scanner`, `reference`, `study`). YAMLs saved by 0.7.0 and earlier will not load — `SessionConfig.from_yaml` and the GUI's `StudyConfig.from_dict` both raise `ValueError` with a clear message instructing you to regenerate via the study editor.
- **`network_assignment` is the single source of truth for role-to-mask mapping.** The YAML keys `differential_pair`, `target_mask`, `top_network`, `bottom_network`, and `mask_names` are gone. For 2-network tasks, `network_assignment[role[0]] = positive` and `network_assignment[role[1]] = negative`; the `differential_pair` passed to `run_feedback_task` is now derived from this at runtime.
- **`task_type` → `task`** on `feedback_runs[*]`; **`yeo_targets` → `networks`** on `localizers[*]`.
- **Dataclass API**: `MaskConfig`, `FeedbackConfig`, `TaskConfig`, and `ScanConfig` are deleted from `pineuro.session`. Replace with the new dataclasses: `LocalizerConfig` (with its `extraction: LocalizerExtractionConfig` sub-block), `FeedbackRunConfig` (with `qc: QCConfig`, `glm: GLMConfig`, `connectivity: ConnectivityConfig`, `intermittent: IntermittentConfig` sub-blocks), and the new top-level blocks `SimulatorConfig` / `ScannerConfig` / `ReferenceConfig` / `StudyInfoConfig`.
- **Per-run `reference_realtime_*` overrides on `FeedbackRunConfig` are dropped.** The top-level `reference:` block applies to every run.

### Fixed

- **GUI positive/negative mask was silently inverted in saved YAML** when the localizer was rest and its `yeo_targets` order disagreed with the user's positive/negative selection in the feedback tab. The fallback that copied `yeo_targets` order into `top_network`/`bottom_network` is gone; `_feedback_tab.collect_config` now always emits `network_assignment` so the role mapping is explicit and round-trips correctly.

### Added

- New `combined_mask_names` property on `SessionConfig` (de-duplicated union of `localizers[*].mask_names`), replacing the legacy stored `mask.mask_names`.
- Explicit `mask_source: <localizer_id>` cross-reference on each `feedback_runs[*]`; validated at load time.
- `id` field on `LocalizerConfig` (defaults to `type`) so multiple localizers can be unambiguously referenced.
- Manual ROI import is now a localizer type — `LocalizerConfig(type="manual", manual_mask_paths=...)` replaces the old `MaskConfig.manual_*` fields.
- Fields previously consumed via raw-dict access are now first-class dataclass fields: `LocalizerConfig.motion_correction`, `PathConfig.example_participant_id`, `ShamConfig.blind_mode`.
- `config_snapshot._deep_diff` recurses into lists of dicts, so per-localizer / per-feedback-run protocol diffs surface with element-indexed paths (e.g. `feedback_runs.0.glm.motion_correction`).

### Changed

- Default ICA `n_components`: 20 → 30.
- Default ball-task `use_transfer_runs`: False → True.
- `simulator.feedback_speed` renamed from `feedback_simulator_speed` (redundant prefix removed inside the new `simulator:` block).
- `simulator`/`scanner`/`reference` blocks strip their old prefixes (e.g. `simulator_enabled` → `simulator.enabled`, `reference_initial_n_volumes` → `reference.initial_n_volumes`, `scanner_dicom_dir` → `scanner.dicom_dir`).

## [0.7.0] - 2026-05-12

### Added

- **Declarative live metrics**: Each task now declares what to surface in the participant workflow monitoring panel via a `live_metrics: list[LiveMetric]` field on its `TaskMetadata`. The GUI, end-of-run summary dialog, and saved QC HTML report all iterate these declarations — no `task_id` branches in shared infra. Ball surfaces Hits, Gauge surfaces Score, RIT triggered-ball / experience-sampling surface cumulative trigger and prompt counts, and tasks declaring nothing hide the row entirely instead of showing a stale "Hits: --".
- **Slope-based trend option for RIT trigger detection** as an alternative to threshold-based triggering.
- **Brain slice live view during task-localizer runs**: task and rest localizers now share the same monitoring panel, so task runs get the live brain slice alongside the task mirror and QC plots.
- **Auto-computed localizer `n_volumes` for variable-duration tasks**: localizer setup derives the volume count from task duration instead of requiring a manual entry.
- **Save prompt before starting a session**: clicking Start Session with unsaved study-editor edits now opens a Cancel / Start without Saving / Save and Start dialog instead of silently using the in-memory config while the on-disk YAML stays stale.
- **Per-participant RIT memory prompts** with randomized trial order.

### Changed

- **Default localizer `collection_speed`**: 10x → 1x.
- **Default localizer `n_volumes`**: 200 → 250.
- **Test suite restructured by source module** (`tests/{tasks,gui,gui/components,gui/screens,integration}/` mirroring `src/pineuro/`), with 174 new tests covering surfaces that previously had no direct test file. 519 → 690 passing unit tests.

### Fixed

- **Windows `os_key` simulator trigger silently dropped**: `SendInput` was returning 0 for both key-down and key-up because the ctypes `INPUT` struct's union only contained `KEYBDINPUT`, making `sizeof(INPUT) = 32` bytes; the real Win32 struct's union also carries the larger `MOUSEINPUT` and is 40 bytes, and `SendInput` rejects any `cbSize` mismatch. Added `MOUSEINPUT` and `HARDWAREINPUT` to the union so sizes match.

## [0.6.0] - 2026-05-04

### Fixed

- **Scanner trigger reliability**: Real scans occasionally never received the scanner's trigger keypress, leaving the task stuck on "Waiting for scanner...". Two intertwined causes are addressed.
  - The unfocused-window key-polling fallback (Windows `GetAsyncKeyState`, Linux `XQueryKeymap`, macOS `CGEventSourceKeyState`) had a one-shot seed-skip that silently dropped the very first detected press. Scanner triggers are short (~50-100ms) and the polling loop runs every 10ms, so the first detection was usually exactly the trigger itself. Replaced the seed-skip flag with a snapshot of currently-held keys taken at reset: a key already held must be released before its next press fires; a fresh press always fires.
  - The PsychoPy task window was never explicitly brought to the OS foreground when launched from the GUI, so pyglet's focused-event path silently failed unless the user manually clicked the task window. Added a best-effort window activation at the start of `wait_for_trigger`, including the `AttachThreadInput` Win32 trick that bypasses Win10/11's foreground-stealing block.
- **Diagnostics for trigger failures**: `wait_for_trigger` now logs the trigger keys, window handle, foreground state, keys held at reset, and which path delivered the trigger (`focused_event` / `poll_win32` / `poll_x11` / `poll_quartz` / `simulator_*`). Warns when a configured trigger key is missing from the active platform's keymap.

### Changed

- **Simulator trigger now exercises the real-scan code path**: previously the simulator bypassed the keyboard entirely by directly appending to PsychoPy's internal `_keyBuffer`, so simulator runs could never reproduce real-scan trigger failures. The simulator now routes through the same focus-dependent path as a real scanner, controlled by a new `scan.simulator_trigger_mode` setting in the study editor:
  - `manual` (default): simulator waits for the operator to press the trigger key, identical to a real scan.
  - `os_key`: simulator injects a synthetic OS-level keypress via `SendInput` (Windows), `Xlib.ext.xtest.fake_input` (X11), or `CGEventPost` (macOS). Pure-Wayland sessions and macOS without Accessibility permission automatically fall back to manual mode with a clear warning.

## [0.5.0] - 2026-04-27

### Added

- **Simulator-mode banner**: A persistent red banner under the participant workflow header now reads "Simulator mode — replaying recorded DICOMs (no live scanner)" whenever `scan.simulator_enabled` is true, so the operator cannot mistake replayed fixtures for a live scan.

### Fixed

- **Renamed studies opening blank**: After renaming a study and saving, reopening it no longer presents a blank editor — the workflow now reloads from the new on-disk study path instead of the stale in-memory handle.
- **Header subtitle simulator hint**: The " | Simulator Mode" suffix in the workflow header was driven by a stale `collection_speed > 1.0` heuristic that disagreed with the canonical `simulator_enabled` flag used elsewhere; both now read from the same flag.

## [0.4.0] - 2026-04-13

### Added

- **Scanner preflight check**: Before each localizer, reference, or feedback collection, `test_scanner_connection` runs in a background thread; if it fails, the operator is prompted with a proceed-anyway warning
- **Persistent Scanner Connection panel**: The panel no longer disappears across step refreshes when simulator mode is off, so connection status stays visible throughout a session

### Changed

- **Localizer volume collection**: `collect_volumes` no longer enforces an elapsed-time timeout — it waits indefinitely for the expected number of volumes and exits only via `stop_event` (user abort) or `task_done_event` (variable-duration task completion). Prevents spurious `TimeoutError`s from slow or stalled scanners mid-run.

### Fixed

- **Held-key screen skipping**: The READY slide in the ball task (and any slide/trigger-wait reached while a key was still physically held) no longer dismisses in a single frame. `_reset_poll_state()` now arms a one-shot seed so the user must release and re-press before the next screen advances.
- **Sham hemisphere-split roles**: In alternative-masks sham mode, the task's non-target role was falling through to the real non-target network instead of the contralateral hemisphere produced by `hemisphere_split`. Activation mode now uses the full sham pair (connectivity mode was already correct), and sham entries that collide with a real mask name are deduped when building the combined mask list instead of silently dropped.

## [0.3.0] - 2026-04-03

### Changed

- **Unified acquisition dialog**: Run selection and acquisition selection merged into a single dialog that lists all acquisitions grouped by run, each with a checkbox

### Fixed

- **Default mask order**: ContA (target) is now correctly placed as the positive mask
- **Collection timeout**: Timeout clock no longer counts pre-trigger wait time, preventing premature TimeoutErrors late in acquisition
- **Ball task display**: Instructions no longer cut off on narrow aspect ratio displays (e.g., 4:3 scanner projectors)

## [0.2.0] - 2026-03-26

### Added

- **Real-time motion monitoring for localizers**: FD, DVARS, and tSNR plots now display during all localizer collection runs (rest and task-based), using the existing `CollectionQCMonitor` infrastructure
- **Per-localizer motion correction config**: Each localizer tab in the study editor now has its own motion correction dropdown (Siemens MoCo / ANTsPy / None)
- **Flexible session modes**: Studies can now have 0 feedback runs (localizer-only, for motion monitoring) or 0 localizer runs (MNI atlas masks via real-time reference, no coregistration needed)
- **Validation**: Study editor prevents saving configs with both 0 localizers and 0 feedback runs
- **mypy `import-untyped` suppression**: Added `disable_error_code` for installed packages without type stubs (psychopy, pandas)

### Fixed

- **Activation plot colors**: Target network is now red and Non-Target is blue (was previously reversed)
- **Software motion correction in QC monitor**: Siemens mosaic DICOMs (2D tiled images) are now properly unpacked to 3D volumes before ANTsPy rigid registration — previously FD was always 0 when using software MoCo
- **Brain slice flicker**: Fixed race condition where raw mosaic DICOM and properly unpacked single-slice images alternated in the localizer monitoring panel
- **Localizer-only UI**: Mask Extraction and Feedback Tasks panels are now hidden when no feedback runs are configured

## [0.1.0] - 2026-03-23

Initial beta release.

### Added

- **Core pipeline**: DICOM-to-NIfTI conversion (dcm2niix), motion correction (Siemens MOCO / ANTsPy software / none), brain extraction, spatial coregistration
- **Mask extraction**: ICA-based (resting-state) and GLM-based (task localizer) network mask identification
- **Manual ROI import**: Support for native, anatomical, and MNI space masks with automatic registration
- **Atlas support**: Yeo 7/17, Schaefer 100/400, Harvard-Oxford Cortical, AAL — on-demand label fetching with caching
- **Real-time feedback**: Incremental GLM (Gentleman's algorithm) with configurable nuisance regressors (drift, 6 motion parameters, WM mean, CSF mean)
- **Feedback modes**: Activation, connectivity (sliding-window correlation), differential (configurable ROI pair), intermittent (mean over N volumes)
- **Real-time QC**: Framewise displacement with in-run alerts, DVARS (RMS temporal derivative), tSNR (Welford's online algorithm) — all logged per-volume
- **Spatial smoothing**: Optional real-time Gaussian smoothing with configurable FWHM
- **Task framework**: PsychoPy-based plugin system with `@pineuro_task` decorator and `register_task()` for external tasks
- **Built-in tasks**: Ball (dual-network, adaptive difficulty), Thermometer, Gauge, Fixation, Self-Other localizer, RIT localizer, RIT experience sampling, RIT triggered ball
- **Session orchestration**: Multi-run, multi-session support with persistent session state and numbered output directories
- **Sham/control conditions**: Replay-based and synthetic sham providers, randomization generator, blind operator mode
- **Volume simulator**: Offline replay at configurable TR intervals for development and testing
- **Flet GUI**: Study editor (6-tab configuration), participant workflow, real-time monitoring (activation plot, FD plot, DVARS chart, tSNR indicator, task mirror, QC viewer)
- **Cross-platform support**: Windows, macOS, Linux — all dependencies are pure pip
- **CI/CD**: GitHub Actions pipeline (lint, typecheck, build, test matrix across 3 OS x 2 Python versions)
- **Test suite**: ~500 unit tests plus integration tests using NKI Rockland data
- **Documentation**: End-to-end tutorial, user guide, contributing guide
