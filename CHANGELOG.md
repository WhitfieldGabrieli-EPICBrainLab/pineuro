# Changelog

All notable changes to pineuro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **Localizer-only UI**: Mask Extraction and Feedback Session panels are now hidden when no feedback runs are configured

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
- **Built-in tasks**: Ball (dual-network, adaptive difficulty), Thermometer, Gauge, Fixation, Self-Other localizer, RNT localizer, RNT experience sampling, RNT triggered ball
- **Session orchestration**: Multi-run, multi-visit support with persistent session state and numbered output directories
- **Sham/control conditions**: Replay-based and synthetic sham providers, randomization generator, blind operator mode
- **Volume simulator**: Offline replay at configurable TR intervals for development and testing
- **Flet GUI**: Study editor (6-tab configuration), participant workflow, real-time monitoring (activation plot, FD plot, DVARS chart, tSNR indicator, task mirror, QC viewer)
- **Cross-platform support**: Windows, macOS, Linux — all dependencies are pure pip
- **CI/CD**: GitHub Actions pipeline (lint, typecheck, build, test matrix across 3 OS x 2 Python versions)
- **Test suite**: ~500 unit tests plus integration tests using NKI Rockland data
- **Documentation**: End-to-end tutorial, user guide, contributing guide
