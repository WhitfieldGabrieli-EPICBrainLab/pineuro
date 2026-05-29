# QC Reports & Output Files

pineuro generates HTML quality control reports at each pipeline stage and logs per-volume data in standardized TSV files. All reports are static HTML with embedded images — no server or JavaScript required.

## Preprocessing QC

**File:** `preprocessing/qc/sub-001_task-rest_preprocessing_qc.html`

Contains:
- **Motion parameter timeseries** — 6 DOF (3 translations, 3 rotations) across all volumes
- **Framewise displacement** — Per-volume FD with 0.5mm threshold line
- **Brain mask coverage** — Overlay of the brain mask on the mean image to verify skull stripping quality

## Mask Extraction QC

**File:** `qc/sub-001_mask_extraction_qc.html`

Contains:
- **Correlation heatmap** — Spatial correlation between each ICA component and each atlas network. Shows which component was selected for each target.
- **Selected mask overlays** — Thresholded masks overlaid on the participant's anatomy in axial/coronal/sagittal views
- **Alternative component overlays** — All other ICA components for each target network, so you can verify the auto-selection or identify a better match
- **Reference coregistered overlays** — Masks after coregistration to the feedback task reference, verifying alignment

The Participant Workflow GUI also shows overlay thumbnails for quick visual verification before proceeding to feedback.

## Feedback Tasks QC

**File:** `feedback/02_feedback_output/qc/sub-001_ses-01_task-ball_run-01_attempt-01_feedback_qc.html`

Contains:
- **Activation timeseries** — Per-volume activation for each network mask, with baseline period shaded
- **FD timeseries** — Framewise displacement during the feedback task
- **DVARS timeseries** — Standardized signal intensity changes
- **Design matrix heatmap** — The GLM design matrix used for real-time estimation
- **Summary statistics** — Mean activation, motion summary, tSNR, outlier count

## Output TSV Files

### Feedbacklog (`*_feedbacklog.tsv`)

One row per volume. The primary data file for post-hoc analysis.

| Column | Description |
|--------|-------------|
| `time` | Seconds since session start |
| `is_baseline` | Whether this volume is in the baseline period |
| `is_outlier` | Whether FD or DVARS exceeded threshold |
| `fd` | Framewise displacement (mm) |
| `dvars` | Raw DVARS value |
| `tsnr` | Temporal signal-to-noise ratio |
| `condition` | Block condition label (e.g., `real`, `baseline`) |
| `{network}_activation` | Activation z-score for each network mask |
| `differential` | Difference between differential pair networks (if configured) |

When sham is enabled, additional columns appear:
- `{network}_activation_real` — True activation (always computed)
- `{network}_activation_delivered` — What was shown to the participant

### Design Matrix (`*_design.tsv`)

One row per volume. Contains the full GLM design matrix used for real-time estimation, with one column per regressor (task, intercept, drift, motion parameters, WM/CSF means).

### Psychopylog (`*_psychopylog.tsv`)

One row per display frame (~60 Hz). Records the task display state for precise timing analysis.

Contents vary by task, but typically include:
- Frame timestamp
- Current condition (e.g., wander/focus/rest/baseline)
- Score or activation value being displayed
- Whether feedback is active
- Button press events
- Task-specific state (e.g., `labels_swapped` for gauge, `hit_target` for ball)

### Events (`*_events.tsv`)

BIDS-format event file with one row per block or trial:

| Column | Description |
|--------|-------------|
| `onset` | Block start time in seconds |
| `duration` | Block duration in seconds |
| `trial_type` | Condition label |

### Task State (`task_state.json`)

Persistent state that carries across runs within a session. Includes:
- Run completion status
- Adaptive difficulty parameters (ball task)
- Hit counters and circle sizes
- Current session/run tracking

## Locating Output Files

All outputs follow a consistent directory structure:

```
base_dir/sub-001/
├── preprocessing/qc/          # Preprocessing QC
├── qc/                        # Mask extraction QC + overlays
├── feedback/02_feedback_output/
│   ├── *_feedbacklog.tsv      # Per-volume data
│   ├── *_design.tsv           # GLM design matrix
│   ├── *_psychopylog.tsv      # Task display log
│   ├── *_events.tsv           # Block events
│   ├── task_state.json     # Task state persistence
│   └── qc/                    # Feedback QC report
└── logs/                      # Session logs (timestamped)
```

## Session Logs

Detailed logging is saved to `logs/session_YYYYMMDD_HHMMSS.log` for each session. These logs include:
- Pipeline step execution and timing
- Per-volume processing details
- Error messages and warnings
- Configuration parameters used
