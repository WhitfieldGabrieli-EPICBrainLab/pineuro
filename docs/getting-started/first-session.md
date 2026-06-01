# First Session Tutorial

This tutorial walks you through a complete neurofeedback task using pineuro's GUI. You'll preprocess resting-state data, extract participant-specific network masks, and run a simulated real-time neurofeedback task — all without a scanner.

We use a public resting-state neurofeedback dataset as example data, but the same workflow applies to any DICOM dataset.

## Prerequisites

- **pineuro installed** — see [Installation](installation.md)
- **Sample DICOM data** — This tutorial uses DICOMs from a public resting-state neurofeedback dataset. Each subject should have three scan directories:
  - `rest_1400/` — Resting-state scan (TR=1.4s, ~404 volumes)
  - `mask/` — Reference scan (TR=2.0s, typically 2-4 volumes)
  - `dmn_tracking_test/` — Neurofeedback run (TR=2.0s, ~412 volumes)

## Step 1: Create a Source Mapping File

The source mapping tells pineuro's simulator where to find DICOMs for each participant. Create a tab-separated file (e.g., `subject_mapping.tsv`):

```
subject	rest1	reference_realtime	feedback_run-01
sub-01	/path/to/A00059346/rest_1400	/path/to/A00059346/mask	/path/to/A00059346/dmn_tracking_test
sub-02	/path/to/A00059428/rest_1400	/path/to/A00059428/mask	/path/to/A00059428/dmn_tracking_test
```

Each row maps a participant ID to the DICOM directories for that subject. Column names must match exactly:

| Column | Purpose |
|--------|---------|
| `subject` | Participant ID you'll enter in the GUI |
| `rest1` | Resting-state DICOM directory (localizer) |
| `reference_realtime` | Reference scan directory (typically 2-4 volumes) |
| `feedback_run-01` | Feedback run DICOM directory |

## Step 2: Create a Study

Launch the GUI:

```bash
pineuro-gui
```

Click **"+ New Study"** to open the Study Editor. It has study-wide tabs (**Study**,
**Display**, **Sham**) and a per-session **Experiment Design** area where you add
localizers and feedback runs — each with its own editor.

### Study tab (study-wide)

- **Name**: `NFB Demo`
- **Simulator**: enable it, point the source mapping at your `subject_mapping.tsv`, set the
  localizer/collection speed to `10.0` (10× real-time) and the feedback speed to `1.0`
  (real-time) — or `10.0` to finish quickly.
- **Reference scans**: enable the **real-time reference** (leave the initial reference off —
  this dataset has only one reference scan).
- **Base data directory**: where participant data is written.

### Localizer (Experiment Design)

Add one localizer and set:

- **Type**: `rest` (resting-state → ICA)
- **TR**: `1.4`, **Volumes**: `404`, **Runs**: `1`
- **Network atlas**: Yeo 7-network; **Target networks**: `Cont`, `Default` (Control and
  Default Mode networks)
- **ICA components**: `30`; **Threshold**: `2000` voxels per mask

### Feedback run (Experiment Design)

Add one feedback run and set:

- **Task**: `gauge`
- **TR**: `2.0`, **Volumes**: `412`, **Baseline volumes**: `15` (30 s at TR=2.0)
- **Motion correction**: `software` (ANTsPy-based — works with any scanner)
- **Nuisance regressors**: drift, motion, WM, and CSF
- **Feedback mode**: `activation` with differential pair `Cont − Default` (feedback =
  Control minus Default-mode activation)
- **Gauge parameters**: block durations `30,90,60,30,90,60,60,90,30,60,90,30`; conditions
  `wander,focus`; swap labels every `3` blocks; transfer runs off

Click **Save**.

## Step 3: Run a Participant

From the Study Manager, click your study card and enter the participant ID: `sub-01`. Click **Start** to open the Participant Workflow.

The pipeline runs in sequential phases. Click each step to execute it:

### Phase 1: Resting-State Collection

The simulator copies DICOMs from the rest_1400 directory at 10x speed. You'll see a progress bar as 404 volumes are collected. This takes about 1 minute at 10x speed.

### Phase 2: Preprocessing

Three substeps run automatically:

1. **DICOM to NIfTI** — Converts DICOMs using dcm2niix
2. **Motion Correction** — Aligns all volumes to a reference using ANTsPy rigid registration
3. **Brain Extraction** — Creates a brain mask using ANTsPy

This step takes several minutes depending on your hardware.

### Phase 3: Mask Extraction

1. **ICA Decomposition** — Runs CanICA (20 components) on the preprocessed resting-state data
2. **MNI Registration** — Aligns components to MNI space using ANTs SyN registration
3. **Network Selection** — Matches ICA components to Yeo atlas networks via spatial correlation
4. **Thresholding** — Selects the top 2000 voxels per network mask

After this step, you can review the mask QC report to verify the selected components look reasonable.

### Phase 4: Reference Scan + Coregistration

1. **Reference Collection** — Simulator copies the reference scan
2. **Mask Coregistration** — Aligns network masks to the current head position using the reference scan

### Phase 5: Feedback Run

The feedback task starts. On the operator monitor, you'll see real-time plots of:
- **Activation timeseries** — Per-volume CEN and DMN activation
- **FD plot** — Framewise displacement (head motion)
- **DVARS plot** — Signal intensity changes
- **tSNR indicator** — Temporal signal-to-noise ratio

On the participant display (second monitor or same screen), the gauge task shows:
- A gauge/dial displaying the differential activation (CEN - DMN)
- Block instructions alternating between "Wander" and "Focus" conditions
- The needle moves in response to real-time brain activation

At TR=2.0 and 412 volumes, a real-time run takes about 14 minutes. Set the feedback simulator speed to 10x if you want to finish quickly.

## Step 4: Inspect Outputs

After the session completes, the output for `sub-01` (under `ses-01/`) contains:

```
data/sub-01/
└── ses-01/
    ├── collection/                              # Raw collected DICOMs
    │   ├── task-rest_run-01_attempt-01/         # Resting-state DICOMs
    │   ├── task-reference_run-01_attempt-01/    # Reference scan DICOMs
    │   └── task-gauge_run-01_attempt-01/        # Feedback run DICOMs
    │
    ├── preprocessing/
    │   ├── 01_nifti/                            # Converted NIfTI files
    │   ├── 02_motion_correction/                # Motion-corrected data
    │   ├── 03_brain_extraction/                 # Brain mask + mean image
    │   ├── sub-01_ses-01_task-rest_preproc.nii.gz       # Final preprocessed 4D
    │   ├── sub-01_ses-01_task-rest_preproc_mask.nii.gz  # Brain mask
    │   ├── sub-01_ses-01_task-rest_boldref.nii.gz       # BOLD reference
    │   └── qc/
    │       └── sub-01_ses-01_task-rest_preprocessing_qc.html
    │
    ├── mask_extraction/
    │   ├── 01_ica/                              # ICA components
    │   ├── 02_mnireg/                           # MNI registration transforms + Yeo atlas in subject space
    │   ├── 03_masks/                            # Network masks
    │   │   ├── sub-01_Cont.nii.gz             # Control network (full)
    │   │   ├── sub-01_Cont_thr.nii.gz         # Control network (thresholded)
    │   │   ├── sub-01_Default.nii.gz          # DMN (full)
    │   │   ├── sub-01_Default_thr.nii.gz      # DMN (thresholded)
    │   │   └── mask_selections.json           # ICA component → network mapping
    │   └── 05_tissue/                           # WM and CSF masks for nuisance regression
    │
    ├── feedback/
    │   ├── 01_reference/                        # Reference scan + coregistered masks
    │   │   ├── masks/                           # Network masks aligned to feedback task
    │   │   └── xfm/                             # Registration transforms
    │   └── 02_feedback_output/
    │       ├── sub-01_ses-01_task-gauge_run-01_attempt-01_feedbacklog.tsv   # Per-volume activations
    │       ├── sub-01_ses-01_task-gauge_run-01_attempt-01_design.tsv        # GLM design matrix
    │       ├── sub-01_ses-01_task-gauge_run-01_attempt-01_psychopylog.tsv   # Task display state
    │       ├── sub-01_ses-01_task-gauge_run-01_attempt-01_events.tsv        # Block events
    │       ├── task_state.json                                              # Task state
    │       └── qc/
    │           └── sub-01_ses-01_task-gauge_run-01_attempt-01_feedback_qc.html
    │
    ├── qc/                                      # Master QC directory
    │   ├── sub-01_mask_extraction_qc.html      # Mask QC with overlay images
    │   └── overlays/                            # Mask overlay PNGs (subject, reference, alternatives)
    │
    └── logs/                                    # Session logs
```

### Key Output Files

**Feedbacklog** (`feedbacklog.tsv`) — One row per volume. Key columns:
- `volume`, `time` — Volume number and seconds since session start
- `is_baseline`, `is_outlier` — Volume classification
- `fd`, `dvars`, `tsnr` — Quality metrics
- `condition` — Current block condition (real/baseline)
- `{network}_activation` — Activation value per network
- `differential` — Control minus Default-mode activation (the feedback signal)

See [QC Reports & Output Files](../user-guide/qc-reports.md) for the full column list.

**Psychopylog** (`psychopylog.tsv`) — One row per display frame (~60 Hz) with:
- Task condition (wander/focus/rest)
- Gauge score and needle position
- Whether feedback is active
- Label swap state

**QC Reports** — HTML reports with embedded plots for preprocessing quality, mask selection overlays, and feedback activation timeseries.

## Step 5: Next Steps

- **Run more participants** — Enter a different participant ID (e.g., `sub-02`) and run the same pipeline. The source mapping automatically selects the correct DICOMs.
- **Adjust parameters** — Edit the study to change ICA components, threshold values, or task settings. Re-run with `overwrite=False` to skip completed steps.
- **Multi-session studies** — A study declares an ordered list of sessions; a later session can reuse an earlier session's masks (selected per feedback run) with reference-scan coregistration. See [Multi-Session Studies](../user-guide/multi-session.md).
- **Sham conditions** — Enable sham mode to deliver alternative mask feedback or replay data for control participants.

See the [Pipeline Overview](../user-guide/pipeline-overview.md) for details on each processing stage.
