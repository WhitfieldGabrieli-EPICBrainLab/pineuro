# First Session Tutorial

This tutorial walks you through a complete neurofeedback session using pineuro's GUI. You'll preprocess resting-state data, extract participant-specific network masks, and run a simulated real-time neurofeedback session — all without a scanner.

We use the NKI Rockland Sample neurofeedback dataset as example data, but the same workflow applies to any DICOM dataset.

## Prerequisites

- **pineuro installed** — see [Installation](installation.md)
- **Sample DICOM data** — This tutorial uses DICOMs from the Enhanced NKI Rockland Sample, available through the [INDI data portal](https://fcon_1000.projects.nitrc.org/indi/enhanced/). Each subject should have three scan directories:
  - `rest_1400/` — Resting-state scan (TR=1.4s, ~404 volumes)
  - `mask/` — 2-volume reference scan (TR=2.0s, 2 volumes)
  - `dmn_tracking_test/` — Neurofeedback run (TR=2.0s, ~412 volumes)

## Step 1: Create a Source Mapping File

The source mapping tells pineuro's simulator where to find DICOMs for each participant. Create a tab-separated file (e.g., `subject_mapping.tsv`):

```
subject	rest1	2vol_realtime	feedback_run-01
sub-rockland01	/path/to/A00059346/rest_1400	/path/to/A00059346/mask	/path/to/A00059346/dmn_tracking_test
sub-rockland02	/path/to/A00059428/rest_1400	/path/to/A00059428/mask	/path/to/A00059428/dmn_tracking_test
```

Each row maps a participant ID to the DICOM directories for that subject. Column names must match exactly:

| Column | Purpose |
|--------|---------|
| `subject` | Participant ID you'll enter in the GUI |
| `rest1` | Resting-state DICOM directory (localizer) |
| `2vol_realtime` | 2-volume reference scan directory |
| `feedback_run-01` | Feedback run DICOM directory |

## Step 2: Create a Study

Launch the GUI:

```bash
pineuro-gui
```

Click **"+ New Study"** to open the Study Editor. Configure each tab:

### General Tab

- **Name**: `Rockland NFB`
- **Description**: A brief description of your study

### Scan Tab

Configure the scan parameters to match the Rockland dataset:

- **Localizer**: TR = `1.4`, Volumes = `404`, Runs = `1`, Task Type = `rest`
- **Feedback**: TR = `2.0`, Volumes = `412`
- **2-Volume Reference**: Enable "Real-time 2vol" (disable "Initial 2vol" — Rockland only has one reference scan)
- **Simulator**: Check "Enable Simulator", set source mapping to your `subject_mapping.tsv`, Collection Speed = `10.0` (10x real-time)
- **Feedback Simulator Speed**: `1.0` (real-time) for a realistic test, or `10.0` to run quickly

### Mask Tab

- **Network Atlas**: Yeo 7-network
- **Target Networks**: `Cont`, `Default` (Control and Default Mode networks)
- **ICA Components**: `20`
- **Threshold**: `2000` voxels per mask

### Feedback Tab

- **Baseline Volumes**: `15` (30 seconds at TR=2.0)
- **Motion Correction**: `software` (ANTsPy-based, works with any scanner)
- **Nuisance Regressors**: Enable drift, motion, WM, and CSF regressors
- **Differential Pair**: `Cont` - `Default` (feedback = CEN activation minus DMN activation)

### Task Tab

- **Task Type**: `gauge`
- **Max Feedback Runs**: `1`
- Configure the gauge task parameters:
  - **Block Durations**: `30,90,60,30,90,60,60,90,30,60,90,30`
  - **Conditions**: `wander,focus`
  - **Swap Labels**: Enabled, every `3` blocks
  - **Baseline Duration**: `30.0` seconds
  - **Transfer Runs**: Disable (not needed for this tutorial)

Click **Save**.

## Step 3: Run a Participant

From the Study Manager, click your study card and enter the participant ID: `sub-rockland01`. Click **Start** to open the Participant Workflow.

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

### Phase 4: 2-Volume Reference + Coregistration

1. **2vol Collection** — Simulator copies the 2-volume reference scan
2. **Mask Coregistration** — Aligns network masks to the current head position using the 2-volume reference

### Phase 5: Feedback Run

The feedback session starts. On the operator monitor, you'll see real-time plots of:
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

After the session completes, the output directory for `sub-rockland01` contains:

```
data/sub-rockland01/
├── collection/                          # Raw collected DICOMs
│   ├── task-rest_acq-01_run-01/         # Resting-state DICOMs
│   ├── task-2vol_acq-01_run-02/         # 2-volume reference DICOMs
│   └── feedback_acq-01_run-01/          # Feedback run DICOMs
│
├── preprocessing/
│   ├── 01_nifti/                        # Converted NIfTI files
│   ├── 02_motion_correction/            # Motion-corrected data
│   ├── 03_brain_extraction/             # Brain mask + mean image
│   ├── sub-rockland01_task-rest_preproc.nii.gz       # Final preprocessed 4D
│   ├── sub-rockland01_task-rest_preproc_mask.nii.gz  # Brain mask
│   ├── sub-rockland01_task-rest_boldref.nii.gz       # BOLD reference
│   └── qc/
│       └── sub-rockland01_task-rest_preprocessing_qc.html
│
├── mask_extraction/
│   ├── 01_ica/                          # ICA components
│   ├── 02_mnireg/                       # MNI registration transforms + Yeo atlas in subject space
│   ├── 03_masks/                        # Network masks
│   │   ├── sub-rockland01_Cont.nii.gz         # Control network (full)
│   │   ├── sub-rockland01_Cont_thr.nii.gz     # Control network (thresholded)
│   │   ├── sub-rockland01_Default.nii.gz      # DMN (full)
│   │   ├── sub-rockland01_Default_thr.nii.gz  # DMN (thresholded)
│   │   └── mask_selections.json               # ICA component → network mapping
│   └── 05_tissue/                       # WM and CSF masks for nuisance regression
│
├── feedback/
│   ├── 01_2vol/                         # 2-volume reference + coregistered masks
│   │   ├── masks/                       # Network masks aligned to feedback session
│   │   └── xfm/                         # Registration transforms
│   └── 02_feedback_output/
│       ├── sub-rockland01_task-feedback_acq-01_run-01_feedbacklog.tsv   # Per-volume activations
│       ├── sub-rockland01_task-feedback_acq-01_run-01_design.tsv        # GLM design matrix
│       ├── sub-rockland01_task-feedback_acq-01_run-01_psychopylog.tsv   # Task display state
│       ├── sub-rockland01_task-feedback_acq-01_run-01_events.tsv        # Block events
│       ├── session_state.json                                            # Session progress
│       └── qc/
│           └── sub-rockland01_task-feedback_acq-01_run-01_feedback_qc.html
│
├── qc/                                  # Master QC directory
│   ├── sub-rockland01_mask_extraction_qc.html  # Mask QC with overlay images
│   └── overlays/                        # Mask overlay PNGs (subject, 2vol, alternatives)
│
└── logs/                                # Session logs
```

### Key Output Files

**Feedbacklog** (`feedbacklog.tsv`) — One row per volume with columns:
- `time` — Seconds since session start
- `is_baseline`, `is_outlier` — Volume classification
- `fd`, `dvars`, `tsnr` — Quality metrics
- `condition` — Current block condition (real/baseline)
- `{network}_activation` — Activation value per network
- `differential` — CEN minus DMN activation (the feedback signal)

**Psychopylog** (`psychopylog.tsv`) — One row per display frame (~60 Hz) with:
- Task condition (wander/focus/rest)
- Gauge score and needle position
- Whether feedback is active
- Label swap state

**QC Reports** — HTML reports with embedded plots for preprocessing quality, mask selection overlays, and feedback activation timeseries.

## Step 5: Next Steps

- **Run more participants** — Enter a different participant ID (e.g., `sub-rockland02`) and run the same pipeline. The source mapping automatically selects the correct DICOMs.
- **Adjust parameters** — Edit the study to change ICA components, threshold values, or task settings. Re-run with `overwrite=False` to skip completed steps.
- **Multi-visit studies** — Enable multi-visit mode in the study editor to reuse masks across sessions with 2-volume coregistration.
- **Sham conditions** — Enable sham mode to deliver alternative mask feedback or replay data for control participants.

See the [Pipeline Overview](../user-guide/pipeline-overview.md) for details on each processing stage.
