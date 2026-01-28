# Pipeline Overview

pineuro implements a complete real-time fMRI neurofeedback pipeline. This page explains how the pieces fit together.

## The Big Picture

A typical neurofeedback study follows this workflow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SETUP PHASE                                  │
│  (Usually one session, before neurofeedback training)               │
│                                                                      │
│  1. Collect Resting-State     →  2. Preprocessing   →  3. Mask      │
│     (5-10 min scan)               (motion, brain       Extraction   │
│                                    extraction)          (ICA-based) │
└─────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                                  │
│  (Multiple sessions over days/weeks)                                │
│                                                                      │
│  4. 2-Volume Coregistration  →  5. Real-Time      →  6. Display    │
│     (Align to current head       Feedback             & Task        │
│      position)                   (GLM estimation)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Pipeline Components

### 1. Resting-State Collection

**Purpose:** Acquire fMRI data for extracting participant-specific network masks.

**What happens:**
- Scanner exports DICOM files to a watch directory
- pineuro collects DICOMs as they arrive
- Typically 200-300 volumes (5-10 minutes)

**Code:**
```python
from pineuro import collect_volumes

dicom_folder = collect_volumes(
    watch_dir='/scanner/incoming',
    output_dir='/data/sub-01/raw/rest',
    n_volumes=200,
)
```

### 2. Preprocessing

**Purpose:** Prepare fMRI data for analysis by correcting artifacts.

**What happens:**
1. **DICOM → NIfTI**: Convert scanner format to analysis format
2. **Motion Correction**: Align all volumes to a reference (mcflirt)
3. **Brain Extraction**: Create mask excluding non-brain tissue (bet)
4. **Coregistration**: Align multiple runs if present

**Code:**
```python
from pineuro import preprocessing

preproc_file, mask_file, ref_file = preprocessing(
    participant_id='sub-01',
    task='rest',
    root_dir='/data/sub-01/preproc',
    input_dicoms=dicom_folder,
)
```

**Outputs:**
```
preproc/
├── 01_nifti/               # Converted NIfTI files
├── 02_motion_correction/   # Motion-corrected data + .par files
├── 03_brain_extraction/    # Temporal means + brain mask
├── 04_coregistered/        # Multi-run alignment (if >1 run)
├── sub-01_task-rest_preproc.nii.gz      # Final masked output
├── sub-01_task-rest_preproc_mask.nii.gz # Brain mask
├── sub-01_task-rest_boldref.nii.gz      # BOLD reference
└── qc/                     # QC report
```

### 3. Mask Extraction

**Purpose:** Identify participant-specific functional network regions.

**What happens:**
1. **ICA Decomposition**: Find spatially independent patterns (CanICA)
2. **Atlas Registration**: Align Yeo atlas to participant space
3. **Network Matching**: Correlate ICA components with atlas networks
4. **Thresholding**: Create binary masks from top-loading voxels

**Why ICA-based masks?**
- Group atlases represent population averages
- Individual brains vary in network topography
- ICA captures participant's actual network boundaries

**Code:**
```python
from pineuro import mask_extraction_rest

mask_files = mask_extraction_rest(
    participant_id='sub-01',
    root_dir='/data/sub-01/masks',
    input_file=preproc_file,
    mask_file=mask_file,
    ref_file=ref_file,
    yeo_targets=['DefaultA', 'ContA'],
)
```

**Outputs:**
```
masks/
├── 01_ica/                 # ICA components + correlations
├── 02_mnireg/              # MNI registration + atlas in subject space
├── 03_masks/               # Network masks + selection metadata
│   ├── sub-01_DefaultA_mask.nii.gz
│   ├── sub-01_ContA_mask.nii.gz
│   └── mask_selections.json
└── qc/                     # QC report with overlays
```

### 4. 2-Volume Coregistration

**Purpose:** Align masks to participant's current head position.

**Why needed:**
- Between sessions, head position changes
- Masks from session 1 won't align with session 2 data
- Quick 2-volume scan provides registration target

**What happens:**
1. Acquire 2 EPI volumes at start of new session
2. Average to create registration target
3. Compute transform from original reference
4. Apply inverse transform to masks

**Code:**
```python
from pineuro import coregister_2vol

aligned_masks = coregister_2vol(
    participant_id='sub-01',
    root_dir='/data/sub-01/feedback',
    input_dicom='/scanner/2vol_dicoms',
    network_files=list(mask_files.values()),
    ref_file=ref_file,
)
```

### 5. Real-Time Feedback

**Purpose:** Compute moment-to-moment brain activation for feedback.

**What happens:**
1. **Volume arrives** from scanner
2. **Motion extraction** from DICOM headers
3. **GLM design row** built (task + nuisance regressors)
4. **Incremental GLM update** using Gentleman's algorithm
5. **Activation computed** for each mask region
6. **Results streamed** to display task

**The GLM Model:**
```
BOLD = β₀ + β₁·drift + β₂₋₇·motion + β₈·task + error
        └────────── nuisance ──────────┘   └─ signal ─┘
```

- Nuisance is subtracted from signal
- Activation = (signal - nuisance) / running_std

**Code:**
```python
from pineuro import run_feedback_session

results = run_feedback_session(
    mask_paths=aligned_masks,
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/data/sub-01/feedback',
    incoming_dir='/scanner/incoming',
    n_baseline=25,
    n_volumes=200,
)
```

### 6. Display Task

**Purpose:** Show feedback to participant in real-time.

**What happens:**
- Task receives activation values from coordinator
- Renders visual feedback (ball, thermometer, etc.)
- Logs responses and timing

**Built-in tasks:**
- **BallTask**: Dual-network ball feedback with adaptive scaling
- **ThermometerTask**: Single-network thermometer display

**Custom tasks:**
- Extend `BaseTask` class
- Register with task framework
- Receive activations via callback

See [Creating Tasks](../task-development/creating-tasks.md) for details.

## Data Flow Summary

```
Scanner → collect_volumes() → preprocessing() → mask_extraction_rest()
                                                          ↓
                                                    Network Masks
                                                          ↓
Scanner → coregister_2vol() → run_feedback_session() → Display Task
                                      ↓
                                  TSV Logs
```

## Timing Considerations

- **TR**: Typically 1-2 seconds
- **Processing latency**: ~100-200ms per volume
- **Display latency**: ~16ms (60 Hz refresh)
- **Total feedback delay**: ~1-2 TRs after neural event

The incremental GLM achieves O(p²) complexity per volume (vs O(np²) for full recomputation), keeping processing within a single TR.

## Next Steps

- [Preprocessing Details](preprocessing.md)
- [Mask Extraction Details](mask-extraction.md)
- [Feedback Session Details](feedback-sessions.md)
