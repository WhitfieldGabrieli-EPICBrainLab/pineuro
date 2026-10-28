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
│  4. Reference Coregistration →  5. Real-Time      →  6. Display    │
│     (Align to current head       Feedback             & Task        │
│      position)                   (GLM estimation)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Pipeline Components

Each stage below is exposed as a public function; for runnable end-to-end code see the
[Quick Start scripting example](../getting-started/quickstart.md#scripting-optional).

### 1. Resting-State Collection

**Purpose:** Acquire fMRI data for extracting participant-specific network masks.

**What happens:**
- Scanner exports DICOM files to a watch directory
- pineuro collects DICOMs as they arrive
- Typically 200-300 volumes (5-10 minutes)

### 2. Preprocessing

**Purpose:** Prepare fMRI data for analysis by correcting artifacts.

**What happens:**
1. **DICOM → NIfTI**: Convert scanner format to analysis format
2. **Motion Correction**: Align all volumes to a reference (ANTsPy rigid registration)
3. **Brain Extraction**: Create mask excluding non-brain tissue (ANTsPy)
4. **Coregistration**: Align multiple runs if present

**Outputs:**
```
preprocessing/
├── 01_nifti/               # Converted NIfTI files
├── 02_motion_correction/   # Motion-corrected data + .par files
├── 03_brain_extraction/    # Temporal means + brain mask
├── 04_coregistered/        # Multi-run alignment (if >1 run)
├── sub-01_ses-01_task-rest_preproc.nii.gz      # Final masked output
├── sub-01_ses-01_task-rest_preproc_mask.nii.gz # Brain mask
├── sub-01_ses-01_task-rest_preproc_mask.json   # Brain-mask parameters and sources
├── sub-01_ses-01_task-rest_boldref.nii.gz      # BOLD reference
└── qc/                     # QC report + brain-mask review images
```

See [Preprocessing](preprocessing.md) for the full step-by-step breakdown.

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

**Outputs:**
```
mask_extraction/
├── 01_ica/                 # ICA components + correlations
├── 02_mnireg/              # MNI registration + atlas in subject space
├── 03_masks/               # Network masks + selection metadata
│   ├── sub-01_DefaultA.nii.gz       # full component map
│   ├── sub-01_DefaultA_thr.nii.gz   # thresholded binary mask
│   ├── sub-01_ContA.nii.gz
│   ├── sub-01_ContA_thr.nii.gz
│   └── mask_selections.json
└── 04_reference/           # coregistration outputs (populated later)
```

ICA is one of three extraction modes — see [Mask Extraction](mask-extraction.md) for the
GLM-localizer and MNI-atlas options, supported atlases, and manual ROI import.

### 4. Reference Coregistration

**Purpose:** Align masks to participant's current head position.

**Why needed:**
- Between sessions, head position changes
- Masks from session 1 won't align with session 2 data
- A short reference scan (typically 2-4 volumes) provides registration target

**What happens:**
1. Acquire a short reference scan (typically 2-4 EPI volumes) at start of new session
2. Average to create registration target
3. Compute transform from original reference
4. Apply inverse transform to masks

### 5. Real-Time Feedback

**Purpose:** Compute moment-to-moment brain activation for feedback.

**What happens:**
1. **Volume arrives** from scanner and is read into memory — DICOM pixel data is read directly with pydicom (~15 ms). dcm2niix is **not** used here; it runs only in offline preprocessing, where its ~206 ms per-volume cost would not fit inside a TR.
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

See [Feedback Tasks](feedback-tasks.md) for feedback modes, the design matrix, and
nuisance/normalization details.

### 6. Display Task

**Purpose:** Show feedback to participant in real-time.

**What happens:**
- Task receives activation values from coordinator
- Renders visual feedback (ball, thermometer, etc.)
- Logs responses and timing

**Built-in tasks:**
- **GaugeTask**: Dual-network gauge with block design and label swapping
- **BallTask**: Dual-network ball feedback with adaptive scaling
- **ThermometerTask**: Single-network thermometer display

**Custom tasks:**
- Extend `BaseTask` class
- Register with task framework
- Receive activations via callback

See [Tasks](tasks.md) for details.

## Data Flow Summary

```
Scanner → collect_volumes() → preprocessing() → mask_extraction_rest()
                                                          ↓
                                                    Network Masks
                                                          ↓
Scanner → coregister_reference() → run_feedback_task() → Display Task
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
- [Feedback Task Details](feedback-tasks.md)
