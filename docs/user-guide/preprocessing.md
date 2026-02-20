# Preprocessing

Preprocessing converts raw DICOM data into analysis-ready NIfTI volumes. pineuro uses a four-step pipeline with numbered output directories for easy tracing.

## Pipeline Steps

### Step 1: DICOM to NIfTI (`01_nifti/`)

Converts DICOM files to compressed NIfTI format using dcm2niix. Also produces a JSON sidecar with acquisition parameters (TR, echo time, slice timing, etc.).

**Output:**
- `sub-001_task-rest_run-01.nii.gz` — 4D NIfTI timeseries
- `sub-001_task-rest_run-01.json` — Acquisition metadata

### Step 2: Motion Correction (`02_motion_correction/`)

Aligns all volumes to a reference (typically the first volume or temporal mean).

**Three modes available:**

| Mode | Description | When to Use |
|------|-------------|-------------|
| `software` | ANTsPy rigid registration (6 DOF) | Any scanner; most robust |
| `siemens_moco` | Parse pre-computed motion from DICOM JSON sidecar | Siemens scanners with MoCo enabled |
| `none` | Skip motion correction | Already corrected data, or debugging |

- `software` mode runs ANTsPy rigid registration for each volume. This takes longer but works with any scanner vendor.
- `siemens_moco` reads the motion parameters that the Siemens scanner has already computed, saving processing time.
- When motion correction is disabled, motion regressors are automatically excluded from the GLM.

**Output:**
- `sub-001_task-rest_run-01_mcf.nii.gz` — Motion-corrected 4D
- Motion parameter files (`.par` format: 3 translations + 3 rotations per volume)

### Step 3: Brain Extraction (`03_brain_extraction/`)

Creates a binary brain mask using ANTsPy's `get_mask()` with conservative erosion to exclude non-brain voxels.

**Output:**
- `sub-001_task-rest_run-01_mean.nii.gz` — Temporal mean image
- `sub-001_task-rest_run-01_mean_brain.nii.gz` — Skull-stripped mean
- `sub-001_task-rest_run-01_mean_brain_mask.nii.gz` — Binary brain mask

### Step 4: Multi-Run Coregistration (`04_coregistered/`)

When multiple localizer runs are collected, this step aligns all runs to a common reference using ANTs rigid registration, then concatenates into a single 4D timeseries.

**Output (only when >1 run):**
- `sub-001_task-rest_run-01_to_ref.nii.gz` — Per-run aligned data
- Concatenated 4D timeseries for mask extraction

## Final Preprocessing Outputs

After all steps complete, three summary files are produced at the preprocessing root:

```
preprocessing/
├── 01_nifti/
├── 02_motion_correction/
├── 03_brain_extraction/
├── 04_coregistered/                    # Only if >1 run
├── sub-001_task-rest_preproc.nii.gz    # Final masked 4D timeseries
├── sub-001_task-rest_preproc_mask.nii.gz  # Brain mask
├── sub-001_task-rest_boldref.nii.gz    # BOLD reference image
└── qc/
    └── sub-001_task-rest_preprocessing_qc.html
```

## Reference Scans

Two types of reference scans are supported:

**Initial Reference** — Collected at the very start of the session, before any localizer. Provides a geometric reference for preprocessing. Useful when the localizer scan geometry differs from the feedback scan.

**Real-time Reference** — Collected at the start of the feedback session (potentially on a different day). Used to coregister masks from the localizer session to the current head position.

Both are configured in the Study Editor Scan tab. The reference scan (typically 2-4 volumes) is averaged to create a mean image, and ANTs registration computes the transform needed to align masks.

## Overwrite Behavior

All preprocessing steps default to `overwrite=False`. If output files already exist, the step is skipped. This enables efficient re-runs when only later stages need recomputation. Force recomputation by deleting the relevant output directory.

## Configuration

| Parameter | GUI Location | Values |
|-----------|-------------|--------|
| Motion correction mode | Feedback tab → Motion Correction | `software`, `siemens_moco`, `none` |
| Localizer TR | Scan tab → Localizer TR | e.g., `1.4` |
| Localizer volumes | Scan tab → Volumes | e.g., `404` |
| Number of runs | Scan tab → Runs | e.g., `1` or `2` |
| Reference initial | Scan tab → Initial Reference | Enable/disable |
| Reference real-time | Scan tab → Real-time Reference | Enable/disable |
