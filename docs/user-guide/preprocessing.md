# Preprocessing

Preprocessing converts raw DICOM data into analysis-ready NIfTI volumes. pineuro uses a four-step pipeline with numbered output directories for easy tracing.

## Pipeline Steps

Intermediate filenames (Steps 1–4) include both the run and the acquisition attempt as `run-{NN}_attempt-{AA}`. This lets you re-preprocess a different attempt of the same run (e.g. after re-collecting due to motion) without colliding with the prior attempt's files, and keeps `{1: [2, 3]}`-style multi-attempt-per-run selections distinguishable on disk. The final `_preproc`, `_preproc_mask`, and `_boldref` outputs do not carry `attempt-` — they represent the canonical merged result for `(participant, session, task)`.

### Step 1: DICOM to NIfTI (`01_nifti/`)

Converts DICOM files to compressed NIfTI format using dcm2niix. Also produces a JSON sidecar with acquisition parameters (TR, echo time, slice timing, etc.).

**Output:**
- `sub-01_ses-01_task-rest_run-01_attempt-01.nii.gz` — 4D NIfTI timeseries
- `sub-01_ses-01_task-rest_run-01_attempt-01.json` — Acquisition metadata

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
- `sub-01_ses-01_task-rest_run-01_attempt-01_mcf.nii.gz` — Motion-corrected 4D
- Motion parameter files (`.par` format: 3 translations + 3 rotations per volume)

### Step 3: Brain Extraction (`03_brain_extraction/`)

Creates a binary brain mask from the temporal mean of the first input with ANTsPy's `get_mask()`. Voxels above the intensity threshold, a fraction of the image mean over the whole field of view (default 0.5), are kept. A morphological cleanup then erodes the mask by the cleanup radius (default 2 voxels), keeps the largest connected piece, grows it back by the same radius and fills holes. The final `_preproc_mask` is eroded by 2 more voxels to drop partial-volume edge voxels.

Raising the intensity threshold excludes more dim non-brain tissue. Raising the cleanup radius detaches tissue joined to the brain by a narrow connection. Both have study defaults on the Localizer tab, and an operator can adjust them for a participant after reviewing the mask (see [Reviewing the Brain Mask](#reviewing-the-brain-mask)).

**Output:**
- `sub-01_ses-01_task-rest_run-01_attempt-01_mean.nii.gz` — Temporal mean image
- `sub-01_ses-01_task-rest_run-01_attempt-01_mean_brain.nii.gz` — Skull-stripped mean
- `sub-01_ses-01_task-rest_run-01_attempt-01_mean_brain_mask.nii.gz` — Binary brain mask

### Step 4: Multi-Run Coregistration (`04_coregistered/`)

When multiple inputs are processed, this step aligns each to the first input's coordinate space using ANTs rigid registration, then concatenates into a single 4D timeseries.

**Output (only when >1 input):**
- `sub-01_ses-01_task-rest_run-02_attempt-01_to_run-01_attempt-01.nii.gz` — Per-input aligned data (filename encodes both source and reference `run-NN_attempt-NN`)
- Concatenated 4D timeseries for mask extraction

## Final Preprocessing Outputs

After all steps complete, three summary files are produced at the preprocessing root:

```
preprocessing/
├── 01_nifti/
├── 02_motion_correction/
├── 03_brain_extraction/
├── 04_coregistered/                            # Only if >1 input
├── sub-01_ses-01_task-rest_preproc.nii.gz     # Final masked 4D timeseries
├── sub-01_ses-01_task-rest_preproc_mask.nii.gz  # Brain mask
├── sub-01_ses-01_task-rest_preproc_mask.json    # How the brain mask was made
├── sub-01_ses-01_task-rest_boldref.nii.gz     # BOLD reference image
└── qc/
    └── sub-01_ses-01_task-rest_preprocessing_qc.html
```

The `_preproc_mask.json` record (written from pineuro 1.1) holds the brain-mask parameters (`threshold_fraction`, `cleanup`, `n_erode`), the files the mask was made from (`source_mean`, `source_timeseries`, relative to `preprocessing/`), whether an operator adjusted it (`adjusted_by_operator`), and when it was written.

## Reviewing the Brain Mask

After preprocessing completes, the participant workflow opens the **Review Brain Mask** window. It is also available from the **Review Brain Mask** button under the preprocessing step. The window shows the mask outline on the unmasked mean EPI in several slices per plane, so both non-brain tissue the mask kept and brain it cut off are visible.

- Change the intensity threshold or the cleanup radius and click **Recompute** to preview the mask. Nothing is written.
- **Accept** with changed values asks for confirmation, then recomputes only the brain mask, the masked timeseries and the BOLD reference. DICOM conversion, motion correction and coregistration are not repeated. The localizer's mask extraction results and the coregistered masks are deleted and mask extraction runs again, so the masks need review and approval again, and coregistration runs again.
- **Close**, or **Accept** with unchanged values, keeps the mask.

The operator's values are recorded in `_preproc_mask.json` and reused if preprocessing is re-run for that participant. Participants preprocessed before pineuro 1.1 have no record; re-run preprocessing with Overwrite to review their mask. The brain mask cannot be changed once a feedback run of the session has been completed.

## Reference Scans

Two types of reference scans are supported:

**Initial Reference** — Collected at the very start of the session, before any localizer. Provides a geometric reference for preprocessing. Useful when the localizer scan geometry differs from the feedback scan.

**Real-time Reference** — Collected at the start of the feedback session (potentially on a different day). Used to coregister masks from the localizer session to the current head position.

Both are configured on the Study Editor's **Study** tab. The reference scan (typically 2-4 volumes) is averaged to create a mean image, and ANTs registration computes the transform needed to align masks.

## Overwrite Behavior

All preprocessing steps default to `overwrite=False`. If output files already exist, the step is skipped. This enables efficient re-runs when only later stages need recomputation. Existing outputs are kept whatever parameters they were made with, so changing the brain-mask defaults affects participants preprocessed afterwards (or re-runs with Overwrite); use the brain-mask review to change one participant's mask.

With `overwrite=True`, Step 1 also deletes any prior dcm2niix outputs for the current `(run, attempt)` label before re-running — including the primary file, collision-suffixed variants (`...a.nii.gz`, `...b.nii.gz` that dcm2niix produces under its default `-w 2` behavior), and multi-echo / phase splits (`..._e1.nii.gz`, `..._ph.nii.gz`). This prevents stale files from accumulating across re-runs. Cleanup is scoped to the current label only; siblings (other runs / other attempts) are untouched.

## Configuration

| Parameter | GUI Location | Values |
|-----------|-------------|--------|
| Motion correction mode | Feedback tab → Motion Correction | `software`, `siemens_moco`, `none` |
| Localizer TR | Localizer tab → TR | e.g., `1.4` |
| Localizer volumes | Localizer tab → Volumes | e.g., `404` |
| Number of runs | Localizer tab → Runs | e.g., `1` or `2` |
| Brain mask intensity threshold | Localizer tab → Brain Mask | `preprocessing.brain_mask_threshold_fraction`, × image mean (default `0.5`) |
| Brain mask cleanup | Localizer tab → Brain Mask | `preprocessing.brain_mask_cleanup`, `0` to `4` voxels (default `2`) |
| Reference initial | Study tab → Initial Reference | Enable/disable |
| Reference real-time | Study tab → Real-time Reference | Enable/disable |
