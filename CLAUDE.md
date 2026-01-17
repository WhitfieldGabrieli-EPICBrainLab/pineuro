# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyrtnf is a Python library for real-time fMRI neurofeedback research. It provides a pipeline from DICOM acquisition through preprocessing and network mask extraction, implementing a real-time approach for moment-to-moment BOLD activation estimation.

## Dependencies

- **FSL** (FMRIB Software Library) - Required for preprocessing (mcflirt, bet, fslmaths, fslmerge, convert_xfm)
- **ANTsPy** - Registration (multi-run coreg, MNI alignment, 2vol coreg)
- **dcm2niix** - DICOM to NIfTI conversion
- **Python 3.11+** with: numpy, nibabel, nilearn, scipy, pandas, pydicom, templateflow, psychopy

## Commands

```bash
# Install in development mode
pip install -e .

# Run integration tests
python tests/test.py

# Run DICOM simulator for offline testing
python -m pyrtnf.simulator --source /path/to/dicoms --target /path/to/incoming --tr 1.5 --speed 2.0
```

## Architecture

### Core Processing Pipeline

The library follows a sequential pipeline with numbered output directories (01_, 02_, etc.) for easy tracing:

1. **DICOM Collection** (`preprocessing.py: collect_dicoms()`) - Watches for incoming DICOMs from scanner, copies to output folder. Similar to MURFI's resting_state step. Works for both resting-state and task data collection.

2. **Preprocessing** (`preprocessing.py`) - DICOM conversion, motion correction (mcflirt), brain extraction (bet), multi-run coregistration. FSL commands go through `run_command()` in `utils.py`; registrations now use ANTs helpers.

3. **Mask Extraction** (`mask_extraction.py`) - ICA-based network mask extraction using nilearn's CanICA:
   - Runs CanICA on preprocessed data (20 components default)
   - Registers participant space to MNI152 via ANTs
   - Pulls Yeo atlas (7 or 17 networks) back to participant space using inverse ANTs transforms
   - Identifies network components via spatial correlation
   - Thresholds masks by voxel count (2000 default)
   - `coregister_2vol()` handles registration to 2-volume feedback space

4. **Feedback Estimation** (`feedback.py`) - Real-time activation computation:
   - `IncrementalGLM` - Gentleman's algorithm for online least squares updates with continuous normalization
   - `FeedbackEstimator` - Processes fMRI volumes, fits GLM to remove nuisance signals
   - `NeurofeedbackCoordinator` - Watches for DICOMs, converts to NIfTI, processes through estimator
   - `run_feedback_session()` - High-level function that handles both simulation and real-time modes

5. **Simulation** (`simulator.py`) - `DicomSimulator` copies DICOMs at TR intervals for offline testing. Auto-extracts TR from DICOM headers.

### Quality Control (`qc.py`)

Comprehensive QC visualization system for preprocessing, mask extraction, and feedback stages. All QC functions generate professional HTML reports with embedded plots.

**High-level API** - One function call per stage:
- `generate_preprocessing_report()` - Motion QC, brain mask coverage, registration quality
- `generate_mask_extraction_report()` - ICA components, network correlations, mask overlays
- `generate_feedback_report()` - Activation timeseries, baseline distributions, design matrix

**Core plotting functions**:
- `plot_mask_overlay_tripanel()` - 3-panel (sagittal/coronal/axial) mask overlays on anatomical
- `plot_corr_heatmap()` - Correlation matrix heatmaps (e.g., ICA × Yeo networks)
- `plot_motion_timeseries()` - 6-panel motion parameters (translation + rotation)
- `plot_motion_framewise_displacement()` - FD timeseries with 0.5mm threshold
- `plot_brain_mask_coverage()` - Brain extraction quality check
- `plot_registration_overlay()` - Multi-run alignment validation
- `plot_ica_component_grid()` - 5×2 grid showing top ICA components
- `plot_activation_timeseries()` - Per-mask activation over time
- `plot_baseline_distributions()` - Histogram of baseline normalization
- `plot_design_matrix()` - Heatmap of GLM design matrix

**Integration**: QC is optional with `qc_enabled` parameter (defaults to `False` for feedback, `True` for preprocessing/mask extraction). Reports auto-discover outputs and generate HTML with all visualizations.

**Output structure**:
```
{stage_dir}/qc/
├── {stage}_qc.html              # Main report
├── {plot1}.png                  # Individual plots
├── {plot2}.png
└── overlays/                    # Mask overlays (if applicable)
    ├── subject/
    ├── mni/
    └── 2vol/
```

### Utilities (`utils.py`)

Central command execution, ANTs-backed registration helpers, and DICOM utilities:
- `run_command()` - Validates inputs/outputs, checks command availability, handles overwrite logic
- Registration/masking helpers: `compute_mean()`, `compute_xfm()`, `apply_xfm()`, `compute_xfm_to_mni()`, `apply_mask()`, `spatial_smoothing()`, `temporal_filtering()`
- ANTs-specific helpers: `compute_mean_ants()`, `compute_xfm_ants()`, `apply_xfm_ants()`
- `get_yeo_labels()` - Returns Yeo network labels for 7 or 17 networks
- Uses templateflow for MNI152 BOLD reference template
- DICOM utilities: `DICOM_EXTENSIONS`, `get_dicom_files()`, `is_valid_dicom()`, `get_tr_from_dicom()` - shared by simulator, feedback, and preprocessing modules

## Key Patterns

### FSL Command Wrapper
All external commands go through `run_command()`:
```python
run_command(
    cmd_name='mcflirt',
    cmd=['mcflirt', '-in', str(input), '-out', str(output)],
    description='Motion correction',
    inputs=input,
    outputs=output,
    overwrite=False  # Skip if output exists
)
```

### Overwrite Control
All major functions support `overwrite=False` (default) - skips computation if outputs exist, enabling efficient pipeline re-runs.

### Output Directory Structure
Preprocessing creates numbered directories:
- `01_nifti/`, `02_motion_correction/`, `03_brain_extraction/`, `04_coregistered/`
- Final outputs: `{participant_task}_preproc.nii.gz`, `{participant_task}_preproc_mask.nii.gz`, `{participant_task}_boldref.nii.gz`
- `qc/` - QC report and plots (if `qc_enabled=True`)

Mask extraction:
- `01_ica/`, `02_mnireg/`, `03_masks/`
- `04_2vol/` - Initial 2vol coregistration (run-01, from setup)
- `qc/` - QC report and plots (if `qc_enabled=True`)

Feedback:
- `01_2vol/` - Realtime 2vol coregistration (run-02+, during session)
- `02_feedback_output/` - Feedback run outputs
  - `nifti_task-{task}_run-{run}/` - Converted NIfTI volumes
  - `{subject}_task-{task}_run-{run}_feedbacklog.tsv` - Activation log
  - `{subject}_task-{task}_run-{run}_design.tsv` - GLM design matrix
- `qc/` - QC report and plots (if `qc_enabled=True`)

## Usage Examples

### Collecting Resting-State DICOMs (Real-Time Mode)
```python
from pyrtnf import collect_dicoms, preprocessing

# Step 1: Collect resting-state DICOMs from scanner
# (Similar to MURFI's resting_state step)
dicom_folder = collect_dicoms(
    watch_dir='/scanner/export/incoming',
    output_dir='/data/sub-01/raw/rest',
    n_volumes=200,
    clear_watch_dir=True,  # Like MURFI's rm -rf tmp/murfi_input
)

# Step 2: Run preprocessing on collected data
preproc_file, mask_file, ref_file = preprocessing(
    participant_id='sub-01',
    task='rest',
    root_dir='/data/sub-01/preproc',
    input_dicoms=dicom_folder,
)
```

### Collecting DICOMs (Simulation Mode for Testing)
```python
from pyrtnf import collect_dicoms

# Offline testing with simulated scanner acquisition
dicom_folder = collect_dicoms(
    watch_dir='/tmp/incoming',
    output_dir='/data/sub-01/raw/rest',
    n_volumes=200,
    simulator_source_dir='/path/to/source_dicoms',
    simulator_speed=10.0,  # 10x speed for testing
)
```

### Running a Feedback Session (Simulation Mode)
```python
from pyrtnf import run_feedback_session

# Offline testing with simulated DICOM acquisition
results = run_feedback_session(
    mask_paths=['/path/to/dmn_mask.nii.gz'],
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/path/to/output',
    n_baseline=25,
    n_volumes=150,
    # Enable simulation mode
    simulator_source_dir='/path/to/source_dicoms',
    simulator_speed=10.0,  # 10x speed for testing
)
```

### Running a Feedback Session (Real-Time Mode)
```python
from pyrtnf import run_feedback_session

# Real-time processing with scanner acquisition
results = run_feedback_session(
    mask_paths=['/path/to/dmn_mask.nii.gz'],
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/path/to/output',
    incoming_dir='/scanner/export/dicoms',  # Scanner DICOM directory
    n_baseline=25,
    n_volumes=200,
)
```

### Generating QC Reports

QC reports can be generated automatically during pipeline execution or manually after completion.

**Automatic QC during pipeline execution**:
```python
from pyrtnf import preprocessing, mask_extraction_rest, run_feedback_session

# Preprocessing with QC (enabled by default)
preproc_file, mask_file, ref_file = preprocessing(
    participant_id='sub-01',
    task='rest',
    root_dir='/data/sub-01/preproc',
    input_dicoms='/data/sub-01/raw',
    qc_enabled=True  # Generate QC report (default: True)
)
# QC report saved to: /data/sub-01/preproc/qc/preprocessing_qc.html

# Mask extraction with QC (enabled by default)
mask_files = mask_extraction_rest(
    participant_id='sub-01',
    root_dir='/data/sub-01/mask_extraction',
    input_file=preproc_file,
    mask_file=mask_file,
    ref_file=ref_file,
    qc_enabled=True  # Generate QC report (default: True)
)
# QC report saved to: /data/sub-01/mask_extraction/qc/mask_extraction_qc.html

# Feedback session with QC (disabled by default)
results = run_feedback_session(
    mask_paths=mask_files,
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/data/sub-01/feedback',
    collection_dir='/data/sub-01/collection',
    n_baseline=25,
    n_volumes=200,
    tr=1.2,
    qc_enabled=True  # Generate QC report (default: False)
)
# QC report saved to: /data/sub-01/feedback/qc/sub-01_task-nback_run-01_feedback_qc.html
```

**Manual QC report generation**:
```python
from pyrtnf import qc
from pathlib import Path

# Generate preprocessing QC report
qc.generate_preprocessing_report(
    preprocessing_dir=Path('/data/sub-01/preproc'),
    participant_id='sub-01',
    task='rest',
    output_dir=Path('/data/sub-01/preproc/qc'),
    open_browser=True  # Open in browser
)

# Generate mask extraction QC report
qc.generate_mask_extraction_report(
    mask_extraction_dir=Path('/data/sub-01/mask_extraction'),
    participant_id='sub-01',
    network_names=['DefaultA', 'ContA'],
    output_dir=Path('/data/sub-01/mask_extraction/qc'),
    open_browser=True
)

# Generate feedback QC report
qc.generate_feedback_report(
    feedback_dir=Path('/data/sub-01/feedback'),
    subject='sub-01',
    task='nback',
    run=1,
    output_dir=Path('/data/sub-01/feedback/qc'),
    open_browser=True
)
```
