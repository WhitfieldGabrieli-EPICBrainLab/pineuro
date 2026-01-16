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

Mask extraction:
- `01_ica/`, `02_mnireg/`, `03_masks/`, `01_2vol/`

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
