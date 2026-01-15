# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyrtnf is a Python library for real-time fMRI neurofeedback research. It provides a pipeline from DICOM acquisition through preprocessing and network mask extraction, implementing the MURFI approach for moment-to-moment BOLD activation estimation.

## Dependencies

- **FSL** (FMRIB Software Library) - Required for preprocessing (mcflirt, bet, flirt, fslmaths, fslmerge, convert_xfm)
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

1. **Preprocessing** (`preprocessing.py`) - DICOM conversion, motion correction (mcflirt), brain extraction (bet), multi-run coregistration. All FSL commands go through `run_command()` in `utils.py`.

2. **Mask Extraction** (`mask_extraction.py`) - ICA-based network mask extraction using nilearn's CanICA:
   - Runs CanICA on preprocessed data (20 components default)
   - Registers participant space to MNI152 via FSL FLIRT
   - Pulls Yeo atlas (7 or 17 networks) back to participant space
   - Identifies network components via spatial correlation
   - Thresholds masks by voxel count (2000 default)
   - `coregister_2vol()` handles registration to 2-volume feedback space

3. **Feedback Estimation** (`feedback.py`) - Real-time activation computation:
   - `IncrementalGLM` - Gentleman's algorithm for online least squares updates with continuous normalization
   - `FeedbackEstimator` - Processes fMRI volumes, fits GLM to remove nuisance signals
   - `NeurofeedbackCoordinator` - Watches for DICOMs, converts to NIfTI, processes through estimator
   - `run_feedback_session()` - High-level function that handles both simulation and real-time modes

4. **Simulation** (`simulator.py`) - `DicomSimulator` copies DICOMs at TR intervals for offline testing. Auto-extracts TR from DICOM headers.

### Utilities (`utils.py`)

Central command execution and FSL wrappers:
- `run_command()` - Validates inputs/outputs, checks command availability, handles overwrite logic
- FSL wrappers: `compute_mean()`, `compute_xfm()`, `apply_xfm()`, `invert_xfm()`, `compute_xfm_to_mni()`, `apply_mask()`, `spatial_smoothing()`, `temporal_filtering()`
- `get_yeo_labels()` - Returns Yeo network labels for 7 or 17 networks
- Uses templateflow for MNI152 BOLD reference template

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
