# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pineuro** (Python for Individualized Neurofeedback) is a library for real-time fMRI neurofeedback research. It provides a complete pipeline from DICOM acquisition through preprocessing, network mask extraction, and moment-to-moment BOLD activation estimation using an incremental GLM approach.

## Dependencies

**External Tools:**
- **FSL** (FMRIB Software Library) - Preprocessing (mcflirt, bet, fslmaths, fslmerge)
- **dcm2niix** - DICOM to NIfTI conversion

**Python (3.11+):**
- **antspyx** - Registration (multi-run coreg, MNI alignment, 2vol coreg)
- **nibabel** - NIfTI I/O
- **nilearn** - ICA decomposition (CanICA)
- **psychopy** - Task display
- **flet** - GUI framework
- **templateflow** - MNI152 templates
- **watchdog** - File watching for real-time acquisition
- numpy, scipy, pandas, matplotlib, pydicom, pyyaml, screeninfo

## Commands

```bash
# Install in development mode
pip install -e .

# Launch GUI application
pineuro-gui

# Run tests
python tests/test_decomposition_methods.py
```

## Architecture

### Project Structure

```
src/pineuro/
├── Core Pipeline
│   ├── preprocessing.py     # DICOM → preprocessed NIfTI
│   ├── mask_extraction.py   # ICA-based network identification
│   ├── feedback.py          # Real-time GLM estimation (~3000 lines)
│   └── simulator.py         # Offline testing with volume replay
├── Session Management
│   ├── session.py           # Multi-run orchestration, multi-visit support
│   └── sham.py              # Sham/control condition delivery
├── Support
│   ├── utils.py             # FSL/ANTs wrappers, DICOM utilities
│   ├── qc.py                # QC visualization and HTML reports
│   ├── config.py            # Global configuration
│   ├── display.py           # Monitor detection
│   └── script_generator.py  # Reproducible script generation
├── tasks/                   # PsychoPy task framework
│   ├── base.py              # BaseTask abstract class
│   ├── ball.py              # BallTask (dual-network game)
│   ├── thermometer.py       # Simple single-network example
│   ├── designs.py           # TaskDesign abstractions (block, event-related)
│   ├── task_registry.py     # Task registration and plugin system
│   └── session.py           # SessionState persistence
└── gui/                     # Flet GUI application
    ├── app.py               # Main application entry
    ├── screens/             # Navigation screens
    └── components/          # Reusable UI components
```

### Core Processing Pipeline

The library uses numbered output directories (01_, 02_, etc.) for easy tracing:

**1. Volume Collection** (`preprocessing.py`)
- `collect_volumes()` / `collect_dicoms()` - Watches for incoming volumes from scanner
- Supports both real-time (scanner) and simulation (replay) modes
- Works with DICOM or NIfTI input

**2. Preprocessing** (`preprocessing.py`)
- `preprocessing()` - Main 4-step pipeline:
  - 01_nifti: DICOM → NIfTI (dcm2niix)
  - 02_motion_correction: Motion correction (FSL mcflirt)
  - 03_brain_extraction: Brain extraction (FSL bet)
  - 04_coregistered: Multi-run alignment (ANTs)
- Output: `{participant}_task-{task}_preproc.nii.gz`, `_mask.nii.gz`, `_boldref.nii.gz`

**3. Mask Extraction** (`mask_extraction.py`)
- `mask_extraction_rest()` - ICA-based network identification:
  - 01_ica: CanICA decomposition (20 components default)
  - 02_mnireg: Register to MNI152 via ANTs
  - 03_masks: Pull Yeo atlas to subject space, correlate with ICA components
  - 04_2vol: Coregister to real-time scanner position
- `coregister_2vol()` - Aligns masks to feedback run scanner position

**4. Feedback Estimation** (`feedback.py`)
- `IncrementalGLM` - Gentleman's algorithm for O(p²) online least squares
- `FeedbackEstimator` - Builds design matrix, computes normalized activation
- `NeurofeedbackCoordinator` - Watches volumes, orchestrates processing
- `run_feedback_session()` - High-level API for complete feedback session
- Design matrix: `[task_regressors] [intercept] [drift] [motion_x6]`
- **Motion correction modes** (`moco_enabled` parameter):
  - `True` (default): Extract motion from Siemens MoCo JSON sidecar
  - `False`: Use ANTsPy multi-resolution rigid registration (~400-500ms/vol)

**5. Simulation** (`simulator.py`)
- `VolumeSimulator` / `DicomSimulator` - Copies volumes at TR intervals
- Deadline-based timing for realistic acquisition simulation

### Session Management

**session.py** - Multi-run and multi-visit orchestration:
- Configuration dataclasses: `SessionConfig`, `PathConfig`, `ScanConfig`, `MaskConfig`, `FeedbackConfig`, `TaskConfig`, `VisitConfig`, `ShamConfig`
- `setup_session()` - Initialize collection, preprocessing, mask extraction
- `run_single_task()` - Run one feedback/transfer task
- `run_full_session()` - Complete multi-run session
- Multi-visit support: `detect_existing_visits()`, `migrate_to_multi_visit()`, `resolve_mask_source()`

**sham.py** - Sham/control condition handling:
- `ShamActivationProvider` - Abstract base for sham sources
- `ReplayActivationProvider` - Replays activation from another participant
- Real activations always computed; alternative values delivered for blinding

### Task Framework (`tasks/`)

**BaseTask** (`base.py`) - Abstract base for PsychoPy tasks:
- Thread-safe activation polling between coordinator and display
- Standard lifecycle: `setup()`, `update(data)`, `cleanup()`
- Optional session state and post-run questions

**BallTask** (`ball.py`) - Dual-network ball game:
- Two target circles (one per network)
- Ball movement driven by relative activation
- Adaptive difficulty scaling persisted via SessionState

**Task Designs** (`designs.py`):
- `BlockDesign` - Block design with HRF-convolved boxcar
- `NuisanceOnlyDesign` - Resting state (no task regressors)
- `EventRelatedDesign` - Event-related designs

**Session State** (`session.py`):
- `SessionState` - JSON-backed persistence across runs
- `RunInfo` - Completed run metadata

### Quality Control (`qc.py`)

High-level report generators:
- `generate_preprocessing_report()` - Motion, brain mask, registration
- `generate_mask_extraction_report()` - ICA components, network correlations
- `generate_feedback_report()` - Activation timeseries, design matrix

Output: Static HTML reports with embedded PNG plots in `{stage_dir}/qc/`

### GUI Application (`gui/`)

Flet-based application (`pineuro-gui` command):
- **Study Editor**: Configure study parameters (6 tabs)
- **Participant Workflow**: Pipeline execution and monitoring
- Real-time visualization: activation plots, FD plots, task mirror

## Key Patterns

### Command Execution
All external commands (FSL, dcm2niix) route through `run_command()`:
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
All functions support `overwrite=False` (default) - skips computation if outputs exist, enabling efficient re-runs.

### Threading Model
- **Coordinator thread** (background): Processes volumes, updates shared state
- **Task thread** (main): Runs PsychoPy at display refresh rate, polls activation
- Thread-safe communication via `threading.Lock`

### Incremental GLM
- Gentleman's algorithm: O(p²) per volume vs O(np²) for full recomputation
- Givens rotations for numerical stability
- Continuous normalization (not post-hoc z-scoring)

## Usage Examples

### Complete Session with GUI
```bash
pineuro-gui
```

### Programmatic Pipeline
```python
from pineuro import preprocessing, mask_extraction_rest, run_feedback_session

# Step 1: Preprocess resting-state data
preproc_file, mask_file, ref_file = preprocessing(
    participant_id='sub-01',
    task='rest',
    root_dir='/data/sub-01/preproc',
    input_dicoms='/data/sub-01/raw',
)

# Step 2: Extract network masks
mask_files = mask_extraction_rest(
    participant_id='sub-01',
    root_dir='/data/sub-01/mask_extraction',
    input_file=preproc_file,
    mask_file=mask_file,
    ref_file=ref_file,
    target_networks=['DefaultA', 'ContA'],
)

# Step 3: Run feedback session (real-time mode)
results = run_feedback_session(
    mask_paths=mask_files,
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/data/sub-01/feedback',
    incoming_dir='/scanner/export/dicoms',
    n_baseline=25,
    n_volumes=200,
)
```

### Simulation Mode (Offline Testing)
```python
from pineuro import collect_volumes, run_feedback_session

# Collect with simulated acquisition
dicom_folder = collect_volumes(
    watch_dir='/tmp/incoming',
    output_dir='/data/sub-01/raw/rest',
    n_volumes=200,
    simulator_source_dir='/path/to/source_dicoms',
    simulator_speed=10.0,  # 10x speed
)

# Feedback with simulation
results = run_feedback_session(
    mask_paths=['/path/to/mask.nii.gz'],
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/path/to/output',
    n_baseline=25,
    n_volumes=150,
    simulator_source_dir='/path/to/source_dicoms',
    simulator_speed=10.0,
    moco_enabled=False,  # Use ANTsPy if Siemens MoCo unavailable
)
```

### Multi-Run Session Orchestration
```python
from pineuro import SessionConfig, PathConfig, ScanConfig, setup_session, run_full_session

config = SessionConfig(
    participant_id='sub-01',
    paths=PathConfig(
        root_dir='/data/study',
        incoming_dir='/scanner/export',
    ),
    scan=ScanConfig(tr=1.2, n_volumes_rest=200, n_volumes_task=150),
)

# Setup: collection, preprocessing, mask extraction
setup_session(config)

# Run: transfer1 → feedback runs → transfer2
run_full_session(config)
```

### Custom Task Creation
```python
from pineuro.tasks import BaseTask
from psychopy import visual

class MyTask(BaseTask):
    def setup(self):
        self.win = visual.Window([800, 600])
        self.stim = visual.Circle(self.win, radius=0.1)

    def update(self, data: dict):
        # data contains: activation, volume, time, is_baseline
        self.stim.radius = 0.1 + data['activation'] * 0.05
        self.stim.draw()
        self.win.flip()

    def cleanup(self):
        self.win.close()
```

## Output Directory Structure

**Preprocessing:**
```
{root}/
├── 01_nifti/
├── 02_motion_correction/
├── 03_brain_extraction/
├── 04_coregistered/
├── {participant}_task-{task}_preproc.nii.gz
├── {participant}_task-{task}_preproc_mask.nii.gz
├── {participant}_task-{task}_boldref.nii.gz
└── qc/
```

**Mask Extraction:**
```
{root}/
├── 01_ica/
├── 02_mnireg/
├── 03_masks/
├── 04_2vol/
└── qc/
```

**Feedback:**
```
{root}/
├── 01_2vol/
├── 02_feedback_output/
│   ├── nifti_task-{task}_run-{run}/
│   ├── {subject}_task-{task}_run-{run}_feedbacklog.tsv
│   └── {subject}_task-{task}_run-{run}_design.tsv
└── qc/
```

## Public API

Core functions exported from `pineuro`:
- `run_feedback_session`, `preprocessing`, `collect_volumes`, `mask_extraction_rest`, `coregister_2vol`
- `BaseTask`, `BallTask`, `ActivationSmoother`, `ActivationNormalizer`, `SessionState`
- `SessionConfig`, `PathConfig`, `ScanConfig`, `MaskConfig`, `FeedbackConfig`, `TaskConfig`, `VisitConfig`, `ShamConfig`
- `setup_session`, `run_single_task`, `run_full_session`, `run_next_task`
- `ShamActivationProvider`, `ReplayActivationProvider`, `load_randomization`
- `Timer`, `enable_timing`, `get_timing_data`, `check_platform_requirements`
