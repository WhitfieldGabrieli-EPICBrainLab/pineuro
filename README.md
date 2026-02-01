# pineuro

**Python Real-Time fMRI Neurofeedback**

A Python library for real-time fMRI neurofeedback research. pineuro provides a complete pipeline from DICOM acquisition through preprocessing, network mask extraction, and moment-to-moment BOLD activation estimation.

## Features

- **Real-time DICOM processing** - Watch for incoming scanner DICOMs and process them as they arrive
- **Robust preprocessing** - Motion correction, brain extraction, and multi-run coregistration using FSL and ANTs
- **Network-based neurofeedback** - ICA-based mask extraction with Yeo atlas integration
- **Incremental GLM estimation** - Online activation computation using Gentleman's algorithm for numerical stability
- **Task framework** - Thread-safe PsychoPy integration for custom neurofeedback tasks
- **Simulation mode** - Offline testing with DICOM replay at realistic timing intervals

## Installation

### Prerequisites

pineuro requires the following external tools:

- **FSL** (FMRIB Software Library) - For preprocessing operations
  - Install from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
  - Required commands: `mcflirt`, `bet`, `fslmaths`, `fslmerge`, `convert_xfm`

- **ANTs** (Advanced Normalization Tools) - For registration
  - Install ANTsPy: `pip install antspyx`

- **dcm2niix** - For DICOM to NIfTI conversion
  - Install from: https://github.com/rordenlab/dcm2niix

### Install pineuro

**Note:** PsychoPy requires wxPython, which may need to be installed separately before running `pip install`. See [PsychoPy installation docs](https://www.psychopy.org/download.html) if you encounter issues.

```bash
# Clone the repository
git clone https://github.com/marlvan/pineuro.git
cd pineuro

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Collect and Preprocess Resting-State Data

```python
from pineuro import collect_dicoms, preprocessing

# Collect DICOMs from scanner (or use simulation mode for testing)
dicom_folder = collect_dicoms(
    watch_dir='/scanner/export/incoming',  # Scanner DICOM export directory
    output_dir='/data/sub-01/raw/rest',
    n_volumes=200,
    clear_watch_dir=True,  # Clean up after collection
)

# Run preprocessing pipeline
preproc_file, mask_file, ref_file = preprocessing(
    participant_id='sub-01',
    task='rest',
    root_dir='/data/sub-01/preproc',
    input_dicoms=dicom_folder,
)
```

### 2. Extract Network Masks

```python
from pineuro import mask_extraction_rest

# Extract network masks using ICA and Yeo atlas
mask_extraction_rest(
    root_dir='/data/sub-01/masks',
    preproc_runs=['/data/sub-01/preproc/sub-01_rest_preproc.nii.gz'],
    n_components=20,  # ICA components
    yeo_networks=[7],  # Default Mode Network (DMN)
    n_atlas=7,  # Use Yeo 7-network atlas
)
```

### 3. Run a Neurofeedback Session

```python
from pineuro import run_feedback_session

# Run real-time feedback with scanner acquisition
results = run_feedback_session(
    mask_paths=['/data/sub-01/masks/03_masks/network_7.nii.gz'],
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/data/sub-01/feedback',
    incoming_dir='/scanner/export/dicoms',  # Scanner DICOM directory
    n_baseline=25,  # Baseline volumes for normalization
    n_volumes=200,  # Total volumes to process
)

# Results contain per-volume activation estimates
print(results['activations'])
print(results['timing'])
```

### 4. Offline Testing with Simulation Mode

```python
from pineuro import run_feedback_session

# Test feedback pipeline offline using DICOM replay
results = run_feedback_session(
    mask_paths=['/data/sub-01/masks/03_masks/network_7.nii.gz'],
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/data/sub-01/feedback_test',
    n_baseline=25,
    n_volumes=150,
    # Enable simulation mode
    simulator_source_dir='/path/to/source_dicoms',  # Pre-collected DICOMs
    simulator_speed=10.0,  # 10x speed for rapid testing
)
```

## Architecture

### Processing Pipeline

pineuro implements a sequential pipeline with numbered output directories for easy tracing:

1. **DICOM Collection** - Watch for and collect incoming DICOMs from scanner
   - Real-time mode: monitors scanner export directory
   - Simulation mode: replays pre-collected DICOMs at TR intervals

2. **Preprocessing** - Convert and prepare fMRI data
   - DICOM → NIfTI conversion (`dcm2niix`)
   - Motion correction (`mcflirt`)
   - Brain extraction (`bet`)
   - Multi-run coregistration (ANTs)
   - Outputs: `01_nifti/`, `02_motion_correction/`, `03_brain_extraction/`, `04_coregistered/`

3. **Mask Extraction** - Identify network regions of interest
   - ICA decomposition (nilearn CanICA)
   - Registration to MNI space (ANTs)
   - Yeo atlas integration (7 or 17 networks)
   - Network identification via spatial correlation
   - Outputs: `01_ica/`, `02_mnireg/`, `03_masks/`

4. **Real-time Feedback** - Compute moment-to-moment activation
   - `IncrementalGLM`: Online least-squares using Gentleman's algorithm
   - `FeedbackEstimator`: Per-volume activation computation with nuisance regression
   - `NeurofeedbackCoordinator`: Orchestrates DICOM watching, conversion, and processing
   - Streams activation values to PsychoPy tasks via thread-safe callbacks

5. **Task Framework** - Present neurofeedback to participants
   - `BaseTask`: Abstract base class for custom tasks
   - `BallTask`: Dual-network ball game with adaptive difficulty
   - Thread-safe activation polling and callback mechanisms

### Key Components

- **`preprocessing.py`** - DICOM collection and preprocessing pipeline
- **`mask_extraction.py`** - Network mask extraction with ICA and atlases
- **`feedback.py`** - Real-time activation estimation with incremental GLM
- **`simulator.py`** - DICOM replay for offline testing
- **`session.py`** - High-level session orchestration and configuration
- **`tasks/`** - PsychoPy task framework with example implementations
- **`utils.py`** - FSL/ANTs command wrappers and DICOM utilities

## Usage Patterns

### Overwrite Control

All major functions support `overwrite=False` (default) to skip computation if outputs already exist, enabling efficient pipeline re-runs:

```python
# First run: performs full preprocessing
preprocessing(participant_id='sub-01', task='rest', root_dir='/data',
              input_dicoms='/data/raw')

# Second run: skips existing outputs
preprocessing(participant_id='sub-01', task='rest', root_dir='/data',
              input_dicoms='/data/raw', overwrite=False)  # Fast!

# Force recomputation
preprocessing(participant_id='sub-01', task='rest', root_dir='/data',
              input_dicoms='/data/raw', overwrite=True)
```

### FSL Command Wrapper

All external commands go through `run_command()` for consistent validation, logging, and error handling:

```python
from pineuro.utils import run_command

run_command(
    cmd_name='mcflirt',
    cmd=['mcflirt', '-in', str(input_path), '-out', str(output_path)],
    description='Motion correction with MCFLIRT',
    inputs=input_path,  # Validated before execution
    outputs=output_path,  # Checked for existence/overwrite
    overwrite=False,
)
```

### Real-time vs Simulation Mode

Most functions support both real-time scanner acquisition and offline simulation:

```python
# Real-time mode: watches incoming_dir for new DICOMs
collect_dicoms(watch_dir='/scanner/export', ...)

# Simulation mode: replays DICOMs from simulator_source_dir
collect_dicoms(
    watch_dir='/tmp/incoming',
    simulator_source_dir='/path/to/source_dicoms',
    simulator_speed=10.0,  # Speed multiplier
)
```

## Creating Custom Tasks

See [src/pineuro/tasks/README.md](src/pineuro/tasks/README.md) for detailed instructions on creating custom neurofeedback tasks.

Quick example:

```python
from pineuro.tasks import BaseTask
from psychopy import visual

class MyTask(BaseTask):
    def setup(self):
        """Initialize PsychoPy stimuli."""
        self.text = visual.TextStim(self.win, text='')

    def update(self, frame_num):
        """Update display each frame (~60 Hz)."""
        # Get latest activation (thread-safe)
        activation = self.get_latest_activation('network_1')

        # Update display
        self.text.text = f'Activation: {activation:.2f}'
        self.text.draw()
```

## Testing

```bash
# Run integration tests
python tests/test.py

# Run DICOM simulator manually for testing other components
python -m pineuro.simulator \
    --source /path/to/source_dicoms \
    --target /tmp/incoming \
    --tr 1.5 \
    --speed 2.0  # 2x real-time speed
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive architecture guide and development patterns
- **[tasks/README.md](src/pineuro/tasks/README.md)** - Task framework integration guide
- **[docs/references.md](docs/references.md)** - Academic references and citations

## Citation

If you use pineuro in your research, please cite:

```
Vandewouw, M. M. (2026). pineuro: Python Real-Time fMRI Neurofeedback.
https://github.com/marlvan/pineuro
```

Key references:
- Gentleman, W. M. (1974). Basic procedures for large, sparse or weighted linear least squares problems. *Applied Statistics*, 23(3), 448-454.
- Hinds, O. et al. (2011). Computing moment-to-moment BOLD activation for real-time neurofeedback. *NeuroImage*, 54(1), 361-368.
- Yeo, B. T. et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125-1165.

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contact

Marlee M Vandewouw - marlee.vandewouw@outlook.com
