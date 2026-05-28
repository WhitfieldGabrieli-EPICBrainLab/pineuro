# Quick Start

Get up and running with pineuro in 5 minutes.

## Prerequisites

- Python 3.10 or 3.11
- pineuro installed (`pip install -e /path/to/pineuro`)

## 1. Preprocess Resting-State Data

First, preprocess the resting-state data that will be used for mask extraction:

```python
from pineuro import preprocessing

# Run preprocessing on DICOM data
preproc_file, mask_file, ref_file = preprocessing(
    participant_id='sub-01',
    task='rest',
    root_dir='/data/sub-01/preproc',
    input_dicoms='/scanner/sub-01/rest_dicoms',
)

print(f"Preprocessed: {preproc_file}")
print(f"Brain mask: {mask_file}")
print(f"BOLD reference: {ref_file}")
```

This will:
- Convert DICOMs to NIfTI
- Run motion correction (ANTsPy rigid registration)
- Extract brain mask (ANTsPy)
- Generate QC report

## 2. Extract Network Masks

Use ICA to identify participant-specific network masks:

```python
from pineuro import mask_extraction_rest

# Extract DMN and Control network masks
mask_files = mask_extraction_rest(
    participant_id='sub-01',
    root_dir='/data/sub-01/masks',
    input_file=preproc_file,
    mask_file=mask_file,
    ref_file=ref_file,
    networks=['DefaultA', 'ContA'],  # Target networks
)

for name, path in mask_files.items():
    print(f"{name}: {path}")
```

This will:
- Run ICA (20 components by default)
- Register to MNI and pull Yeo atlas to participant space
- Select best ICA component for each target network
- Threshold and save masks

## 3. Run a Feedback Task (Simulation)

Test the feedback pipeline with simulated data:

```python
from pineuro import run_feedback_task

# Run simulated feedback session
results = run_feedback_task(
    mask_paths=list(mask_files.values()),
    subject='sub-01',
    task='nback',
    run=1,
    output_dir='/data/sub-01/feedback',
    n_baseline=25,
    n_volumes=150,
    # Enable simulation mode
    simulator_source_dir='/path/to/source_dicoms',
    simulator_speed=10.0,  # 10x speed for testing
)
```

This will:
- Simulate scanner acquisition (copying DICOMs at TR intervals)
- Process each volume in real-time
- Compute activation for each network
- Log results to TSV files

## 4. View Results

After the session completes, check the outputs:

```
/data/sub-01/feedback/
├── 02_feedback_output/
│   ├── sub-01_task-nback_run-01_feedbacklog.tsv  # Activations
│   └── sub-01_task-nback_run-01_design.tsv       # GLM design
└── qc/
    └── sub-01_task-nback_run-01_feedback_qc.html # QC report
```

The `feedbacklog.tsv` contains columns:
- `volume`: Volume number
- `time`: Timestamp
- `is_baseline`: Whether volume is in baseline period
- `{network}_activation`: Activation value for each network

## Next Steps

- [First Session Tutorial](first-session.md) — Complete walkthrough with the GUI
- [Pipeline Overview](../user-guide/pipeline-overview.md) — Understand the full workflow
- [Tasks](../user-guide/tasks.md) — Built-in and custom feedback displays

## Common Issues

### "DICOM folder not found"
Ensure the path to DICOMs is correct and contains .dcm files.

### "No components found with sufficient correlation"
The ICA may need adjustment. Try:
- Increasing `n_components` (e.g., 30)
- Adjusting `smooth_fwhm` (e.g., 6.0)
- Checking QC report for data quality issues
