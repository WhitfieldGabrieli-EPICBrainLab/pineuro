# Quick Start

Get pineuro running in a few minutes using the desktop GUI and the built-in
simulator — no scanner required.

## Prerequisites

- **Python 3.10 or 3.11**
- pineuro installed — see the [Installation Guide](installation.md)
- Some pre-collected DICOMs to replay (a resting-state run for mask extraction and a
  feedback run; e.g. a public neurofeedback dataset)

## 1. Launch the GUI

```bash
pineuro-gui
```

## 2. Create a study

Click **"+ New Study"** to open the Study Editor and set the essentials (tabs:
**Study · Experiment Design · Localizer · Feedback · Display · Sham**):

- **Study** — a name; enable the **Simulator** and point it at your DICOMs (a source
  mapping file or per-run paths); set the base data directory.
- **Localizer** — a `rest` localizer (TR, volumes); pick a network atlas (Yeo 7 or 17)
  and target networks (e.g. `Cont`, `Default`).
- **Feedback** — baseline volumes, nuisance regressors, and a task to display
  (e.g. `gauge`, `ball`, `thermometer`).

Click **Save**. See [Study Configuration](../user-guide/study-configuration.md) for every
setting.

## 3. Run a participant

From the Study Manager, open the study, enter a participant ID (`sub-01`), and click
**Start**. Step through the pipeline phases — collection → preprocessing → mask
extraction → reference coregistration → feedback. The simulator replays your DICOMs at
the configured speed; the operator window shows live activation, motion (FD/DVARS), and
tSNR plots.

## 4. Inspect outputs

Outputs are written under `base_dir/sub-01/ses-01/`. The key files and QC reports are
described in [QC Reports & Output Files](../user-guide/qc-reports.md).

## Scripting (optional)

Most users drive pineuro through the GUI, but the pipeline is also a plain Python API.
A minimal end-to-end run:

```python
from pineuro import preprocessing, mask_extraction_rest, run_feedback_task

# 1. Preprocess the resting-state localizer
preproc_file, mask_file, ref_file = preprocessing(
    participant_id="sub-01", session_id="ses-01", task="rest",
    root_dir="/data/sub-01/ses-01/preprocessing",
    input_dicoms="/scanner/sub-01/rest",
)

# 2. Extract participant-specific network masks (ICA + Yeo atlas) -> list of Paths
mask_paths = mask_extraction_rest(
    participant_id="sub-01", session_id="ses-01",
    root_dir="/data/sub-01/ses-01/mask_extraction",
    input_file=preproc_file, mask_file=mask_file, ref_file=ref_file,
    networks=["DefaultA", "ContA"],
)

# 3. Run a feedback task, replaying DICOMs through the simulator
run_feedback_task(
    mask_paths=mask_paths, subject="sub-01", session="ses-01",
    task="ball", run=1,
    output_dir="/data/sub-01/ses-01/feedback",
    collection_dir="/scanner/incoming",
    n_baseline=15, n_volumes=200, tr=2.0,
    simulator_source_dir="/data/sub-01/feedback_dicoms", simulator_speed=10.0,
)
```

See the [Pipeline Overview](../user-guide/pipeline-overview.md) for what each stage does.

## Next Steps

- [First Session Tutorial](first-session.md) — full GUI walkthrough with sample data
- [Study Configuration](../user-guide/study-configuration.md) — every study setting
- [Tasks](../user-guide/tasks.md) — built-in and custom task displays
