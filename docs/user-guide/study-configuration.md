# Study Configuration

Studies are the central configuration unit in pineuro. A study defines all parameters for a neurofeedback experiment: scan acquisition, preprocessing, mask extraction, feedback, task display, and sham conditions. Studies are saved as YAML files in `~/.pineuro/studies/`.

## Creating a Study

Launch the GUI (`pineuro-gui`) and click **"+ New Study"** to open the Study Editor. Studies can also be duplicated from existing ones.

## Study Editor Tabs

### General

- **Name** — Display name for the study card
- **Description** — Brief description of the study protocol

### Paths

- **Base Directory** — Root directory where participant data will be stored. Each participant gets a subdirectory: `base_dir/sub-001/`, `base_dir/sub-002/`, etc.
- **Example Participant ID** — Used for previewing the directory structure

### Scan Parameters

Configure acquisition parameters for each scan type:

**Localizer (resting-state or task)**
- **TR** — Repetition time in seconds
- **Volumes** — Number of volumes per localizer run
- **Runs** — Number of localizer runs (typically 1-2)
- **Task Type** — `rest` (resting-state), `fixation` (fixation cross), or a task-based localizer (e.g., `self_other`)

**Feedback**
- **TR** — Repetition time for feedback runs
- **Volumes** — Expected volumes per feedback run

**Reference Scans**
- **Initial Reference** — Collected before the localizer (preprocessing reference). Enable if your protocol acquires a reference scan at session start.
- **Real-time Reference** — Collected before feedback runs (coregistration reference). Enable to align masks to current head position.

### Simulator

When testing offline (no scanner), the simulator replays pre-collected DICOMs:

- **Enable Simulator** — Toggle simulation mode
- **Source Mapping** — Path to a TSV/CSV file mapping participant IDs to DICOM directories (see [Simulation Mode](simulation-mode.md))
- **Collection Speed** — Speed multiplier for localizer/preprocessing (e.g., `10.0` = 10x real-time)
- **Feedback Simulator Speed** — Speed multiplier for feedback runs (e.g., `1.0` = real-time, `10.0` = fast)
- **Per-Run DICOM Sources** — Alternatively, specify DICOM directories directly per run (localizer, feedback, transfer)

### Mask Extraction

- **Network Atlas** — `Yeo 7` or `Yeo 17` for ICA-based extraction. See [Mask Extraction](mask-extraction.md) for other atlas options.
- **Target Networks** — Which networks to extract (e.g., `Cont`, `Default` for Yeo 7; `ContA`, `DefaultA` for Yeo 17)
- **ICA Components** — Number of independent components to extract (default: 20)
- **Smoothing FWHM** — Spatial smoothing in mm before ICA (default: 5.0)
- **High-Pass Filter** — Cutoff in Hz (default: 0.01)
- **Threshold** — Number of voxels per mask (default: 2000)
- **Threshold Method** — `top_n` (select top N voxels by weight) or `cluster` (atlas-guided iterative clustering)

### Feedback

See [Feedback Tasks](feedback-tasks.md) for full details.

- **Baseline Volumes** — Volumes for baseline statistics (typically 15-25)
- **Motion Correction** — `software` (ANTsPy), `siemens_moco` (parse from DICOM sidecar), or `none`
- **Nuisance Regressors** — Toggle drift, motion (6 DOF), white matter, and CSF regressors independently
- **Feedback Mode** — `activation` (single/differential network activation) or `connectivity` (sliding-window correlation)
- **Differential Pair** — Which two networks to contrast (e.g., Cont - Default)
- **QC Thresholds** — FD (mm), DVARS (standardized), tSNR cutoffs for outlier detection

### Task

- **Task Type** — Any registered task (`gauge`, `ball`, `thermometer`, `rit_triggered_ball`, etc. — see [Tasks](tasks.md))
- **Max Feedback Runs** — Number of feedback runs per session
- **Transfer Runs** — Enable/disable practice runs without real feedback
- **Task-Specific Parameters** — Dynamically populated based on selected task type

### Sham

See [Sham & Blinding](sham-and-blinding.md) for full details.

- **Enable Sham** — Toggle sham/control condition
- **Blind Mode** — Hide network names in operator GUI (show "Target"/"Non-Target")
- **Delivery Method** — `alternative_masks` or `replay`
- **Randomization File** — TSV assigning participants to real/sham conditions

### Display

See [Display Setup](display-setup.md).

- **Control Monitor** — `left` or `right` (operator GUI)
- **Presenter Monitor** — `left` or `right` (participant PsychoPy display)

### Multi-Session

- **Enable Multi-Session** — Toggle longitudinal mode (see [Multi-Session](multi-session.md))

## Study File Format

Studies are stored as YAML in `~/.pineuro/studies/{name}.yaml`. The file contains all configuration organized by section (paths, scan, mask, feedback, task, sham, display, study). Studies can be manually edited if needed, though the GUI is the primary interface.

## Managing Studies

From the Study Manager screen:
- **Edit** — Open in Study Editor
- **Duplicate** — Create a copy with a new name
- **Delete** — Remove the study (does not delete participant data)
- **Start Session** — Begin a participant workflow using this study's configuration
