# Study Configuration

Studies are the central configuration unit in pineuro. A study defines all parameters for a neurofeedback experiment: scan acquisition, preprocessing, mask extraction, feedback, task display, and sham conditions. Studies are saved as YAML files in `~/.pineuro/studies/`.

## Creating a Study

Launch the GUI (`pineuro-gui`) and click **"+ New Study"** to open the Study Editor. Studies can also be duplicated from existing ones.

## Study Editor Tabs

The editor has study-wide tabs (**Study**, **Display**, **Sham**) and a per-session
**Experiment Design** area where you add localizers and feedback runs — each opens its own
**Localizer** and **Feedback** editor tab.

### Study (study-wide)

- **Name / Description** — shown on the study card.
- **Base data directory** — root for participant data. Each participant/session is written
  to `base_dir/sub-XX/ses-XX/`.
- **Example Participant ID** — previews the directory structure.
- **Simulator** — enable offline replay; set the localizer/collection and feedback speeds,
  the trigger mode, and a source mapping file (see [Simulation Mode](simulation-mode.md)).
- **Scanner** — watch (DICOM) directory, scanner IP, and trigger keys (apply to every task).
- **Reference scans** — enable the **initial reference** (before localizers) and/or the
  **real-time reference** (before feedback), each with a volume count.

### Experiment Design (per session)

A study declares an ordered list of **sessions** (the Sessions strip — add, duplicate,
select, remove). For the selected session you define:

- **Localizer tasks** — add one or more; each opens a **Localizer** editor tab.
- **Feedback tasks** — add one or more; each opens a **Feedback** editor tab.
- **Session notes** and, in per-subject simulator mode, the session's source DICOMs.

### Localizer (per localizer)

- **Type** — `rest` (resting-state → ICA), a task localizer (e.g. `selfother`, `rit` → GLM),
  or `manual` (import pre-built ROIs).
- **TR**, **Volumes**, **Runs**, **Motion correction**.
- **Extraction** (the `extraction` block): **Network atlas** (`yeo_17` default; also Yeo 7,
  Schaefer, Harvard-Oxford, AAL, gray-matter — see [Mask Extraction](mask-extraction.md)),
  **Target networks**, **ICA components** (default 30), **Threshold** (default 2000 voxels),
  **Threshold method** (`top_n` / `cluster`), **Smoothing FWHM** (default 5.0),
  **High-pass** (default 0.01 Hz).

### Feedback (per feedback run)

- **Task** — any registered task (`gauge`, `ball`, `thermometer`, `rittriggeredball`, … —
  see [Tasks](tasks.md)). Task-specific parameters appear below the selection.
- **Mask source** — which localizer (this session) or **earlier session's masks** drive this
  run. Selecting an earlier session's masks is how cross-session reuse is configured (see
  [Multi-Session](multi-session.md)).
- **TR**, **Volumes**, **Baseline volumes** (default 25).
- **Feedback mode** — `activation` (single/differential) or `connectivity` (sliding-window
  correlation). **Differential pair** for dual-network activation (e.g. `Cont − Default`).
- **Motion correction** — `siemens_moco` (default), `software` (ANTsPy), or `none`.
- **Nuisance regressors** — drift, motion (6 DOF), white matter, CSF (toggled independently).
- **ROI combine** — `voxel_efficiency` (default) or `mean`. **QC thresholds** — FD, DVARS, tSNR.
- **Transfer runs** — practice runs without active feedback.

See [Feedback Tasks](feedback-tasks.md) for the feedback computation details.

### Display

- **Control Monitor** — `left` or `right` (operator GUI).
- **Presenter Monitor** — `left` or `right` (participant PsychoPy display).

### Sham

- **Enable Sham**, **Blind Mode** (hide the condition from the operator), **Delivery method**
  (`alternative_masks` / `replay`), **Randomization file**. See
  [Sham & Blinding](sham-and-blinding.md).

## Study File Format

Studies are stored as YAML in `~/.pineuro/studies/{name}.yaml`. Study-wide/environment
blocks (`paths`, `simulator`, `scanner`, `reference`, `sham`, `display`) live at the top
level; the per-session protocol lives under a `sessions:` list, where each entry has its
own `localizers` and `feedback_runs`. Studies can be manually edited if needed, though the
GUI is the primary interface.

## Managing Studies

From the Study Manager screen:
- **Edit** — Open in Study Editor
- **Duplicate** — Create a copy with a new name
- **Delete** — Remove the study (does not delete participant data)
- **Start Session** — Begin a participant workflow using this study's configuration
