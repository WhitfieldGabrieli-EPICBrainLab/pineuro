# Feedback Tasks

A feedback task is the core of neurofeedback training. pineuro processes each fMRI volume in real-time, estimates brain activation using an incremental GLM, and streams the result to a visual task display.

## Feedback Modes

### Activation Mode

The default mode. Computes moment-to-moment activation for each network mask.

**Single network:** Activation is the z-scored beta estimate for the task regressor, extracted from the mask region.

**Differential feedback:** When two masks are configured with a differential pair (e.g., `Cont - Default`), feedback is the difference between the two network activations. This is common in dual-network paradigms where participants regulate the balance between two networks.

Configure the differential pair in the Study Editor Feedback tab by selecting the two networks and their order.

### Connectivity Mode

Computes sliding-window Pearson correlation between two network timeseries.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `connectivity_window` | 15 | Number of TRs in the sliding window |
| `connectivity_target` | `increase` | Whether the participant should increase or decrease the correlation |

Connectivity mode requires exactly 2 masks. The feedback value is the correlation coefficient (range: -1 to +1).

## The Incremental GLM

pineuro uses Gentleman's algorithm for numerically stable, online least-squares estimation. Each incoming volume updates the running estimate in O(p^2) time (where p = number of regressors), compared to O(np^2) for full recomputation.

### Design Matrix

The GLM design matrix includes:

| Regressor | Type | Configurable | Description |
|-----------|------|-------------|-------------|
| Task | Signal | Automatic | Task condition regressor(s) |
| Intercept | Nuisance | Always on | Constant term |
| Drift | Nuisance | Toggle | 3rd-order polynomial drift |
| Motion (6 DOF) | Nuisance | Toggle | Translation (x, y, z) + rotation (x, y, z) |
| White matter mean | Nuisance | Toggle | Mean signal from WM mask |
| CSF mean | Nuisance | Toggle | Mean signal from CSF mask |

**Activation computation:** For each volume, the task beta is extracted and the nuisance contribution is subtracted. The result is z-scored against baseline statistics.

### Baseline Period

The first N volumes (configurable via `n_baseline`) serve as the baseline period:
- Task regressor is frozen at 0
- Running statistics (mean, variance) are collected
- Feedback display is inactive or shows a baseline indicator

After baseline, activation values are normalized relative to baseline statistics.

## Nuisance Regression

Each nuisance regressor can be independently enabled/disabled:

- **Drift** — Removes slow signal drift (scanner heating, physiological low-frequency oscillation)
- **Motion (6 DOF)** — Removes signal correlated with head movement. Automatically disabled when motion correction mode is `none`.
- **White Matter** — Removes global signal from white matter (requires WM tissue mask from mask extraction)
- **CSF** — Removes CSF signal fluctuations (requires CSF tissue mask)

WM and CSF regressors require tissue masks, which are automatically extracted during the mask extraction step when these regressors are enabled.

## Spatial Smoothing

Optional Gaussian spatial smoothing can be applied to each incoming volume before GLM estimation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `smooth_fwhm` | None (disabled) | Full-width at half-maximum in mm |

Set in the Study Editor Feedback tab. Typical values: 4-6 mm.

## Intermittent Feedback

By default, feedback is delivered every TR. For intermittent feedback, set `feedback_interval` to deliver the mean activation every N volumes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feedback_interval` | None (every TR) | Deliver feedback every N volumes |

When set, activations are accumulated and the mean is delivered to the task every N volumes. Per-volume data is still logged in the feedbacklog at full temporal resolution.

## ROI Combining

Per-voxel activations within each mask are combined into a single feedback value per volume. Set `roi_combine_method` on a feedback run to choose how:

| Method | Description |
|--------|-------------|
| `inverse_sd` (default) | Voxel efficiency weighting (Hinds et al., 2011): each voxel is weighted by `1/σ`, the inverse of its residual standard deviation, so noisier voxels contribute less. |
| `mean` | Unweighted mean — every mask voxel contributes equally. |

## QC Thresholds

Real-time quality metrics are computed per volume and used for outlier detection:

| Metric | Default Threshold | Description |
|--------|-------------------|-------------|
| FD (Framewise Displacement) | 0.5 mm | Head motion between consecutive volumes |
| DVARS | 1.5 (standardized) | Intensity change between consecutive volumes |
| tSNR | 20.0 | Temporal signal-to-noise ratio |

Volumes exceeding FD or DVARS thresholds are flagged as outliers. Outlier volumes are logged but their activation values are not delivered to the task display (the previous valid value is held).

## Real-Time Monitoring

During a feedback task, the operator GUI displays:
- **Activation plot** — Per-volume activation for each network (and differential if configured)
- **FD plot** — Framewise displacement with threshold line
- **DVARS plot** — Signal intensity change with threshold line
- **tSNR indicator** — Color-coded signal quality (green >= 40, orange >= 20, red < 20)
- **Terminal log** — Per-volume processing time and drift

## Configuration Summary

| Parameter | GUI Location | Default |
|-----------|-------------|---------|
| Baseline volumes | Feedback tab | 15-25 |
| Motion correction | Feedback tab | `siemens_moco` |
| Nuisance regressors | Feedback tab (checkboxes) | drift + motion on |
| Feedback mode | Feedback tab | `activation` |
| Differential pair | Feedback tab | None |
| Smooth FWHM | Feedback tab | None |
| Feedback interval | Feedback tab | None |
| Connectivity window | Feedback tab | 15 TRs |
| Connectivity target | Feedback tab | `increase` |
| Efficiency weights | Feedback tab | On |
| FD threshold | Feedback tab | 0.5 |
| DVARS threshold | Feedback tab | 1.5 |
| tSNR threshold | Feedback tab | 20.0 |
