# pineuro Feature Gap Analysis & Accessibility Review

## Context

pineuro is a well-architected real-time fMRI neurofeedback library with a strong foundation: incremental GLM (Gentleman's algorithm), ICA-based mask extraction, PsychoPy task framework, multi-visit/multi-run session orchestration, sham/control conditions, blind operator mode, and a Flet GUI. FSL dependency has already been removed — all dependencies are pip-installable.

This review compares pineuro against the fMRI neurofeedback literature (CRED-nf checklist, Fede et al. 2020, Sulzer et al. 2013) and existing tools (OpenNFT, FRIEND, RTPSpy, Pyneal, Turbo-BrainVoyager) to identify must-have gaps and accessibility improvements.

---

## Tier 1: High Priority (Must-Have Gaps)

### 1.1 Multi-Vendor Scanner Support
- **Gap**: Motion parameter extraction is Siemens-specific (MoCo JSON sidecar). ANTsPy fallback works but adds ~400-500ms latency. No native GE or Philips DICOM parsing.
- **Why**: Pyneal supports GE/Siemens/Philips. Single-vendor lock-in excludes ~60% of potential users.
- **Scope**: Medium
- **Module**: `feedback.py`, `preprocessing.py`, `utils.py`

### 1.2 Real-Time Nuisance Regression (WM/CSF Signals)
- **Gap**: No white matter or CSF signal regression. Current nuisance model: intercept + drift + 6 motion params.
- **Why**: RTPSpy demonstrates <400ms with WM/CSF regression. CRED-nf encourages comprehensive confound removal (Sulzer et al. 2013).
- **Scope**: Medium
- **Module**: `feedback.py` (`FeedbackEstimator._build_design_row`), `preprocessing.py`

### ~~1.3 Real-Time QC Metrics: tSNR, DVARS, Motion Alerts~~ ✅
- **Done**: Added real-time tSNR (Welford's online algorithm, median across voxels) and DVARS (RMS of temporal derivative across mask voxels) to `FeedbackEstimator`. Both metrics logged per-volume in `feedbacklog.tsv` (`dvars` and `tsnr` columns). DVARS displayed in real-time via `DVARSPlot` chart in monitoring panel. tSNR displayed as color-coded text indicator (green ≥40, orange ≥20, red <20). Post-run summary dialog includes Signal Quality card with DVARS stats and tSNR quality rating. QC report includes DVARS timeseries plot and tSNR/DVARS stats in summary box. FD monitoring with in-run alerting was already implemented.

### ~~1.4 Differential Feedback at Pipeline Level~~ ✅
- **Done**: Computes differential signal in `NeurofeedbackCoordinator` whenever 2+ masks are used. Direction explicitly configurable via `differential_pair` in `FeedbackConfig` (e.g., `["ContA", "DefaultA"]` = ContA - DefaultA). GUI shows two dropdowns in feedback tab when 2+ masks configured. Logged as dedicated column in `feedbacklog.tsv`. Included in activations dict delivered to tasks and in console output. In sham mode, both delivered and real differentials are logged. Falls back to first two masks if no pair configured.

### 1.5 Manual ROI Import (NIfTI Masks)
- **Gap**: No streamlined path to import pre-defined NIfTI masks (FreeSurfer, fMRIPrep, hand-drawn) directly into the feedback pipeline.
- **Why**: Turbo-BrainVoyager supports VOI import. Fundamental flexibility requirement for researchers with existing ROIs.
- **Scope**: Small
- **Module**: `mask_extraction.py`, `session.py`

### ~~1.6 Broader Atlas Support (AAL, Schaefer, Harvard-Oxford)~~ ✅
- **Done**: Added `ATLAS_SOURCES` registry with Yeo 7/17, Schaefer 100/400 (7/17-network), Harvard-Oxford Cortical, and AAL (SPM12). On-demand label fetching via `get_atlas_labels()` with caching (eliminates GUI startup lag). Schaefer atlas supported in resting-state ICA pathway (parcels grouped by network for spatial correlation). Generalized `fetch_atlas_mask()` supports all atlas types. Two-tier atlas selection in GUI (atlas → region/network). `network_atlas` param added to `mask_extraction_rest()`, `atlas_source`/`atlas_regions` params added to `mask_extraction_task()` and `MaskConfig`.

### 1.7 Formal Test Suite
- **Gap**: No pytest-based tests. Only validation scripts (compare_murfi, compare_fmriprep, compare_melodic).
- **Why**: Essential for open-source credibility, NeuroImage reviewers, and safe refactoring.
- **Scope**: Large
- **Module**: New `tests/` directory, `pyproject.toml` (dev deps)

---

## Tier 2: Medium Priority (Should-Have)

### ~~2.1 Spatial Smoothing (Real-Time)~~ ✅
- **Done**: Added optional `smooth_fwhm` parameter to `FeedbackConfig`, threaded through `FeedbackEstimator` → `NeurofeedbackCoordinator` → `run_feedback_session()`. Uses `nilearn.image.smooth_img()` per volume. GUI field in Study Editor Feedback tab. Default `None` (disabled).

### 2.2 Connectivity-Based Feedback (Sliding-Window Correlation)
- **Gap**: No functional connectivity feedback. Only activation-based (univariate).
- **Why**: OpenNFT supports DCM, FRIEND supports dual-ROI correlation. Growing clinical area.
- **Approach**: Sliding-window Pearson correlation only (no DCM).
- **Scope**: Medium
- **Module**: `feedback.py` (new `ConnectivityEstimator` class)

### ~~2.3 Intermittent Feedback Mode~~ ✅
- **Done**: Added `feedback_interval` to `FeedbackConfig`. When set, coordinator accumulates activations and delivers the mean every N volumes after baseline. CSV logging retains per-volume resolution. GUI checkbox + interval field in Study Editor feedback tab. Baseline always delivered continuously.

### ~~2.7 Randomization File Generator~~ ✅
- **Done**: Added `generate_randomization()` to `sham.py` with simple and replay methods. Added Generate button + dialog in Study Editor sham tab. Exported from `__init__.py`.

### ~~2.9 Windows Support: Add Classifier, Remove WSL Code~~ ✅
- **Done**: Added Windows classifier to pyproject.toml. Removed `is_wsl()`, WSL-specific warnings, WSL code paths, and WSL documentation. Updated README and installation docs to remove stale FSL references.

---

## Accessibility Improvements

### A.1 End-to-End Tutorial with Sample Data
- **Gap**: README has API examples but no walkthrough tutorial with real data.
- **Approach**: Tutorial (Markdown in `docs/`) walking through: download OpenNeuro dataset → preprocess → extract masks → run simulated feedback → inspect QC reports. Link to an existing OpenNeuro rt-fMRI dataset rather than hosting custom data.
- **Impact**: HIGH

### A.2 Expand In-Repo Documentation
- **Gap**: Docs exist but are sparse. No API reference, no FAQ, no troubleshooting.
- **Approach**: Keep docs in-repo on GitHub (no external hosting site). Expand `docs/` with: comprehensive user guide, API reference, task development guide, FAQ/troubleshooting.
- **Impact**: HIGH

### A.5 CI/CD Pipeline
- **Gap**: No automated testing or build pipeline.
- **Approach**: GitHub Actions with linting, pytest, type checking, build verification. Pairs with 1.7.
- **Impact**: MEDIUM

### A.6 Contributing Guide
- **Gap**: No CONTRIBUTING.md.
- **Approach**: Development setup, code style, PR process, how to add tasks/atlases.
- **Impact**: MEDIUM

### A.8 API Stability & Versioning
- **Gap**: Version 0.1.0 with no changelog or versioning strategy.
- **Approach**: Add CHANGELOG.md, follow semantic versioning.
- **Impact**: MEDIUM

---

## Future Steps (Not Current Scope)

These items are deferred for later consideration:

- **MVPA/Classification-Based Feedback (DecNef)** — Large scope, requires decoder training workflow and classification pipeline
- **Physiological Noise Regression (RETROICOR, RVT)** — No ability to test currently; requires physiological recording interface
- **PyPI Publication** — Once everything is finalized
- **Auditory Feedback Modality** — Niche use case
- **Signal Dropout Detection** — Not a standard feature in any existing tool
- **Closed-Loop Adaptive Protocols** — Emerging research frontier, not standard yet

---

## Removed (Not Applicable)

- ~~Docker/Container Image~~ — All deps are pip-installable, users can manage their own environments
- ~~Web-Based Monitoring Dashboard~~ — Everything monitored via GUI
- ~~BIDS-App Compliance~~ — pineuro is interactive/real-time, not batch; already uses BIDS naming
- ~~EEG-fMRI Integration~~ — Out of scope
- ~~CRED-nf Report Auto-Generation~~ — Out of scope
- ~~Double-Blind Operator Support~~ — Already implemented (blind mode in GUI)
- ~~Reduce FSL Dependency~~ — Already done; FSL removed
- ~~Installation Simplification~~ — Already pure pip

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `src/pineuro/feedback.py` | 1.1 vendor parsing, 1.2 WM/CSF regressors, 1.3 QC metrics, ~~1.4 differential feedback~~ ✅, ~~2.1 smoothing~~ ✅, 2.2 connectivity estimator, ~~2.3 intermittent mode~~ ✅ |
| `src/pineuro/session.py` | Config dataclass extensions for all new features |
| `src/pineuro/mask_extraction.py` | 1.5 manual ROI import, ~~1.6 atlas registry expansion~~ ✅ |
| `src/pineuro/preprocessing.py` | 1.1 vendor auto-detection, 1.2 WM/CSF mask extraction |
| `src/pineuro/qc.py` | 1.3 tSNR/DVARS reporting |
| `src/pineuro/gui/components/fd_plot.py` | 1.3 motion alerts in GUI |
| `src/pineuro/sham.py` | ~~2.7 randomization generator~~ ✅ |
| `src/pineuro/display.py` | ~~2.9 remove WSL code~~ ✅ |
| `pyproject.toml` | 1.7 dev deps, ~~2.9 Windows classifier~~ ✅ |
| `tests/` (new) | 1.7 formal test suite |
| `docs/` | A.1 tutorial, A.2 expanded docs |
| `CONTRIBUTING.md` (new) | A.6 |
| `CHANGELOG.md` (new) | A.8 |

---

## References

- Davydov et al. (2022). Real-time quality assessment estimators. *Frontiers in Neuroscience*.
- Fede et al. (2020). A guide to literature informed decisions in the design of real time fMRI neurofeedback studies. *Frontiers in Human Neuroscience*.
- Koush et al. (2017). OpenNFT: An open-source Python/Matlab framework. *NeuroImage*.
- MacInnes & Adcock (2020). Pyneal: Open source real-time fMRI software. *Frontiers in Neuroscience*.
- Mamisaki et al. (2022). RTPSpy: A library for fMRI real-time processing systems in Python. *Frontiers in Neuroscience*.
- Ros et al. (2020). CRED-nf checklist. *Brain*.
- Sato et al. (2013). FRIEND: An FSL-integrated BCI toolbox. *PLOS ONE*.
- Sitaram et al. (2017). Closed-loop brain training. *Nature Reviews Neuroscience*.
- Stoeckel et al. (2014). Optimizing real-time fMRI neurofeedback. *NeuroImage: Clinical*.
- Sulzer et al. (2013). Real-time fMRI neurofeedback: Progress and challenges. *NeuroImage*.
- Thibault et al. (2018). Neurofeedback with fMRI: A critical systematic review.
