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

### 1.3 Real-Time QC Metrics: tSNR, DVARS, Motion Alerts
- **Gap**: No tSNR, no DVARS, no automated motion threshold alerts. FD is displayed but not acted upon.
- **Why**: Davydov et al. 2022 standard rtQA metrics. OpenNFT includes rtQA. Operators need warnings when quality degrades mid-run.
- **Scope**: Small-Medium
- **Module**: `feedback.py` (`NeurofeedbackCoordinator`), `qc.py`, GUI `components/fd_plot.py`

### 1.4 Differential Feedback at Pipeline Level
- **Gap**: Differential feedback (target minus control ROI) exists at the task level (BallTask computes DMN - CEN) but not as a first-class pipeline-level signal in `FeedbackEstimator`. Not logged in `feedbacklog.tsv` as a computed value.
- **Why**: Most common approach in the literature (Fede et al. 2020). Controls for global signal changes. Should be available to any task without reimplementation.
- **Scope**: Small
- **Module**: `feedback.py` (`FeedbackEstimator`), `session.py` (config)

### 1.5 Manual ROI Import (NIfTI Masks)
- **Gap**: No streamlined path to import pre-defined NIfTI masks (FreeSurfer, fMRIPrep, hand-drawn) directly into the feedback pipeline.
- **Why**: Turbo-BrainVoyager supports VOI import. Fundamental flexibility requirement for researchers with existing ROIs.
- **Scope**: Small
- **Module**: `mask_extraction.py`, `session.py`

### 1.6 Broader Atlas Support (AAL, Schaefer, Harvard-Oxford, Glasser)
- **Gap**: Only Yeo 7/17 atlases. No AAL, Schaefer, Harvard-Oxford, Desikan-Killiany, or Glasser.
- **Why**: Schaefer 400 increasingly standard. Harvard-Oxford common in clinical studies. Architecture already supports extension via `ATLAS_REGION_REGISTRY` and `fetch_atlas_mask()`.
- **Scope**: Medium
- **Module**: `mask_extraction.py`

### 1.7 Formal Test Suite
- **Gap**: No pytest-based tests. Only validation scripts (compare_murfi, compare_fmriprep, compare_melodic).
- **Why**: Essential for open-source credibility, NeuroImage reviewers, and safe refactoring.
- **Scope**: Large
- **Module**: New `tests/` directory, `pyproject.toml` (dev deps)

---

## Tier 2: Medium Priority (Should-Have)

### 2.1 Spatial Smoothing (Real-Time)
- **Gap**: No Gaussian spatial smoothing in real-time pipeline.
- **Why**: RTPSpy includes it. Improves SNR for smaller ROIs.
- **Scope**: Small
- **Module**: `feedback.py`

### 2.2 Connectivity-Based Feedback (Sliding-Window Correlation)
- **Gap**: No functional connectivity feedback. Only activation-based (univariate).
- **Why**: OpenNFT supports DCM, FRIEND supports dual-ROI correlation. Growing clinical area.
- **Approach**: Sliding-window Pearson correlation only (no DCM).
- **Scope**: Medium
- **Module**: `feedback.py` (new `ConnectivityEstimator` class)

### 2.3 Intermittent Feedback Mode
- **Gap**: Feedback is continuous (every TR). No block-summary or periodic mode.
- **Why**: Stoeckel et al. 2014 found intermittent may be superior for imagery-based strategies.
- **Scope**: Small
- **Module**: `feedback.py`, `session.py` (config)

### 2.7 Randomization File Generator
- **Gap**: No built-in tool for group assignment randomization (participant → sham/real).
- **Why**: Standard experimental methodology. Currently requires external creation.
- **Scope**: Small
- **Module**: `sham.py` or new `randomization.py`

### 2.9 Windows Support: Add Classifier, Remove WSL Code
- **Gap**: Windows not in pyproject.toml classifiers. WSL-specific workarounds exist but are unnecessary since all deps are pip-installable natively.
- **Action**: Add Windows classifier to pyproject.toml. Remove all WSL-specific code/workarounds.
- **Scope**: Small
- **Module**: `pyproject.toml`, `display.py`, docs

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
| `src/pineuro/feedback.py` | 1.1 vendor parsing, 1.2 WM/CSF regressors, 1.3 QC metrics, 1.4 differential feedback, 2.1 smoothing, 2.2 connectivity estimator, 2.3 intermittent mode |
| `src/pineuro/session.py` | Config dataclass extensions for all new features |
| `src/pineuro/mask_extraction.py` | 1.5 manual ROI import, 1.6 atlas registry expansion |
| `src/pineuro/preprocessing.py` | 1.1 vendor auto-detection, 1.2 WM/CSF mask extraction |
| `src/pineuro/qc.py` | 1.3 tSNR/DVARS reporting |
| `src/pineuro/gui/components/fd_plot.py` | 1.3 motion alerts in GUI |
| `src/pineuro/sham.py` | 2.7 randomization generator |
| `src/pineuro/display.py` | 2.9 remove WSL code |
| `pyproject.toml` | 1.7 dev deps, 2.9 Windows classifier |
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
