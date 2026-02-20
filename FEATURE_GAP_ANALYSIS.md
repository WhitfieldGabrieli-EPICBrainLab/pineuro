# pineuro Feature Gap Analysis & Accessibility Review

## Context

pineuro is a well-architected real-time fMRI neurofeedback library with a strong foundation: incremental GLM (Gentleman's algorithm), ICA-based mask extraction, PsychoPy task framework, multi-visit/multi-run session orchestration, sham/control conditions, blind operator mode, and a Flet GUI. FSL dependency has already been removed — all dependencies are pip-installable.

This review compares pineuro against the fMRI neurofeedback literature (CRED-nf checklist, Fede et al. 2020, Sulzer et al. 2013) and existing tools (OpenNFT, FRIEND, RTPSpy, Pyneal, Turbo-BrainVoyager) to identify must-have gaps and accessibility improvements.

---

## Tier 1: High Priority (Must-Have Gaps)

### ~~1.1 Multi-Vendor Scanner Support~~ ✅
- **Done**: Replaced binary `moco_enabled` flag with 3-way `motion_correction` mode: `"siemens_moco"` (parse from JSON sidecar), `"software"` (ANTsPy registration, ~400-500ms/vol, any scanner), `"none"` (disabled). Motion correction decoupled from GLM — each nuisance regressor (drift, 6 motion params, WM mean, CSF mean) is independently toggleable. Intercept always included. GUI: motion correction dropdown + individual regressor checkboxes (motion checkbox auto-disabled when mode is "none"). Dynamic design matrix columns match active regressors. FD = NaN when motion correction disabled; FD plot gracefully skips NaN. Backward compatible: old `moco_enabled` and `nuisance_regression` config keys auto-migrated.

### ~~1.2 Real-Time Nuisance Regression (WM/CSF Signals)~~ ✅
- **Done**: Added `create_tissue_masks()` to `mask_extraction.py` using ANTsPy segmentation (Atropos 3-class). WM and CSF mean signals added as independently toggleable nuisance regressors in `FeedbackEstimator` design matrix (`use_wm_regressor`, `use_csf_regressor`). `FeedbackConfig` extended with `wm_mask_path`/`csf_mask_path`. Tissue masks auto-created during preprocessing, coregistered per-run, and resolved automatically by session orchestration. GUI checkboxes in Study Editor feedback tab. WM/CSF means logged per-volume in `feedbacklog.tsv`.

### ~~1.3 Real-Time QC Metrics: tSNR, DVARS, Motion Alerts~~ ✅
- **Done**: Added real-time tSNR (Welford's online algorithm, median across voxels) and DVARS (RMS of temporal derivative across mask voxels) to `FeedbackEstimator`. Both metrics logged per-volume in `feedbacklog.tsv` (`dvars` and `tsnr` columns). DVARS displayed in real-time via `DVARSPlot` chart in monitoring panel. tSNR displayed as color-coded text indicator (green ≥40, orange ≥20, red <20). Post-run summary dialog includes Signal Quality card with DVARS stats and tSNR quality rating. QC report includes DVARS timeseries plot and tSNR/DVARS stats in summary box. FD monitoring with in-run alerting was already implemented.

### ~~1.4 Differential Feedback at Pipeline Level~~ ✅
- **Done**: Computes differential signal in `NeurofeedbackCoordinator` whenever 2+ masks are used. Direction explicitly configurable via `differential_pair` in `FeedbackConfig` (e.g., `["ContA", "DefaultA"]` = ContA - DefaultA). GUI shows two dropdowns in feedback tab when 2+ masks configured. Logged as dedicated column in `feedbacklog.tsv`. Included in activations dict delivered to tasks and in console output. In sham mode, both delivered and real differentials are logged. Falls back to first two masks if no pair configured.

### ~~1.5 Manual ROI Import (NIfTI Masks)~~ ✅
- **Done**: Added `import_manual_masks()` to `mask_extraction.py` supporting three input spaces: native (functional/EPI), anatomical (T1→EPI via ANTs Rigid), and MNI152 (SyN registration). Masks are validated, optionally thresholded (top-N voxels), binarized, and saved to `03_masks/` with standard naming. `MaskConfig` dataclass extended with `manual_mask_paths`, `manual_ref_file`, `manual_anat_file`, `manual_mask_space` fields. `localizer_type="manual"` accepted throughout pipeline. GUI: "Manual ROI Import" option in study editor localizer dropdown with mask space selection and optional study-wide mask paths. Participant workflow shows per-participant file pickers (mask files, BOLD reference, T1 structural) with "Import Masks" button. Localizer/preprocessing steps automatically skipped. Pipeline runner and coregistration handle manual ref_file resolution. Exported from `__init__.py`.

### ~~1.6 Broader Atlas Support (AAL, Schaefer, Harvard-Oxford)~~ ✅
- **Done**: Added `ATLAS_SOURCES` registry with Yeo 7/17, Schaefer 100/400 (7/17-network), Harvard-Oxford Cortical, and AAL (SPM12). On-demand label fetching via `get_atlas_labels()` with caching (eliminates GUI startup lag). Schaefer atlas supported in resting-state ICA pathway (parcels grouped by network for spatial correlation). Generalized `fetch_atlas_mask()` supports all atlas types. Two-tier atlas selection in GUI (atlas → region/network). `network_atlas` param added to `mask_extraction_rest()`, `atlas_source`/`atlas_regions` params added to `mask_extraction_task()` and `MaskConfig`.

### 1.7 Formal Test Suite
- **Gap**: No pytest-based tests. Only validation scripts (compare_murfi, compare_fmriprep, compare_melodic).
- **Why**: Essential for open-source credibility, NeuroImage reviewers, and safe refactoring.
- **Scope**: Large
- **Module**: New `tests/` directory, `pyproject.toml` (dev deps)
- **Also update**: `CONTRIBUTING.md` Testing section — replace "no pytest suite yet" with how to run tests, write tests, fixture conventions

---

## Tier 2: Medium Priority (Should-Have)

### ~~2.1 Spatial Smoothing (Real-Time)~~ ✅
- **Done**: Added optional `smooth_fwhm` parameter to `FeedbackConfig`, threaded through `FeedbackEstimator` → `NeurofeedbackCoordinator` → `run_feedback_session()`. Uses `nilearn.image.smooth_img()` per volume. GUI field in Study Editor Feedback tab. Default `None` (disabled).

### ~~2.2 Connectivity-Based Feedback (Sliding-Window Correlation)~~ ✅
- **Done**: Added `ConnectivityEstimator` class to `feedback.py` with configurable window size and ROI pairs. GUI dropdown (Activation vs Connectivity) replaces old differential/connectivity checkboxes. Ball task supports `signal_mode` and `connectivity_target` (increase/decrease) for direction control. Psychopylog logs connectivity value + target/nontarget hits. Monitoring plot shows correlation line(s) with y-axis [-1, 1], sham-aware (alt_masks shows both pairs), blind-aware (generic label). Sham support: alternative_masks swaps both masks and adds sham connectivity pair; replay accumulates replayed values via `_sham_timeseries`. GUI hides "Target Hemisphere" in connectivity mode (correlation is symmetric).

### ~~2.3 Intermittent Feedback Mode~~ ✅
- **Done**: Added `feedback_interval` to `FeedbackConfig`. When set, coordinator accumulates activations and delivers the mean every N volumes after baseline. CSV logging retains per-volume resolution. GUI checkbox + interval field in Study Editor feedback tab. Baseline always delivered continuously.

### ~~2.7 Randomization File Generator~~ ✅
- **Done**: Added `generate_randomization()` to `sham.py` with simple and replay methods. Added Generate button + dialog in Study Editor sham tab. Exported from `__init__.py`.

### ~~2.9 Windows Support: Add Classifier, Remove WSL Code~~ ✅
- **Done**: Added Windows classifier to pyproject.toml. Removed `is_wsl()`, WSL-specific warnings, WSL code paths, and WSL documentation. Updated README and installation docs to remove stale FSL references.

---

## Accessibility Improvements

### ~~A.1 End-to-End Tutorial with Sample Data~~ ✅
- **Done**: Created `docs/getting-started/first-session.md` — full GUI walkthrough using NKI Rockland Sample data (NITRC INDI portal). Covers source mapping setup, study creation, all pipeline phases (preprocessing, mask extraction, reference coregistration, feedback with gauge task), and output inspection. Fixed stale docs: removed FSL/external dcm2niix references from installation.md, quickstart.md, pipeline-overview.md, README.md. Fixed GitHub Issues link in index.md.

### ~~A.2 Expand In-Repo Documentation~~ ✅
- **Done**: Created 10 user guide pages covering every feature: study-configuration.md, preprocessing.md, mask-extraction.md, feedback-sessions.md, tasks.md, sham-and-blinding.md, multi-visit.md, display-setup.md, simulation-mode.md, qc-reports.md. Updated docs/index.md with complete navigation. All task types (gauge, ball, thermometer, fixation, self_other), feedback modes (activation, connectivity, differential, intermittent), mask methods (ICA, GLM, manual ROI), atlas options, sham/blinding, multi-visit, display setup, and output formats documented.

### ~~A.5 CI/CD Pipeline~~ ✅
- **Done**: Added `.github/workflows/ci.yml` with 4 jobs: lint (Ruff check + format), typecheck (mypy, lenient), build (sdist + wheel artifact upload), test (pytest on Ubuntu/Windows/macOS × Python 3.10/3.11/3.12 matrix). Added `pytest`, `mypy`, `build` to dev dependencies. Added `[tool.pytest.ini_options]` and `[tool.mypy]` config to `pyproject.toml`. Smoke test `tests/test_import.py` verifies package imports on all platforms. Updated `CONTRIBUTING.md` Testing section (pytest instructions) and PR Process section (CI checks that must pass).

### ~~A.6 Contributing Guide~~ ✅
- **Done**: Created `CONTRIBUTING.md` covering development setup, Ruff linting, code conventions, project architecture overview, how to add tasks (plugin system) and atlases (registry), PR process, testing, and platform notes.

### A.8 API Stability & Versioning
- **Gap**: Version 0.1.0 with no changelog or versioning strategy.
- **Approach**: Add CHANGELOG.md, follow semantic versioning.
- **Impact**: MEDIUM
- **Also update**: `CONTRIBUTING.md` PR Process section — add changelog entry requirement, versioning policy

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
| `src/pineuro/feedback.py` | 1.1 vendor parsing, ~~1.2 WM/CSF regressors~~ ✅, 1.3 QC metrics, ~~1.4 differential feedback~~ ✅, ~~2.1 smoothing~~ ✅, ~~2.2 connectivity estimator~~ ✅, ~~2.3 intermittent mode~~ ✅ |
| `src/pineuro/session.py` | Config dataclass extensions for all new features |
| `src/pineuro/mask_extraction.py` | ~~1.5 manual ROI import~~ ✅, ~~1.6 atlas registry expansion~~ ✅ |
| `src/pineuro/preprocessing.py` | 1.1 vendor auto-detection, ~~1.2 WM/CSF mask extraction~~ ✅ |
| `src/pineuro/qc.py` | 1.3 tSNR/DVARS reporting |
| `src/pineuro/gui/components/fd_plot.py` | 1.3 motion alerts in GUI |
| `src/pineuro/sham.py` | ~~2.7 randomization generator~~ ✅ |
| `src/pineuro/display.py` | ~~2.9 remove WSL code~~ ✅ |
| `pyproject.toml` | 1.7 dev deps, ~~2.9 Windows classifier~~ ✅ |
| `tests/` (new) | 1.7 formal test suite |
| `docs/` | ~~A.1 tutorial~~ ✅, ~~A.2 expanded docs~~ ✅ |
| `CONTRIBUTING.md` (new) | ~~A.6~~ ✅ |
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
