# Changelog

All notable changes to pineuro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-02-20

Initial beta release.

### Added

- **Core pipeline**: DICOM-to-NIfTI conversion (dcm2niix), motion correction (Siemens MOCO / ANTsPy software / none), brain extraction, spatial coregistration
- **Mask extraction**: ICA-based (resting-state) and GLM-based (task localizer) network mask identification
- **Manual ROI import**: Support for native, anatomical, and MNI space masks with automatic registration
- **Atlas support**: Yeo 7/17, Schaefer 100/400, Harvard-Oxford Cortical, AAL — on-demand label fetching with caching
- **Real-time feedback**: Incremental GLM (Gentleman's algorithm) with configurable nuisance regressors (drift, 6 motion parameters, WM mean, CSF mean)
- **Feedback modes**: Activation, connectivity (sliding-window correlation), differential (configurable ROI pair), intermittent (mean over N volumes)
- **Real-time QC**: Framewise displacement with in-run alerts, DVARS (RMS temporal derivative), tSNR (Welford's online algorithm) — all logged per-volume
- **Spatial smoothing**: Optional real-time Gaussian smoothing with configurable FWHM
- **Task framework**: PsychoPy-based plugin system with `@pineuro_task` decorator and `register_task()` for external tasks
- **Built-in tasks**: Ball (dual-network, adaptive difficulty), Thermometer, Gauge, Fixation, Self-Other localizer
- **Session orchestration**: Multi-run, multi-visit support with persistent session state and numbered output directories
- **Sham/control conditions**: Replay-based and synthetic sham providers, randomization generator, blind operator mode
- **Volume simulator**: Offline replay at configurable TR intervals for development and testing
- **Flet GUI**: Study editor (6-tab configuration), participant workflow, real-time monitoring (activation plot, FD plot, DVARS chart, tSNR indicator, task mirror, QC viewer)
- **Cross-platform support**: Windows, macOS, Linux — all dependencies are pure pip
- **CI/CD**: GitHub Actions pipeline (lint, typecheck, build, test matrix across 3 OS x 3 Python versions)
- **Test suite**: 162 unit tests plus integration tests using NKI Rockland data
- **Documentation**: End-to-end tutorial, 10-page user guide, contributing guide
