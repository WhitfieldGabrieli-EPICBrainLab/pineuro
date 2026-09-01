# pineuro Documentation

Welcome to the pineuro documentation. pineuro is a Python library for real-time fMRI neurofeedback research.

## What is pineuro?

pineuro provides a complete pipeline for real-time fMRI neurofeedback experiments:

1. **Preprocessing** — DICOM conversion, motion correction, brain extraction
2. **Mask Extraction** — Participant-specific network masks: ICA (resting-state), GLM (task localizer), or MNI atlas
3. **Feedback Tasks** — Real-time activation estimation and display

The library is designed for:
- **Research flexibility** — Customize every stage of the pipeline
- **Real-time performance** — Sub-second latency from volume acquisition to feedback
- **Reproducibility** — Consistent output structure and parameter logging

## Getting Started

- [Installation](getting-started/installation.md) — System requirements and setup
- [Quick Start](getting-started/quickstart.md) — Get running in a few minutes (GUI-first)
- [First Session Tutorial](getting-started/first-session.md) — Step-by-step GUI walkthrough with sample data

## User Guide

- [Pipeline Overview](user-guide/pipeline-overview.md) — How the pieces fit together
- [Study Configuration](user-guide/study-configuration.md) — Creating and configuring studies
- [Preprocessing](user-guide/preprocessing.md) — Motion correction, brain extraction, coregistration
- [Mask Extraction](user-guide/mask-extraction.md) — ICA, GLM, manual ROI import, atlas options
- [Feedback Tasks](user-guide/feedback-tasks.md) — Activation, connectivity, differential, intermittent feedback
- [Tasks](user-guide/tasks.md) — Gauge, ball, thermometer, RIT variants, rest & self/other localizers; transfer runs
- [Sham & Blinding](user-guide/sham-and-blinding.md) — Control conditions, blind operator mode, randomization
- [Multi-Session Studies](user-guide/multi-session.md) — Longitudinal protocols with mask reuse
- [Display Setup](user-guide/display-setup.md) — Multi-monitor and projector configuration
- [Simulation Mode](user-guide/simulation-mode.md) — Offline testing with DICOM replay
- [QC Reports & Output Files](user-guide/qc-reports.md) — Quality reports, feedbacklog, psychopylog
- [Network Testing](user-guide/network-testing.md) — Two-machine scanner connectivity testing

## Support

- **Issues**: [GitHub Issues](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/issues)
- **Questions**: Open a discussion on GitHub
