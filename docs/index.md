# pineuro Documentation

Welcome to the pineuro documentation. pineuro is a Python library for real-time fMRI neurofeedback research.

## Quick Links

- [Installation Guide](getting-started/installation.md) - Set up pineuro and dependencies
- [Quick Start](getting-started/quickstart.md) - Get running in 5 minutes
- [Pipeline Overview](user-guide/pipeline-overview.md) - Understand the full workflow

## What is pineuro?

pineuro provides a complete pipeline for real-time fMRI neurofeedback experiments:

1. **Preprocessing** - DICOM conversion, motion correction, brain extraction
2. **Mask Extraction** - ICA-based network identification from resting-state data
3. **Feedback Sessions** - Real-time activation estimation and display

The library is designed for:
- **Research flexibility** - Customize every stage of the pipeline
- **Real-time performance** - Sub-second latency from volume acquisition to feedback
- **Reproducibility** - Consistent output structure and parameter logging

## Documentation Sections

### Getting Started
- [Installation](getting-started/installation.md) - System requirements and setup
- [Quick Start](getting-started/quickstart.md) - Your first feedback session
- [First Session Tutorial](getting-started/first-session.md) - Step-by-step walkthrough

### User Guide
- [Pipeline Overview](user-guide/pipeline-overview.md) - How the pieces fit together
- [Preprocessing](user-guide/preprocessing.md) - DICOM handling and motion correction
- [Mask Extraction](user-guide/mask-extraction.md) - ICA and network selection
- [Feedback Sessions](user-guide/feedback-sessions.md) - Running real-time experiments
- [Simulation Mode](user-guide/simulation-mode.md) - Offline testing

### Task Development
- [Creating Tasks](task-development/creating-tasks.md) - Build custom feedback displays
- [Task API Reference](task-development/task-api-reference.md) - BaseTask and utilities
- [Example Tasks](task-development/example-tasks.md) - Ball and thermometer walkthrough

### API Reference
- [Core Functions](api-reference/core-functions.md) - preprocessing, mask_extraction, feedback
- [Configuration](api-reference/configuration.md) - SessionConfig and settings
- [Data Classes](api-reference/dataclasses.md) - MaskSelection, RunInfo, etc.

### Technical
- [Architecture](technical/architecture.md) - System design and data flow
- [Incremental GLM](technical/incremental-glm.md) - The feedback algorithm
- [Platform Requirements](technical/platform-requirements.md) - OS-specific notes
- [References](technical/references.md) - Academic citations

## Support

- **Issues**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Questions**: Open a discussion on GitHub
