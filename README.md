# pineuro

**Python for Individualized Neurofeedback**

A Python library for real-time fMRI neurofeedback research. pineuro provides a complete pipeline from DICOM acquisition through preprocessing, network mask extraction, and moment-to-moment BOLD activation estimation, with a desktop GUI for study management and participant workflows.

## Features

**Core pipeline**
- **DICOM processing**: Watch for incoming scanner DICOMs and process volumes as they arrive
- **Preprocessing**: Motion correction, brain extraction, and multi-run coregistration (ANTsPy)
- **ICA mask extraction**: Participant-specific network masks via ICA and Yeo atlas
- **GLM mask extraction**: Network masks from task localizer contrasts
- **MNI atlas masks**: Atlas-based ROIs without a localizer run
- **Incremental GLM**: Online activation estimation using Gentleman's algorithm
- **Connectivity feedback**: Real-time functional connectivity between network pairs

**Study management**
- **Desktop GUI**: Study configuration and participant workflows (`pineuro-gui`)
- **Session orchestration**: Multi-run and multi-session protocols
- **Sham / control conditions**: Sham feedback, replay providers, and randomization
- **QC reports**: HTML quality reports for each pipeline stage

**Task framework**
- **Built-in tasks**: Gauge (dual-network block design), ball (dual-network adaptive), thermometer (single-network), and RIT variants
- **Task plugin system**: `@pineuro_task` decorator and entry-point registration
- **Simulation mode**: Offline DICOM replay at configurable speed

## Installation

**Requires Python 3.10 or 3.11** (`>=3.10,<3.12`). All dependencies are pure pip; no external tools required.

```bash
pip install pineuro
```

See the [Installation Guide](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/getting-started/installation.md) for platform-specific notes (on Linux, install wxPython first), installing from source, and troubleshooting.

## GUI

```bash
pineuro-gui
```

The desktop GUI provides study configuration, participant workflows, live activation plots, and QC reports. See the [First Session Tutorial](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/getting-started/first-session.md) for a complete walkthrough.

## Documentation

**Getting started**
- [Installation](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/getting-started/installation.md): Setup and dependencies
- [Quick Start](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/getting-started/quickstart.md): GUI quick start (with an API snippet)
- [First Session](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/getting-started/first-session.md): End-to-end GUI walkthrough

**User guide**: [Full index](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/index.md)

[Pipeline Overview](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/pipeline-overview.md) · [Study Configuration](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/study-configuration.md) · [Preprocessing](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/preprocessing.md) · [Mask Extraction](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/mask-extraction.md) · [Feedback Tasks](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/feedback-tasks.md) · [Tasks](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/tasks.md) · [Sham & Blinding](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/sham-and-blinding.md) · [Multi-Session](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/multi-session.md) · [Display Setup](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/display-setup.md) · [Simulation Mode](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/simulation-mode.md) · [QC Reports](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/qc-reports.md) · [Network Testing](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/docs/user-guide/network-testing.md)

## Citation

If you use pineuro in your research, please cite:

```
Vandewouw, M. M. (2026). pineuro: Python for Individualized Neurofeedback.
https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro
```

## License

Apache License 2.0. See [LICENSE](https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro/blob/main/LICENSE) for details.

