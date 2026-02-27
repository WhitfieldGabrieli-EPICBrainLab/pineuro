# pineuro

**Python for Individualized Neurofeedback**

A Python library for real-time fMRI neurofeedback research. pineuro provides a complete pipeline from DICOM acquisition through preprocessing, network mask extraction, and moment-to-moment BOLD activation estimation, with a desktop GUI for study management and participant workflows.

## Features

**Core pipeline**
- **DICOM processing** — Watch for incoming scanner DICOMs and process volumes as they arrive
- **Preprocessing** — Motion correction, brain extraction, and multi-run coregistration (ANTsPy)
- **ICA mask extraction** — Participant-specific network masks via ICA and Yeo atlas
- **GLM mask extraction** — Network masks from task localizer contrasts
- **MNI atlas masks** — Atlas-based ROIs without a localizer run
- **Incremental GLM** — Online activation estimation using Gentleman's algorithm
- **Connectivity feedback** — Real-time functional connectivity between network pairs

**Study management**
- **Desktop GUI** — Study configuration and participant workflows (`pineuro-gui`)
- **Session orchestration** — Multi-run and multi-visit protocols
- **Sham / control conditions** — Sham feedback, replay providers, and randomization
- **QC reports** — HTML quality reports for each pipeline stage

**Task framework**
- **Built-in tasks** — Ball (dual-network) and thermometer (single-network) feedback displays
- **Task plugin system** — `@pineuro_task` decorator and entry-point registration
- **Simulation mode** — Offline DICOM replay at configurable speed

## Installation

**Requires Python 3.10 or 3.11** (`>=3.10,<3.12`). All dependencies are pure pip — no external tools required.

```bash
git clone https://github.com/marlvan/pineuro.git
cd pineuro
pip install -e .
```

See the [Installation Guide](docs/getting-started/installation.md) for platform-specific notes and troubleshooting.

## GUI

```bash
pineuro-gui
```

The desktop GUI provides study configuration, participant workflows, live activation plots, and QC reports. See the [First Session Tutorial](docs/getting-started/first-session.md) for a complete walkthrough.

## Documentation

**Getting started**
- [Installation](docs/getting-started/installation.md) — Setup and dependencies
- [Quick Start](docs/getting-started/quickstart.md) — Python API quick start
- [First Session](docs/getting-started/first-session.md) — End-to-end GUI walkthrough

**User guide** — [Full index](docs/index.md)

[Pipeline Overview](docs/user-guide/pipeline-overview.md) · [Study Configuration](docs/user-guide/study-configuration.md) · [Preprocessing](docs/user-guide/preprocessing.md) · [Mask Extraction](docs/user-guide/mask-extraction.md) · [Feedback Sessions](docs/user-guide/feedback-sessions.md) · [Tasks](docs/user-guide/tasks.md) · [Sham & Blinding](docs/user-guide/sham-and-blinding.md) · [Multi-Visit](docs/user-guide/multi-visit.md) · [Simulation Mode](docs/user-guide/simulation-mode.md) · [QC Reports](docs/user-guide/qc-reports.md)

## Citation

If you use pineuro in your research, please cite:

```
Vandewouw, M. M. (2026). pineuro: Python for Individualized Neurofeedback.
https://github.com/marlvan/pineuro
```

Key references:
- Gentleman, W. M. (1974). Basic procedures for large, sparse or weighted linear least squares problems. *Applied Statistics*, 23(3), 448-454.
- Hinds, O. et al. (2011). Computing moment-to-moment BOLD activation for real-time neurofeedback. *NeuroImage*, 54(1), 361-368.
- Yeo, B. T. et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125-1165.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Contact

Marlee M Vandewouw — marlee.vandewouw@outlook.com
