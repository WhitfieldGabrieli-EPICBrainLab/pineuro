# Installation

This guide covers installing pineuro and its dependencies.

## System Requirements

### Supported Platforms
- Linux (Ubuntu 20.04+, Debian 11+)
- macOS (11.0+)
- Windows (10+)

### Software
- **Python 3.10 or 3.11** (`>=3.10,<3.12`) — Python 3.12+ is not supported.

### Hardware
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for dependencies + space for data
- **Display**: Multi-monitor support for PsychoPy tasks

## Quick Install

```bash
# Create virtual environment
python -m venv pineuro_env
source pineuro_env/bin/activate  # Linux/macOS
# pineuro_env\Scripts\activate   # Windows

# Clone and install pineuro (all dependencies included)
git clone https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro.git
cd pineuro
pip install -e .
```

All dependencies — including dcm2niix, ANTsPy, PsychoPy, and nilearn — are installed automatically via pip. No external tools are required.

## Running at the Scanner

If the scanner sends DICOMs to this computer through the Siemens BOLD Dot Add-In, log in to
the account named in the add-in's **User Name** (the account that owns the shared folder) and
run pineuro from that account, every session. The scanner's files are readable only by that
account, so under any other login pineuro cannot read the incoming DICOMs even though they
appear in the folder. The scanner connection check (run before each acquisition) warns when
pineuro is running under a different account. See
[Network Testing → Scanner-Side Configuration](../user-guide/network-testing.md#scanner-side-configuration-siemens-bold-dot-add-in)
for the full setup.

## Dependencies

All Python dependencies are installed automatically:
- **dcm2niix** — DICOM to NIfTI conversion
- **antspyx** — Motion correction, brain extraction, registration
- **nibabel** — NIfTI I/O
- **nilearn** — Neuroimaging analysis (ICA, atlas fetching)
- **psychopy** — Visual stimulus display
- **flet** — GUI framework
- numpy, scipy, pandas, matplotlib, pydicom, pyyaml, screeninfo, pygame, templateflow, watchdog
- **Linux only**: `evdev` and `python-xlib` — keyboard polling and window positioning

## Development Installation

For contributing to pineuro:

```bash
# Clone repository
git clone https://github.com/WhitfieldGabrieli-EPICBrainLab/pineuro.git
cd pineuro

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Troubleshooting

### ANTsPy installation fails

ANTsPy can be slow to install (compiles from source). If it fails:

```bash
# Try installing pre-built wheels
pip install antspyx --no-cache-dir

# Or build with specific options
CMAKE_BUILD_PARALLEL_LEVEL=4 pip install antspyx
```

### PsychoPy installation issues

PsychoPy requires wxPython, which may need to be installed separately on some platforms. See [PsychoPy installation docs](https://www.psychopy.org/download.html) if you encounter issues.

### Linux: scanner triggers not detected under Wayland

Under Wayland, receiving scanner triggers / advance keys while the operator is
focused on the control monitor requires read access to `/dev/input` — add your
user to the `input` group (`sudo usermod -aG input $USER`, then log out and back
in), or use an Xorg session. See
[Display Setup → Linux](../user-guide/display-setup.md#scanner-triggers-while-the-control-monitor-is-focused-wayland)
for the full explanation.

## Next Steps

- [Quick Start](quickstart.md) — Run your first preprocessing
- [First Session Tutorial](first-session.md) — Complete end-to-end walkthrough
