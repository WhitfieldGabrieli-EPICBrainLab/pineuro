# Installation

This guide covers installing pineuro and its dependencies.

## System Requirements

### Supported Platforms
- Linux (Ubuntu 20.04+, Debian 11+)
- macOS (11.0+)
- Windows (10+)

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

# Install system dependency: dcm2niix
# Linux:   sudo apt install dcm2niix
# macOS:   brew install dcm2niix
# Windows: winget install dcm2niix

# Install pineuro
pip install -e /path/to/pineuro
```

## Dependencies

### dcm2niix

dcm2niix converts DICOM files to NIfTI format. This is the only external (non-Python) dependency.

**Ubuntu/Debian:**
```bash
sudo apt install dcm2niix
```

**macOS:**
```bash
brew install dcm2niix
```

**Windows:**
```bash
winget install dcm2niix
```

**Verify installation:**
```bash
dcm2niix --version  # Should show version
```

### Python Dependencies

All Python dependencies are installed automatically via pip:
- antspyx - Motion correction, brain extraction, registration
- nibabel - NIfTI I/O
- nilearn - Neuroimaging analysis (ICA, atlas fetching)
- psychopy - Visual stimulus display
- flet - GUI framework
- numpy, scipy, pandas, matplotlib, pydicom, pyyaml, screeninfo

## Development Installation

For contributing to pineuro:

```bash
# Clone repository
git clone https://github.com/marlvan/pineuro.git
cd pineuro

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Verify Installation

Run the test script to verify everything is working:

```bash
python tests/test.py
```

This will run through the core pipeline with simulated data.

## Troubleshooting

### ANTsPy installation fails

ANTsPy can be slow to install (compiles from source). If it fails:

```bash
# Try installing pre-built wheels
pip install antspyx --no-cache-dir

# Or build with specific options
CMAKE_BUILD_PARALLEL_LEVEL=4 pip install antspyx
```

## Next Steps

- [Quick Start](quickstart.md) - Run your first preprocessing
- [First Session](first-session.md) - Complete tutorial
