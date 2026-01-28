# Installation

This guide covers installing pineuro and its dependencies.

## System Requirements

### Supported Platforms
- Linux (Ubuntu 20.04+, Debian 11+)
- macOS (11.0+)
- Windows via WSL2

### Hardware
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for dependencies + space for data
- **Display**: Multi-monitor support for PsychoPy tasks

## Quick Install (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv \
    dcm2niix \
    fsl-core

# Set up FSL environment
echo 'export FSLDIR=/usr/share/fsl' >> ~/.bashrc
echo 'source $FSLDIR/etc/fslconf/fsl.sh' >> ~/.bashrc
source ~/.bashrc

# Create virtual environment
python3.11 -m venv pineuro_env
source pineuro_env/bin/activate

# Install pineuro
pip install -e /path/to/pineuro
```

## Dependencies

### FSL (FMRIB Software Library)

FSL provides motion correction (mcflirt) and brain extraction (bet).

**Ubuntu/Debian:**
```bash
sudo apt install fsl-core
```

**macOS (via Homebrew):**
```bash
brew install --cask fsl
```

**Verify installation:**
```bash
mcflirt --help  # Should show usage
bet --help      # Should show usage
```

### dcm2niix

dcm2niix converts DICOM files to NIfTI format.

**Ubuntu/Debian:**
```bash
sudo apt install dcm2niix
```

**macOS:**
```bash
brew install dcm2niix
```

**Verify installation:**
```bash
dcm2niix --version  # Should show version
```

### ANTsPy

ANTsPy provides image registration. It's installed automatically with pineuro.

```bash
pip install antspyx
```

### Python Dependencies

All Python dependencies are installed automatically:
- numpy, scipy - Numerical operations
- nibabel - NIfTI I/O
- nilearn - Neuroimaging analysis (ICA, etc.)
- pydicom - DICOM header reading
- psychopy - Visual stimulus display
- flet - GUI framework

## Development Installation

For contributing to pineuro:

```bash
# Clone repository
git clone https://github.com/your-org/pineuro.git
cd pineuro

# Create environment
python3.11 -m venv venv
source venv/bin/activate

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

### FSL not found

If `mcflirt` or `bet` commands aren't found:

1. Ensure FSL is installed: `ls /usr/share/fsl` or `ls $FSLDIR`
2. Source the FSL setup: `source $FSLDIR/etc/fslconf/fsl.sh`
3. Add to your shell profile (~/.bashrc or ~/.zshrc)

### ANTsPy installation fails

ANTsPy can be slow to install (compiles from source). If it fails:

```bash
# Try installing pre-built wheels
pip install antspyx --no-cache-dir

# Or build with specific options
CMAKE_BUILD_PARALLEL_LEVEL=4 pip install antspyx
```

### PsychoPy display issues on WSL2

WSL2 requires special configuration for PsychoPy displays. See [Platform Requirements](../technical/platform-requirements.md) for details.

## Next Steps

- [Quick Start](quickstart.md) - Run your first preprocessing
- [First Session](first-session.md) - Complete tutorial
