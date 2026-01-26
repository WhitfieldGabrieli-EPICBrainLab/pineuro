# Platform Requirements and Compatibility

This document describes platform-specific requirements, known issues, and configuration options for pineuro.

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (X11) | Fully supported | Primary development platform |
| macOS | Fully supported | Tested on Intel and Apple Silicon |
| WSL2 | Partial support | Single-monitor mode recommended |
| Windows (native) | Not supported | FSL not available natively |
| Linux (Wayland) | Partial support | May require XWayland |

## System Dependencies

### Required - Core Pipeline

| Package | Purpose | Installation |
|---------|---------|--------------|
| **FSL** | Motion correction (`mcflirt`), brain extraction (`bet`), registration (`flirt`) | See [FSL Installation](#fsl-fmrib-software-library) |
| **dcm2niix** | DICOM to NIfTI conversion | See [dcm2niix Installation](#dcm2niix) |

### Required - GUI (if using `pip install .[gui]`)

| Package | Purpose | Installation |
|---------|---------|--------------|
| **python3-tk** | File browser dialogs in study editor | `sudo apt install python3-tk` |
| **libgtk-3-0** | Flet GUI framework backend | `sudo apt install libgtk-3-0` |
| **libwebkit2gtk-4.1-0** | Flet webview rendering | `sudo apt install libwebkit2gtk-4.1-0` |

### Optional - QC Browser Positioning (Linux X11 only)

These utilities enable automatic positioning of QC report browser windows on multi-monitor setups. If not installed, QC reports still open but may appear on the wrong monitor.

| Package | Purpose | Installation |
|---------|---------|--------------|
| **wmctrl** | Window positioning | `sudo apt install wmctrl` |
| **x11-xserver-utils** | Monitor geometry detection (`xrandr`) | `sudo apt install x11-xserver-utils` |

## Quick Install

### Ubuntu/Debian - Core Only

```bash
# System packages
sudo apt install dcm2niix

# FSL (via NeuroDebian)
wget -O- http://neuro.debian.net/lists/$(lsb_release -cs).us-nh.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9
sudo apt update
sudo apt install fsl-complete

# Python package
pip install -e /path/to/pineuro
```

### Ubuntu/Debian - Core + GUI

```bash
# System packages (core + GUI)
sudo apt install dcm2niix python3-tk libgtk-3-0 libwebkit2gtk-4.1-0

# Optional: QC browser positioning
sudo apt install wmctrl x11-xserver-utils

# FSL (via NeuroDebian)
wget -O- http://neuro.debian.net/lists/$(lsb_release -cs).us-nh.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9
sudo apt update
sudo apt install fsl-complete

# Python package with GUI
pip install -e /path/to/pineuro[gui]
```

### macOS

```bash
# System packages
brew install dcm2niix

# FSL - download installer from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

# Python package
pip install -e /path/to/pineuro
# Or with GUI:
pip install -e /path/to/pineuro[gui]
```

## Detailed Installation

### FSL (FMRIB Software Library)

FSL is required for preprocessing (motion correction, brain extraction, registration).

**Required commands:** `mcflirt`, `bet`, `flirt`

**Linux (Debian/Ubuntu via NeuroDebian):**
```bash
wget -O- http://neuro.debian.net/lists/$(lsb_release -cs).us-nh.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9
sudo apt update
sudo apt install fsl-complete
```

**macOS:**
Download installer from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

**Environment setup** (add to `~/.bashrc` or `~/.zshrc`):
```bash
export FSLDIR=/usr/local/fsl  # or your FSL installation path
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH
```

### dcm2niix

Required for DICOM to NIfTI conversion.

```bash
# Linux
sudo apt install dcm2niix

# macOS
brew install dcm2niix

# WSL2
sudo apt install dcm2niix
```

## Python Dependencies

Python dependencies are managed via `pyproject.toml`. Install with:

```bash
# Core functionality
pip install -e .

# With GUI
pip install -e .[gui]
```

**Core dependencies:** antspyx, matplotlib, nibabel, nilearn, numpy, pandas, psychopy, pydicom, pyyaml, scipy, templateflow, watchdog

**GUI dependencies:** flet

**Note:** ANTsPy (`antspyx`) does not have native Windows wheels. Use WSL2 for Windows environments.

## Display Configuration

### Multi-Monitor Setup

pineuro supports dual-monitor configurations for neurofeedback experiments:
- **Monitor 0 (Primary):** Experimenter GUI, QC reports
- **Monitor 1 (Secondary):** Participant-facing PsychoPy task

Configure via `pineuro.config`:
```python
import pineuro.config as config
config.PSYCHOPY_SCREEN = 1    # PsychoPy on secondary monitor
config.QC_BROWSER_SCREEN = 0  # QC reports on primary monitor

# Import pineuro after configuring
from pineuro import run_feedback_session
```

### Platform-Specific Display Notes

#### Linux (X11)
- Full multi-monitor support
- QC browser positioning uses `xrandr` and `wmctrl` (install with `sudo apt install x11-xserver-utils wmctrl`)
- If these utilities are missing, QC reports still open but window positioning is skipped

#### macOS
- Full multi-monitor support
- PsychoPy uses native screen parameter
- QC browser opens on default screen (automatic positioning not available)

#### WSL2 (Windows Subsystem for Linux)
- **Single-monitor mode is recommended**
- pineuro automatically defaults to `PSYCHOPY_SCREEN = 0` on WSL2
- Multi-monitor PsychoPy may not work correctly due to WSLg/XWayland limitations

**Known WSL2 Issues:**
1. PsychoPy's `screen` parameter may be ignored by XWayland
2. Xlib window positioning workarounds don't work reliably
3. Flet (GUI framework) may have display issues with XWayland

**Workaround:** Run in single-monitor mode (default on WSL2):
```python
import pineuro.config as config
config.PSYCHOPY_SCREEN = 0
config.QC_BROWSER_SCREEN = 0
```

#### Wayland
- Some X11-based features may not work without XWayland
- Window positioning is automatically skipped on Wayland
- Consider using an X11/Xorg session for full compatibility

## Verifying Installation

### Built-in Check

Use the built-in function to verify your environment:
```python
from pineuro.utils import check_platform_requirements

warnings = check_platform_requirements()
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")
else:
    print("All requirements satisfied")
```

### Manual Verification

```bash
# Check FSL
which mcflirt bet flirt

# Check dcm2niix
which dcm2niix

# Check optional utilities (Linux)
which wmctrl xrandr

# Check Python imports
python -c "import pineuro; print('pineuro OK')"
python -c "import ants; print('ANTsPy OK')"
python -c "from psychopy import visual; print('PsychoPy OK')"
```

## Troubleshooting

### FSL commands not found

Ensure FSL is installed and environment is set up:
```bash
# Add to ~/.bashrc
export FSLDIR=/usr/local/fsl  # or your FSL path
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH
```

Then restart your terminal or run `source ~/.bashrc`.

### PsychoPy doesn't open on the correct monitor

1. Check your display configuration:
   ```bash
   # Linux
   xrandr --current

   # Check screen indices in Python
   import pyglet
   screens = pyglet.canvas.Display().get_screens()
   for i, s in enumerate(screens):
       print(f"Screen {i}: {s.width}x{s.height}")
   ```

2. Set the screen explicitly before importing pineuro:
   ```python
   import pineuro.config as config
   config.PSYCHOPY_SCREEN = 0  # or 1
   ```

### QC browser opens on wrong monitor (Linux)

Install wmctrl and xrandr:
```bash
sudo apt install x11-xserver-utils wmctrl
```

### WSL2: GUI applications don't appear

1. Ensure WSLg is working:
   ```bash
   # Test with a simple GUI app
   sudo apt install x11-apps
   xclock
   ```

2. If using older WSL2 without WSLg, install an X server on Windows (VcXsrv, X410) and set:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   ```

### ImportError for antspyx on Windows

ANTsPy doesn't have native Windows wheels. Use WSL2 for Windows environments.

### tkinter not available (GUI file dialogs)

Install the system package:
```bash
sudo apt install python3-tk
```

### Flet GUI doesn't start (Linux)

Ensure GTK and WebKit are installed:
```bash
sudo apt install libgtk-3-0 libwebkit2gtk-4.1-0
```
