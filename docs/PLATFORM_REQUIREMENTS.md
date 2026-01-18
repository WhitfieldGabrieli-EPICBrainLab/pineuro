# Platform Requirements and Compatibility

This document describes platform-specific requirements, known issues, and configuration options for pyrtnf.

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (X11) | Fully supported | Primary development platform |
| macOS | Fully supported | Tested on Intel and Apple Silicon |
| WSL2 | Partial support | Single-monitor mode recommended |
| Windows (native) | Not supported | FSL not available natively |
| Linux (Wayland) | Partial support | May require XWayland |

## Required Dependencies

### FSL (FMRIB Software Library)

FSL is required for preprocessing (motion correction, brain extraction, registration).

**Availability:**
- Linux: Install via NeuroDebian or FSL installer
- macOS: Install via FSL installer
- Windows: Not available natively (use WSL2)

**Required commands:** `mcflirt`, `bet`, `fslmaths`, `fslmerge`, `convert_xfm`

**Installation:**
```bash
# Linux (Debian/Ubuntu via NeuroDebian)
wget -O- http://neuro.debian.net/lists/$(lsb_release -cs).us-nh.full | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9
sudo apt-get update
sudo apt-get install fsl-complete

# macOS
# Download installer from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
```

### dcm2niix

Required for DICOM to NIfTI conversion.

**Installation:**
```bash
# Linux
sudo apt-get install dcm2niix

# macOS
brew install dcm2niix

# Windows (if using WSL2)
sudo apt-get install dcm2niix
```

### ANTsPy

Required for registration (multi-run coregistration, MNI alignment).

**Installation:**
```bash
pip install antspyx
```

Note: ANTsPy wheels are available for Linux and macOS. Windows support is limited.

### Python Dependencies

Install via pip:
```bash
pip install -e .  # From pyrtnf directory
```

Core dependencies: numpy, nibabel, nilearn, scipy, pandas, pydicom, templateflow, psychopy, flet

## Display Configuration

### Multi-Monitor Setup

pyrtnf supports dual-monitor configurations for neurofeedback experiments:
- **Monitor 0 (Primary):** Experimenter GUI, QC reports
- **Monitor 1 (Secondary):** Participant-facing PsychoPy task

Configure via `pyrtnf.config`:
```python
import pyrtnf.config as config
config.PSYCHOPY_SCREEN = 1    # PsychoPy on secondary monitor
config.QC_BROWSER_SCREEN = 0  # QC reports on primary monitor

# Import pyrtnf after configuring
from pyrtnf import run_feedback_session
```

### Platform-Specific Display Notes

#### Linux (X11)
- Full multi-monitor support
- QC browser positioning uses `xrandr` and `wmctrl`
- Install these utilities: `sudo apt-get install x11-xserver-utils wmctrl`

#### macOS
- Full multi-monitor support
- PsychoPy uses native screen parameter
- QC browser opens on default screen (positioning not automated)

#### WSL2 (Windows Subsystem for Linux)
- **Single-monitor mode is recommended**
- pyrtnf automatically defaults to `PSYCHOPY_SCREEN = 0` on WSL2
- Multi-monitor PsychoPy may not work correctly due to WSLg/XWayland limitations

**Known WSL2 Issues:**
1. PsychoPy's `screen` parameter may be ignored by XWayland
2. Xlib window positioning workarounds don't work reliably
3. Flet (GUI framework) may have display issues with XWayland

**Workaround for WSL2:**
Run in single-monitor mode (default on WSL2):
```python
import pyrtnf.config as config
config.PSYCHOPY_SCREEN = 0
config.QC_BROWSER_SCREEN = 0
```

#### Wayland
- Some X11-based features may not work without XWayland
- Xlib window positioning is skipped on Wayland
- Consider using an X11/Xorg session for full compatibility

## Checking Platform Requirements

Use the built-in check function to verify your environment:
```python
from pyrtnf.utils import check_platform_requirements

warnings = check_platform_requirements()
for warning in warnings:
    print(f"Warning: {warning}")
```

## Troubleshooting

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

2. Set the screen explicitly before importing pyrtnf:
   ```python
   import pyrtnf.config as config
   config.PSYCHOPY_SCREEN = 0  # or 1
   ```

### QC browser opens on wrong monitor (Linux)

Install wmctrl and xrandr:
```bash
sudo apt-get install x11-xserver-utils wmctrl
```

### FSL commands not found

Ensure FSL is installed and environment is set up:
```bash
# Add to ~/.bashrc
export FSLDIR=/usr/local/fsl  # or your FSL path
source $FSLDIR/etc/fslconf/fsl.sh
export PATH=$FSLDIR/bin:$PATH
```

### WSL2: GUI applications don't appear

1. Ensure WSLg is working:
   ```bash
   # Test with a simple GUI app
   sudo apt-get install x11-apps
   xclock
   ```

2. If using older WSL2 without WSLg, install an X server on Windows (VcXsrv, X410) and set:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   ```

### ImportError for antspyx on Windows

ANTsPy doesn't have native Windows wheels. Use WSL2 for Windows environments.
