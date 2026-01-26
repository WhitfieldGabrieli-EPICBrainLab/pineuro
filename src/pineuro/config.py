"""Global configuration settings for pineuro.

This module provides centralized configuration for display settings,
platform detection, and cross-platform defaults. Settings can be modified
before importing other pineuro modules to customize behavior.

Display Settings
----------------
QC_BROWSER_SCREEN : int
    Screen index for QC report browser windows (0 = primary, 1 = secondary)
PSYCHOPY_SCREEN : int
    Screen index for PsychoPy task windows (0 = primary, 1 = secondary)

Platform Detection
------------------
is_wsl() : Check if running under Windows Subsystem for Linux
is_wayland() : Check if running under Wayland display server
get_platform() : Get normalized platform name

Examples
--------
>>> import pineuro.config as config
>>> # Configure before importing other modules
>>> config.PSYCHOPY_SCREEN = 0  # Use primary monitor for tasks
>>> config.QC_BROWSER_SCREEN = 0  # Use primary monitor for QC
>>>
>>> # Now import and use pineuro
>>> from pineuro import run_feedback_session
"""

import os
import sys
from typing import Literal

# =============================================================================
# Display Configuration
# =============================================================================

# Screen index for QC report browser windows
# 0 = primary monitor, 1 = secondary monitor
QC_BROWSER_SCREEN: int = 0

# Screen index for PsychoPy task windows
# 0 = primary monitor, 1 = secondary monitor
PSYCHOPY_SCREEN: int = 1


# =============================================================================
# Platform Detection
# =============================================================================

def is_wsl() -> bool:
    """Detect if running under Windows Subsystem for Linux (WSL).

    Returns
    -------
    bool
        True if running under WSL1 or WSL2, False otherwise.

    Notes
    -----
    WSL2 with WSLg supports GUI applications but has known issues with
    multi-monitor configurations and some X11 features.
    """
    if sys.platform != 'linux':
        return False
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            return 'microsoft' in version_info or 'wsl' in version_info
    except (OSError, IOError):
        return False


def is_wayland() -> bool:
    """Detect if running under Wayland display server.

    Returns
    -------
    bool
        True if XDG_SESSION_TYPE is 'wayland', False otherwise.

    Notes
    -----
    Some X11-dependent features (like Xlib window positioning) may not
    work correctly under Wayland without XWayland compatibility layer.
    """
    return os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland'


def get_platform() -> Literal['darwin', 'linux', 'wsl', 'windows']:
    """Get normalized platform name.

    Returns
    -------
    str
        One of: 'darwin' (macOS), 'linux', 'wsl', 'windows'

    Notes
    -----
    WSL is detected as a distinct platform from native Linux because
    it has different display server behavior and limitations.
    """
    if sys.platform == 'darwin':
        return 'darwin'
    elif sys.platform == 'win32':
        return 'windows'
    elif is_wsl():
        return 'wsl'
    else:
        return 'linux'


# =============================================================================
# Platform-Specific Defaults
# =============================================================================

# Note: Default settings (PSYCHOPY_SCREEN=1, QC_BROWSER_SCREEN=0) are kept
# for all platforms. On WSL2, multi-monitor may require additional setup
# but the Xlib workaround in ball.py handles window positioning.
