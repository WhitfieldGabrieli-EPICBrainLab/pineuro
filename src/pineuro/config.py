"""Global configuration settings for pineuro.

This module provides centralized configuration for display settings,
platform detection, and cross-platform defaults. Settings can be modified
before importing other pineuro modules to customize behavior.

Display Settings
----------------
QC_BROWSER_SCREEN : int
    Screen index for QC report browser windows (0 = primary, 1 = secondary)

Platform Detection
------------------
is_wayland() : Check if running under Wayland display server
get_platform() : Get normalized platform name

Examples
--------
>>> import pineuro.config as config
>>> # Configure before importing other modules
>>> config.QC_BROWSER_SCREEN = 0  # Use primary monitor for QC
>>>
>>> # Now import and use pineuro
>>> from pineuro import run_feedback_task
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


# =============================================================================
# Platform Detection
# =============================================================================


def is_wayland() -> bool:
    """Detect if running under Wayland display server.

    Returns
    -------
    bool
        True if XDG_SESSION_TYPE is 'wayland' or WAYLAND_DISPLAY is set,
        False otherwise.

    Notes
    -----
    WAYLAND_DISPLAY is checked in addition to XDG_SESSION_TYPE because a
    spawned task subprocess does not always inherit XDG_SESSION_TYPE, whereas
    WAYLAND_DISPLAY is present whenever a Wayland compositor is in use (it is
    also set for XWayland clients, which is exactly what we want -- they must
    poll keys via evdev). A pure Xorg session sets neither, so the X11 path is
    unaffected.

    Some X11-dependent features (like Xlib window positioning) may not
    work correctly under Wayland without XWayland compatibility layer.
    """
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def get_platform() -> Literal["darwin", "linux", "windows"]:
    """Get normalized platform name.

    Returns
    -------
    str
        One of: 'darwin' (macOS), 'linux', 'windows'
    """
    if sys.platform == "darwin":
        return "darwin"
    elif sys.platform == "win32":
        return "windows"
    else:
        return "linux"


# =============================================================================
# Platform-Specific Defaults
# =============================================================================

# Note: The default QC_BROWSER_SCREEN=0 is kept for all platforms. On
# Linux/Wayland, the Xlib workaround in task_base.py handles window positioning.
