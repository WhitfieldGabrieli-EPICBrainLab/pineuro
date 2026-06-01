"""Cross-platform monitor detection utilities.

This module provides functions to detect and select monitors across
different platforms (macOS, Windows, Linux).

Functions
---------
get_monitors() : Get all monitors sorted by x position (left-to-right)
get_monitor_by_position(position) : Get monitor by "left" or "right" label

Examples
--------
>>> from pineuro.display import get_monitors, get_monitor_by_position
>>> monitors = get_monitors()
>>> for m in monitors:
...     print(f"{m.name}: {m.width}x{m.height} at ({m.x}, {m.y})")
>>> left_monitor = get_monitor_by_position("left")
>>> right_monitor = get_monitor_by_position("right")
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonitorInfo:
    """Information about a display monitor.

    Attributes
    ----------
    index : int
        Index of the monitor (0-based, sorted by x position)
    x : int
        X coordinate of the monitor's left edge
    y : int
        Y coordinate of the monitor's top edge
    width : int
        Width in pixels
    height : int
        Height in pixels
    name : str
        Monitor name/identifier (e.g., 'XWAYLAND0', 'HDMI-1')
    is_primary : bool
        Whether this is the primary monitor (unreliable on XWayland)
    """

    index: int
    x: int
    y: int
    width: int
    height: int
    name: str
    is_primary: bool


def get_monitors() -> list[MonitorInfo]:
    """Get all monitors sorted by x position (left-to-right).

    Returns
    -------
    list of MonitorInfo
        List of detected monitors, sorted by x position.
        Empty list if no monitors detected or screeninfo unavailable.

    """
    try:
        from screeninfo import get_monitors as si_get_monitors

        raw_monitors = si_get_monitors()

        # Sort by x position (left to right)
        sorted_monitors = sorted(raw_monitors, key=lambda m: (m.x, m.y))

        monitors = []
        for i, m in enumerate(sorted_monitors):
            monitors.append(
                MonitorInfo(
                    index=i,
                    x=m.x,
                    y=m.y,
                    width=m.width,
                    height=m.height,
                    name=m.name or f"Monitor {i}",
                    is_primary=m.is_primary if m.is_primary is not None else False,
                )
            )

        return monitors

    except ImportError:
        logger.warning("screeninfo not available - monitor detection disabled")
        return []
    except Exception as e:
        logger.warning(f"Failed to detect monitors: {e}")
        return []


def get_monitor_by_position(position: str) -> MonitorInfo | None:
    """Get monitor by position label.

    Parameters
    ----------
    position : {"left", "right"}
        Position label. "left" returns the leftmost monitor (lowest x),
        "right" returns the rightmost monitor (highest x).

    Returns
    -------
    MonitorInfo or None
        Monitor info if found, None if no monitors detected.
        For single-monitor setups, both "left" and "right" return the same monitor.
    """
    monitors = get_monitors()

    if not monitors:
        logger.warning(f"No monitors detected, cannot get {position} monitor")
        return None

    if len(monitors) == 1:
        return monitors[0]

    if position == "left":
        return monitors[0]  # Already sorted by x, leftmost first
    elif position == "right":
        return monitors[-1]  # Rightmost
    else:
        logger.warning(f"Invalid position '{position}', defaulting to left")
        return monitors[0]
