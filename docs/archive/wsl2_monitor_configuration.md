# PsychoPy Monitor Configuration on WSL2

## Problem

PsychoPy windows ignore the `screen` parameter when running from within a Flet GUI on WSL2, always appearing on the left monitor regardless of settings.

## Root Cause

WSL2 uses WSLg (Windows Subsystem for Linux GUI), which runs on Wayland with XWayland for X11 compatibility. When a Flet/Flutter GUI is running, the Wayland compositor places new windows on the same monitor as the focused application (Flet), ignoring application requests for specific screen placement.

## Solution

The working solution uses python-xlib to move the window after creation:

1. Create window at target monitor's full resolution (windowed mode, not fullscreen)
2. Call `win.flip()` to ensure the window appears
3. Use python-xlib to find the PsychoPy window by name and move it to the target x position

This is implemented in `BallTask._move_window_xlib()` in [ball.py](../src/pineuro/tasks/ball.py).

```python
# Create window at target monitor's size
self.win = visual.Window(
    fullscr=False,
    size=(target_monitor.width, target_monitor.height),
    pos=(0, 0),
    screen=0,
    units='height',
    color='black',
    allowGUI=False,
)
self.win.flip()

# Move to target monitor using Xlib
time.sleep(0.2)  # Give window time to appear
self._move_window_xlib(target_monitor.x)
```

## Platform-Specific Behavior

| Platform | Solution |
|----------|----------|
| macOS | Use `screen` parameter directly - works correctly |
| Native Linux (X11) | Use `screen` parameter directly - works correctly |
| WSL2 (WSLg) | Use Xlib workaround to move window after creation |

## Dependencies

- `python-xlib` - Required for the WSL2 workaround (already a dependency of PsychoPy)

## What Doesn't Work on WSL2

These approaches were tested and do NOT work when Flet is running:

- PsychoPy's `screen` parameter (ignored by Wayland)
- PsychoPy's `pos` parameter (ignored by Wayland)
- Pyglet's `set_fullscreen(screen=...)` (ignored by Wayland)
- Creating window in subprocess (still placed on focused monitor)
- Direct GLFW with monitor parameter (still placed on focused monitor)
- Minimizing Flet before launch (doesn't change compositor behavior)
- xdotool click to change focus (Wayland ignores it)

The only working approach is to move the window after creation using X11 calls.

## Environment

- WSL2 with WSLg
- Linux kernel: 6.6.87.1-microsoft-standard-WSL2
- Display server: Wayland (via XWayland)
