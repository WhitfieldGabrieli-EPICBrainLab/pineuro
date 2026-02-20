# Display Setup

Neurofeedback experiments typically use two monitors: one for the operator (GUI with real-time plots) and one for the participant (PsychoPy task display). pineuro supports multi-monitor and single-monitor configurations.

## Multi-Monitor Configuration

In the Study Editor Display section, assign monitors by position:

| Setting | Description |
|---------|-------------|
| **Control Monitor** | `left` or `right` — The operator's GUI with real-time plots, logs, and controls |
| **Presenter Monitor** | `left` or `right` — The participant's PsychoPy fullscreen task display |

These should be set to different monitors. The GUI opens on the control monitor; the PsychoPy task opens fullscreen on the presenter monitor.

## Single-Monitor Mode

When only one monitor is detected, pineuro runs both the GUI and the task display on the same screen. The PsychoPy window opens in a smaller windowed mode rather than fullscreen.

## Platform Notes

### Windows

pineuro uses position-based matching between screeninfo (which detects monitors by physical position) and pyglet (which PsychoPy uses internally, ordered by Windows adapter). This ensures the correct monitor is selected regardless of which monitor is set as the Windows primary display.

### macOS

Monitor enumeration by position matches between screeninfo and pyglet. No special handling needed.

### Linux

On X11/GNOME, pineuro uses Xlib to position the PsychoPy window at the correct monitor coordinates. Wayland support detects the display server and adjusts accordingly.

## Projector Setups

For MRI scanner setups where the participant views a projection screen via a mirror:
- Connect the projector as a second display
- Set `presenter_monitor` to the projector's position (left or right)
- The PsychoPy task opens fullscreen on the projector
- The operator views the GUI on the other monitor
