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

Monitor enumeration by position matches between screeninfo and pyglet.

#### Scanner triggers and button presses while the control monitor is focused

During a run the operator works in the control-monitor GUI, so the participant's stimulus window is usually unfocused, and macOS delivers key presses only to the focused app. Whenever pineuro waits for a scanner trigger or a participant response (instructions, ratings, sliders, catch trials), it therefore also reads global key state through Quartz (`CGEventSourceKeyState`, installed with PsychoPy), so the button box works without clicking on the stimulus screen.

macOS may require **Input Monitoring** permission for this. At task start pineuro logs `Unfocused-key poll backend=quartz (input_monitoring_access=...)`. If it reports `False` and presses are missed while the stimulus window is unfocused, open **System Settings → Privacy & Security → Input Monitoring**, enable the app you launch pineuro from (Terminal, iTerm, or your IDE), and restart that app.

### Linux

pineuro uses Xlib to position the PsychoPy window at the correct monitor coordinates; it detects the display server (X11 vs Wayland) and adjusts accordingly.

#### Scanner triggers while the control monitor is focused (Wayland)

During a run the operator works in the control-monitor GUI, so the participant's stimulus window is unfocused. To still receive scanner triggers / advance keys there, pineuro reads global key state:

- **X11 / Xorg** — works out of the box.
- **Wayland** — the compositor does not deliver keys to unfocused windows, so pineuro reads `/dev/input/event*` directly via `evdev`, which requires read access to those device nodes. On many Wayland setups the logged-in user is **not** granted that access automatically, so triggers are silently ignored while the stimulus window is unfocused. Add your user to the `input` group:

  ```bash
  sudo usermod -aG input $USER   # then log out and back in
  ```

  Verify with `id | grep input`. Alternatively, log in with an **Xorg/X11** session, which needs no extra permissions.

The detected session type and chosen backend are logged at task start (`Unfocused-key poll backend=evdev|x11 ...`); if evdev cannot read the keyboard you'll see a `permission denied on /dev/input/event*` warning pointing to this same fix.

## Projector Setups

For MRI scanner setups where the participant views a projection screen via a mirror:
- Connect the projector as a second display
- Set `presenter_monitor` to the projector's position (left or right)
- The PsychoPy task opens fullscreen on the projector
- The operator views the GUI on the other monitor
