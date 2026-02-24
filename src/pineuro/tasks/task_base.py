"""
Common base class for all PsychoPy tasks (feedback and localizer).

Summary
-------
Provides shared infrastructure for PsychoPy window management, screenshot
mirroring, instruction display, trigger waiting, and simulator integration.
Both BaseTask (neurofeedback) and BaseLocalizerTask (stimulus presentation)
inherit from TaskBase.

Components
----------
- TaskBase: ABC with window, mirror, trigger, and simulator support

Dependencies
------------
PsychoPy (for concrete tasks)
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from abc import ABC
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class TaskBase(ABC):  # noqa: B024
    """Common abstract base for all PsychoPy tasks.

    Provides shared functionality:
    - PsychoPy window creation (cross-platform)
    - Screenshot-based GUI mirroring
    - Instruction slide display
    - Scanner trigger waiting (keyboard-based, with simulator support)
    - Simulator thread management

    Subclasses (BaseTask, BaseLocalizerTask) add their specific lifecycle.

    Parameters
    ----------
    trigger_keys : list of str, optional
        Keys that trigger acquisition start (default: ["t", "+", "5", "equal"])
    simulator : VolumeSimulator, optional
        Volume simulator for offline testing. Replicates real scanner behavior:
        injects a synthetic trigger key, then produces DICOMs at TR intervals.
    """

    task_id: str = "task_base"

    def __init__(
        self,
        trigger_keys: list[str] | None = None,
        simulator=None,
        **kwargs,
    ):
        self.trigger_keys = trigger_keys or ["t", "+", "5", "equal"]
        self._simulator = simulator
        self._simulator_thread: threading.Thread | None = None

        self._running = False
        self.win: Any = None

        # Shared event fired when trigger is received (for external timing sync)
        self.trigger_event: threading.Event | None = None

        # Mirror callback for GUI visualization
        self._mirror_callback: Callable[[dict[str, Any]], None] | None = None
        self._mirror_frame_counter = 0
        self._mirror_frame_skip = 4  # Only capture every Nth frame

    # =========================================================================
    # Mirror callback support
    # =========================================================================

    def set_mirror_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Set callback for GUI mirroring.

        The callback receives visual state each frame for rendering in a GUI mirror.

        Parameters
        ----------
        callback : Callable[[Dict[str, Any]], None]
            Function that receives the visual state dictionary
        """
        self._mirror_callback = callback

    def _capture_and_emit_screenshot(self, force: bool = False) -> None:
        """Capture current PsychoPy frame and emit to mirror callback.

        This provides a true mirror of exactly what participants see on the
        PsychoPy window. Works for any task without task-specific code.

        IMPORTANT: Call this BEFORE win.flip(), not after. This captures the
        back buffer (what has been drawn but not yet displayed), which works
        reliably across all platforms.

        To reduce flickering from capture overhead, this method only captures
        every Nth frame (controlled by _mirror_frame_skip) unless force=True.

        Parameters
        ----------
        force : bool
            If True, capture immediately regardless of frame counter.
            Use for important screens like instructions that need immediate update.
        """
        if self._mirror_callback is None:
            return
        if not hasattr(self, "win") or self.win is None:
            return

        # Frame skipping: only capture every Nth frame to reduce overhead
        # unless force=True (for important screens like instructions)
        if not force:
            self._mirror_frame_counter += 1
            if self._mirror_frame_counter < self._mirror_frame_skip:
                return
            self._mirror_frame_counter = 0

        frame = None
        frame_resized = None
        buffer = None

        try:
            import base64
            import io

            # Capture back buffer (drawn content before flip)
            frame = self.win.getMovieFrame(buffer="back")

            # Resize to mirror dimensions (16:9 aspect ratio)
            frame_resized = frame.resize((480, 270))

            # Convert to JPEG for faster encoding (less overhead than PNG)
            buffer = io.BytesIO()
            frame_resized.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Emit to callback
            self._mirror_callback({"type": "screenshot", "image": f"data:image/jpeg;base64,{img_base64}"})
        except Exception as e:
            logger.debug(f"Screenshot capture failed: {e}")
        finally:
            # Clean up PIL Images and BytesIO to prevent memory leaks
            if buffer is not None:
                buffer.close()
            if frame_resized is not None:
                frame_resized.close()
            if frame is not None:
                frame.close()

    # =========================================================================
    # Window Management (Cross-Platform)
    # =========================================================================

    def _move_window_xlib(self, target_x: int) -> None:
        """Move PsychoPy window to a specific x position using Xlib.

        This is a workaround for Wayland ignoring window position requests.
        Uses the actual window ID from PsychoPy's window handle.

        Parameters
        ----------
        target_x : int
            X coordinate to move window to (e.g., 1920 for right monitor)
        """
        try:
            from Xlib import display as xlib_display

            # Get the X11 window ID from PsychoPy's pyglet window
            window_id = self.win.winHandle._window

            d = xlib_display.Display()
            xlib_win = d.create_resource_object("window", window_id)
            xlib_win.configure(x=target_x, y=0)
            d.sync()
            d.close()

            logger.info(f"Moved window (id={window_id}) to x={target_x} using Xlib")

        except ImportError:
            logger.warning("Xlib not available - window positioning disabled")
        except Exception as e:
            logger.warning(f"Could not move window: {e}")

    def create_window(
        self,
        monitor_position: str = "right",
        color: str = "black",
        units: str = "height",
        **kwargs,
    ):
        """Create a fullscreen PsychoPy window with cross-platform monitor targeting.

        Handles platform differences:
        - macOS: Uses screeninfo index directly (pyglet enumerates by position)
        - Windows: Matches screeninfo (x,y) to pyglet screen index (adapter order may differ)
        - Linux: Creates window then moves it using Xlib

        Parameters
        ----------
        monitor_position : str
            Monitor position: "left", "right", or "primary" (default: "right")
        color : str
            Background color (default: 'black')
        units : str
            PsychoPy units (default: 'height')
        **kwargs
            Additional arguments passed to visual.Window

        Returns
        -------
        psychopy.visual.Window
            The created window (also stored as self.win)
        """
        from psychopy import monitors, visual

        from pineuro import config
        from pineuro.display import get_monitor_by_position

        platform = config.get_platform()

        # Create minimal monitor to avoid "Monitor specification not found" warning
        mon = monitors.Monitor("neurofeedback")

        # Get target screen using display module
        target_monitor = get_monitor_by_position(monitor_position)
        if target_monitor:
            target_screen = target_monitor.index
        else:
            target_screen = 0
            logger.warning("No monitors detected, using screen 0")

        # Reset stale pyglet state from previous runs (may be from a dead thread)
        self._reset_pyglet_for_new_window()

        # Platform-specific window creation
        if platform == "darwin":
            # macOS: pyglet enumerates by position, index matches screeninfo
            self.win = visual.Window(
                fullscr=True,
                monitor=mon,
                screen=target_screen,
                units=units,
                color=color,
                allowGUI=False,
                waitBlanking=False,
                **kwargs,
            )
        elif platform == "windows":
            # Windows: screeninfo sorts by x-position but pyglet uses adapter
            # order (DISPLAY1, DISPLAY2...). Resolve by matching (x,y) position.
            if target_monitor:
                pyglet_idx = self._resolve_pyglet_screen_index(target_monitor.x, target_monitor.y)
            else:
                pyglet_idx = 0
            self.win = visual.Window(
                fullscr=True,
                monitor=mon,
                screen=pyglet_idx,
                units=units,
                color=color,
                allowGUI=False,
                waitBlanking=False,
                **kwargs,
            )
        else:
            # Linux: Wayland/XWayland ignores screen parameter.
            # Workaround: Create window at target size, then move it using Xlib.
            if target_monitor:
                win_width = target_monitor.width
                win_height = target_monitor.height
                target_x = target_monitor.x
            else:
                win_width, win_height = (1920, 1080)
                target_x = 0

            logger.info(f"Creating {win_width}x{win_height} window, will move to x={target_x}")
            self.win = visual.Window(
                fullscr=False,
                size=(win_width, win_height),
                pos=(0, 0),
                monitor=mon,
                screen=0,
                units=units,
                color=color,
                allowGUI=False,
                waitBlanking=False,
                **kwargs,
            )
            self.win.flip()

            # Move window to target monitor using Xlib
            if target_x > 0:
                time.sleep(0.2)  # Give window time to appear
                self._move_window_xlib(target_x)

        return self.win

    def _reset_pyglet_for_new_window(self) -> None:
        """Reset pyglet global state so a new PsychoPy window can be created.

        Each task run happens in a fresh thread, but pyglet binds global state
        to the thread that first imported it.  After a run is stopped, the old
        thread is dead and pyglet's globals are stale.  Two things break:

        1. ``wglShareLists`` fails ("Unable to share contexts") because the
           shadow window's GL context was created in the old (dead) thread.
        2. ``platform_event_loop.start()`` fails ("must be called from the
           same thread that imports pyglet.app") because ``_event_thread``
           still records the old thread ID.

        This method patches both before ``visual.Window()`` is called.
        Only runs when the current thread differs from pyglet's recorded thread
        (i.e. this is a re-run, not the first run).
        """
        import sys

        if sys.platform != "win32":
            return

        from ctypes import windll  # type: ignore[attr-defined]  # Windows-only, guarded above

        current_thread = windll.kernel32.GetCurrentThreadId()

        # Check if we're in the same thread that created pyglet's state.
        # If so, everything is fine — no reset needed.
        try:
            import pyglet.app

            if pyglet.app.platform_event_loop._event_thread == current_thread:
                return
        except Exception:
            return

        logger.debug("Detected new thread for PsychoPy window — resetting pyglet state")

        try:
            import pyglet
            import pyglet.gl as gl
            from pyglet.gl import gl_info

            # 1. Clear stale GL context so new window won't try to share
            #    with the dead shadow window's context via wglShareLists.
            gl.current_context = None

            # Orphan the shadow window (don't close() it — that triggers
            # event loop thread checks that fail from the new thread).
            gl._shadow_window = None
            pyglet.options["shadow_window"] = False

            # Reset gl_info so Win32Config.match() uses the basic pixel
            # format path instead of the ARB path (which needs an active
            # GL context that no longer exists).
            gl_info.remove_active_context()

            # 2. Re-bind the platform event loop to the current thread.
            pyglet.app.platform_event_loop._event_thread = current_thread
        except Exception as e:
            logger.debug(f"Could not reset pyglet state: {e}")

    def _reset_pyglet_gl_info(self) -> None:
        """Reset pyglet's gl_info cached state after closing a window.

        Without this, gl_info retains stale OpenGL version info that causes
        "string index out of range" errors when creating a new window.
        Should be called in cleanup after closing the window.
        """
        try:
            import pyglet.gl.gl_info as gl_info

            # Reset the global GLInfo instance to initial state
            gl_info.remove_active_context()
            # Also reset the cached version/renderer info so it re-queries on next window
            gl_info._have_info = False
            gl_info.version = "0.0.0"
            gl_info.vendor = ""
            gl_info.renderer = ""
            gl_info.extensions = set()
            logger.debug("Reset pyglet gl_info state for clean window recreation")
        except Exception as e:
            logger.debug(f"Could not reset pyglet gl_info state: {e}")

    def _resolve_pyglet_screen_index(self, target_x: int, target_y: int) -> int:
        """Find pyglet screen index matching a screeninfo monitor position.

        screeninfo sorts by x-position; pyglet uses Windows adapter order.
        Both report identical (x,y) from EnumDisplayMonitors, so we match
        on position to bridge the two enumeration orders.
        """
        import pyglet

        display = pyglet.canvas.get_display()
        screens = display.get_screens()

        # Exact match on position
        for i, scr in enumerate(screens):
            if scr.x == target_x and scr.y == target_y:
                logger.info(f"Resolved pyglet screen: ({target_x},{target_y}) -> index {i} ({scr.width}x{scr.height})")
                return i

        # Fallback: closest by distance
        best_i = 0
        best_dist = float("inf")
        for i, scr in enumerate(screens):
            dist = (scr.x - target_x) ** 2 + (scr.y - target_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_i = i
        scr = screens[best_i]
        logger.warning(f"No exact pyglet match for ({target_x},{target_y}), using closest: index {best_i}")
        return best_i

    # =========================================================================
    # Instructions Display
    # =========================================================================

    def show_instructions(
        self,
        slides: list[str],
        advance_keys: list[str] | None = None,
        text_height: float = 0.05,
        wrap_width: float = 1.5,
    ) -> None:
        """Display instruction slides, advancing on key press.

        Shows a sequence of text slides. User advances through them by pressing
        any of the advance keys. Escape aborts the task.

        Parameters
        ----------
        slides : list of str
            Text content for each slide
        advance_keys : list of str
            Keys that advance to next slide (default: ["space", "return", "1", "2", "3"])
        text_height : float
            Height of text (default: 0.05)
        wrap_width : float
            Width for text wrapping (default: 1.5)

        Raises
        ------
        KeyboardInterrupt
            If escape is pressed or stop signal received
        """
        from psychopy import core, event, visual

        advance_keys = advance_keys or ["space", "return", "1", "2", "3"]

        # Create instruction text stimulus
        instruction_text = visual.TextStim(
            self.win,
            text="",
            pos=(0, 0),
            height=text_height,
            wrapWidth=wrap_width,
            color="white",
            alignText="center",
        )

        for slide_text in slides:
            instruction_text.text = slide_text
            instruction_text.draw()
            self._capture_and_emit_screenshot(force=True)
            self.win.flip()

            # Wait for advance key
            event.clearEvents("keyboard")
            while True:
                if not self._running:
                    raise KeyboardInterrupt("Stopped by user")

                keys = event.getKeys(keyList=advance_keys + ["escape"])
                if "escape" in keys:
                    raise KeyboardInterrupt("Escaped during instructions")
                if keys:
                    break

                core.wait(0.01)

        logger.info(f"Instructions complete ({len(slides)} slides)")

    # =========================================================================
    # Trigger Waiting
    # =========================================================================

    def wait_for_trigger(self, message: str = "Waiting for scanner...") -> None:
        """Wait for scanner trigger before starting task.

        Displays a message and waits for a trigger key press. In simulation
        mode, the simulator injects a synthetic trigger key (replicating the
        real scanner behavior) before starting to produce DICOMs.

        The same keyboard-polling code path runs in both real and sim mode.

        Parameters
        ----------
        message : str
            Text to display while waiting (default: "Waiting for scanner...")

        Raises
        ------
        KeyboardInterrupt
            If escape is pressed or stop signal received
        """
        from psychopy import core, event, visual

        # Display waiting message
        waiting_text = visual.TextStim(
            self.win,
            text=message,
            pos=(0, 0),
            height=0.05,
            color="white",
        )
        waiting_text.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        # Clear stale keyboard events before waiting
        logger.info(f"Waiting for trigger (keys: {self.trigger_keys})...")
        event.clearEvents("keyboard")

        # Start simulator if present (replicates real scanner: trigger key + DICOMs)
        # Must be AFTER clearEvents to avoid wiping the injected trigger key
        if self._simulator is not None:
            self._start_simulator_thread()
        while True:
            if not self._running:
                raise KeyboardInterrupt("Stopped by user")

            keys = event.getKeys(keyList=self.trigger_keys + ["escape"])
            if "escape" in keys:
                raise KeyboardInterrupt("Escaped during trigger wait")
            if any(k in keys for k in self.trigger_keys):
                logger.info("Trigger received, starting task")
                # Signal external listeners (e.g., collect_dicoms timing)
                if self.trigger_event is not None:
                    self.trigger_event.set()
                break

            core.wait(0.01)

        # Post-trigger hook (e.g., BaseTask calls coordinator.send_trigger())
        self._on_trigger()

    def _on_trigger(self) -> None:  # noqa: B027
        """Hook called after trigger is received.

        Default is no-op. BaseTask overrides to call coordinator.send_trigger().
        """
        pass

    # =========================================================================
    # Simulator Thread Management
    # =========================================================================

    def _inject_trigger_key(self) -> None:
        """Inject synthetic trigger key into PsychoPy's event buffer.

        Called by the simulator's trigger_callback to replicate the real
        scanner sending a trigger key before acquiring volumes.
        """
        from psychopy import event as psychopy_event

        key = self.trigger_keys[0]  # e.g. 't'
        psychopy_event._keyBuffer.append((key, 0.0))
        logger.info(f"Simulator injected synthetic trigger key: '{key}'")

    def _start_simulator_thread(self) -> None:
        """Start the simulator in a background thread.

        The simulator replicates real scanner behavior:
        1. Injects a synthetic trigger key (via trigger_callback)
        2. Copies DICOMs at TR intervals
        """
        logger.info("Starting simulator thread")

        # Clear target directory if requested
        if self._simulator.clear_target:
            self._simulator._clear_target_directory()
            self._simulator.clear_target = False

        # Set trigger callback to inject synthetic key press
        self._simulator.trigger_callback = self._inject_trigger_key

        # Set start time for deadline-based timing
        self._simulator.start_time = time.monotonic()

        def _run_simulator():
            try:
                # Brief delay so "Waiting for scanner..." is visible before
                # the synthetic trigger key is injected
                time.sleep(2.0)
                logger.info("Simulator thread starting")
                for vol_num in self._simulator.copy_volumes_iter():
                    if not self._running:
                        logger.info("Simulator stopped (task not running)")
                        break
                logger.info("Simulator thread complete")
            except Exception as e:
                logger.exception(f"Simulator thread error: {e}")

        self._simulator_thread = threading.Thread(
            target=_run_simulator,
            daemon=False,
            name="SimulatorThread",
        )
        self._simulator_thread.start()
        logger.info("Simulator thread started")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def stop(self) -> None:
        """Signal task to stop."""
        self._running = False

    def cleanup(self) -> None:
        """Release shared resources after task completes.

        Joins simulator thread, closes PsychoPy window, clears events,
        and resets pyglet GL state. Subclasses should call super().cleanup()
        at the end of their own cleanup.
        """
        # Wait for simulator thread to complete if it exists
        if self._simulator_thread is not None and self._simulator_thread.is_alive():
            logger.info("Waiting for simulator thread to complete...")
            self._simulator_thread.join(timeout=5.0)
            if self._simulator_thread.is_alive():
                logger.warning("Simulator thread did not finish within timeout")
            else:
                logger.info("Simulator thread joined successfully")

        # Close window
        if hasattr(self, "win") and self.win is not None:
            self.win.close()
            self.win = None

        # Force-close any remaining pyglet windows (prevents stale GL contexts
        # from causing "Unable to share contexts" on the next run)
        try:
            import pyglet

            for window in list(pyglet.app.windows):
                with contextlib.suppress(Exception):
                    window.close()
        except Exception:
            pass

        # Clear pending events for clean state
        try:
            from psychopy import event

            event.clearEvents()
        except Exception:
            pass

        # Reset pyglet gl_info for clean window recreation
        self._reset_pyglet_gl_info()

        # Force garbage collection to release GL context references
        import gc

        gc.collect()
