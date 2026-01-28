"""
Base class for PsychoPy neurofeedback tasks.

Summary
-------
Thread-safe task scaffold that polls the coordinator for activations and drives
PsychoPy displays at a target refresh rate.

Components
----------
- BaseTask: Abstract base with setup/update/cleanup hooks

Workflow
--------
- setup(): create window/stimuli
- run(): poll coordinator at refresh_rate, call update() each frame
- cleanup(): release resources when coordinator signals completion

Dependencies
------------
threading (stdlib), PsychoPy (for concrete tasks)

Notes
-----
The run() loop calls setup(), then updates at refresh_rate Hz until the
coordinator reports completion, then calls cleanup(). Use this when building
PsychoPy tasks that need thread-safe access to live activations.
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import logging
import threading
import time

if TYPE_CHECKING:
    from .session import SessionState

# Configure module-level logger
logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Abstract base class for neurofeedback tasks.

    Subclasses must implement setup() and update() methods. The cleanup()
    method has a default implementation that shows post-run questions and
    closes the window.

    The task automatically handles thread-safe access to activation data
    from the NeurofeedbackCoordinator running in a background thread.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator object providing activation data
    mask_name : str, optional
        Name of mask to display. If None, uses first mask.
    refresh_rate : float, optional
        Display refresh rate in Hz (default: 60.0)
    feedback_mode : bool, optional
        Whether feedback is active (default: True)
    session_state : SessionState, optional
        Session state for persistence across runs

    Attributes
    ----------
    coordinator : NeurofeedbackCoordinator
        Reference to coordinator for polling latest activations
    mask_name : str
        Name of the mask being displayed
    refresh_rate : float
        Target refresh rate in Hz
    feedback_mode : bool
        Whether feedback is active
    _session_state : SessionState or None
        Session state for persistence

    Examples
    --------
    >>> class MyTask(BaseTask):
    ...     def setup(self):
    ...         self.win = visual.Window([800, 600])
    ...
    ...     def update(self, data: dict):
    ...         # Draw based on data['activation']
    ...         pass
    ...
    ...     # cleanup() is optional - base handles questions + window close
    """

    # Subclasses should set this to match their registered task_id
    # Used for accessing task metadata (parameters, session state, questions)
    task_id: str = "base"

    def __init__(
        self,
        coordinator,
        mask_name: Optional[str] = None,
        refresh_rate: float = 60.0,
        feedback_mode: bool = True,
        session_state: Optional["SessionState"] = None,
        **kwargs,  # Accept additional kwargs for subclass compatibility
    ):
        self.coordinator = coordinator
        self.mask_name = mask_name or coordinator.mask_names[0]
        self.refresh_rate = refresh_rate
        self.frame_duration = 1.0 / refresh_rate

        # Question support
        self.feedback_mode = feedback_mode
        self._session_state = session_state
        self._questions_file = None  # Optional file handle for logging responses
        self._questions_writer = None  # Optional CSV writer

        # Thread-safe access to latest data
        self._lock = threading.Lock()
        self._latest_data = {
            'activations': {},  # All mask activations
            'activation': 0.0,  # Primary mask (backwards compat)
            'volume': 0,
            'time': 0.0,
            'is_baseline': True,
        }

        self._running = False

        # Mirror callback for GUI visualization
        self._mirror_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._mirror_frame_counter = 0
        self._mirror_frame_skip = 4  # Only capture every Nth frame to reduce overhead

    def get_task_metadata(self):
        """Get metadata for this task type.

        Returns
        -------
        TaskMetadata
            Metadata object with parameters, session state fields, and questions.
            Returns None if task is not registered.
        """
        from .task_registry import get_task_metadata
        try:
            return get_task_metadata(self.task_id)
        except ValueError:
            return None

    def set_mirror_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback for GUI mirroring.

        The callback receives visual state each frame for rendering in a GUI mirror.

        Parameters
        ----------
        callback : Callable[[Dict[str, Any]], None]
            Function that receives the visual state dictionary from get_mirror_state()
        """
        self._mirror_callback = callback

    def get_mirror_state(self) -> Dict[str, Any]:
        """Return current visual state for GUI mirroring.

        Subclasses should override this to provide their visual representation.
        Default implementation returns an empty state.

        Returns
        -------
        Dict[str, Any]
            Dictionary with:
            - 'background': str - background color (default: 'black')
            - 'shapes': List[dict] - list of drawable primitives

            Each shape dict has 'type' and type-specific properties.
            All coordinates are normalized 0-1 (x: left-right, y: top-bottom).

            Supported shape types:
            - circle: x, y, radius, fill or stroke, stroke_width
            - line: x1, y1, x2, y2, stroke, stroke_width
            - rect: x, y, width, height, fill or stroke, stroke_width
            - text: x, y, content, size, color
        """
        return {'background': 'black', 'shapes': []}

    def _emit_mirror_state(self) -> None:
        """Emit current visual state to mirror callback if set."""
        if self._mirror_callback is not None:
            try:
                self._mirror_callback(self.get_mirror_state())
            except Exception as e:
                logger.debug(f"Mirror callback failed: {e}")

    def _capture_and_emit_screenshot(self, force: bool = False) -> None:
        """Capture current PsychoPy frame and emit to mirror callback.

        This provides a true mirror of exactly what participants see on the
        PsychoPy window. Works for any task without task-specific code.

        IMPORTANT: Call this BEFORE win.flip(), not after. This captures the
        back buffer (what has been drawn but not yet displayed), which works
        reliably across all platforms including WSL2/XWayland.

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
        if not hasattr(self, 'win'):
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
            import io
            import base64

            # Capture back buffer (drawn content before flip)
            # This works reliably on WSL2/XWayland where front buffer capture fails
            frame = self.win.getMovieFrame(buffer='back')

            # Resize to mirror dimensions (16:9 aspect ratio)
            frame_resized = frame.resize((480, 270))

            # Convert to JPEG for faster encoding (less overhead than PNG)
            buffer = io.BytesIO()
            frame_resized.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Emit to callback
            self._mirror_callback({'type': 'screenshot', 'image': f'data:image/jpeg;base64,{img_base64}'})
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

    def _update_data(
        self,
        activations: dict,
        volume: int,
        elapsed: float,
        is_baseline: bool,
        is_outlier: bool
    ):
        """Internal callback to update latest data (thread-safe).

        Called by coordinator after each volume is processed.
        Stores all activations so multi-network tasks can access them.
        """
        with self._lock:
            self._latest_data = {
                'activations': activations,  # All mask activations
                'activation': activations.get(self.mask_name, 0.0),  # Primary mask (backwards compat)
                'volume': volume,
                'time': elapsed,
                'is_baseline': is_baseline,
            }

    def get_latest_data(self) -> dict:
        """Get latest activation data (thread-safe).

        Returns
        -------
        dict
            Dictionary containing:
            - 'activation': float - current activation value
            - 'volume': int - current volume number
            - 'time': float - elapsed time in seconds
            - 'is_baseline': bool - whether in baseline period
        """
        with self._lock:
            return self._latest_data.copy()

    # =========================================================================
    # Post-Run Questions Support
    # =========================================================================

    def _run_slider(
        self,
        question_text: str,
        left_label: str = "Never",
        right_label: str = "Always",
        left_button: str = "1",
        right_button: str = "2",
        enter_button: str = "3",
        scale_min: int = 1,
        scale_max: int = 9,
    ) -> int:
        """Display a slider question and return the user's response.

        Creates a PsychoPy slider, waits for user input, and optionally
        logs the response to file if _questions_writer is configured.

        Parameters
        ----------
        question_text : str
            The question to display above the slider
        left_label : str
            Label for the low end of the scale
        right_label : str
            Label for the high end of the scale
        left_button : str
            Key to move slider left
        right_button : str
            Key to move slider right
        enter_button : str
            Key to confirm selection
        scale_min : int
            Minimum value on scale (default: 1)
        scale_max : int
            Maximum value on scale (default: 9)

        Returns
        -------
        int
            The user's rating
        """
        from psychopy import visual, event, core

        # Create slider
        slider = visual.Slider(
            win=self.win,
            size=(0.85, 0.1),
            ticks=(scale_min, scale_max),
            labels=(left_label, right_label),
            granularity=1,
            color='white',
            fillColor='white',
            labelHeight=0.06,
        )

        # Create question text
        question_stim = visual.TextStim(
            self.win,
            text=question_text,
            pos=(0, 0.3),
            height=0.05,
            wrapWidth=1.5,
            color='white',
        )

        # Initialize slider position at center
        event.clearEvents('keyboard')
        center = (scale_min + scale_max) // 2
        slider.markerPos = center

        # Draw initial state
        slider.draw()
        question_stim.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        # Track timing
        response_clock = core.Clock()

        # Wait for response
        while True:
            # Check for stop signal from GUI
            if hasattr(self, 'coordinator') and self.coordinator.was_stopped():
                logger.info("Stop signal received during slider question")
                raise KeyboardInterrupt("Stopped by user")

            keys = event.getKeys(keyList=[left_button, right_button, enter_button, 'escape'])

            for key in keys:
                if key == 'escape':
                    raise KeyboardInterrupt("Escaped during question")
                elif key == left_button:
                    slider.markerPos = max(scale_min, slider.markerPos - 1)
                elif key == right_button:
                    slider.markerPos = min(scale_max, slider.markerPos + 1)
                elif key == enter_button:
                    slider.rating = slider.markerPos
                    rt = response_clock.getTime()
                    rating = int(slider.rating)

                    # Log to questions file if configured
                    if self._questions_writer is not None:
                        self._questions_writer.writerow([
                            getattr(self.coordinator, 'subject', 'unknown'),
                            getattr(self.coordinator, 'run_number', 0),
                            'feedback' if self.feedback_mode else 'transfer',
                            question_text,
                            rating,
                            f'{rt:.3f}',
                        ])
                        if self._questions_file is not None:
                            self._questions_file.flush()

                    logger.info(f"Question response: '{question_text[:40]}...' -> {rating} (RT: {rt:.2f}s)")
                    return rating

            # Redraw if keys were pressed
            if keys:
                slider.draw()
                question_stim.draw()
                self._capture_and_emit_screenshot(force=True)
                self.win.flip()

            # Small sleep to avoid busy-waiting
            core.wait(0.01)

    def _show_post_run_questions(
        self,
        left_button: str = "1",
        right_button: str = "2",
        enter_button: str = "3",
    ) -> List[Dict[str, Any]]:
        """Display post-run questions defined in task metadata.

        Called automatically by cleanup(). Tasks with no questions defined
        in metadata will skip this (no-op).

        Reads questions from TaskMetadata.post_run_questions and displays
        each one based on its condition:
        - "always": Show for all runs
        - "feedback_only": Only show when feedback_mode is True
        - "last_feedback_only": Only show on final feedback run

        Parameters
        ----------
        left_button : str
            Key to move slider left
        right_button : str
            Key to move slider right
        enter_button : str
            Key to confirm selection

        Returns
        -------
        list of dict
            List of {"question": str, "response": int} for each answered question
        """
        metadata = self.get_task_metadata()
        if not metadata or not metadata.post_run_questions:
            return []  # No-op for tasks without questions

        # Check if window exists
        if not hasattr(self, 'win') or self.win is None:
            logger.warning("Cannot show questions: no PsychoPy window")
            return []

        responses = []

        for question in metadata.post_run_questions:
            # Check condition
            if question.condition == "feedback_only":
                if not self.feedback_mode:
                    continue

            elif question.condition == "last_feedback_only":
                is_last = False
                if self._session_state is not None:
                    is_last = getattr(self._session_state, 'is_last_feedback_run', False)
                if not (self.feedback_mode and is_last):
                    continue

            # Display slider and collect response
            response = self._run_slider(
                question_text=question.text,
                left_label=question.left_label,
                right_label=question.right_label,
                left_button=left_button,
                right_button=right_button,
                enter_button=enter_button,
            )

            responses.append({
                "question": question.text,
                "response": response,
            })

        logger.info(f"Post-run questions complete: {len(responses)} responses")
        return responses

    @abstractmethod
    def setup(self):
        """Initialize PsychoPy window and stimuli.

        Called once before run loop starts. Create window, visual stimuli,
        and any other resources needed for the task.
        """
        pass

    @abstractmethod
    def update(self, data: dict):
        """Update display based on latest activation data.

        Called every display frame with latest data. Update stimulus
        properties (position, color, size, etc.) based on activation.

        Parameters
        ----------
        data : dict
            Dictionary containing:
            - 'activation': float - current activation value
            - 'volume': int - current volume number
            - 'time': float - elapsed time in seconds
            - 'is_baseline': bool - whether in baseline period
        """
        pass

    def cleanup(self):
        """Clean up resources after task completes.

        Default implementation:
        1. Records run completion to session state (if configured)
        2. Shows post-run questions (if defined in metadata)
        3. Closes the PsychoPy window (if exists)
        4. Clears pending keyboard events
        5. Resets pyglet gl_info for clean window recreation

        Subclasses can override to add custom cleanup logic, but should
        call super().cleanup() at the end. Set self._task_state_snapshot
        before calling super().cleanup() to include task-specific state.

        For custom question behavior (like BallTask's instruction screen),
        override _show_post_run_questions() instead.
        """
        # Record run completion (no-op if no session state)
        self._record_run_completion()

        # Show post-run questions if coordinator says to (respects stop() with show_questions flag)
        # Falls back to True if coordinator doesn't have the method (standalone task)
        should_show = True
        if hasattr(self, 'coordinator') and hasattr(self.coordinator, 'should_show_questions'):
            should_show = self.coordinator.should_show_questions()

        if should_show:
            self._show_post_run_questions()
        else:
            logger.info("Session was stopped - skipping post-run questions")

        # Close window
        if hasattr(self, 'win') and self.win is not None:
            self.win.close()
            self.win = None

        # Clear pending events for clean state
        try:
            from psychopy import event
            event.clearEvents()
        except Exception:
            pass

        # Reset pyglet gl_info for clean window recreation
        self._reset_pyglet_gl_info()

    def run(self):
        """Main task loop - runs in main thread for PsychoPy.

        Calls setup(), then loops calling update() at the target refresh rate
        until the session completes, then calls cleanup().
        """
        self._running = True
        self.setup()

        try:
            while self._running and not self.coordinator.is_complete():
                frame_start = time.time()

                # Check for errors from coordinator/simulator
                self.coordinator.check_error()

                # Get latest data and update display
                data = self.get_latest_data()
                self.update(data)

                # Emit visual state to GUI mirror if callback is set
                self._emit_mirror_state()

                # Frame timing
                elapsed = time.time() - frame_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.cleanup()

    def stop(self):
        """Signal task to stop."""
        self._running = False

    # =========================================================================
    # Improvement A: Trigger Waiting
    # =========================================================================

    def wait_for_trigger(
        self,
        message: str = "Waiting for scanner...",
        trigger_keys: list = None,
    ) -> None:
        """Wait for scanner trigger before starting task.

        Displays a message and waits for a trigger key press. Handles escape key
        to abort, and simulation mode for automatic triggering.

        Parameters
        ----------
        message : str
            Text to display while waiting (default: "Waiting for scanner...")
        trigger_keys : list of str
            Keys that trigger acquisition start (default: ["t", "+", "5", "equal"])

        Raises
        ------
        KeyboardInterrupt
            If escape is pressed or stop signal received
        """
        from psychopy import visual, event, core

        trigger_keys = trigger_keys or ["t", "+", "5", "equal"]

        # Create and display waiting text
        waiting_text = visual.TextStim(
            self.win,
            text=message,
            pos=(0, 0),
            height=0.05,
            color='white',
        )
        waiting_text.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        # Check if we're in simulation mode (coordinator has a simulator)
        if hasattr(self.coordinator, '_simulator') and self.coordinator._simulator is not None:
            logger.info("Simulation mode: auto-triggering")
            core.wait(0.1)  # Brief pause to show waiting screen
            self.coordinator.send_trigger()
        else:
            # Real-time mode: wait for trigger key
            logger.info(f"Waiting for trigger (keys: {trigger_keys})...")
            event.clearEvents('keyboard')
            while True:
                # Check for stop signal from GUI
                if self.coordinator.was_stopped():
                    logger.info("Stop signal received during trigger wait")
                    raise KeyboardInterrupt("Stopped by user")

                keys = event.getKeys(keyList=trigger_keys + ['escape'])
                if 'escape' in keys:
                    raise KeyboardInterrupt("Escaped during trigger wait")
                if any(k in keys for k in trigger_keys):
                    logger.info("Trigger received, starting task")
                    self.coordinator.send_trigger()
                    break

                core.wait(0.01)

    # =========================================================================
    # Improvement B: Instructions Display
    # =========================================================================

    def show_instructions(
        self,
        slides: list,
        advance_keys: list = None,
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
        from psychopy import visual, event, core

        advance_keys = advance_keys or ["space", "return", "1", "2", "3"]

        # Create instruction text stimulus
        instruction_text = visual.TextStim(
            self.win,
            text='',
            pos=(0, 0),
            height=text_height,
            wrapWidth=wrap_width,
            color='white',
            alignText='center',
        )

        for i, slide_text in enumerate(slides):
            instruction_text.text = slide_text
            instruction_text.draw()
            self._capture_and_emit_screenshot(force=True)
            self.win.flip()

            # Wait for advance key
            event.clearEvents('keyboard')
            while True:
                # Check for stop signal from GUI
                if hasattr(self, 'coordinator') and self.coordinator.was_stopped():
                    logger.info("Stop signal received during instructions")
                    raise KeyboardInterrupt("Stopped by user")

                keys = event.getKeys(keyList=advance_keys + ['escape'])
                if 'escape' in keys:
                    raise KeyboardInterrupt("Escaped during instructions")
                if keys:
                    break

                core.wait(0.01)

        logger.info(f"Instructions complete ({len(slides)} slides)")

    # =========================================================================
    # Improvement C: Automatic Session State Recording
    # =========================================================================

    def _record_run_completion(self) -> None:
        """Record run completion to session state.

        Called automatically by cleanup() if session state is configured.
        Subclasses should set self._task_state_snapshot dict before cleanup
        to include task-specific state in the recording.

        Override this method if you need custom recording logic.
        """
        if self._session_state is None:
            return

        self._session_state.record_run(
            run_type="feedback" if self.feedback_mode else "transfer",
            volumes_processed=getattr(self, '_volumes_processed', 0),
            task_state_snapshot=getattr(self, '_task_state_snapshot', {}),
        )

    # =========================================================================
    # Window Management (Cross-Platform)
    # =========================================================================

    def _move_window_xlib(self, target_x: int) -> None:
        """Move PsychoPy window to a specific x position using Xlib.

        This is a workaround for WSL2/Wayland ignoring window position requests.
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
            xlib_win = d.create_resource_object('window', window_id)
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
        fullscreen: bool = True,
        monitor_position: str = "right",
        size: tuple = None,
        color: str = 'black',
        units: str = 'height',
        **kwargs,
    ):
        """Create a PsychoPy window with cross-platform monitor targeting.

        Handles platform differences:
        - macOS: Uses screen parameter directly
        - Linux/WSL2: Creates window then moves it using Xlib

        Parameters
        ----------
        fullscreen : bool
            Whether to create fullscreen window (default: True)
        monitor_position : str
            Monitor position: "left", "right", or "primary" (default: "right")
        size : tuple of int, optional
            Window size (width, height). If None, uses monitor dimensions.
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
        from psychopy import visual, monitors
        from pineuro import config
        from pineuro.display import get_monitor_by_position

        platform = config.get_platform()

        # Create minimal monitor to avoid "Monitor specification not found" warning
        mon = monitors.Monitor('neurofeedback')

        # Get target screen using display module
        target_monitor = get_monitor_by_position(monitor_position)
        if target_monitor:
            target_screen = target_monitor.index
            logger.info(f"Task window targeting {monitor_position} monitor: {target_monitor.name} (screen={target_screen})")
        else:
            target_screen = 0
            logger.warning("No monitors detected, using screen 0")

        # Platform-specific window creation
        if platform == 'darwin':
            # macOS: screen parameter works directly
            self.win = visual.Window(
                fullscr=fullscreen,
                monitor=mon,
                screen=target_screen,
                units=units,
                color=color,
                allowGUI=False,
                waitBlanking=False,
                **kwargs,
            )
        else:
            # Linux (including WSL2): Wayland/XWayland ignores screen parameter.
            # Workaround: Create window at target size, then move it using Xlib.
            if target_monitor:
                win_width = target_monitor.width
                win_height = target_monitor.height
                target_x = target_monitor.x
            else:
                win_width, win_height = size or (1920, 1080)
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
                import time
                time.sleep(0.2)  # Give window time to appear
                self._move_window_xlib(target_x)

        return self.win

    def _reset_pyglet_gl_info(self) -> None:
        """Reset pyglet's gl_info state to allow clean window recreation.

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
            gl_info.version = '0.0.0'
            gl_info.vendor = ''
            gl_info.renderer = ''
            gl_info.extensions = set()
            logger.debug("Reset pyglet gl_info state for clean window recreation")
        except Exception as e:
            logger.debug(f"Could not reset pyglet gl_info state: {e}")
