"""
Base class for localizer tasks (stimulus presentation without feedback).

Summary
-------
Provides a scaffold for PsychoPy tasks that present stimuli on a fixed time
schedule and record BIDS-compatible events. Unlike BaseTask, localizer tasks
do not require a NeurofeedbackCoordinator — they are time-driven, not
data-driven.

Components
----------
- BaseLocalizerTask: Abstract base with setup/run_paradigm/cleanup lifecycle

Workflow
--------
- setup(): create window/stimuli, load stimulus files
- run(): setup → instructions → trigger wait → run_paradigm → save events → cleanup
- cleanup(): close window, release resources

Dependencies
------------
PsychoPy (for concrete tasks), pandas (for events TSV output)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseLocalizerTask(ABC):
    """Abstract base class for localizer tasks.

    Localizer tasks present stimuli in a fixed schedule (block design,
    event-related, etc.) and record onset/duration/trial_type events for
    downstream GLM analysis via mask_extraction_task().

    Unlike BaseTask (for real-time feedback), localizer tasks:
    - Do NOT require a NeurofeedbackCoordinator
    - Are time-driven (present stimulus at t=X seconds)
    - Output BIDS events TSV as their primary product

    Subclasses must implement setup() and run_paradigm().

    Parameters
    ----------
    output_dir : str or Path
        Directory for events TSV and any other output files.
    participant_id : str
        Participant identifier (e.g., "sub-01").
    run : int
        Run number (1-based).
    trigger_keys : list of str, optional
        Keys that trigger acquisition start.
    fullscreen : bool
        Whether to create fullscreen window.
    monitor_position : str
        Monitor position: "left", "right", or "primary".
    watch_dir : str or Path, optional
        Directory to watch for incoming DICOM files (auto-trigger mode).
    simulator : VolumeSimulator, optional
        Volume simulator instance for offline testing. If provided, the task
        will automatically start the simulator thread and wait for volumes
        to arrive in the simulator's target directory.
    """

    task_id: str = "base_localizer"

    def __init__(
        self,
        output_dir: Union[str, Path],
        participant_id: str,
        run: int = 1,
        acq: Optional[int] = None,
        trigger_keys: Optional[List[str]] = None,
        fullscreen: bool = True,
        monitor_position: str = "right",
        watch_dir: Optional[Union[str, Path]] = None,
        simulator=None,  # VolumeSimulator instance for simulation mode
        **kwargs,
    ):
        self.output_dir = Path(output_dir)
        self.participant_id = participant_id
        self.run_number = run
        self.acq_number = acq
        self.trigger_keys = trigger_keys or ["t", "+", "5", "equal"]
        self.fullscreen = fullscreen
        self.monitor_position = monitor_position
        self.watch_dir = Path(watch_dir) if watch_dir else None
        self._simulator = simulator

        self._events: List[Dict[str, Any]] = []
        self._running = False
        self._mirror_callback: Optional[Callable] = None
        self._mirror_frame_counter = 0
        self._mirror_frame_skip = 4
        self._experiment_clock = None
        self.win = None
        self._simulator_thread = None

    # =========================================================================
    # Mirror callback support
    # =========================================================================

    def set_mirror_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback for GUI mirroring."""
        self._mirror_callback = callback

    def _capture_and_emit_screenshot(self, force: bool = False) -> None:
        """Capture current PsychoPy frame and emit to mirror callback.

        Call BEFORE win.flip(). Only captures every Nth frame unless force=True.
        """
        if self._mirror_callback is None:
            return
        if not hasattr(self, 'win') or self.win is None:
            return

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

            frame = self.win.getMovieFrame(buffer='back')
            frame_resized = frame.resize((480, 270))
            buffer = io.BytesIO()
            frame_resized.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            self._mirror_callback({
                'type': 'screenshot',
                'image': f'data:image/jpeg;base64,{img_base64}',
            })
        except Exception as e:
            logger.debug(f"Screenshot capture failed: {e}")
        finally:
            if buffer is not None:
                buffer.close()
            if frame_resized is not None:
                frame_resized.close()
            if frame is not None:
                frame.close()

    # =========================================================================
    # Window management (adapted from BaseTask)
    # =========================================================================

    def _move_window_xlib(self, target_x: int) -> None:
        """Move PsychoPy window to a specific x position using Xlib.

        Workaround for WSL2/Wayland ignoring window position requests.
        """
        try:
            from Xlib import display as xlib_display

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
        size: Optional[tuple] = None,
        color: str = 'black',
        units: str = 'height',
        **kwargs,
    ):
        """Create a PsychoPy window with cross-platform monitor targeting.

        Handles platform differences:
        - macOS: Uses screen parameter directly
        - Linux/WSL2: Creates window then moves it using Xlib
        """
        from psychopy import visual, monitors
        from pineuro import config
        from pineuro.display import get_monitor_by_position

        platform = config.get_platform()
        mon = monitors.Monitor('neurofeedback')

        target_monitor = get_monitor_by_position(monitor_position)
        if target_monitor:
            target_screen = target_monitor.index
            logger.info(f"Localizer window targeting {monitor_position} monitor: "
                        f"{target_monitor.name} (screen={target_screen})")
        else:
            target_screen = 0
            logger.warning("No monitors detected, using screen 0")

        if platform == 'darwin':
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

            if target_x > 0:
                import time
                time.sleep(0.2)
                self._move_window_xlib(target_x)

        return self.win

    def _reset_pyglet_gl_info(self) -> None:
        """Reset pyglet's gl_info state to allow clean window recreation."""
        try:
            import pyglet.gl.gl_info as gl_info
            gl_info.remove_active_context()
            gl_info._have_info = False
            gl_info.version = '0.0.0'
            gl_info.vendor = ''
            gl_info.renderer = ''
            gl_info.extensions = set()
            logger.debug("Reset pyglet gl_info state for clean window recreation")
        except Exception as e:
            logger.debug(f"Could not reset pyglet gl_info state: {e}")

    # =========================================================================
    # Stimulus presentation utilities
    # =========================================================================

    def show_instructions(
        self,
        slides: List[str],
        advance_keys: Optional[List[str]] = None,
        text_height: float = 0.05,
        wrap_width: float = 1.5,
    ) -> None:
        """Display instruction slides, advancing on key press.

        Parameters
        ----------
        slides : list of str
            Text content for each slide.
        advance_keys : list of str, optional
            Keys that advance to next slide.
        text_height : float
            Height of text.
        wrap_width : float
            Width for text wrapping.
        """
        from psychopy import visual, event, core

        advance_keys = advance_keys or ["space", "return", "1", "2", "3"]

        instruction_text = visual.TextStim(
            self.win,
            text='',
            pos=(0, 0),
            height=text_height,
            wrapWidth=wrap_width,
            color='white',
            alignText='center',
        )

        for slide_text in slides:
            instruction_text.text = slide_text
            instruction_text.draw()
            self._capture_and_emit_screenshot(force=True)
            self.win.flip()

            event.clearEvents('keyboard')
            while True:
                if not self._running:
                    raise KeyboardInterrupt("Stopped by user")

                keys = event.getKeys(keyList=advance_keys + ['escape'])
                if 'escape' in keys:
                    raise KeyboardInterrupt("Escaped during instructions")
                if keys:
                    break

                core.wait(0.01)

        logger.info(f"Instructions complete ({len(slides)} slides)")

    def wait_for_trigger(self, message: str = "Waiting for scanner...") -> None:
        """Wait for scanner trigger before starting paradigm.

        Supports three modes:
        1. Simulation mode (if simulator is set): starts simulator thread and waits for first DICOM
        2. Watch mode (if watch_dir is set): waits for first DICOM file to appear (auto-trigger)
        3. Keyboard mode: waits for manual keyboard trigger

        Parameters
        ----------
        message : str
            Text to display while waiting.
        """
        from psychopy import visual, event, core
        import threading
        import time

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

        # Check if we're in simulation mode
        if self._simulator is not None:
            logger.info("Simulation mode detected: starting simulator thread")

            # Clear target directory if requested
            if self._simulator.clear_target:
                self._simulator._clear_target_directory()
                self._simulator.clear_target = False

            # Set watch_dir to simulator's target for auto-trigger
            self.watch_dir = self._simulator.target_dir

            # Start simulator thread to begin copying volumes
            # Simulator will start immediately and copy volumes at TR intervals
            self._simulator.start_time = time.monotonic()

            def _run_simulator():
                try:
                    logger.info("Simulator thread starting volume delivery")
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
                name="LocalizerSimulatorThread"
            )
            self._simulator_thread.start()
            logger.info("Simulator thread started, waiting for first volume...")

        # If watch_dir is set (either from simulator or manual), watch for first DICOM file
        if self.watch_dir is not None:
            logger.info(f"Waiting for first DICOM in {self.watch_dir}...")
            event.clearEvents('keyboard')
            while True:
                if not self._running:
                    raise KeyboardInterrupt("Stopped by user")

                # Check for escape key
                keys = event.getKeys(keyList=['escape'])
                if 'escape' in keys:
                    raise KeyboardInterrupt("Escaped during trigger wait")

                # Check for DICOM files in watch directory
                if self.watch_dir.exists():
                    dicom_files = list(self.watch_dir.glob('*.dcm')) + list(self.watch_dir.glob('*.IMA'))
                    if dicom_files:
                        logger.info(f"First DICOM detected, starting paradigm")
                        break

                core.wait(0.1)  # Check every 100ms
        else:
            # Manual keyboard trigger
            logger.info(f"Waiting for trigger (keys: {self.trigger_keys})...")
            event.clearEvents('keyboard')
            while True:
                if not self._running:
                    raise KeyboardInterrupt("Stopped by user")

                keys = event.getKeys(keyList=self.trigger_keys + ['escape'])
                if 'escape' in keys:
                    raise KeyboardInterrupt("Escaped during trigger wait")
                if keys:
                    logger.info("Trigger received, starting paradigm")
                    break

                core.wait(0.01)

    def show_fixation(self, duration: float, text: str = "+", height: float = 0.3) -> None:
        """Display fixation cross for a specified duration.

        Parameters
        ----------
        duration : float
            Duration in seconds.
        text : str
            Fixation character(s).
        height : float
            Text height.
        """
        from psychopy import visual, event, core

        fixation = visual.TextStim(
            self.win, text=text, height=height, color='white',
        )
        fixation.draw()
        self._capture_and_emit_screenshot(force=True)
        self.win.flip()

        timer = core.CountdownTimer(duration)
        while timer.getTime() > 0 and self._running:
            if event.getKeys(keyList=['escape']):
                raise KeyboardInterrupt("Escaped during fixation")
            core.wait(0.01)

    # =========================================================================
    # BIDS events recording
    # =========================================================================

    def record_event(
        self,
        onset: float,
        duration: float,
        trial_type: str,
        **extras,
    ) -> None:
        """Record a single event for the BIDS events TSV.

        Parameters
        ----------
        onset : float
            Event onset time in seconds (relative to first trigger).
        duration : float
            Event duration in seconds.
        trial_type : str
            Event type (e.g., "self", "other", "rest", "question").
        **extras
            Additional columns (e.g., response, response_time, stim_file).
        """
        ev = {"onset": round(onset, 3), "duration": round(duration, 3),
              "trial_type": trial_type}
        ev.update(extras)
        self._events.append(ev)
        logger.debug(f"Event: {trial_type} onset={onset:.1f} dur={duration:.1f}")

    def save_events(self, path: Optional[Path] = None) -> Path:
        """Write recorded events to a BIDS-compatible TSV file.

        Parameters
        ----------
        path : Path, optional
            Output path. Defaults to {output_dir}/{participant}_task-{task_id}_run-{run}_events.tsv.

        Returns
        -------
        Path
            Path to the saved events TSV.
        """
        import pandas as pd

        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.acq_number is not None:
                path = (self.output_dir /
                        f"{self.participant_id}_task-{self.task_id}_acq-{self.acq_number:02d}_run-{self.run_number:02d}_events.tsv")
            else:
                path = (self.output_dir /
                        f"{self.participant_id}_task-{self.task_id}_run-{self.run_number:02d}_events.tsv")

        if not self._events:
            logger.warning("No events recorded, saving empty events file")
            df = pd.DataFrame(columns=["onset", "duration", "trial_type"])
        else:
            df = pd.DataFrame(self._events)
            # Ensure standard BIDS columns come first
            bids_cols = ["onset", "duration", "trial_type"]
            extra_cols = [c for c in df.columns if c not in bids_cols]
            df = df[bids_cols + extra_cols]

        df.to_csv(path, sep='\t', index=False)
        logger.info(f"Saved {len(self._events)} events to {path}")
        return path

    # =========================================================================
    # Lifecycle
    # =========================================================================

    @abstractmethod
    def setup(self) -> None:
        """Initialize PsychoPy window, load stimuli, create visual objects."""
        pass

    @abstractmethod
    def run_paradigm(self) -> None:
        """Execute the stimulus presentation paradigm.

        This is where subclasses implement their block/event schedule.
        Should call record_event() for each stimulus onset.
        """
        pass

    def cleanup(self) -> None:
        """Release resources after task completes.

        Closes PsychoPy window, waits for simulator thread, and resets GL state.
        Called automatically by run() in the finally block.
        """
        # Wait for simulator thread to complete if it exists
        if self._simulator_thread is not None and self._simulator_thread.is_alive():
            logger.info("Waiting for simulator thread to complete...")
            self._simulator_thread.join(timeout=5.0)
            if self._simulator_thread.is_alive():
                logger.warning("Simulator thread did not finish within timeout")
            else:
                logger.info("Simulator thread joined successfully")

        if hasattr(self, 'win') and self.win is not None:
            self.win.close()
            self.win = None

        try:
            from psychopy import event
            event.clearEvents()
        except Exception:
            pass

        self._reset_pyglet_gl_info()

    def run(self) -> Path:
        """Full task lifecycle: setup → paradigm → save events → cleanup.

        Returns
        -------
        Path
            Path to the saved BIDS events TSV file.
        """
        self._running = True
        events_path = None

        try:
            self.setup()
            self.run_paradigm()
        except KeyboardInterrupt:
            logger.info("Task interrupted")
        finally:
            self._running = False
            events_path = self.save_events()
            self.cleanup()

        return events_path

    def stop(self) -> None:
        """Signal task to stop (for GUI integration)."""
        self._running = False
