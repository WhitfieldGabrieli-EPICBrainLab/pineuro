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
import logging
import threading
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from .task_base import TaskBase

if TYPE_CHECKING:
    from .session import SessionState

# Configure module-level logger
logger = logging.getLogger(__name__)


class BaseTask(TaskBase):
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
    simulator : VolumeSimulator, optional
        Volume simulator for offline testing

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
        mask_name: str | None = None,
        refresh_rate: float = 60.0,
        feedback_mode: bool = True,
        session_state: Optional["SessionState"] = None,
        simulator=None,
        **kwargs,  # Accept additional kwargs for subclass compatibility
    ):
        super().__init__(
            trigger_keys=kwargs.pop("trigger_keys", None),
            simulator=simulator,
            **kwargs,
        )

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
            "activations": {},  # All mask activations
            "activation": 0.0,  # Primary mask (backwards compat)
            "volume": 0,
            "time": 0.0,
            "is_baseline": True,
        }

        # Block accumulation state (optional — call _start_block_accumulation() to enable)
        self._accumulating: bool = False
        self._block_activations: list[float] = []

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

    # =========================================================================
    # Mirror state (vector-based, for structured GUI rendering)
    # =========================================================================

    def get_mirror_state(self) -> dict[str, Any]:
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
        return {"background": "black", "shapes": []}

    def _emit_mirror_state(self) -> None:
        """Emit current visual state to mirror callback if set."""
        if self._mirror_callback is not None:
            try:
                self._mirror_callback(self.get_mirror_state())
            except Exception as e:
                logger.debug(f"Mirror callback failed: {e}")

    # =========================================================================
    # Thread-safe Coordinator Data Access
    # =========================================================================

    def _update_data(self, activations: dict, volume: int, elapsed: float, is_baseline: bool, is_outlier: bool):
        """Internal callback to update latest data (thread-safe).

        Called by coordinator after each volume is processed.
        Stores all activations so multi-network tasks can access them.
        """
        with self._lock:
            self._latest_data = {
                "activations": activations,  # All mask activations
                "activation": activations.get(self.mask_name, 0.0),  # Primary mask (backwards compat)
                "volume": volume,
                "time": elapsed,
                "is_baseline": is_baseline,
            }

    def get_volume_log_suffix(self) -> str:
        """Optional suffix appended to the coordinator's per-volume log line.

        Override in subclasses to add task-specific info inline.
        Called from the coordinator's background thread immediately after
        _update_data() returns.
        """
        return ""

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
    # Block Accumulation (optional per-block activation averaging)
    # =========================================================================

    def _start_block_accumulation(self) -> None:
        """Begin collecting per-volume activations for the current block.

        Resets the accumulation buffer and sets the accumulation flag.
        Call this at the start of a feedback block; call
        _stop_block_accumulation() at the end to retrieve the values.

        Subclasses that override _update_data() should call
        ``super()._update_data(...)`` first, then check self._accumulating
        and append to self._block_activations if needed. Accumulation state
        is not individually locked; access is safe when _update_data() is
        called exclusively from the coordinator thread.
        """
        self._block_activations = []
        self._accumulating = True

    def _stop_block_accumulation(self) -> list[float]:
        """Stop accumulation and return the collected activation values.

        Returns
        -------
        list[float]
            A copy of all activation values appended since
            _start_block_accumulation() was last called.
        """
        self._accumulating = False
        return self._block_activations.copy()

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
        from psychopy import core, event, visual

        # Create slider
        slider = visual.Slider(
            win=self.win,
            size=(0.85, 0.1),
            ticks=(scale_min, scale_max),
            labels=(left_label, right_label),
            granularity=1,
            color="white",
            fillColor="white",
            labelHeight=0.06,
        )

        # Create question text
        question_stim = visual.TextStim(
            self.win,
            text=question_text,
            pos=(0, 0.3),
            height=0.05,
            wrapWidth=1.5,
            color="white",
        )

        # Initialize slider position at center
        event.clearEvents("keyboard")
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
            if hasattr(self, "coordinator") and self.coordinator.was_stopped():
                logger.info("Stop signal received during slider question")
                raise KeyboardInterrupt("Stopped by user")

            self._inject_keys_if_unfocused([left_button, right_button, enter_button])
            keys = event.getKeys(keyList=[left_button, right_button, enter_button, "escape"])

            for key in keys:
                if key == "escape":
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
                        self._questions_writer.writerow(
                            [
                                getattr(self.coordinator, "subject", "unknown"),
                                getattr(self.coordinator, "run_number", 0),
                                "feedback" if self.feedback_mode else "transfer",
                                question_text,
                                rating,
                                f"{rt:.3f}",
                            ]
                        )
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
    ) -> list[dict[str, Any]]:
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
        if not hasattr(self, "win") or self.win is None:
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
                    is_last = getattr(self._session_state, "is_last_feedback_run", False)
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

            responses.append(
                {
                    "question": question.text,
                    "response": response,
                }
            )

        logger.info(f"Post-run questions complete: {len(responses)} responses")
        return responses

    # =========================================================================
    # Run Results
    # =========================================================================

    def get_run_results(self) -> dict[str, Any]:
        """Return task-specific results from the completed run.

        The framework calls this after run() completes to extract results
        without needing to know about task-specific attributes. Subclasses
        should override to include their own metrics.

        Returns
        -------
        dict
            Task-specific results. Base implementation includes:
            - 'volumes_processed': int - number of volumes processed
        """
        return {
            "volumes_processed": getattr(self, "current_volume", 0) or getattr(self, "_volumes_processed", 0),
        }

    # =========================================================================
    # Abstract Methods
    # =========================================================================

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

    # =========================================================================
    # Trigger Hook
    # =========================================================================

    def _on_trigger(self) -> None:
        """Notify coordinator that trigger has been received."""
        self.coordinator.send_trigger()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def cleanup(self):
        """Clean up resources after task completes.

        Default implementation:
        1. Records run completion to session state (if configured)
        2. Shows post-run questions (if defined in metadata)
        3. Delegates to TaskBase.cleanup() for window/simulator/pyglet cleanup

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
        if hasattr(self, "coordinator") and hasattr(self.coordinator, "should_show_questions"):
            should_show = self.coordinator.should_show_questions()

        if should_show and hasattr(self, "win") and self.win is not None:
            self._show_post_run_questions()
        elif not should_show:
            logger.info("Session was stopped - skipping post-run questions")

        # Delegate to TaskBase for window/simulator/pyglet cleanup
        super().cleanup()

    def run(self):
        """Main task loop - runs in main thread for PsychoPy.

        Calls setup(), then loops calling update() at the target refresh rate
        until the session completes, then calls cleanup().
        """
        self._running = True

        try:
            self.setup()

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

            # Process final data after loop exits (race condition fix)
            # The loop exits when is_complete() returns True, which may happen
            # before we've had a chance to process the final volume.
            data = self.get_latest_data()
            self.update(data)
            self._emit_mirror_state()

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.cleanup()

    # =========================================================================
    # Automatic Session State Recording
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
            volumes_processed=getattr(self, "_volumes_processed", 0),
            task_state_snapshot=getattr(self, "_task_state_snapshot", {}),
        )
