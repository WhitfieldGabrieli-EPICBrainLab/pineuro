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

Threading Model
---------------
Two threads interact with each task during a feedback task:

Coordinator thread (background)
    Runs incremental GLM and activation computation.  Calls
    ``_update_data()`` after each processed volume.  Writes ``_latest_data``
    (protected by ``self._lock``) and reads ``_accumulating`` /
    ``_block_activations`` (lock-free, coordinator-thread only).

Task thread (main thread, required by PsychoPy)
    Runs ``run()`` → ``setup()`` → ``update()`` loop.  Reads ``_latest_data``
    via ``get_latest_data()`` (acquires ``self._lock``).  Writes
    ``_accumulating`` and ``_block_activations`` via
    ``_start_block_accumulation()`` / ``_stop_block_accumulation()``
    (lock-free, task-thread only).

Rule: **Only call PsychoPy APIs from the task (main) thread.**  Never call
PsychoPy drawing functions from the coordinator thread.

See Also
--------
pineuro.tasks.protocol : Protocols documenting the full task/coordinator API.

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
    from .state import TaskState
    from .task_registry import PostRunQuestion

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
    task_state : TaskState, optional
        Task state for persistence across runs
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
    _task_state : TaskState or None
        Task state for persistence

    Examples
    --------
    >>> class MyTask(BaseTask):
    ...     def setup(self):
    ...         self.win = visual.Window([800, 600])
    ...
    ...     def update(self, data: dict):
    ...         # Draw based on data['activations'][self.mask_name]
    ...         pass
    ...
    ...     # cleanup() is optional - base handles questions + window close
    """

    # Subclasses should set this to match their registered task_id
    # Used for accessing task metadata (parameters, task state, questions)
    task_id: str = "base"

    def __init__(
        self,
        coordinator,
        mask_name: str | None = None,
        refresh_rate: float = 60.0,
        feedback_mode: bool = True,
        task_state: Optional["TaskState"] = None,
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
        self._task_state = task_state
        # Set by the runtime (session.py / GUI pipeline_runner) before run():
        # True when this is the last feedback run for THIS task. Gates
        # "last_feedback_only" post-run questions. Default False covers
        # standalone/test tasks, localizers, and transfer runs.
        self._is_last_feedback_run: bool = False
        self._questions_file = None  # Optional file handle for logging responses
        self._questions_writer = None  # Optional CSV writer

        # Thread-safe access to latest data
        self._lock = threading.Lock()
        self._latest_data = {
            "activations": {},  # All mask activations
            "volume": 0,
            "time": 0.0,
            "is_baseline": True,
        }

        # Block accumulation state (optional — call _start_block_accumulation() to enable)
        self._accumulating: bool = False
        self._block_activations: list[float] = []

        # Volume counter for run results (subclasses may overwrite it).
        self._volumes_processed: int = 0

    def get_task_metadata(self):
        """Get metadata for this task type.

        Returns
        -------
        TaskMetadata
            Metadata object with parameters, task state fields, and questions.
            Returns None if task is not registered.
        """
        from .task_registry import get_task_metadata

        try:
            return get_task_metadata(self.task_id)
        except ValueError:
            return None

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
            - 'activations': dict[str, float] - current activation per mask name
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
        self._reset_poll_state()
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

    def _eligible_post_run_questions(self) -> list["PostRunQuestion"]:
        """Post-run questions whose condition passes for the current run context.

        Mirrors the per-question gating documented on
        :class:`~pineuro.tasks.task_registry.PostRunQuestion`:
        - "always": always shown
        - "feedback_only": only when ``feedback_mode`` (feedback run, not transfer)
        - "last_feedback_only": only when ``feedback_mode`` and this is the final
          feedback run for this task (``self._is_last_feedback_run``, set by the
          runtime before the run)

        Reads only init-time / run-stable state (no PsychoPy window), so it is safe
        to call off the task thread — e.g. from the GUI to decide whether a manual
        stop should offer to show questions.
        """
        metadata = self.get_task_metadata()
        if not metadata or not metadata.post_run_questions:
            return []

        is_last = getattr(self, "_is_last_feedback_run", False)

        eligible = []
        for question in metadata.post_run_questions:
            if question.condition == "feedback_only" and not self.feedback_mode:
                continue
            if question.condition == "last_feedback_only" and not (self.feedback_mode and is_last):
                continue
            eligible.append(question)
        return eligible

    def has_pending_post_run_questions(self) -> bool:
        """Whether any post-run question would be shown for the current run."""
        return bool(self._eligible_post_run_questions())

    def _show_post_run_questions(
        self,
        left_button: str = "1",
        right_button: str = "2",
        enter_button: str = "3",
    ) -> list[dict[str, Any]]:
        """Display post-run questions defined in task metadata.

        Called automatically by cleanup(). Tasks with no eligible questions for
        the current run (see :meth:`_eligible_post_run_questions`) skip this (no-op).

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
        questions = self._eligible_post_run_questions()
        if not questions:
            return []  # No-op for tasks without (eligible) questions

        # Check if window exists
        if not hasattr(self, "win") or self.win is None:
            logger.warning("Cannot show questions: no PsychoPy window")
            return []

        responses = []

        for question in questions:
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
    # OVERRIDE GUIDE
    # =========================================================================
    # Must override (abstract — no super()):
    #   setup()                     Create window and stimuli.
    #   update(data)                Draw each frame.
    #
    # Should override (call super()):
    #   cleanup()                   Release resources. Call super().cleanup() at END.
    #   get_run_results()           Return run metrics. Merge with super().get_run_results().
    #   _record_run_completion()    Custom task-state recording. Call super() first.
    #
    # May override (no super() needed):
    #   get_volume_log_suffix()     String appended to coordinator volume log line.
    #   _show_post_run_questions()  Replace the post-run question display entirely.
    #
    # Do not override:
    #   run()                       Main loop; framework-controlled.
    #   _update_data()              Coordinator callback — only with super() called first.
    #   get_latest_data()           Thread-safe data accessor.
    # =========================================================================

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
            - 'activations': dict[str, float] - current activation per mask name
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
        if self.coordinator is not None:
            self.coordinator.send_trigger()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def cleanup(self):
        """Clean up resources after task completes.

        Default implementation:
        1. Records run completion to task state (if configured)
        2. Shows post-run questions (if defined in metadata)
        3. Delegates to TaskBase.cleanup() for window/simulator/pyglet cleanup

        Subclasses can override to add custom cleanup logic, but should
        call super().cleanup() at the end. Set self._task_state_snapshot
        before calling super().cleanup() to include task-specific state.

        For custom question behavior (like BallTask's instruction screen),
        override _show_post_run_questions() instead.
        """
        # Record run completion (no-op if no task state)
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

                # Frame timing
                elapsed = time.time() - frame_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)

            # Process final data after loop exits (race condition fix)
            # The loop exits when is_complete() returns True, which may happen
            # before we've had a chance to process the final volume.
            data = self.get_latest_data()
            self.update(data)

        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.cleanup()

    # =========================================================================
    # Automatic task state recording
    # =========================================================================

    def _run_record_fields(self) -> dict[str, Any]:
        """Source run-completion coordinates from the coordinator.

        Overrides TaskBase's instance-attribute default so feedback runs
        record the correct run type and the coordinator's attempt/slot.
        """
        return {
            "run_type": "feedback" if self.feedback_mode else "transfer",
            "volumes_processed": getattr(self, "_volumes_processed", 0),
            "attempt": self.coordinator.attempt,
            "slot_run_number": self.coordinator.run_number,
            "merge_into_task_state": True,
        }
