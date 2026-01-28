"""
Example single-network neurofeedback task demonstrating the task-agnostic framework.

This module shows how to create a custom task that plugs into pineuro:
1. Define a task class extending BaseTask
2. Define TaskMetadata with parameters, session state, and questions
3. Register both with the task registry and factory

The ThermometerTask displays a simple vertical bar that fills based on
a single network's activation level, with a target zone to hit.

Usage
-----
Once registered, this task will appear in the GUI task dropdown.
Users can configure parameters in the study editor and run the task.
"""

import logging
from typing import Optional, TYPE_CHECKING

from .base import BaseTask
from .task_registry import (
    TaskMetadata,
    TaskParameter,
    SessionStateField,
    PostRunQuestion,
    register_task,
)
from .factory import register_task_class

if TYPE_CHECKING:
    from .session import SessionState

logger = logging.getLogger(__name__)


class ThermometerTask(BaseTask):
    """Single-network thermometer feedback task.

    Displays a vertical bar ("thermometer") that fills based on activation.
    A target zone is shown, and participants try to keep activation within it.

    This is a minimal example demonstrating how to create a custom task.
    For a more complex dual-network example, see BallTask.

    Parameters
    ----------
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data
    mask_names : list of str, optional
        Names of masks to use. Only the first mask is used for this task.
    target_level : float, default=0.5
        Target activation level (0-1 normalized)
    target_width : float, default=0.2
        Width of target zone around target_level
    bar_color : str, default="green"
        Color of the thermometer bar
    feedback_mode : bool, default=True
        If True, bar moves based on activation.
        If False, bar stays at baseline.
    session_state : SessionState, optional
        Session state for persistence across runs
    fullscreen : bool, default=True
        Whether to run in fullscreen mode
    trigger_keys : list of str, optional
        Keys that trigger acquisition start
    """

    # Task identifier for registry lookup
    task_id = "thermometer"

    def __init__(
        self,
        coordinator,
        mask_names: Optional[list[str]] = None,
        target_level: float = 0.5,
        target_width: float = 0.2,
        bar_color: str = "green",
        feedback_mode: bool = True,
        session_state: Optional["SessionState"] = None,
        fullscreen: bool = True,
        trigger_keys: Optional[list[str]] = None,
        presenter_monitor: str = "right",
        **kwargs,  # Accept additional kwargs for factory compatibility
    ):
        # Use first mask for single-network feedback
        mask_names = mask_names or coordinator.mask_names[:1]

        # Initialize base class (handles feedback_mode, session_state, mirror callback)
        super().__init__(
            coordinator=coordinator,
            mask_name=mask_names[0],
            feedback_mode=feedback_mode,
            session_state=session_state,
            **kwargs,
        )

        # Store mask_names for compatibility
        self.mask_names = mask_names

        # Task-specific parameters
        self.target_level = target_level
        self.target_width = target_width
        self.bar_color = bar_color
        self.fullscreen = fullscreen
        self.trigger_keys = trigger_keys or ["t", "+", "5", "equal"]
        self.presenter_monitor = presenter_monitor

        # Runtime state
        self._current_activation = 0.0
        self._time_at_target = 0.0
        self._volumes_processed = 0

        logger.info(f"ThermometerTask initialized: mask={self.mask_name}, target={target_level}")

    def setup(self):
        """Initialize PsychoPy window and stimuli."""
        from psychopy import visual

        # Create window with cross-platform monitor targeting (from BaseTask)
        self.create_window(
            fullscreen=self.fullscreen,
            monitor_position=self.presenter_monitor,
            color='black',
            units='height',
        )

        # Create thermometer background (outline)
        self.bar_outline = visual.Rect(
            self.win,
            width=0.15,
            height=0.7,
            pos=(0, 0),
            lineColor='white',
            lineWidth=3,
            fillColor=None,
        )

        # Create thermometer fill (the bar that moves)
        self.bar_fill = visual.Rect(
            self.win,
            width=0.13,
            height=0.0,  # Starts empty
            pos=(0, -0.35),  # Bottom of outline
            fillColor=self.bar_color,
            lineColor=None,
        )

        # Create target zone
        target_y = (self.target_level - 0.5) * 0.7  # Map 0-1 to outline range
        self.target_zone = visual.Rect(
            self.win,
            width=0.17,
            height=self.target_width * 0.7,
            pos=(0, target_y),
            fillColor=None,
            lineColor='yellow',
            lineWidth=2,
            opacity=0.5,
        )

        # Create instruction text
        self.instruction_text = visual.TextStim(
            self.win,
            text="Focus to raise the bar into the yellow zone",
            pos=(0, 0.42),
            height=0.03,
            color='white',
        )

        # Wait for trigger (uses base class method)
        self.wait_for_trigger(trigger_keys=self.trigger_keys)

        logger.info("ThermometerTask setup complete")

    def update(self, data: dict):
        """Update display based on latest activation data."""
        from psychopy import event

        # Check for escape
        if event.getKeys(keyList=['escape']):
            self._running = False
            return

        # Get activation for our mask
        activations = self.coordinator.get_latest_activations()
        if activations and 'activations' in activations:
            self._current_activation = activations['activations'].get(self.mask_name, 0.0)
            self._volumes_processed = activations.get('volume', 0)

        # Map activation to bar height (clamp to 0-1)
        if self.feedback_mode and not activations.get('is_baseline', True):
            # Normalize activation to 0-1 range (assuming activation is roughly -2 to 2)
            normalized = (self._current_activation + 2) / 4
            normalized = max(0.0, min(1.0, normalized))
        else:
            normalized = 0.0  # Baseline or non-feedback mode

        # Update bar fill
        bar_height = normalized * 0.68  # Slightly less than outline height
        self.bar_fill.height = bar_height
        self.bar_fill.pos = (0, -0.35 + bar_height / 2)

        # Check if in target zone
        if abs(normalized - self.target_level) < self.target_width / 2:
            self._time_at_target += 1 / 60  # Assume 60 Hz
            self.bar_fill.fillColor = 'lime'  # Bright green when in zone
        else:
            self.bar_fill.fillColor = self.bar_color

        # Draw everything
        self.bar_outline.draw()
        self.target_zone.draw()
        self.bar_fill.draw()
        self.instruction_text.draw()

        # Capture screenshot for mirror (before flip)
        self._capture_and_emit_screenshot()

        self.win.flip()

    def cleanup(self):
        """Clean up resources and save state."""
        logger.info(f"ThermometerTask cleanup: time_at_target={self._time_at_target:.1f}s")

        # Update cumulative time at target in session state
        if self._session_state is not None:
            current_time = self._session_state.get("time_at_target", 0.0)
            self._session_state.set("time_at_target", current_time + self._time_at_target)

        # Set snapshot for base class's automatic run recording
        self._task_state_snapshot = {
            "time_at_target": self._time_at_target,
            "target_level": self.target_level,
        }

        # Let base class handle run recording, questions, and window close
        super().cleanup()

    def run(self):
        """Main task loop."""
        from psychopy import core

        self._running = True
        self.setup()

        try:
            while self._running and not self.coordinator.is_complete():
                # Check for errors
                self.coordinator.check_error()

                # Update display
                data = self.coordinator.get_latest_activations() or {}
                self.update(data)

                # Small sleep to avoid busy-waiting
                core.wait(0.001)

        except KeyboardInterrupt:
            logger.info("ThermometerTask interrupted")
        finally:
            self._running = False
            self.cleanup()

    def stop(self):
        """Signal task to stop."""
        self._running = False


# =============================================================================
# Task Metadata and Registration
# =============================================================================

THERMOMETER_METADATA = TaskMetadata(
    task_id="thermometer",
    display_name="Thermometer Task",
    description="Single-network feedback with vertical bar display",
    n_networks=1,  # Only requires one network
    parameters=[
        TaskParameter(
            key="target_level",
            label="Target Level",
            type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            hint="Target activation level (0-1 normalized)",
        ),
        TaskParameter(
            key="target_width",
            label="Target Width",
            type="float",
            default=0.2,
            min_value=0.05,
            max_value=0.5,
            hint="Width of target zone (larger = easier)",
        ),
        TaskParameter(
            key="bar_color",
            label="Bar Color",
            type="str",
            default="green",
            hint="Color of the thermometer bar",
        ),
        TaskParameter(
            key="fullscreen",
            label="Fullscreen Mode",
            type="bool",
            default=True,
            hint="Run task in fullscreen",
        ),
        TaskParameter(
            key="trigger_keys",
            label="Scanner Trigger Keys",
            type="list[str]",
            default=["t", "+", "5", "equal"],
            hint="Keys that trigger volume acquisition",
        ),
    ],
    session_state_fields=[
        SessionStateField(
            key="time_at_target",
            type="float",
            default=0.0,
            description="Cumulative time spent in target zone (seconds)",
        ),
    ],
    post_run_questions=[
        PostRunQuestion(
            text="How focused did you feel during the task?",
            left_label="Not at all",
            right_label="Extremely",
            condition="always",
        ),
        PostRunQuestion(
            text="How well could you control the bar?",
            left_label="Not at all",
            right_label="Very well",
            condition="feedback_only",
        ),
    ],
)

# Auto-register on import
register_task(THERMOMETER_METADATA)
register_task_class("thermometer", ThermometerTask)
