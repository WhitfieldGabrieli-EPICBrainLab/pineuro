"""Task registration system for pluggable neurofeedback tasks.

This module provides the infrastructure for adding new task types to pineuro
without modifying the GUI code. Tasks register themselves by defining metadata
that describes their configurable parameters.

Example:
    # Define task metadata
    ball_metadata = TaskMetadata(
        task_id="ball",
        display_name="Ball Task",
        description="Dual-network neurofeedback with ball control",
        parameters=[
            TaskParameter(
                key="initial_scale_factor",
                label="Initial Scale Factor",
                type="float",
                default=10.0,
                min_value=1.0,
                max_value=50.0,
                hint="Starting difficulty (larger = easier)",
            ),
        ],
    )

    # Register task
    register_task(ball_metadata)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Callable

# Task registry (task_id -> metadata)
_TASK_REGISTRY: Dict[str, 'TaskMetadata'] = {}


@dataclass
class TaskParameter:
    """Definition of a single task parameter for GUI configuration.

    Attributes:
        key: Configuration key (e.g., "initial_scale_factor")
        label: Display label in GUI (e.g., "Initial Scale Factor")
        type: Parameter type (determines GUI control type)
        default: Default value
        hint: Help text shown below field (optional)
        min_value: Minimum value for numeric types (optional)
        max_value: Maximum value for numeric types (optional)
        validator: Custom validation function (optional)
            Should take value and raise ValueError if invalid
    """
    key: str
    label: str
    type: Literal["float", "int", "bool", "str", "list[str]"]
    default: Any
    hint: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    validator: Optional[Callable[[Any], None]] = None

    def validate(self, value: Any) -> None:
        """Validate a value for this parameter.

        Args:
            value: Value to validate

        Raises:
            ValueError: If validation fails
        """
        # Type-specific validation
        if self.type in ["int", "float"]:
            try:
                num = float(value) if self.type == "float" else int(value)
            except (ValueError, TypeError):
                raise ValueError(f"{self.label} must be a number")

            if self.min_value is not None and num < self.min_value:
                raise ValueError(f"{self.label} must be >= {self.min_value}")

            if self.max_value is not None and num > self.max_value:
                raise ValueError(f"{self.label} must be <= {self.max_value}")

        # Custom validation
        if self.validator is not None:
            self.validator(value)


@dataclass
class SessionStateField:
    """Definition of a task-specific persistent state field.

    Session state fields are persisted across runs and can be used to track
    cumulative statistics, adaptive difficulty parameters, or any other
    state that should survive between task runs.

    Attributes:
        key: Field name used in session state dictionary (e.g., "hit_counters")
        type: Data type for validation and serialization
        default: Default value when session state is first created
        description: Documentation for this field (optional)
    """
    key: str
    type: Literal["float", "int", "bool", "list[float]", "list[int]"]
    default: Any
    description: str = ""


@dataclass
class PostRunQuestion:
    """Definition of a post-run slider question.

    Questions are displayed after each run completes. They can be conditional
    based on run type (feedback vs transfer) or run position (last feedback only).

    Attributes:
        text: Question text displayed to participant
        left_label: Label at left end of slider (e.g., "Never", "False")
        right_label: Label at right end of slider (e.g., "Always", "True")
        condition: When to show this question:
            - "always": Show after every run
            - "feedback_only": Show only after feedback runs (not transfer)
            - "last_feedback_only": Show only after the final feedback run
    """
    text: str
    left_label: str = "Never"
    right_label: str = "Always"
    condition: Literal["always", "feedback_only", "last_feedback_only"] = "always"


@dataclass
class TaskMetadata:
    """Metadata for a task type.

    Defines everything a task needs to integrate with the pineuro framework:
    parameters for GUI configuration, session state for persistence across runs,
    post-run questions for participant feedback, and network requirements.

    Attributes:
        task_id: Unique identifier (e.g., "ball", "thermometer")
        display_name: Human-readable name shown in GUI (e.g., "Ball Task")
        description: Brief description of the task
        parameters: List of configurable parameters for GUI
        cross_parameter_validator: Optional function to validate relationships
            between parameters (e.g., "min_hits < max_hits")
        n_networks: Number of brain networks (masks) required by this task.
            Default is 1. Ball task requires 2 (top/bottom networks).
        session_state_fields: List of fields to persist across runs.
            These are stored in session_state.json and available to the task.
        post_run_questions: List of slider questions to display after each run.
            Questions can be conditional on run type or position.
    """
    task_id: str
    display_name: str
    description: str
    parameters: List[TaskParameter]
    cross_parameter_validator: Optional[Callable[[Dict[str, Any]], None]] = None
    n_networks: int = 1
    session_state_fields: List[SessionStateField] = field(default_factory=list)
    post_run_questions: List[PostRunQuestion] = field(default_factory=list)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate a complete task configuration.

        Args:
            config: Task configuration dictionary

        Raises:
            ValueError: If validation fails
        """
        # Validate individual parameters
        for param in self.parameters:
            if param.key in config:
                param.validate(config[param.key])

        # Cross-parameter validation
        if self.cross_parameter_validator is not None:
            self.cross_parameter_validator(config)


def register_task(metadata: TaskMetadata) -> None:
    """Register a task type.

    Args:
        metadata: TaskMetadata object describing the task
    """
    _TASK_REGISTRY[metadata.task_id] = metadata


def get_task_metadata(task_id: str) -> TaskMetadata:
    """Get metadata for a task type.

    Args:
        task_id: Task identifier (e.g., "ball")

    Returns:
        TaskMetadata object

    Raises:
        ValueError: If task_id is not registered
    """
    if task_id not in _TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}")
    return _TASK_REGISTRY[task_id]


def list_available_tasks() -> Dict[str, TaskMetadata]:
    """List all registered tasks.

    Returns:
        Dictionary mapping task_id to TaskMetadata
    """
    return _TASK_REGISTRY.copy()


def pineuro_task(metadata: TaskMetadata):
    """Decorator for registering a task class with its metadata.

    Simplifies task registration by combining register_task() and
    register_task_class() into a single decorator. Also verifies that
    the class's task_id attribute matches the metadata.

    Parameters
    ----------
    metadata : TaskMetadata
        Metadata describing the task's parameters, questions, etc.

    Returns
    -------
    callable
        Class decorator that registers the task

    Raises
    ------
    ValueError
        If class.task_id doesn't match metadata.task_id

    Example
    -------
    ```python
    MY_TASK_METADATA = TaskMetadata(
        task_id="my_task",
        display_name="My Task",
        ...
    )

    @pineuro_task(MY_TASK_METADATA)
    class MyTask(BaseTask):
        task_id = "my_task"
        ...
    ```
    """
    def decorator(cls):
        # Verify task_id matches
        if hasattr(cls, 'task_id') and cls.task_id != metadata.task_id:
            raise ValueError(
                f"Task class '{cls.__name__}' has task_id='{cls.task_id}' "
                f"but metadata has task_id='{metadata.task_id}'. They must match."
            )

        # Register metadata
        register_task(metadata)

        # Register class (import here to avoid circular imports)
        from .factory import register_task_class
        register_task_class(metadata.task_id, cls)

        return cls

    return decorator
