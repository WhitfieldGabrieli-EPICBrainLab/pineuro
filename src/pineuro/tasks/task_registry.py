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

from dataclasses import dataclass
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
class TaskMetadata:
    """Metadata for a task type.

    Attributes:
        task_id: Unique identifier (e.g., "ball", "nback")
        display_name: Human-readable name shown in GUI (e.g., "Ball Task")
        description: Brief description of the task
        parameters: List of configurable parameters
        cross_parameter_validator: Optional function to validate relationships
            between parameters (e.g., "min_hits < max_hits")
    """
    task_id: str
    display_name: str
    description: str
    parameters: List[TaskParameter]
    cross_parameter_validator: Optional[Callable[[Dict[str, Any]], None]] = None

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
