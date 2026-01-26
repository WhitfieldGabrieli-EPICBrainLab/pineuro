"""Task factory for creating neurofeedback tasks from configuration.

This module provides a factory function for creating task instances from
registered task types. It uses the task registry to validate configuration
and instantiate the appropriate task class.

Examples
--------
>>> from pineuro.tasks.factory import create_task
>>> task = create_task(
...     task_id='ball',
...     coordinator=coordinator,
...     config={'initial_scale_factor': 10.0, 'fullscreen': True},
...     session_state=session_state,
... )
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import BaseTask
from .task_registry import get_task_metadata

if TYPE_CHECKING:
    from .session import SessionState


# Map task_id to task class
# This is populated when task modules are imported
_TASK_CLASSES: Dict[str, type] = {}


def register_task_class(task_id: str, task_class: type) -> None:
    """Register a task class for factory creation.

    This should be called by task modules after defining their task class.

    Parameters
    ----------
    task_id : str
        Unique task identifier (must match registry metadata)
    task_class : type
        Task class (must be subclass of BaseTask)
    """
    _TASK_CLASSES[task_id] = task_class


def create_task(
    task_id: str,
    coordinator,
    config: Dict[str, Any],
    session_state: Optional['SessionState'] = None,
    output_dir: Optional[str] = None,
) -> BaseTask:
    """Create a task instance from registered task type.

    Parameters
    ----------
    task_id : str
        Registered task identifier (e.g., 'ball')
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data
    config : dict
        Task configuration parameters (validated against registry metadata)
    session_state : SessionState, optional
        Session state for adaptive difficulty
    output_dir : str, optional
        Output directory for task logs

    Returns
    -------
    BaseTask
        Configured task instance

    Raises
    ------
    ValueError
        If task_id is not registered or config validation fails
    """
    # Get metadata and validate config
    metadata = get_task_metadata(task_id)
    metadata.validate_config(config)

    # Get task class
    if task_id not in _TASK_CLASSES:
        raise ValueError(
            f"Task class not registered for '{task_id}'. "
            f"Ensure the task module is imported."
        )

    task_class = _TASK_CLASSES[task_id]

    # Filter config to only include parameters defined in metadata
    valid_keys = {p.key for p in metadata.parameters}
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}

    # Create task instance
    return task_class(
        coordinator=coordinator,
        session_state=session_state,
        output_dir=output_dir,
        **filtered_config,
    )


def get_registered_task_ids() -> list[str]:
    """Get list of task IDs that have both metadata and class registered.

    Returns
    -------
    list[str]
        Task IDs that can be created via create_task()
    """
    from .task_registry import list_available_tasks
    metadata_ids = set(list_available_tasks().keys())
    class_ids = set(_TASK_CLASSES.keys())
    return sorted(metadata_ids & class_ids)
