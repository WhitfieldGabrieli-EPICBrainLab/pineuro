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
...     config={'initial_scale_factor': 10.0},
...     task_state=task_state,
... )
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseTask
from .task_registry import TaskConfigError, get_task_metadata

__all__ = ["create_task", "create_localizer_task", "get_registered_task_ids", "TaskConfigError"]

if TYPE_CHECKING:
    from .state import TaskState

logger = logging.getLogger(__name__)

# Map task_id to task class
# This is populated when task modules are imported
_TASK_CLASSES: dict[str, type] = {}

# Runtime parameters that can be passed to any task (not defined in metadata)
# These are runtime-specific settings that the framework provides
RUNTIME_PARAMETERS = {
    "mask_names",  # List of mask names to use
    "feedback_mode",  # Whether feedback is on/off
    "signal_mode",  # "activation" or "connectivity"
    "connectivity_target",  # "increase" or "decrease" (ball direction for connectivity)
    "presenter_monitor",  # Which monitor to use for display
    "trigger_keys",  # Scanner trigger keys (study-level, applies to all tasks)
}


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
    config: dict[str, Any],
    task_state: Optional["TaskState"] = None,
    output_dir: str | None = None,
    simulator=None,
) -> BaseTask:
    """Create a task instance from registered task type.

    Parameters
    ----------
    task_id : str
        Registered task identifier (e.g., 'ball', 'thermometer')
    coordinator : NeurofeedbackCoordinator
        Coordinator providing activation data
    config : dict
        Task configuration parameters. Includes:
        - Parameters defined in TaskMetadata (validated)
        - Runtime parameters: mask_names, feedback_mode, presenter_monitor
    task_state : TaskState, optional
        Session state for adaptive difficulty
    output_dir : str, optional
        Output directory for task logs
    simulator : VolumeSimulator, optional
        Volume simulator for offline testing

    Returns
    -------
    BaseTask
        Configured task instance

    Raises
    ------
    ValueError
        If task_id is not registered or insufficient networks are provided.
    TaskConfigError
        If config validation fails (subclass of ValueError). Contains all
        validation errors, not just the first failure.
    """
    # Get metadata and validate config
    metadata = get_task_metadata(task_id)
    metadata.validate_config(config)

    # Get task class
    if task_id not in _TASK_CLASSES:
        raise ValueError(f"Task class not registered for '{task_id}'. Ensure the task module is imported.")

    task_class = _TASK_CLASSES[task_id]
    if not (isinstance(task_class, type) and issubclass(task_class, BaseTask)):
        raise TypeError(f"Registered task class for '{task_id}' is not a BaseTask subclass: {task_class!r}")

    # Validate network count
    mask_names = config.get("mask_names", coordinator.mask_names)
    if len(mask_names) < metadata.n_networks:
        raise ValueError(
            f"Task '{task_id}' requires {metadata.n_networks} network(s), "
            f"but only {len(mask_names)} mask(s) provided: {mask_names}"
        )

    # Filter config to include:
    # 1. Parameters defined in metadata
    # 2. Runtime parameters (mask_names, feedback_mode, etc.)
    valid_keys = {p.key for p in metadata.parameters} | RUNTIME_PARAMETERS
    unknown_keys = set(config) - valid_keys
    if unknown_keys:
        logger.warning(f"Task '{task_id}': ignoring unrecognized config keys: {sorted(unknown_keys)}")
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}

    logger.info(f"Creating task '{task_id}' with {len(mask_names)} mask(s)")

    # Create task instance
    task: BaseTask = task_class(
        coordinator=coordinator,
        task_state=task_state,
        output_dir=output_dir,
        simulator=simulator,
        **filtered_config,
    )
    return task


def create_localizer_task(
    task_id: str,
    output_dir,
    participant_id: str,
    session_id: str,
    config: dict[str, Any],
    **kwargs,
):
    """Create a localizer task instance (no coordinator needed).

    Parameters
    ----------
    task_id : str
        Registered localizer task identifier (e.g., 'self_other')
    output_dir : str or Path
        Directory for events TSV output
    participant_id : str
        Participant identifier (e.g., 'sub-01')
    session_id : str
        BIDS session identifier (e.g., 'ses-01')
    config : dict
        Task configuration parameters (validated against metadata)
    **kwargs
        Additional keyword arguments passed to the task constructor
        (e.g., run, trigger_keys, monitor_position)

    Returns
    -------
    BaseLocalizerTask
        Configured localizer task instance

    Raises
    ------
    ValueError
        If task_id is not registered or config validation fails
    """
    metadata = get_task_metadata(task_id)
    metadata.validate_config(config)

    if task_id not in _TASK_CLASSES:
        raise ValueError(f"Task class not registered for '{task_id}'. Ensure the task module is imported.")

    task_class = _TASK_CLASSES[task_id]

    # Filter config to metadata-defined parameters only
    filtered = {p.key: config.get(p.key, p.default) for p in metadata.parameters}

    logger.info(f"Creating localizer task '{task_id}' for {participant_id} / {session_id}")

    return task_class(
        output_dir=output_dir,
        participant_id=participant_id,
        session_id=session_id,
        **filtered,
        **kwargs,
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
