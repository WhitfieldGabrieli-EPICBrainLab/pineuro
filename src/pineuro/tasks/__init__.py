"""
PsychoPy task integration for real-time neurofeedback.

This module provides base classes and utilities for creating PsychoPy tasks
that receive real-time fMRI activation feedback from the pineuro processing pipeline.

Key Classes
-----------
BaseTask : Abstract base class for all neurofeedback tasks
BallTask : Example task with dual-network feedback and ball movement
GaugeTask : Analog gauge (brain-o-meter) with differential feedback
SessionState : Persistent state management for multi-run sessions
ActivationSmoother : Exponential moving average smoother
ActivationNormalizer : Percentile-based activation normalizer

Task Designs
------------
TaskDesign : Abstract base class for GLM task designs
BlockDesign : Block design with baseline and active periods
NuisanceOnlyDesign : No task regressors (resting state)
EventRelatedDesign : Event-related design with discrete onsets

Task Registry
-------------
TaskMetadata : Metadata describing a task's configurable parameters
TaskParameter : Definition of a single task parameter
SessionStateField : Definition of a persistent session state field
PostRunQuestion : Definition of a post-run slider question
register_task : Register a new task type
pineuro_task : Decorator for easy task registration
get_task_metadata : Get metadata for a registered task
list_available_tasks : List all registered tasks
discover_tasks : Discover and register tasks from plugin directories

UI Helpers
----------
create_fixation_cross : Create a fixation cross stimulus
create_progress_bar : Create a progress bar with outline and fill
create_circle : Create a circle stimulus
create_target_zone : Create a horizontal target zone indicator
display_text : Display text with optional duration or keypress wait

Examples
--------
See examples/run_feedback_with_task.py for complete integration example.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Optional

from .base import BaseTask
from .designs import (
    BlockDesign,
    EventRelatedDesign,
    NuisanceOnlyDesign,
    TaskDesign,
)
from .factory import (
    create_localizer_task,
    create_task,
    get_registered_task_ids,
    register_task_class,
)
from .localizer_base import BaseLocalizerTask
from .session import SessionState
from .task_base import TaskBase
from .task_registry import (
    NetworkRole,
    PostRunQuestion,
    SessionStateField,
    TaskMetadata,
    TaskParameter,
    get_task_metadata,
    list_available_tasks,
    pineuro_task,
    register_task,
)
from .ui_helpers import (
    create_circle,
    create_fixation_cross,
    create_progress_bar,
    create_target_zone,
    display_text,
    update_progress_bar,
    wait_with_escape,
)
from .utils import ActivationNormalizer, ActivationSmoother

logger = logging.getLogger(__name__)


def discover_tasks(plugin_dirs: list[Path] | None = None) -> list[str]:
    """Discover and register tasks from plugin directories.

    Scans specified directories for Python modules containing task classes
    decorated with @pineuro_task. This enables loading external tasks
    without modifying pineuro's source code.

    Parameters
    ----------
    plugin_dirs : list of Path, optional
        Directories to scan for task modules. Each Python file (.py) in these
        directories will be imported, triggering any @pineuro_task decorators.
        If None, no directories are scanned.

    Returns
    -------
    list of str
        List of task_ids that were discovered and registered

    Example
    -------
    ```python
    from pathlib import Path
    from pineuro.tasks import discover_tasks, list_available_tasks

    # Discover tasks from a custom directory
    discovered = discover_tasks([Path('/path/to/my/tasks')])
    print(f"Discovered tasks: {discovered}")

    # Now these tasks are available
    all_tasks = list_available_tasks()
    ```

    Notes
    -----
    - Files starting with '_' or '__' are skipped
    - Import errors are logged but don't stop discovery
    - Tasks must use the @pineuro_task decorator to be registered
    """
    if plugin_dirs is None:
        return []

    discovered: list[str] = []
    existing_tasks = set(get_registered_task_ids())

    for plugin_dir in plugin_dirs:
        plugin_dir = Path(plugin_dir)
        if not plugin_dir.is_dir():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            continue

        for py_file in plugin_dir.glob("*.py"):
            # Skip private modules
            if py_file.name.startswith("_"):
                continue

            module_name = py_file.stem

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(f"pineuro_plugin_{module_name}", py_file)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                logger.debug(f"Loaded plugin module: {py_file.name}")

            except Exception as e:
                logger.warning(f"Failed to load plugin {py_file.name}: {e}")
                continue

    # Find newly registered tasks
    new_tasks = set(get_registered_task_ids()) - existing_tasks
    discovered.extend(new_tasks)

    if discovered:
        logger.info(f"Discovered {len(discovered)} tasks: {discovered}")

    return discovered


# Import tasks to trigger auto-registration
# Task modules register themselves via register_task() at import time
from .ball import BallTask  # Registers "ball" task
from .fixation import FixationTask  # Registers "fixation" localizer task
from .gauge import GaugeTask  # Registers "gauge" task
from .self_other import SelfOtherTask  # Registers "self_other" localizer task
from .thermometer import ThermometerTask  # Registers "thermometer" task (example)
from .rnt import RNTTask  # Registers "rnt" localizer task

__all__ = [
    # Base classes
    "TaskBase",
    "BaseTask",
    "BaseLocalizerTask",
    "BallTask",
    "GaugeTask",
    "ThermometerTask",
    "SelfOtherTask",
    "FixationTask",
    "RNTTask",
    "SessionState",
    "ActivationSmoother",
    "ActivationNormalizer",
    # Task designs
    "TaskDesign",
    "BlockDesign",
    "NuisanceOnlyDesign",
    "EventRelatedDesign",
    # Task registry
    "TaskMetadata",
    "TaskParameter",
    "SessionStateField",
    "PostRunQuestion",
    "NetworkRole",
    "register_task",
    "pineuro_task",
    "get_task_metadata",
    "list_available_tasks",
    "discover_tasks",
    # Task factory
    "create_task",
    "create_localizer_task",
    "register_task_class",
    "get_registered_task_ids",
    # UI helpers
    "create_fixation_cross",
    "create_progress_bar",
    "update_progress_bar",
    "create_circle",
    "create_target_zone",
    "display_text",
    "wait_with_escape",
]
