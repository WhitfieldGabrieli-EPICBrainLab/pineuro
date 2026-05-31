"""
PsychoPy task integration for real-time neurofeedback.

This module provides base classes and utilities for creating PsychoPy tasks
that receive real-time fMRI activation feedback from the pineuro processing pipeline.

Key Classes
-----------
BaseTask : Abstract base class for all neurofeedback tasks
BallTask : Example task with dual-network feedback and ball movement
GaugeTask : Analog gauge (brain-o-meter) with differential feedback
TaskState : Persistent state management for multi-run sessions
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
TaskStateField : Definition of a persistent task state field
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
from .state import TaskState
from .task_base import TaskBase
from .task_registry import (
    NetworkRole,
    PostRunQuestion,
    TaskMetadata,
    TaskParameter,
    TaskStateField,
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

            except Exception as e:
                logger.warning(f"Failed to load plugin {py_file.name}: {e}")
                continue

    # Find newly registered tasks
    new_tasks = set(get_registered_task_ids()) - existing_tasks
    discovered.extend(new_tasks)

    if discovered:
        logger.info(f"Discovered {len(discovered)} tasks: {discovered}")

    return discovered


# Lazy-load built-in task modules to avoid pulling in numpy/PsychoPy at import
# time.  Tasks register themselves via @pineuro_task when first imported.
def _load_builtin_tasks():
    from . import (  # noqa: F401
        ball,
        gauge,
        rest,
        rit,
        rit_experience_sampling,
        rit_triggered_ball,
        self_other,
        thermometer,
    )


from .task_registry import add_lazy_loader  # noqa: E402

add_lazy_loader(_load_builtin_tasks)

# Names that live in lazily-loaded task modules.  A module-level __getattr__
# triggers the lazy loader so that ``from pineuro.tasks import BallTask`` still
# works without eagerly importing numpy/PsychoPy at package-init time.
_LAZY_TASK_CLASS_NAMES = {
    "BallTask",
    "GaugeTask",
    "ThermometerTask",
    "SelfOtherTask",
    "RestTask",
    "RITTask",
    "RITExperienceSamplingTask",
    "RITTriggeredBallTask",
}


def __getattr__(name: str):
    if name in _LAZY_TASK_CLASS_NAMES:
        _load_builtin_tasks()
        # After loading, the name should be importable from the submodules
        # that were just imported.  Look it up in the factory registry.
        from .factory import _TASK_CLASSES

        for cls in _TASK_CLASSES.values():
            if cls.__name__ == name:
                return cls
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "TaskBase",
    "BaseTask",
    "BaseLocalizerTask",
    "BallTask",
    "GaugeTask",
    "ThermometerTask",
    "SelfOtherTask",
    "RestTask",
    "RITTask",
    "RITExperienceSamplingTask",
    "RITTriggeredBallTask",
    "TaskState",
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
    "TaskStateField",
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
