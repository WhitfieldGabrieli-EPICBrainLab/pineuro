"""
PsychoPy task integration for real-time neurofeedback.

This module provides base classes and utilities for creating PsychoPy tasks
that receive real-time fMRI activation feedback from the pineuro processing pipeline.

Key Classes
-----------
BaseTask : Abstract base class for all neurofeedback tasks
BallTask : Example task with dual-network feedback and ball movement
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
register_task : Register a new task type
get_task_metadata : Get metadata for a registered task
list_available_tasks : List all registered tasks

Examples
--------
See examples/run_feedback_with_task.py for complete integration example.
"""

from .base import BaseTask
from .session import SessionState
from .utils import ActivationSmoother, ActivationNormalizer
from .designs import (
    TaskDesign,
    BlockDesign,
    NuisanceOnlyDesign,
    EventRelatedDesign,
)
from .task_registry import (
    TaskMetadata,
    TaskParameter,
    register_task,
    get_task_metadata,
    list_available_tasks,
)
from .factory import (
    create_task,
    register_task_class,
    get_registered_task_ids,
)

# Import tasks to trigger auto-registration
# Task modules register themselves via register_task() at import time
from .ball import BallTask  # Registers "ball" task

__all__ = [
    # Base classes
    'BaseTask',
    'BallTask',
    'SessionState',
    'ActivationSmoother',
    'ActivationNormalizer',
    # Task designs
    'TaskDesign',
    'BlockDesign',
    'NuisanceOnlyDesign',
    'EventRelatedDesign',
    # Task registry
    'TaskMetadata',
    'TaskParameter',
    'register_task',
    'get_task_metadata',
    'list_available_tasks',
    # Task factory
    'create_task',
    'register_task_class',
    'get_registered_task_ids',
]
