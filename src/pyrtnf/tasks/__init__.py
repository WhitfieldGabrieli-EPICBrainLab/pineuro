"""
PsychoPy task integration for real-time neurofeedback.

This module provides base classes and utilities for creating PsychoPy tasks
that receive real-time fMRI activation feedback from the pyrtnf processing pipeline.

Key Classes
-----------
BaseTask : Abstract base class for all neurofeedback tasks
BallTask : Example task with dual-network feedback and ball movement
SessionState : Persistent state management for multi-run sessions
ActivationSmoother : Exponential moving average smoother
ActivationNormalizer : Percentile-based activation normalizer

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
from .task_registry import (
    TaskMetadata,
    TaskParameter,
    register_task,
    get_task_metadata,
    list_available_tasks,
)

# Import tasks to trigger auto-registration
# Task modules register themselves via register_task() at import time
from .ball import BallTask  # Registers "ball" task

__all__ = [
    'BaseTask',
    'BallTask',
    'SessionState',
    'ActivationSmoother',
    'ActivationNormalizer',
    'TaskMetadata',
    'TaskParameter',
    'register_task',
    'get_task_metadata',
    'list_available_tasks',
]
