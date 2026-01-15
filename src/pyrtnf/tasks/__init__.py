"""
PsychoPy task integration for real-time neurofeedback.

This module provides base classes and utilities for creating PsychoPy tasks
that receive real-time fMRI activation feedback from the pyrtnf processing pipeline.

Key Classes
-----------
BaseTask : Abstract base class for all neurofeedback tasks
BallTask : Example task with dual-network feedback and ball movement
ActivationSmoother : Exponential moving average smoother
ActivationNormalizer : Percentile-based activation normalizer

Examples
--------
See examples/run_feedback_with_task.py for complete integration example.
"""

from .base import BaseTask
from .ball import BallTask
from .utils import ActivationSmoother, ActivationNormalizer

__all__ = [
    'BaseTask',
    'BallTask',
    'ActivationSmoother',
    'ActivationNormalizer',
]
