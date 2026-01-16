"""pyrtnf - Python Real-Time Neurofeedback library."""

from .feedback import run_feedback_session
from .preprocessing import preprocessing, collect_dicoms
from .mask_extraction import mask_extraction_rest, coregister_2vol
from .tasks import BaseTask, BallTask, ActivationSmoother, ActivationNormalizer
from .tasks.session import SessionState, RunInfo
from .session import (
    # Configuration dataclasses
    SessionConfig,
    PathConfig,
    ScanConfig,
    MaskConfig,
    FeedbackConfig,
    TaskConfig,
    # Orchestration functions
    setup_session,
    run_single_task,
    run_full_session,
    run_next_task,
    reset_session_state,
)

__all__ = [
    # Core functions
    'run_feedback_session',
    'preprocessing',
    'collect_dicoms',
    'mask_extraction_rest',
    'coregister_2vol',
    # Task classes
    'BaseTask',
    'BallTask',
    'ActivationSmoother',
    'ActivationNormalizer',
    # Session state
    'SessionState',
    'RunInfo',
    # Configuration dataclasses
    'SessionConfig',
    'PathConfig',
    'ScanConfig',
    'MaskConfig',
    'FeedbackConfig',
    'TaskConfig',
    # Orchestration functions
    'setup_session',
    'run_single_task',
    'run_full_session',
    'run_next_task',
    'reset_session_state',
]
