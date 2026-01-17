"""pyrtnf - Python Real-Time fMRI Neurofeedback

A complete pipeline for real-time fMRI neurofeedback research, from DICOM acquisition
through preprocessing, network mask extraction, and moment-to-moment BOLD activation estimation.

Main Functions
--------------
run_feedback_session : Run complete real-time feedback session
preprocessing : DICOM conversion, motion correction, brain extraction
collect_dicoms : Watch for and collect incoming scanner DICOMs
mask_extraction_rest : Extract network masks using ICA and Yeo atlas
coregister_2vol : Coregister masks to real-time scanner position

Task Framework
--------------
BaseTask : Abstract base for custom neurofeedback tasks
BallTask : Dual-network ball game with adaptive difficulty
ActivationSmoother : Exponential moving average smoothing
ActivationNormalizer : Percentile-based normalization

Session Orchestration
--------------------
SessionConfig : High-level session configuration
run_full_session : Run complete multi-run session (transfer + feedback + transfer)
setup_session : Initialize session (collection, preprocessing, mask extraction)
run_single_task : Run one feedback/transfer task
SessionState : Persistent state across runs

Examples
--------
>>> from pyrtnf import preprocessing, mask_extraction_rest, run_feedback_session
>>>
>>> # Step 1: Preprocess resting-state data
>>> preproc_file, mask_file, ref_file = preprocessing(
...     participant_id='sub-01', task='rest',
...     root_dir='/data/preprocessing',
...     input_dicoms='/data/raw/rest'
... )
>>>
>>> # Step 2: Extract network masks
>>> mask_extraction_rest(
...     root_dir='/data/masks',
...     preproc_runs=[preproc_file],
...     yeo_networks=[7],  # Default Mode Network
... )
>>>
>>> # Step 3: Run feedback session
>>> results = run_feedback_session(
...     mask_paths=['/data/masks/03_masks/network_7.nii.gz'],
...     subject='sub-01', task='nback', run=1,
...     output_dir='/data/feedback',
...     incoming_dir='/scanner/export/dicoms',
...     n_baseline=25, n_volumes=200
... )
"""

from .feedback import run_feedback_session, get_next_run_number
from .preprocessing import preprocessing, collect_dicoms
from .mask_extraction import mask_extraction_rest, coregister_2vol
from .tasks import BaseTask, BallTask, ActivationSmoother, ActivationNormalizer
from .tasks.ball import compute_hrf, compute_boxcar_task_regressor
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
from .utils import (
    # Timing functions
    enable_timing,
    disable_timing,
    get_timing_data,
    save_timing_report,
    Timer,
)

__all__ = [
    # Core functions
    'run_feedback_session',
    'get_next_run_number',
    'preprocessing',
    'collect_dicoms',
    'mask_extraction_rest',
    'coregister_2vol',
    # Task classes
    'BaseTask',
    'BallTask',
    'ActivationSmoother',
    'ActivationNormalizer',
    # Task regressor computation (ball task)
    'compute_hrf',
    'compute_boxcar_task_regressor',
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
    # Timing functions
    'enable_timing',
    'disable_timing',
    'get_timing_data',
    'save_timing_report',
    'Timer',
]
