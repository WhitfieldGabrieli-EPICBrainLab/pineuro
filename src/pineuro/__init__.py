"""pineuro - Python Real-Time fMRI Neurofeedback

A complete pipeline for real-time fMRI neurofeedback research, from volume acquisition
through preprocessing, network mask extraction, and moment-to-moment BOLD activation estimation.

Main Functions
--------------
run_feedback_session : Run complete real-time feedback session
preprocessing : DICOM conversion, motion correction, brain extraction
collect_volumes : Watch for and collect incoming scanner volumes (DICOM or NIfTI)
collect_dicoms : Alias for collect_volumes (backward compatibility)
mask_extraction_rest : Extract network masks using ICA and Yeo atlas
mask_extraction_task : Extract network masks using GLM contrasts from task localizer
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
>>> from pineuro import preprocessing, mask_extraction_rest, run_feedback_session
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

__version__ = "0.1.0"

from .feedback import run_feedback_session, get_next_run_number, get_slot_run_info, STEP_TO_RUN_NUMBER
from .preprocessing import preprocessing, collect_volumes, collect_dicoms
from .mask_extraction import mask_extraction_rest, mask_extraction_task, coregister_2vol
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
    VisitConfig,
    ShamConfig,
    SimulatorSourceMapping,
    # Orchestration functions
    setup_session,
    run_single_task,
    run_full_session,
    run_next_task,
    reset_session_state,
    # Multi-visit support functions
    resolve_mask_source,
    detect_existing_visits,
    next_visit_id,
    is_single_visit_data,
    migrate_to_multi_visit,
    setup_visit_with_reused_masks,
)
from .sham import (
    ShamActivationProvider,
    ReplayActivationProvider,
    load_randomization,
    get_condition_for_participant,
    resolve_sham_masks,
    find_feedbacklog_for_replay,
)
from .utils import (
    # Timing functions
    enable_timing,
    disable_timing,
    get_timing_data,
    save_timing_report,
    Timer,
    # Platform checks
    check_platform_requirements,
    # Utilities
    outputs_exist,
    ProgressEmitter,
    # HRF computation
    compute_hrf,
    compute_boxcar_task_regressor,
)
from .script_generator import (
    generate_mask_extraction_script,
    generate_task_mask_extraction_script,
    save_script,
)
from . import config

__all__ = [
    # Version
    '__version__',
    # Core functions
    'run_feedback_session',
    'get_next_run_number',
    'get_slot_run_info',
    'STEP_TO_RUN_NUMBER',
    'preprocessing',
    'collect_volumes',
    'collect_dicoms',  # Backward compatibility alias for collect_volumes
    'mask_extraction_rest',
    'mask_extraction_task',
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
    'VisitConfig',
    'ShamConfig',
    'SimulatorSourceMapping',
    # Sham functionality
    'ShamActivationProvider',
    'ReplayActivationProvider',
    'load_randomization',
    'get_condition_for_participant',
    'resolve_sham_masks',
    'find_feedbacklog_for_replay',
    # Orchestration functions
    'setup_session',
    'run_single_task',
    'run_full_session',
    'run_next_task',
    'reset_session_state',
    # Multi-visit support functions
    'resolve_mask_source',
    'detect_existing_visits',
    'next_visit_id',
    'is_single_visit_data',
    'migrate_to_multi_visit',
    'setup_visit_with_reused_masks',
    # Timing functions
    'enable_timing',
    'disable_timing',
    'get_timing_data',
    'save_timing_report',
    'Timer',
    # Platform checks
    'check_platform_requirements',
    # Utilities
    'outputs_exist',
    'ProgressEmitter',
    # Script generation
    'generate_mask_extraction_script',
    'generate_task_mask_extraction_script',
    'save_script',
    # Configuration
    'config',
]
