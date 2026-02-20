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
coregister_reference : Coregister masks to real-time scanner position

Task Framework
--------------
BaseTask : Abstract base for custom neurofeedback tasks
BaseLocalizerTask : Abstract base for localizer/stimulus tasks
BallTask : Built-in dual-network ball game with adaptive difficulty
ThermometerTask : Built-in single-network thermometer feedback
create_task : Factory function to create tasks from registry
NetworkRole : Declare network slot semantics for tasks
ActivationSmoother : Exponential moving average smoothing
ActivationNormalizer : Percentile-based normalization

Session Orchestration
--------------------
SessionConfig : High-level session configuration
TaskConfig : Generic task configuration (task_type, params, network assignment)
run_full_session : Run complete multi-run session
setup_session : Initialize session (collection, preprocessing, mask extraction)
run_single_task : Run one feedback/transfer task via task factory
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

import importlib as _importlib

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Lazy import system (PEP 562)
#
# Heavy neuroimaging libraries (ANTsPy, nilearn, scipy, nibabel) take 8-17s
# to import on cold start. By deferring submodule imports until first access,
# the GUI starts instantly and scripts only pay for what they actually use.
# ---------------------------------------------------------------------------

# Mapping: attribute_name -> (submodule_path, attribute_name_in_submodule)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # --- feedback.py (heavy: nibabel, scipy, watchdog, pydicom) ---
    "run_feedback_session": (".feedback", "run_feedback_session"),
    "ConnectivityEstimator": (".feedback", "ConnectivityEstimator"),
    "get_next_run_number": (".feedback", "get_next_run_number"),
    "get_slot_run_info": (".feedback", "get_slot_run_info"),
    "STEP_TO_RUN_NUMBER": (".feedback", "STEP_TO_RUN_NUMBER"),
    # --- preprocessing.py (heavy: nibabel, scipy) ---
    "preprocessing": (".preprocessing", "preprocessing"),
    "collect_volumes": (".preprocessing", "collect_volumes"),
    "collect_dicoms": (".preprocessing", "collect_dicoms"),
    # --- mask_extraction.py (heavy: nibabel, scipy, pandas) ---
    "mask_extraction_rest": (".mask_extraction", "mask_extraction_rest"),
    "mask_extraction_task": (".mask_extraction", "mask_extraction_task"),
    "import_manual_masks": (".mask_extraction", "import_manual_masks"),
    "coregister_reference": (".mask_extraction", "coregister_reference"),
    "create_tissue_masks": (".mask_extraction", "create_tissue_masks"),
    "ATLAS_SOURCES": (".atlas_registry", "ATLAS_SOURCES"),
    "get_atlas_labels": (".mask_extraction", "get_atlas_labels"),
    # --- tasks (lightweight but deferred to avoid loading task modules) ---
    "BaseTask": (".tasks", "BaseTask"),
    "BaseLocalizerTask": (".tasks", "BaseLocalizerTask"),
    "BallTask": (".tasks", "BallTask"),
    "ThermometerTask": (".tasks", "ThermometerTask"),
    "ActivationSmoother": (".tasks", "ActivationSmoother"),
    "ActivationNormalizer": (".tasks", "ActivationNormalizer"),
    "NetworkRole": (".tasks", "NetworkRole"),
    "create_task": (".tasks", "create_task"),
    "create_localizer_task": (".tasks", "create_localizer_task"),
    # --- tasks.session ---
    "SessionState": (".tasks.session", "SessionState"),
    "RunInfo": (".tasks.session", "RunInfo"),
    # --- session.py (lightweight: stdlib only) ---
    "SessionConfig": (".session", "SessionConfig"),
    "PathConfig": (".session", "PathConfig"),
    "ScanConfig": (".session", "ScanConfig"),
    "MaskConfig": (".session", "MaskConfig"),
    "FeedbackConfig": (".session", "FeedbackConfig"),
    "TaskConfig": (".session", "TaskConfig"),
    "VisitConfig": (".session", "VisitConfig"),
    "ShamConfig": (".session", "ShamConfig"),
    "SimulatorSourceMapping": (".session", "SimulatorSourceMapping"),
    "setup_session": (".session", "setup_session"),
    "run_single_task": (".session", "run_single_task"),
    "run_full_session": (".session", "run_full_session"),
    "run_next_task": (".session", "run_next_task"),
    "reset_session_state": (".session", "reset_session_state"),
    "resolve_mask_source": (".session", "resolve_mask_source"),
    "detect_existing_visits": (".session", "detect_existing_visits"),
    "next_visit_id": (".session", "next_visit_id"),
    "is_single_visit_data": (".session", "is_single_visit_data"),
    "migrate_to_multi_visit": (".session", "migrate_to_multi_visit"),
    "setup_visit_with_reused_masks": (".session", "setup_visit_with_reused_masks"),
    # --- sham.py ---
    "ShamActivationProvider": (".sham", "ShamActivationProvider"),
    "ReplayActivationProvider": (".sham", "ReplayActivationProvider"),
    "load_randomization": (".sham", "load_randomization"),
    "generate_randomization": (".sham", "generate_randomization"),
    "get_condition_for_participant": (".sham", "get_condition_for_participant"),
    "resolve_sham_masks": (".sham", "resolve_sham_masks"),
    "find_feedbacklog_for_replay": (".sham", "find_feedbacklog_for_replay"),
    # --- utils.py (heavy: numpy, nibabel, scipy, templateflow) ---
    "enable_timing": (".utils", "enable_timing"),
    "disable_timing": (".utils", "disable_timing"),
    "get_timing_data": (".utils", "get_timing_data"),
    "save_timing_report": (".utils", "save_timing_report"),
    "Timer": (".utils", "Timer"),
    "check_platform_requirements": (".utils", "check_platform_requirements"),
    "outputs_exist": (".utils", "outputs_exist"),
    "ProgressEmitter": (".utils", "ProgressEmitter"),
    "compute_hrf": (".utils", "compute_hrf"),
    "compute_boxcar_task_regressor": (".utils", "compute_boxcar_task_regressor"),
    # --- script_generator.py ---
    "generate_mask_extraction_script": (".script_generator", "generate_mask_extraction_script"),
    "generate_task_mask_extraction_script": (".script_generator", "generate_task_mask_extraction_script"),
    "save_script": (".script_generator", "save_script"),
    # --- config module ---
    "config": (".config", ""),  # Module import, not attribute
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        if attr_name is None:
            # Module-level import (e.g., config)
            val = module
        else:
            val = getattr(module, attr_name)
        globals()[name] = val  # Cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Core functions
    "run_feedback_session",
    "ConnectivityEstimator",
    "get_next_run_number",
    "get_slot_run_info",
    "STEP_TO_RUN_NUMBER",
    "preprocessing",
    "collect_volumes",
    "collect_dicoms",
    "mask_extraction_rest",
    "mask_extraction_task",
    "import_manual_masks",
    "coregister_reference",
    "ATLAS_SOURCES",
    "get_atlas_labels",
    # Task classes
    "BaseTask",
    "BaseLocalizerTask",
    "BallTask",
    "ThermometerTask",
    "ActivationSmoother",
    "ActivationNormalizer",
    # Task registry & factory
    "NetworkRole",
    "create_task",
    "create_localizer_task",
    # Task regressor computation
    "compute_hrf",
    "compute_boxcar_task_regressor",
    # Session state
    "SessionState",
    "RunInfo",
    # Configuration dataclasses
    "SessionConfig",
    "PathConfig",
    "ScanConfig",
    "MaskConfig",
    "FeedbackConfig",
    "TaskConfig",
    "VisitConfig",
    "ShamConfig",
    "SimulatorSourceMapping",
    # Sham functionality
    "ShamActivationProvider",
    "ReplayActivationProvider",
    "load_randomization",
    "generate_randomization",
    "get_condition_for_participant",
    "resolve_sham_masks",
    "find_feedbacklog_for_replay",
    # Orchestration functions
    "setup_session",
    "run_single_task",
    "run_full_session",
    "run_next_task",
    "reset_session_state",
    # Multi-visit support functions
    "resolve_mask_source",
    "detect_existing_visits",
    "next_visit_id",
    "is_single_visit_data",
    "migrate_to_multi_visit",
    "setup_visit_with_reused_masks",
    # Timing functions
    "enable_timing",
    "disable_timing",
    "get_timing_data",
    "save_timing_report",
    "Timer",
    # Platform checks
    "check_platform_requirements",
    # Utilities
    "outputs_exist",
    "ProgressEmitter",
    # Script generation
    "generate_mask_extraction_script",
    "generate_task_mask_extraction_script",
    "save_script",
    # Configuration
    "config",
]
