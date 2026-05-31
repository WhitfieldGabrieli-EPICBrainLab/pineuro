"""pineuro - Python Real-Time fMRI Neurofeedback

A complete pipeline for real-time fMRI neurofeedback research, from volume acquisition
through preprocessing, network mask extraction, and moment-to-moment BOLD activation estimation.

Main Functions
--------------
run_feedback_task : Run a complete real-time feedback task
preprocessing : DICOM conversion, motion correction, brain extraction
collect_volumes : Watch for and collect incoming scanner volumes (DICOM or NIfTI)
mask_extraction_rest : Extract network masks using ICA and Yeo atlas
mask_extraction_mni : Extract network masks directly from MNI atlas (no localizer needed)
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
LocalizerConfig / FeedbackRunConfig : Per-localizer and per-feedback-run configuration
run_full_session : Run complete multi-run session
setup_session : Initialize session (collection, preprocessing, mask extraction)
run_single_task : Run one feedback/transfer task via task factory
TaskState : Persistent state across runs

Examples
--------
>>> from pineuro import preprocessing, mask_extraction_rest, run_feedback_task
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
>>> # Step 3: Run feedback task
>>> results = run_feedback_task(
...     mask_paths=['/data/masks/03_masks/network_7.nii.gz'],
...     subject='sub-01', task='nback', run=1,
...     output_dir='/data/feedback',
...     incoming_dir='/scanner/export/dicoms',
...     n_baseline=25, n_volumes=200
... )
"""

import importlib as _importlib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the version declared in pyproject.toml and
    # recorded in the installed package metadata.
    __version__ = _pkg_version("pineuro")
except _PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"

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
    "run_feedback_task": (".feedback", "run_feedback_task"),
    "ConnectivityEstimator": (".feedback", "ConnectivityEstimator"),
    "get_next_run_number": (".feedback", "get_next_run_number"),
    "get_slot_run_info": (".feedback", "get_slot_run_info"),
    # --- preprocessing.py (heavy: nibabel, scipy) ---
    "preprocessing": (".preprocessing", "preprocessing"),
    "collect_volumes": (".preprocessing", "collect_volumes"),
    # --- mask_extraction.py (heavy: nibabel, scipy, pandas) ---
    "mask_extraction_rest": (".mask_extraction", "mask_extraction_rest"),
    "mask_extraction_mni": (".mask_extraction", "mask_extraction_mni"),
    "mask_extraction_task": (".mask_extraction", "mask_extraction_task"),
    "run_mask_extraction_from_config": (".mask_extraction", "run_mask_extraction_from_config"),
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
    # --- tasks.state ---
    "TaskState": (".tasks.state", "TaskState"),
    "RunInfo": (".tasks.state", "RunInfo"),
    # --- session.py (lightweight: stdlib only) ---
    "SessionConfig": (".session", "SessionConfig"),
    "PathConfig": (".session", "PathConfig"),
    "LocalizerConfig": (".session", "LocalizerConfig"),
    "LocalizerExtractionConfig": (".session", "LocalizerExtractionConfig"),
    "FeedbackRunConfig": (".session", "FeedbackRunConfig"),
    "QCConfig": (".session", "QCConfig"),
    "GLMConfig": (".session", "GLMConfig"),
    "ConnectivityConfig": (".session", "ConnectivityConfig"),
    "IntermittentConfig": (".session", "IntermittentConfig"),
    "SimulatorConfig": (".session", "SimulatorConfig"),
    "ScannerConfig": (".session", "ScannerConfig"),
    "ReferenceConfig": (".session", "ReferenceConfig"),
    "SessionMetadata": (".session", "SessionMetadata"),
    "ShamConfig": (".session", "ShamConfig"),
    "ShamRunConfig": (".session", "ShamRunConfig"),
    "SimulatorSourceMapping": (".session", "SimulatorSourceMapping"),
    "setup_session": (".session", "setup_session"),
    "run_single_task": (".session", "run_single_task"),
    "run_full_session": (".session", "run_full_session"),
    "run_next_task": (".session", "run_next_task"),
    "reset_task_state": (".session", "reset_task_state"),
    "resolve_localizer_masks": (".session", "resolve_localizer_masks"),
    "detect_existing_sessions": (".session", "detect_existing_sessions"),
    "next_session_id": (".session", "next_session_id"),
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
    "save_script": (".script_generator", "save_script"),
    # --- scanner_setup.py (lightweight: stdlib only) ---
    "test_scanner_connection": (".scanner_setup", "test_scanner_connection"),
    "ConnectionTestResult": (".scanner_setup", "ConnectionTestResult"),
    "ping_scanner": (".scanner_setup", "ping_scanner"),
    "check_watch_directory": (".scanner_setup", "check_watch_directory"),
    "get_setup_instructions": (".scanner_setup", "get_setup_instructions"),
    # --- config module ---
    "config": (".config", ""),  # Module import, not attribute
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        if not attr_name:
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
    "run_feedback_task",
    "ConnectivityEstimator",
    "get_next_run_number",
    "get_slot_run_info",
    "preprocessing",
    "collect_volumes",
    "mask_extraction_rest",
    "mask_extraction_mni",
    "mask_extraction_task",
    "run_mask_extraction_from_config",
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
    # Task state
    "TaskState",
    "RunInfo",
    # Configuration dataclasses
    "SessionConfig",
    "PathConfig",
    "LocalizerConfig",
    "LocalizerExtractionConfig",
    "FeedbackRunConfig",
    "QCConfig",
    "GLMConfig",
    "ConnectivityConfig",
    "IntermittentConfig",
    "SimulatorConfig",
    "ScannerConfig",
    "ReferenceConfig",
    "SessionMetadata",
    "ShamConfig",
    "ShamRunConfig",
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
    "reset_task_state",
    # Multi-session support functions
    "resolve_localizer_masks",
    "detect_existing_sessions",
    "next_session_id",
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
    "save_script",
    # Scanner setup
    "test_scanner_connection",
    "ConnectionTestResult",
    "ping_scanner",
    "check_watch_directory",
    "get_setup_instructions",
    # Configuration
    "config",
]
