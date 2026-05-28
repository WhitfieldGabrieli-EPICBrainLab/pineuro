"""
Session configuration and orchestration for multi-run neurofeedback workflows.

Summary
-------
This module provides type-safe configuration dataclasses and high-level orchestration
functions for running neurofeedback sessions. It separates configuration from execution,
making it suitable for both scripted usage and future GUI integration.

Session Layout
--------------
Every participant's data is organized under ``base_dir/sub-XX/ses-XX/``. A new
participant defaults to ``ses-01``; later sessions become ``ses-02``, ``ses-03``,
etc. Masks for each session can be:
- Computed fresh (``mask_source='compute'``)
- Reused from a previous session (``mask_source='reuse'``)

Components
----------
Configuration Dataclasses:
- PathConfig: Paths for session directories
- ScanConfig: Scanner/acquisition parameters
- MaskConfig: Mask extraction parameters
- FeedbackConfig: Feedback task parameters
- TaskConfig: Generic task parameters (task_type, task_params, network_assignment)
- SessionMetadata: Session-level configuration (visit number, mask source)
- ShamConfig: Sham/control condition configuration
- SessionConfig: Top-level config combining all above

Orchestration Functions:
- setup_session(): Run all setup steps (collection, preprocessing, mask extraction)
- setup_session_with_reused_masks(): Set up session reusing masks from another session
- run_single_task(): Run one transfer/feedback task
- run_full_session(): Run complete workflow (Transfer -> Feedback x N -> Transfer)
- run_next_task(): Run next task in sequence (for incremental progression)
- reset_task_state(): Reset task state for a new session

Session Helpers:
- resolve_mask_source(): Get mask paths based on session configuration
- detect_existing_sessions(): List sessions for a participant
- next_session_id(): Get next session ID (ses-01, ses-02, etc.)

Workflow
--------
First session, fresh masks:
1. Create SessionConfig with PathConfig(..., session_id='ses-01') and
   SessionMetadata(mask_source='compute')
2. Call setup_session() for the full pipeline
3. Run feedback task

Later session, reusing masks from an earlier session:
1. Create SessionConfig with PathConfig(..., session_id='ses-02') and
   SessionMetadata(mask_source='reuse', mask_source_session='ses-01')
2. Call setup_session() - only runs reference collection + coregistration
3. Run feedback task

Dependencies
------------
dataclasses, pathlib, json, shutil, datetime, yaml (optional), logging

Notes
-----
This module works alongside tasks/state.py which contains TaskState for
persisting task-level state (hits, scale factors) between runs.
Task state resets each session (sessions are independent).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from .tasks.state import TaskState

logger = logging.getLogger(__name__)


# =============================================================================
# Simulator Source Mapping
# =============================================================================


@dataclass
class SimulatorSourceMapping:
    """Loads and queries per-subject simulator DICOM source paths.

    This class enables different subjects to use different DICOM source directories
    for simulation, loaded from a CSV or TSV mapping file.

    Parameters
    ----------
    _data : dict[str, dict[str, Path]]
        Internal mapping: {subject_id: {run_type: dicom_path}}

    Column Name Convention
    ----------------------
    The mapping file should have these column names:
    - subject: Subject/participant identifier (required, first column)
    - rest1, rest2, ...: Localizer run DICOM sources
    - reference_initial: Initial reference scan DICOM source
    - reference_realtime: Realtime reference scan DICOM source
    - feedback_run-01, feedback_run-02, ...: Feedback run DICOM sources
    - transfer_run-01, transfer_run-02, ...: Transfer run DICOM sources

    Examples
    --------
    >>> mapping = SimulatorSourceMapping.from_file(Path("sources.tsv"))
    >>> source = mapping.get_source("sub-001", "feedback_run-01")
    >>> localizer_sources = mapping.get_localizer_sources("sub-001", n_runs=2)
    """

    _data: dict[str, dict[str, Path]]

    @classmethod
    def from_file(cls, path: Path) -> SimulatorSourceMapping:
        """Load mapping from CSV or TSV file.

        Parameters
        ----------
        path : Path
            Path to mapping file. Format is auto-detected:
            - .tsv files use tab delimiter
            - .csv files use comma delimiter
            - Other extensions default to tab delimiter

        Returns
        -------
        SimulatorSourceMapping
            Loaded mapping instance

        Raises
        ------
        FileNotFoundError
            If the mapping file does not exist
        ValueError
            If the file is missing the 'subject' column
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Simulator source mapping file not found: {path}")

        # Detect delimiter from extension
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"

        data: dict[str, dict[str, Path]] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            # Validate header
            if reader.fieldnames is None or "subject" not in reader.fieldnames:
                raise ValueError(f"Mapping file must have a 'subject' column: {path}")

            for row in reader:
                subject = row.pop("subject").strip()
                if not subject:
                    continue

                # Convert remaining columns to Path objects (skip empty values)
                subject_paths = {}
                for run_type, dicom_path in row.items():
                    if dicom_path and dicom_path.strip():
                        subject_paths[run_type.strip()] = Path(dicom_path.strip())

                data[subject] = subject_paths

        logger.info(f"Loaded simulator source mapping for {len(data)} subjects from {path}")
        return cls(_data=data)

    def get_source(self, subject: str, run_type: str) -> Path | None:
        """Get DICOM source path for a subject and run type.

        Parameters
        ----------
        subject : str
            Subject/participant identifier
        run_type : str
            Run type column name (e.g., 'rest1', 'feedback_run-01', 'reference_realtime')

        Returns
        -------
        Path or None
            DICOM source path, or None if not found
        """
        subject_data = self._data.get(subject, {})
        return subject_data.get(run_type)

    def get_localizer_sources(self, subject: str, n_runs: int) -> list[Path | None]:
        """Get localizer/resting-state DICOM sources for a subject.

        Parameters
        ----------
        subject : str
            Subject/participant identifier
        n_runs : int
            Number of localizer runs to retrieve

        Returns
        -------
        list[Path or None]
            List of DICOM source paths for rest1, rest2, ..., restN
        """
        return [self.get_source(subject, f"rest{i + 1}") for i in range(n_runs)]

    def get_feedback_source(self, subject: str, run: int) -> Path | None:
        """Get feedback run DICOM source for a subject.

        Parameters
        ----------
        subject : str
            Subject/participant identifier
        run : int
            Run number (1-indexed)

        Returns
        -------
        Path or None
            DICOM source path for feedback_run-{run:02d}
        """
        return self.get_source(subject, f"feedback_run-{run:02d}")

    def get_transfer_source(self, subject: str, run: int) -> Path | None:
        """Get transfer run DICOM source for a subject.

        Parameters
        ----------
        subject : str
            Subject/participant identifier
        run : int
            Run number (1-indexed)

        Returns
        -------
        Path or None
            DICOM source path for transfer_run-{run:02d}
        """
        return self.get_source(subject, f"transfer_run-{run:02d}")

    def has_subject(self, subject: str) -> bool:
        """Check if a subject exists in the mapping."""
        return subject in self._data

    def subjects(self) -> list[str]:
        """Return list of all subjects in the mapping."""
        return list(self._data.keys())


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class SessionMetadata:
    """Session-level configuration for multi-session studies.

    Parameters
    ----------
    session_id : str
        Session identifier (e.g., 'ses-01', 'ses-02')
    session_number : int
        1-indexed session number
    mask_source : str
        How masks are obtained: 'compute' (run localizer + extraction)
        or 'reuse' (reuse from another session)
    mask_source_session : str, optional
        If mask_source='reuse', the session ID to copy masks from (e.g., 'ses-01')
    localizer_enabled : bool
        Whether to run localizer (resting-state) collection
    notes : str
        Optional notes about this session
    created : str
        ISO timestamp of session creation
    """

    session_id: str
    session_number: int
    mask_source: str = "compute"  # "compute" or "reuse"
    mask_source_session: str | None = None
    localizer_enabled: bool = True
    notes: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.mask_source not in ("compute", "reuse"):
            raise ValueError(f"mask_source must be 'compute' or 'reuse', got '{self.mask_source}'")
        if self.mask_source == "reuse" and not self.mask_source_session:
            raise ValueError("mask_source_session is required when mask_source='reuse'")
        if self.mask_source == "reuse":
            # Disable localizer when reusing masks
            self.localizer_enabled = False


@dataclass
class ShamConfig:
    """Configuration for sham/control neurofeedback conditions.

    Supports two delivery methods for sham feedback:
    1. 'alternative_masks': Use different network masks (e.g., motor instead of DMN)
    2. 'replay': Replay activation values from another participant's feedbacklog.tsv

    For 'alternative_masks', two modes are available:
    1. 'hemisphere_split': Split a single network into left/right hemispheres (default)
    2. 'two_networks': Use two distinct Yeo networks

    Critical blinding requirements:
    - All masks (target + alternative) are extracted for ALL participants
    - Real activations are always computed (even for sham participants)
    - Both real and delivered activations are logged in feedbacklog.tsv
    - Task and GUI display show delivered values only (maintains blinding)

    Parameters
    ----------
    enabled : bool
        Whether sham functionality is enabled for this study.
    delivery_method : str
        How sham feedback is delivered:
        - 'alternative_masks': Use different network masks
        - 'replay': Replay activation values from another participant's feedbacklog
    sham_mask_mode : str
        For 'alternative_masks' method, how masks are selected:
        - 'hemisphere_split': Split a single network into left/right hemispheres
        - 'two_networks': Use two distinct Yeo networks
    hemisphere_network : str
        For 'hemisphere_split' mode: the network to split (e.g., "SomMotA").
        Can be any network from the current Yeo parcellation (7 or 17).
    target_hemisphere : str
        For 'hemisphere_split' mode: which hemisphere is the target ("right" or "left").
        The other hemisphere becomes the non-target/sham mask.
    alternative_mask_targets : list[str]
        For 'two_networks' mode: network names to use for sham participants
        (e.g., ["SomMotA", "SomMotB"] instead of ["DefaultA", "ContA"])
    randomization_file : Path, optional
        Path to TSV file containing subject -> condition assignments.
        Columns: subject_id, condition, matched_participant, matched_session, notes
        For 'replay' method: matched_participant specifies which participant's
        feedbacklog to replay from (looked up in the study's base_dir).
    """

    enabled: bool = False
    blind_mode: bool = False  # Hide condition assignment from operator (double-blind)
    delivery_method: str = "alternative_masks"  # "alternative_masks" or "replay"
    sham_mask_mode: str = "hemisphere_split"  # "hemisphere_split" or "two_networks"
    hemisphere_network: str = "SomMotA"  # Network to split for hemisphere_split mode
    target_hemisphere: str = "right"  # "right" or "left"
    alternative_mask_targets: list[str] = field(default_factory=lambda: ["SomMotA", "SomMotB"])
    randomization_file: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration and convert paths."""
        if self.enabled:
            if self.delivery_method not in ("alternative_masks", "replay"):
                raise ValueError(
                    f"delivery_method must be 'alternative_masks' or 'replay', got '{self.delivery_method}'"
                )
            if self.delivery_method == "replay" and self.randomization_file is None:
                raise ValueError(
                    "randomization_file is required when delivery_method='replay' "
                    "(must contain matched_participant column)"
                )
            if self.delivery_method == "alternative_masks":
                if self.sham_mask_mode not in ("hemisphere_split", "two_networks"):
                    raise ValueError(
                        f"sham_mask_mode must be 'hemisphere_split' or 'two_networks', got '{self.sham_mask_mode}'"
                    )
                if self.sham_mask_mode == "hemisphere_split" and self.target_hemisphere not in ("right", "left"):
                    raise ValueError(f"target_hemisphere must be 'right' or 'left', got '{self.target_hemisphere}'")
        if self.randomization_file is not None:
            self.randomization_file = Path(self.randomization_file)


@dataclass
class PathConfig:
    """Path configuration for session directories.

    Parameters
    ----------
    base_dir : Path
        Base directory for all session data (study-level, not participant-specific)
    participant_id : str
        Participant identifier (e.g., 'sub-01')
    session_id : str
        Session identifier in BIDS format (e.g., 'ses-01'). Defaults to 'ses-01';
        every participant always has at least one session.
    feedback_dir : Path, optional
        Directory for task state and feedback outputs. Defaults to participant_session_dir/feedback
    preprocessing_dir : Path, optional
        Directory for preprocessing outputs. Defaults to participant_session_dir/preprocessing
    mask_extraction_dir : Path, optional
        Directory for mask extraction outputs. Defaults to participant_session_dir/mask_extraction
    feedback_output_dir : Path, optional
        Directory for feedback run outputs. Defaults to feedback_dir/02_feedback_output
    collection_dir : Path, optional
        Directory for DICOM collection. Defaults to participant_session_dir/collection
    mask_source_dir : Path, optional
        If reusing masks from another session, the source mask_extraction directory
    """

    base_dir: Path
    participant_id: str
    session_id: str = "ses-01"
    example_participant_id: str = ""  # GUI placeholder text (e.g. "sub-001")
    feedback_dir: Path | None = None
    preprocessing_dir: Path | None = None
    mask_extraction_dir: Path | None = None
    feedback_output_dir: Path | None = None
    collection_dir: Path | None = None
    mask_source_dir: Path | None = None  # For mask reuse across sessions

    def __post_init__(self) -> None:
        """Convert all paths to Path objects and set defaults for optional directories."""
        self.base_dir = Path(self.base_dir)

        if not re.match(r"^ses-\d+$", self.session_id):
            raise ValueError(f"session_id must be in BIDS format 'ses-NN' (e.g., 'ses-01'), got '{self.session_id}'")

        self._participant_session_dir = self.base_dir / self.participant_id / self.session_id

        # Set default paths relative to participant_session_dir if not provided
        if self.feedback_dir is None:
            self.feedback_dir = self._participant_session_dir / "feedback"
        else:
            self.feedback_dir = Path(self.feedback_dir)

        if self.preprocessing_dir is None:
            self.preprocessing_dir = self._participant_session_dir / "preprocessing"
        else:
            self.preprocessing_dir = Path(self.preprocessing_dir)

        if self.mask_extraction_dir is None:
            self.mask_extraction_dir = self._participant_session_dir / "mask_extraction"
        else:
            self.mask_extraction_dir = Path(self.mask_extraction_dir)

        if self.feedback_output_dir is None:
            self.feedback_output_dir = self.feedback_dir / "02_feedback_output"
        else:
            self.feedback_output_dir = Path(self.feedback_output_dir)

        if self.collection_dir is None:
            self.collection_dir = self._participant_session_dir / "collection"
        else:
            self.collection_dir = Path(self.collection_dir)

        if self.mask_source_dir is not None:
            self.mask_source_dir = Path(self.mask_source_dir)

    @property
    def participant_session_dir(self) -> Path:
        """Return the participant + session directory."""
        return cast(Path, self._participant_session_dir)


# =============================================================================
# Top-level configuration blocks
# =============================================================================


@dataclass
class SimulatorConfig:
    """Simulator-mode settings (the YAML ``simulator:`` block).

    Parameters
    ----------
    enabled : bool
        Whether simulator mode is active.
    collection_speed : float
        Speed multiplier for DICOM collection simulation (localizer/setup runs).
    feedback_speed : float
        Speed multiplier for feedback DICOM simulation.
    trigger_mode : str
        ``"manual"`` (user presses trigger key) or ``"os_key"`` (synthetic keypress).
    source_mapping : str
        Optional path to CSV/TSV mapping subjects to per-run DICOM source paths.
        When set, takes priority over individual DICOM source fields.
    """

    enabled: bool = True
    collection_speed: float = 1.0
    feedback_speed: float = 10.0
    trigger_mode: str = "manual"
    source_mapping: str = ""


@dataclass
class ScannerConfig:
    """Scanner connectivity settings (the YAML ``scanner:`` block)."""

    dicom_dir: str = "/BOLD_DICOM"
    ip: str = ""
    trigger_keys: list[str] = field(default_factory=lambda: ["t", "+", "5", "equal"])


@dataclass
class ReferenceConfig:
    """Reference-scan acquisition settings (the YAML ``reference:`` block).

    The reference scan(s) provide a stable baseline volume used for
    real-time registration. ``initial`` runs before the localizer; ``realtime``
    runs immediately before feedback (in scanner head pose).
    """

    initial_enabled: bool = True
    initial_n_volumes: int = 2
    initial_dicoms: str = ""
    realtime_enabled: bool = True
    realtime_n_volumes: int = 2
    realtime_dicoms: str = ""


@dataclass
class LocalizerExtractionConfig:
    """Mask-extraction parameters for a single localizer.

    Grouped under ``localizers[*].extraction``.
    """

    network_atlas: str = "yeo_17"
    n_components: int = 30
    threshold: int = 2000
    threshold_method: str = "top_n"
    smooth_fwhm: float = 5.0
    highpass: float = 0.01
    lowpass: float | None = None
    atlas_mask: str | None = None
    atlas_source: str | None = None
    atlas_regions: list[str] | None = None
    cluster_start_threshold: float = 2.0
    cluster_step: float = 0.1
    cluster_return_all: bool = False
    min_cluster_size: int = 50
    max_total_voxels: int | None = None

    def __post_init__(self) -> None:
        if self.threshold_method not in ("top_n", "cluster", "cluster_extent"):
            raise ValueError(
                f"threshold_method must be 'top_n', 'cluster', or 'cluster_extent', got '{self.threshold_method}'"
            )


@dataclass
class QCConfig:
    """Per-run quality-control thresholds (``feedback_runs[*].qc``)."""

    fd_threshold: float = 0.5
    dvars_threshold: float = 1.5
    tsnr_threshold: float = 20.0


@dataclass
class GLMConfig:
    """Per-run GLM and motion-correction settings (``feedback_runs[*].glm``)."""

    VALID_MOTION_CORRECTION = {"siemens_moco", "software", "none"}

    motion_correction: str = "siemens_moco"
    include_drift: bool = True
    include_motion_regressors: bool = True
    include_wm_regressor: bool = False
    include_csf_regressor: bool = False
    smooth_fwhm: float | None = None

    def __post_init__(self) -> None:
        if self.motion_correction not in self.VALID_MOTION_CORRECTION:
            raise ValueError(
                f"motion_correction must be one of {self.VALID_MOTION_CORRECTION}, got '{self.motion_correction}'"
            )


@dataclass
class ConnectivityConfig:
    """Per-run connectivity-mode settings (``feedback_runs[*].connectivity``).

    Only used when ``FeedbackRunConfig.feedback_mode == "connectivity"``.
    """

    pairs: list[list[str]] | None = None
    window: int = 15
    target: str = "increase"


@dataclass
class IntermittentConfig:
    """Per-run intermittent-feedback settings (``feedback_runs[*].intermittent``).

    Omit (or ``interval=None``) for continuous feedback.
    """

    interval: int | None = None


@dataclass
class FeedbackRunConfig:
    """Configuration for a single feedback task type in a session.

    A session can have multiple feedback task types (e.g., triggered experience
    sampling + ball task), each with independent scan parameters, GLM settings,
    and feedback mode. Each ``FeedbackRunConfig`` is repeated for ``max_runs``
    iterations during the session.

    Parameters
    ----------
    task : str
        Registered task identifier (e.g., ``"ball"``, ``"thermometer"``).
    mask_source : str
        Identifier of the :class:`LocalizerConfig` (its ``id`` field) that
        supplies masks for this run.
    max_runs : int
        Number of feedback runs for this task type.
    tr : float
        Repetition time in seconds.
    n_volumes : int
        Total volumes per run.
    n_baseline : int
        Number of baseline volumes before task onset.
    feedback_mode : str
        ``"activation"`` or ``"connectivity"``.
    roi_combine_method : str
        How per-voxel activations are combined per ROI: ``"inverse_sd"``
        (Hinds 2011 efficiency weighting, w = 1/sigma; the default) or ``"mean"``.
    qc : QCConfig
        Per-run QC thresholds (fd, dvars, tsnr).
    glm : GLMConfig
        Per-run GLM and motion-correction settings.
    connectivity : ConnectivityConfig
        Per-run connectivity settings (only used when feedback_mode == "connectivity").
    intermittent : IntermittentConfig
        Per-run intermittent-feedback settings (omit / interval=None for continuous).
    use_transfer_runs : bool
        Whether to include transfer runs for this task.
    task_params : dict
        Free-form task-specific parameters passed to ``create_task()``.
    network_assignment : dict
        Maps network role keys (declared by ``TaskMetadata.network_roles``) to mask names.
    sham : dict
        Optional per-run sham override. Empty dict means use the top-level
        ``SessionConfig.sham`` block.
    source_dicoms : str
        Simulator DICOM source for feedback runs.
    transfer_source_dicoms : str
        Simulator DICOM source for transfer runs.
    """

    VALID_ROI_COMBINE_METHODS = {"mean", "inverse_sd"}

    task: str = "ball"
    mask_source: str = ""
    max_runs: int = 1
    tr: float = 1.2
    n_volumes: int = 150
    n_baseline: int = 25

    # Feedback signal
    feedback_mode: str = "activation"
    roi_combine_method: str = "inverse_sd"

    # Grouped sub-blocks
    qc: QCConfig = field(default_factory=QCConfig)
    glm: GLMConfig = field(default_factory=GLMConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    intermittent: IntermittentConfig = field(default_factory=IntermittentConfig)

    # Transfer runs
    use_transfer_runs: bool = True

    # Task-specific
    task_params: dict[str, Any] = field(default_factory=dict)
    network_assignment: dict[str, str] = field(default_factory=dict)

    # Sham (per-run override; if empty, the top-level SessionConfig.sham applies)
    sham: dict[str, Any] = field(default_factory=dict)

    # Simulator sources
    source_dicoms: str = ""
    transfer_source_dicoms: str = ""

    def __post_init__(self) -> None:
        if self.roi_combine_method not in self.VALID_ROI_COMBINE_METHODS:
            raise ValueError(
                f"roi_combine_method must be one of {self.VALID_ROI_COMBINE_METHODS}, got '{self.roi_combine_method}'"
            )


@dataclass
class LocalizerConfig:
    """Configuration for a single localizer run.

    Each localizer defines its type (rest or a registered task ID), scan
    parameters, and extraction settings. A session can have multiple
    localizers (e.g., one rest + one task), each producing its own masks.

    Parameters
    ----------
    type : str
        ``"rest"`` for ICA-based extraction, or a registered task ID
        (e.g. ``"rit"``) for GLM-based extraction.
    tr : float
        Repetition time in seconds for this localizer.
    n_volumes : int
        Number of volumes per run.
    n_runs : int
        Number of runs for this localizer.
    source_dicoms : list[Path]
        DICOM source folders (for simulator).
    networks : list[str]
        Target network names to extract. Rest only.
    network_atlas : str
        Network atlas for ICA component identification. Rest only.
    smooth_fwhm : float
        Spatial smoothing FWHM in mm.
    highpass : float
        High-pass filter cutoff in Hz.
    lowpass : float or None
        Low-pass filter cutoff in Hz. Rest only.
    contrasts : dict[str, str]
        Contrast expressions for task extraction. Task only.
    include_reverse_contrasts : bool
        Generate reverse (negated) masks for each contrast. Task only.
    events_file : Path or None
        BIDS events file for task extraction. Task only.
    stat_type : str
        Statistical map type (``"t"`` or ``"z"``). Task only.
    hrf_model : str
        HRF model for task extraction. Task only.
    drift_model : str
        Drift model for task extraction. Task only.
    threshold : int
        Number of voxels per mask.
    threshold_method : str
        ``"top_n"``, ``"cluster"``, or ``"cluster_extent"``.
    atlas_mask : str or None
        Binary atlas NIfTI path for cluster thresholding.
    atlas_source : str or None
        Atlas source key for cluster thresholding.
    atlas_regions : list[str] or None
        Region labels for cluster mask.
    cluster_start_threshold : float
        Starting stat threshold for cluster search.
    cluster_step : float
        Threshold increment per iteration.
    """

    type: str = "rest"
    id: str = ""  # stable identifier referenced by feedback_runs[*].mask_source

    # Scan parameters
    tr: float = 1.2
    n_volumes: int = 250
    n_runs: int = 2
    motion_correction: str = "none"
    source_dicoms: list[Path] = field(default_factory=list)

    # Rest-specific (ICA): list of Yeo network short names (e.g. ["DefaultA", "ContA"])
    networks: list[str] = field(default_factory=lambda: ["DefaultA", "ContA"])

    # Mask-extraction parameters (grouped)
    extraction: LocalizerExtractionConfig = field(default_factory=LocalizerExtractionConfig)

    # Task-specific (GLM)
    contrasts: dict[str, str] = field(default_factory=dict)
    include_reverse_contrasts: bool = False
    events_file: Path | None = None
    stat_type: str = "t"
    hrf_model: str = "spm"
    drift_model: str = "cosine"

    # Task-localizer-only: PsychoPy/task-side parameters for task-type localizers
    task_params: dict[str, Any] = field(default_factory=dict)

    # Manual-import fields (used when type == "manual")
    manual_mask_paths: list[Path] = field(default_factory=list)
    manual_mask_names: list[str] = field(default_factory=list)
    manual_ref_file: Path | None = None
    manual_anat_file: Path | None = None
    manual_mask_space: str = "native"

    @property
    def is_rest(self) -> bool:
        """Whether this is a resting-state localizer."""
        return self.type == "rest"

    @property
    def is_manual(self) -> bool:
        """Whether this localizer imports pre-built masks rather than extracting them."""
        return self.type == "manual"

    @property
    def task_name(self) -> str:
        """Name used for file naming and directory structure."""
        return "rest" if self.is_rest else self.type

    @property
    def mask_names(self) -> list[str]:
        """Mask names this localizer produces."""
        if self.is_rest:
            return list(self.networks)
        if self.is_manual:
            # Prefer explicit names; otherwise derive from filenames (strip
            # extensions like .nii.gz / .nii).
            if self.manual_mask_names:
                return list(self.manual_mask_names)
            names: list[str] = []
            for p in self.manual_mask_paths:
                stem = Path(p).name
                for suffix in (".nii.gz", ".nii"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                names.append(stem)
            return names
        # Task localizer: contrasts.keys() plus _reverse variants when enabled
        names = list(self.contrasts.keys())
        if self.include_reverse_contrasts:
            names += [f"{k}_reverse" for k in self.contrasts]
        return names

    def __post_init__(self) -> None:
        self.source_dicoms = [Path(p) for p in self.source_dicoms]
        if self.events_file is not None:
            self.events_file = Path(self.events_file)
        if not self.id:
            self.id = self.type
        if self.is_manual:
            if self.manual_mask_space not in ("native", "anat", "mni"):
                raise ValueError(
                    f"manual_mask_space must be 'native', 'anat', or 'mni', got '{self.manual_mask_space}'"
                )
            if self.manual_mask_paths:
                # Direct-path mode (Python API): mask files supplied up front —
                # validate them fully.
                if self.manual_mask_space == "anat" and self.manual_anat_file is None:
                    raise ValueError("manual_anat_file is required when manual_mask_space='anat'")
                if self.manual_mask_names and len(self.manual_mask_names) != len(self.manual_mask_paths):
                    raise ValueError(
                        f"manual_mask_names ({len(self.manual_mask_names)}) and manual_mask_paths "
                        f"({len(self.manual_mask_paths)}) must have the same length"
                    )
                self.manual_mask_paths = [Path(p) for p in self.manual_mask_paths]
            elif not self.manual_mask_names:
                # Names-only mode (GUI study config): the per-participant mask
                # files are chosen later in the workflow, but the network names
                # are still required to define the masks.
                raise ValueError(
                    "type='manual' requires manual_mask_names (network names) when no manual_mask_paths are given"
                )
            if self.manual_ref_file is not None:
                self.manual_ref_file = Path(self.manual_ref_file)
            if self.manual_anat_file is not None:
                self.manual_anat_file = Path(self.manual_anat_file)
        elif self.type != "rest" and not self.contrasts:
            raise ValueError(f"contrasts must be provided for task localizer type='{self.type}'")


def _path_to_str(obj: Any) -> Any:
    """Convert Path objects to strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_path_to_str(v) for v in obj]
    return obj


@dataclass
class SelectedAttempts:
    """Store user-selected run/attempt combinations for reproducibility.

    This allows manual selection of which attempt to use for each run when
    multiple attempts exist (e.g., after interruptions/retries).
    """

    localizer: dict[int, int] | None = None  # run_num -> attempt_num for localizer runs
    reference: dict[int, int] | None = None  # run_num -> attempt_num for reference scans
    feedback: dict[str, dict[str, Any]] | None = None  # step_id -> {run, attempt} for feedback runs

    def __post_init__(self) -> None:
        """Initialize empty dicts if None."""
        if self.localizer is None:
            self.localizer = {}
        if self.reference is None:
            self.reference = {}
        if self.feedback is None:
            self.feedback = {}


@dataclass
class SessionConfig:
    """Top-level configuration combining all session settings.

    Parameters
    ----------
    paths : PathConfig
        Path configuration
    scan : ScanConfig
        Scanner/acquisition parameters
    mask : MaskConfig
        Mask extraction parameters
    feedback : FeedbackConfig
        Feedback task parameters
    task : TaskConfig
        BallTask parameters
    session : SessionMetadata, optional
        Session-specific configuration (visit number, mask source).
    sham : ShamConfig, optional
        Sham/control condition configuration.
        If None or sham.enabled=False, all participants receive real feedback.

    Examples
    --------
    >>> # First session, fresh masks
    >>> config = SessionConfig(
    ...     paths=PathConfig(base_dir="/data/study", participant_id="sub-01", session_id="ses-01"),
    ...     session=SessionMetadata(session_id="ses-01", session_number=1, mask_source="compute"),
    ... )

    >>> # Multi-session with mask reuse
    >>> config = SessionConfig(
    ...     paths=PathConfig(base_dir="/data/study", participant_id="sub-01", session_id="ses-02"),
    ...     session=SessionMetadata(session_id="ses-02", session_number=2, mask_source="reuse", mask_source_session="ses-01"),
    ... )
    """

    paths: PathConfig
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    session: SessionMetadata | None = None
    sham: ShamConfig | None = None
    localizers: list[LocalizerConfig] = field(default_factory=list)
    feedback_runs: list[FeedbackRunConfig] = field(default_factory=list)

    # Runtime-only fields (not serialized to YAML). Set by coregister_before_first_transfer
    # / tissue-mask resolution and consumed by run_feedback_task. Previously lived on
    # FeedbackConfig.wm_mask_path / csf_mask_path; moved here so they don't appear in YAML.
    _wm_mask_path: Path | None = field(default=None, repr=False)
    _csf_mask_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: Path) -> SessionConfig:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml") from e

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: Path) -> SessionConfig:
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SessionConfig:
        """Create config from dictionary."""
        paths_data = data.get("paths", {})
        paths = PathConfig(**paths_data)

        simulator_data = data.get("simulator", {})
        simulator = SimulatorConfig(**simulator_data) if simulator_data else SimulatorConfig()

        scanner_data = data.get("scanner", {})
        scanner = ScannerConfig(**scanner_data) if scanner_data else ScannerConfig()

        reference_data = data.get("reference", {})
        reference = ReferenceConfig(**reference_data) if reference_data else ReferenceConfig()

        session_data = data.get("session")
        session = SessionMetadata(**session_data) if session_data else None

        sham_data = data.get("sham")
        sham = ShamConfig(**sham_data) if sham_data else None

        # Parse localizers list (each may carry a nested `extraction:` block)
        localizers_data = data.get("localizers", [])
        localizers = [cls._build_localizer(loc_data) for loc_data in localizers_data]

        # Parse feedback_runs list (each may carry nested qc/glm/connectivity/intermittent blocks)
        feedback_runs_data = data.get("feedback_runs", [])
        feedback_runs = [cls._build_feedback_run(frc) for frc in feedback_runs_data]

        # Validate cross-references: each feedback_runs[*].mask_source must match some localizer.id
        localizer_ids = {loc.id for loc in localizers}
        for i, frc in enumerate(feedback_runs):
            if frc.mask_source and frc.mask_source not in localizer_ids:
                raise ValueError(
                    f"feedback_runs[{i}].mask_source={frc.mask_source!r} does not match any "
                    f"localizer.id (available: {sorted(localizer_ids)})"
                )

        return cls(
            paths=paths,
            simulator=simulator,
            scanner=scanner,
            reference=reference,
            session=session,
            sham=sham,
            localizers=localizers,
            feedback_runs=feedback_runs,
        )

    @staticmethod
    def _build_localizer(loc_data: dict[str, Any]) -> LocalizerConfig:
        """Build a LocalizerConfig, unpacking the nested ``extraction:`` block."""
        loc_data = dict(loc_data)  # don't mutate caller's dict
        extraction_data = loc_data.pop("extraction", None)
        if extraction_data:
            loc_data["extraction"] = LocalizerExtractionConfig(**extraction_data)
        return LocalizerConfig(**loc_data)

    @staticmethod
    def _build_feedback_run(frc_data: dict[str, Any]) -> FeedbackRunConfig:
        """Build a FeedbackRunConfig, unpacking the nested qc/glm/connectivity/intermittent blocks."""
        frc_data = dict(frc_data)  # don't mutate caller's dict
        for key, klass in (
            ("qc", QCConfig),
            ("glm", GLMConfig),
            ("connectivity", ConnectivityConfig),
            ("intermittent", IntermittentConfig),
        ):
            block = frc_data.pop(key, None)
            if block:
                frc_data[key] = klass(**block)
        return FeedbackRunConfig(**frc_data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to save YAML configs: pip install pyyaml") from e

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    def _to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with Path objects as strings."""
        result: dict[str, Any] = {
            "paths": _path_to_str(asdict(self.paths)),
            "simulator": asdict(self.simulator),
            "scanner": asdict(self.scanner),
            "reference": _path_to_str(asdict(self.reference)),
        }
        if self.session is not None:
            result["session"] = asdict(self.session)
        if self.sham is not None:
            result["sham"] = _path_to_str(asdict(self.sham))
        if self.localizers:
            result["localizers"] = [_path_to_str(asdict(loc)) for loc in self.localizers]
        if self.feedback_runs:
            result["feedback_runs"] = [_path_to_str(asdict(frc)) for frc in self.feedback_runs]
        return result

    @property
    def combined_mask_names(self) -> list[str]:
        """De-duplicated union of every localizer's mask names.

        Returns the ordered union (first occurrence wins) so callers get a
        stable, deterministic list.
        """
        seen: set[str] = set()
        out: list[str] = []
        for loc in self.localizers:
            for name in loc.mask_names:
                if name not in seen:
                    seen.add(name)
                    out.append(name)
        return out

    def validate(self) -> list[str]:
        """Validate configuration, returning list of error messages.

        Returns
        -------
        list[str]
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.paths.participant_id:
            errors.append("paths.participant_id is required")

        if not self.feedback_runs:
            errors.append("at least one feedback_runs entry is required")
        for i, frc in enumerate(self.feedback_runs):
            if frc.tr <= 0:
                errors.append(f"feedback_runs[{i}].tr must be positive, got {frc.tr}")
            if frc.n_volumes <= 0:
                errors.append(f"feedback_runs[{i}].n_volumes must be positive, got {frc.n_volumes}")
            if frc.max_runs <= 0:
                errors.append(f"feedback_runs[{i}].max_runs must be positive, got {frc.max_runs}")

        from .atlas_registry import ATLAS_SOURCES

        for i, loc in enumerate(self.localizers):
            if loc.is_rest and loc.extraction.network_atlas not in ATLAS_SOURCES:
                errors.append(f"localizers[{i}].extraction.network_atlas is invalid: {loc.extraction.network_atlas!r}")

        return errors

    def resolve_simulator_source(self, run_type: str) -> Path | None:
        """Resolve simulator DICOM source path for a given run type.

        This method checks the simulator_source_mapping file first (if configured),
        falling back to the individual DICOM source fields if no mapping is found.

        Parameters
        ----------
        run_type : str
            Run type identifier. Supported values:
            - 'rest1', 'rest2', ...: Localizer runs
            - 'reference_initial': Initial reference scan
            - 'reference_realtime': Realtime reference scan
            - 'feedback_run-01', 'feedback_run-02', ...: Feedback runs
            - 'transfer_run-01', 'transfer_run-02', ...: Transfer runs

        Returns
        -------
        Path or None
            Resolved DICOM source path, or None if not configured

        Examples
        --------
        >>> config = SessionConfig(paths=PathConfig(participant_id="sub-001"), ...)
        >>> source = config.resolve_simulator_source("feedback_run-01")
        """
        # Try mapping file first
        if self.simulator.source_mapping:
            try:
                mapping = SimulatorSourceMapping.from_file(Path(self.simulator.source_mapping))
                source = mapping.get_source(self.paths.participant_id, run_type)
                if source is not None:
                    logger.debug(f"Resolved {run_type} for {self.paths.participant_id} from mapping: {source}")
                    return source
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load simulator source mapping: {e}")

        # Fall back to individual fields
        primary_localizer = self.localizers[0] if self.localizers else None
        if run_type.startswith("rest"):
            # Extract run number (1-indexed); resolve from the first localizer's source_dicoms
            try:
                run_num = int(run_type[4:])
                if primary_localizer and 0 < run_num <= len(primary_localizer.source_dicoms):
                    return primary_localizer.source_dicoms[run_num - 1]
            except (ValueError, IndexError):
                pass
        elif run_type == "reference_initial":
            return Path(self.reference.initial_dicoms) if self.reference.initial_dicoms else None
        elif run_type == "reference_realtime":
            return Path(self.reference.realtime_dicoms) if self.reference.realtime_dicoms else None
        elif run_type.startswith("feedback_run-") or run_type.startswith("transfer_run-"):
            # All feedback/transfer runs use the first feedback_run's source_dicoms
            if self.feedback_runs:
                frc = self.feedback_runs[0]
                src = frc.transfer_source_dicoms if run_type.startswith("transfer_run-") else frc.source_dicoms
                return Path(src) if src else None
            return None

        return None

    def resolve_localizer_sources(self) -> list[Path]:
        """Resolve simulator DICOM sources for all localizer runs.

        Returns the source DICOM paths for the first localizer's runs.
        """
        if not self.localizers:
            return []
        return [Path(p) for p in self.localizers[0].source_dicoms]


# =============================================================================
# Helper Functions
# =============================================================================


def ref_file_path(config: SessionConfig, task: str = "rest", source_session: str | None = None) -> Path:
    """Return the expected BOLD reference path based on config.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    task : str
        Task name (default: 'rest')
    source_session : str, optional
        If specified, return ref path from this session instead of current session

    Returns
    -------
    Path
        Path to BOLD reference file
    """
    session_id = source_session or config.paths.session_id
    if source_session:
        preprocessing_dir = config.paths.base_dir / config.paths.participant_id / source_session / "preprocessing"
    else:
        assert config.paths.preprocessing_dir is not None
        preprocessing_dir = config.paths.preprocessing_dir
    return preprocessing_dir / f"{config.paths.participant_id}_{session_id}_task-{task}_boldref.nii.gz"


def get_mask_run_number(mask_files: list[Path]) -> int | None:
    """Extract run number from mask file paths.

    Parameters
    ----------
    mask_files : list[Path]
        List of mask file paths

    Returns
    -------
    int or None
        Run number extracted from first mask file, or None if not found
    """
    if not mask_files:
        return None
    match = re.search(r"run-(\d+)", Path(mask_files[0]).name)
    return int(match.group(1)) if match else None


def next_collection_run_number(collection_dir: Path, task_name: str) -> int:
    """Return the next run number based on existing collection folders.

    Parameters
    ----------
    collection_dir : Path
        Directory containing collection folders
    task_name : str
        Task name prefix (e.g., 'rest', 'reference')

    Returns
    -------
    int
        Next available run number (1-indexed)
    """
    max_run = 0
    if collection_dir.exists():
        # Collection folders are named "{task}_run-NN_attempt-NN".
        pattern = re.compile(rf"{re.escape(task_name)}_run-(\d+)_attempt-\d+$")
        for entry in collection_dir.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    max_run = max(max_run, int(match.group(1)))
    return max_run + 1


def get_localizer_run_info(collection_dir: Path, task: str, run: int) -> dict[str, Any]:
    """Get run info for localizer/reference with retry detection using attempt entity.

    Scans for existing {task}_run-{run:02d}_attempt-* folders and determines
    next attempt number. Previous attempts are preserved in place.

    Parameters
    ----------
    collection_dir : Path
        Collection directory to scan
    task : str
        Task name (e.g., 'rest', 'reference', 'self_other')
    run : int
        Run number

    Returns
    -------
    dict
        Dictionary containing:
        - 'run': int - The run number (same as input)
        - 'attempt': int - The attempt number for this try (1, 2, 3, ...)
        - 'run_attempt_string': str - Combined string like 'run-01_attempt-01'
        - 'existing_attempts': list - List of existing attempt numbers for this run

    Examples
    --------
    >>> info = get_localizer_run_info(collection_dir, 'task-rest', 1)
    >>> info
    {'run': 1, 'attempt': 1, 'run_attempt_string': 'run-01_attempt-01', 'existing_attempts': []}
    >>> # After one collection:
    >>> info = get_localizer_run_info(collection_dir, 'task-rest', 1)
    >>> info
    {'run': 1, 'attempt': 2, 'run_attempt_string': 'run-01_attempt-02', 'existing_attempts': [1]}
    """
    if not collection_dir.exists():
        return {
            "run": run,
            "attempt": 1,
            "run_attempt_string": f"run-{run:02d}_attempt-01",
            "existing_attempts": [],
        }

    # Scan for existing collection folders for this run
    # Pattern: {task}_run-{run:02d}_attempt-*/
    pattern = f"{task}_run-{run:02d}_attempt-*"

    existing_attempts: list[int] = []

    # Check main collection directory
    for folder in collection_dir.glob(pattern):
        if folder.is_dir():
            match = re.search(r"attempt-(\d+)", folder.name)
            if match:
                existing_attempts.append(int(match.group(1)))

    # Remove duplicates and sort
    existing_attempts = sorted(set(existing_attempts))

    # Determine next attempt number
    if not existing_attempts:
        attempt = 1
    else:
        attempt = max(existing_attempts) + 1

    return {
        "run": run,
        "attempt": attempt,
        "run_attempt_string": f"run-{run:02d}_attempt-{attempt:02d}",
        "existing_attempts": existing_attempts,
    }


def discard_attempt(
    base_dir: Path,
    *,
    subject: str,
    session: str,
    task: str,
    attempt: int,
    run: int,
    kind: Literal["localizer", "feedback"],
) -> list[Path]:
    """Delete all artifacts for one ``run-NN_attempt-NN`` attempt.

    Used after the operator stops a run mid-acquisition and chooses to discard
    the partial data so the next attempt reuses the same ``attempt`` number
    (rather than having the failed try permanently occupy an ``attempt`` slot).

    For ``kind="localizer"``, deletes the entire
    ``base_dir/collection/task-{task}_run-NN_attempt-NN/`` directory plus any
    per-attempt files the localizer task wrote into ``base_dir/preprocessing/``
    matching ``{subject}_task-{task}_run-NN_attempt-NN_*`` (e.g. events.tsv).

    For ``kind="feedback"``, deletes every file under
    ``base_dir/feedback/02_feedback_output/`` whose name starts with
    ``{subject}_task-{task}_run-NN_attempt-NN`` plus the sibling
    ``nifti_task-{task}_run-NN_attempt-NN/`` directory.

    Parameters
    ----------
    base_dir : Path
        Participant base directory (the one containing ``collection/`` and
        ``feedback/02_feedback_output/`` subtrees).
    subject : str
        Subject identifier (e.g. ``"sub-01"``). Only used for the feedback kind.
    task : str
        Task name (e.g. ``"rest"``, ``"feedback"``, ``"transfer"``).
    attempt : int
        Attempt number (1, 2, 3, ...) of the attempt to discard.
    run : int
        Run number (1, 2, ...) of the attempt to discard.
    kind : {"localizer", "feedback"}
        Which output convention to clean up.

    Returns
    -------
    list[Path]
        The paths that were actually removed (empty if nothing matched).

    Raises
    ------
    OSError
        If one or more matched paths failed to delete. The error message
        aggregates every failure. Files that were successfully removed remain
        removed.
    """
    base_dir = Path(base_dir)
    run_attempt = f"run-{run:02d}_attempt-{attempt:02d}"

    if kind == "localizer":
        targets = [base_dir / "collection" / f"task-{task}_{run_attempt}"]
        # Localizer tasks also write an events TSV (and any per-attempt
        # outputs the task subclass adds) into preprocessing/.
        preprocessing_dir = base_dir / "preprocessing"
        if preprocessing_dir.exists():
            targets.extend(preprocessing_dir.glob(f"{subject}_{session}_task-{task}_{run_attempt}_*"))
    elif kind == "feedback":
        feedback_dir = base_dir / "feedback" / "02_feedback_output"
        if not feedback_dir.exists():
            return []
        prefix = f"{subject}_{session}_task-{task}_{run_attempt}"
        targets = list(feedback_dir.glob(f"{prefix}*"))
        nifti_dir = feedback_dir / f"nifti_{session}_task-{task}_{run_attempt}"
        if nifti_dir.exists() and nifti_dir not in targets:
            targets.append(nifti_dir)
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    removed: list[Path] = []
    failures: list[tuple[Path, Exception]] = []
    for target in targets:
        if not target.exists():
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed.append(target)
            logger.info(f"Discarded attempt artifact: {target}")
        except OSError as exc:
            failures.append((target, exc))
            logger.error(f"Failed to remove {target}: {exc}")

    if failures:
        msg = "; ".join(f"{path}: {exc}" for path, exc in failures)
        raise OSError(f"Failed to discard {len(failures)} path(s): {msg}")

    return removed


# =============================================================================
# Multi-Session Support Functions
# =============================================================================


def resolve_mask_source(config: SessionConfig) -> tuple[list[Path], Path]:
    """Resolve mask file paths based on session configuration.

    For sessions that compute their own masks, returns paths from the current session's
    mask_extraction directory. For sessions that reuse masks from another session,
    returns paths from the source session.

    Parameters
    ----------
    config : SessionConfig
        Session configuration

    Returns
    -------
    tuple[list[Path], Path]
        (mask_files, ref_file) where mask_files are thresholded network masks
        and ref_file is the BOLD reference from preprocessing
    """
    if config.session is None or config.session.mask_source == "compute":
        # Masks computed in this session
        if config.paths.mask_extraction_dir is None:
            raise ValueError("config.paths.mask_extraction_dir must be set when mask_source='compute'")
        mask_dir = config.paths.mask_extraction_dir / "03_masks"
        mask_files = sorted(mask_dir.glob("*_thr.nii.gz")) if mask_dir.exists() else []
        ref = ref_file_path(config)
        return (mask_files, ref)
    elif config.session.mask_source == "reuse":
        # Masks from another session
        if config.session.mask_source_session is None:
            raise ValueError("config.session.mask_source_session must be set when mask_source='reuse'")
        source_session = config.session.mask_source_session
        source_base = config.paths.base_dir / config.paths.participant_id / source_session / "mask_extraction"
        source_mask_dir = source_base / "03_masks"
        mask_files = sorted(source_mask_dir.glob("*_thr.nii.gz")) if source_mask_dir.exists() else []
        ref = ref_file_path(config, source_session=source_session)
        return (mask_files, ref)
    else:
        raise ValueError(f"Unknown mask_source: {config.session.mask_source}")


def detect_existing_sessions(base_dir: Path, participant_id: str) -> list[dict[str, Any]]:
    """Detect existing session directories for a participant.

    Parameters
    ----------
    base_dir : Path
        Study base directory
    participant_id : str
        Participant identifier

    Returns
    -------
    list[dict]
        List of session info dicts with keys:
        - session_id: str (BIDS ``ses-NN`` directory name)
        - has_masks: bool
        - has_feedback: bool
        - masks_approved: bool
    """
    participant_dir = base_dir / participant_id
    sessions: list[dict[str, Any]] = []

    if not participant_dir.exists():
        return sessions

    for entry in sorted(participant_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("ses-"):
            me_dir = entry / "mask_extraction"
            sessions.append(
                {
                    "session_id": entry.name,
                    "has_masks": (me_dir / "03_masks").exists(),
                    "has_feedback": (entry / "feedback" / "task_state.json").exists(),
                    "masks_approved": (me_dir / ".masks_approved").exists(),
                }
            )

    return sessions


def next_session_id(base_dir: Path, participant_id: str) -> str:
    """Get the next session ID for a participant.

    Parameters
    ----------
    base_dir : Path
        Study base directory
    participant_id : str
        Participant identifier

    Returns
    -------
    str
        Next session ID (e.g., 'ses-01', 'ses-02')
    """
    sessions = detect_existing_sessions(base_dir, participant_id)
    session_nums: list[int] = []
    for v in sessions:
        parts = str(v["session_id"]).split("-")
        if len(parts) == 2 and parts[1].isdigit():
            session_nums.append(int(parts[1]))
    next_num = max(session_nums) + 1 if session_nums else 1
    return f"ses-{next_num:02d}"


# =============================================================================
# Setup Functions (run once per participant)
# =============================================================================


def run_dicom_collection(
    config: SessionConfig,
    use_simulator: bool = True,
    task_name: str = "rest",
    source_dirs: list[Path] | None = None,
    n_volumes_list: list[int] | None = None,
    start_run: int | None = None,
    max_runs: int | None = None,
    qc_log_dir: Path | None = None,
    qc_callback: Callable | None = None,
) -> list[Path]:
    """Collect DICOMs for a task (rest, reference, etc.) using scanner or simulator.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    use_simulator : bool
        If True, use VolumeSimulator to feed DICOMs (for testing)
        If False, watch for real scanner DICOMs (real-time mode)
    task_name : str
        Name used for logging and output directory prefix
    source_dirs : list[Path], optional
        Source DICOM folders for simulator mode
    n_volumes_list : list[int], optional
        Number of volumes to collect per run
    start_run : int, optional
        Explicit starting run number. If None, uses next available
    max_runs : int, optional
        Maximum run number to collect (inclusive)
    qc_log_dir : Path, optional
        Directory for QC log files. If provided, real-time QC metrics
        (FD, DVARS, tSNR) are computed during collection.
    qc_callback : callable, optional
        ``(volume, fd, dvars, tsnr, slice_b64) -> None``
        Called after each volume with QC data for GUI display.

    Returns
    -------
    list[Path]
        Paths to collected DICOM folders (one per run)
    """
    from .preprocessing import collect_volumes

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"DICOM COLLECTION ({task_name})")
    logger.info("=" * 60)

    assert config.paths.collection_dir is not None
    collection_dir = config.paths.collection_dir
    if source_dirs is None:
        source_dirs = config.resolve_localizer_sources()
    if n_volumes_list is None:
        loc = config.localizers[0] if config.localizers else None
        n_volumes_list = [loc.n_volumes] * loc.n_runs if loc else [250]

    if len(source_dirs) != len(n_volumes_list):
        raise ValueError(f"Mismatch: {len(source_dirs)} source dirs vs {len(n_volumes_list)} n_volumes entries")

    collected_folders = []

    start_run_candidate = next_collection_run_number(collection_dir, task_name) if start_run is None else start_run
    if max_runs is not None and start_run_candidate > max_runs:
        start_run_candidate = max_runs
    start_run = start_run_candidate

    for run_idx, (source_dir, n_volumes) in enumerate(zip(source_dirs, n_volumes_list, strict=False)):
        run_num = start_run + run_idx
        if max_runs is not None and run_num > max_runs:
            logger.info(f"Stopping {task_name} collection at run {run_num}: exceeds max_runs={max_runs}")
            break

        # Get run info with retry detection
        run_info = get_localizer_run_info(collection_dir=collection_dir, task=task_name, run=run_num)
        attempt = run_info["attempt"]

        logger.info(f"Collecting {task_name} run {run_num} (attempt-{attempt:02d})...")
        if run_info["existing_attempts"]:
            logger.info(f"Previous attempts: {', '.join(f'attempt-{a:02d}' for a in run_info['existing_attempts'])}")

        output_dir = collection_dir / f"{task_name}_run-{run_num:02d}_attempt-{attempt:02d}"

        # Skip if simulator outputs already present
        if use_simulator and output_dir.exists():
            # Get all files (supports extensionless DICOMs)
            existing_volumes = [f for f in output_dir.iterdir() if f.is_file()]
            if len(existing_volumes) >= n_volumes:
                logger.info(
                    f"Skipping simulator collection for run {run_num}: {len(existing_volumes)} files already exist in {output_dir}"
                )
                collected_folders.append(output_dir)
                continue

        # Create QC monitor if log dir provided
        qc_mon = None
        if qc_log_dir is not None:
            from .collection_qc import CollectionQCMonitor

            qc_log_path = qc_log_dir / f"{task_name}_run-{run_num:02d}_attempt-{attempt:02d}_collectionlog.tsv"
            motion_mode = config.localizers[0].motion_correction if config.localizers else "siemens_moco"
            qc_mon = CollectionQCMonitor(
                log_path=qc_log_path,
                motion_correction=motion_mode,
                on_qc_data=qc_callback,
            )

        if use_simulator:
            watch_dir = collection_dir / "incoming"
            volume_folder = collect_volumes(
                watch_dir=watch_dir,
                output_dir=output_dir,
                n_volumes=n_volumes,
                clear_watch_dir=True,
                clear_output_dir=True,
                simulator_source_dir=source_dir,
                simulator_speed=config.simulator.collection_speed,
                qc_monitor=qc_mon,
            )
        else:
            watch_dir = collection_dir / "incoming"
            volume_folder = collect_volumes(
                watch_dir=watch_dir,
                output_dir=output_dir,
                n_volumes=n_volumes,
                clear_watch_dir=True,
                clear_output_dir=True,
                qc_monitor=qc_mon,
            )

        collected_folders.append(volume_folder)
        logger.info(f"  Collected {n_volumes} volumes to {volume_folder}")

    logger.info(f"Collection complete: {len(collected_folders)} runs")
    return collected_folders


def run_preprocessing(
    config: SessionConfig,
    rest_dicoms: list[Path] | None = None,
    overwrite: bool = False,
    task_name: str | None = None,
) -> tuple[Path, Path, Path]:
    """Run preprocessing on localizer scan.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    rest_dicoms : list[Path], optional
        Paths to DICOM folders. If None, resolved from config
    overwrite : bool
        If True, rerun preprocessing even if outputs exist
    task_name : str, optional
        Explicit task name for output file naming (e.g. ``"rest"``,
        ``"rit"``). If None, inferred from ``config.scan.localizer_task_type``.

    Returns
    -------
    tuple[Path, Path, Path]
        (preproc_file, mask_file, ref_file)
    """
    from .preprocessing import preprocessing

    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING")
    logger.info("=" * 60)

    if rest_dicoms is None:
        rest_dicoms = config.resolve_localizer_sources()

    if task_name is None:
        task_type = config.localizers[0].type if config.localizers else "rest"
        task_name = task_type if task_type != "rest" else "rest"

    return preprocessing(
        participant_id=config.paths.participant_id,
        session_id=config.paths.session_id,
        task=task_name,
        root_dir=str(config.paths.preprocessing_dir),
        input_dicoms=[str(p) for p in rest_dicoms],
        overwrite=overwrite,
    )


def run_mask_extraction(
    config: SessionConfig,
    preproc_file: Path | None,
    mask_file: Path | None,
    ref_file: Path | None,
    overwrite: bool = False,
) -> list[Path]:
    """Run mask extraction (ICA-based or GLM-based depending on config).

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    preproc_file : Path or None
        Preprocessed 4D NIfTI file (None for manual import)
    mask_file : Path or None
        Brain mask file (None for manual import)
    ref_file : Path or None
        BOLD reference file (None for manual import)
    overwrite : bool
        If True, rerun even if outputs exist

    Returns
    -------
    list[Path]
        Paths to thresholded network mask files
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("MASK EXTRACTION")
    logger.info("=" * 60)

    if not config.localizers:
        raise ValueError("config.localizers must contain at least one localizer for mask extraction")
    loc = config.localizers[0]
    localizer_type = loc.type
    extraction = loc.extraction

    if localizer_type == "manual":
        from .mask_extraction import import_manual_masks

        logger.info("Using manual ROI import")
        manual_ref = loc.manual_ref_file or ref_file
        assert manual_ref is not None, "ref_file or manual_ref_file is required for manual import"
        network_files = import_manual_masks(
            participant_id=config.paths.participant_id,
            session_id=config.paths.session_id,
            root_dir=str(config.paths.mask_extraction_dir),
            mask_paths=list(loc.manual_mask_paths),
            mask_names=loc.mask_names,
            ref_file=manual_ref,
            mask_space=loc.manual_mask_space,
            anat_file=loc.manual_anat_file,
            overwrite=overwrite,
            qc_enabled=True,
        )
    elif localizer_type != "rest":
        from .mask_extraction import mask_extraction_task

        assert preproc_file is not None and mask_file is not None and ref_file is not None
        logger.info("Using task-based (GLM) mask extraction")
        network_files = mask_extraction_task(
            participant_id=config.paths.participant_id,
            session_id=config.paths.session_id,
            root_dir=str(config.paths.mask_extraction_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            events=loc.events_file,
            contrasts=loc.contrasts,
            tr=loc.tr,
            threshold=extraction.threshold,
            threshold_method=extraction.threshold_method,
            atlas_mask=extraction.atlas_mask,
            atlas_source=extraction.atlas_source,
            atlas_regions=extraction.atlas_regions,
            ref_file=ref_file,
            cluster_start_threshold=extraction.cluster_start_threshold,
            cluster_step=extraction.cluster_step,
            cluster_return_all=extraction.cluster_return_all,
            min_cluster_size=extraction.min_cluster_size,
            max_total_voxels=extraction.max_total_voxels,
            stat_type=loc.stat_type,
            smoothing_fwhm=extraction.smooth_fwhm,
            high_pass=extraction.highpass,
            drift_model=loc.drift_model,
            hrf_model=loc.hrf_model,
            include_reverse=loc.include_reverse_contrasts,
            overwrite=overwrite,
        )
    else:
        from .mask_extraction import mask_extraction_rest

        assert preproc_file is not None and mask_file is not None and ref_file is not None
        logger.info("Using ICA-based (resting-state) mask extraction")
        network_files = mask_extraction_rest(
            participant_id=config.paths.participant_id,
            session_id=config.paths.session_id,
            root_dir=str(config.paths.mask_extraction_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            ref_file=ref_file,
            networks=loc.networks,
            network_atlas=extraction.network_atlas,
            smooth_fwhm=extraction.smooth_fwhm,
            highpass=extraction.highpass,
            lowpass=extraction.lowpass,
            overwrite=overwrite,
        )

    # Create WM/CSF tissue masks for nuisance regression (if either is enabled in any feedback run)
    _any_tissue = any(frc.glm.include_wm_regressor or frc.glm.include_csf_regressor for frc in config.feedback_runs)
    if _any_tissue:
        from .mask_extraction import create_tissue_masks

        assert config.paths.mask_extraction_dir is not None
        logger.info("")
        logger.info("Creating WM/CSF tissue masks for nuisance regression...")
        tissue_dir = config.paths.mask_extraction_dir / "05_tissue"

        # Try to reuse MNI inverse transforms from mask_extraction_rest
        inv_xfm = None
        mnireg_dir = config.paths.mask_extraction_dir / "02_mnireg"
        if mnireg_dir.exists():
            inv_files = sorted(mnireg_dir.glob("*_inv_*"))
            if inv_files:
                inv_xfm = [str(f) for f in inv_files]
                logger.info(f"Reusing MNI inverse transforms from {mnireg_dir.name}/")

        assert ref_file is not None
        create_tissue_masks(
            ref_file=ref_file,
            output_dir=tissue_dir,
            participant_id=config.paths.participant_id,
            session_id=config.paths.session_id,
            inv_transform_files=inv_xfm,
            overwrite=overwrite,
        )

    return network_files


def run_localizer_extraction(
    config: SessionConfig,
    localizer: LocalizerConfig,
    loc_index: int,
    preproc_file: Path,
    mask_file: Path,
    ref_file: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Run mask extraction for a single localizer.

    Each localizer gets its own subdirectory under ``mask_extraction/`` to
    avoid file collisions. The resulting thresholded masks are also copied
    into the shared ``03_masks/`` directory so that :func:`resolve_mask_source`
    can find them.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    localizer : LocalizerConfig
        Configuration for this localizer
    loc_index : int
        Index of this localizer (for directory naming)
    preproc_file : Path
        Preprocessed 4D NIfTI file
    mask_file : Path
        Brain mask file
    ref_file : Path
        BOLD reference file
    overwrite : bool
        If True, rerun even if outputs exist

    Returns
    -------
    list[Path]
        Paths to thresholded network mask files (in the shared 03_masks dir)
    """
    assert config.paths.mask_extraction_dir is not None
    loc_dir = config.paths.mask_extraction_dir / f"loc_{loc_index}_{localizer.task_name}"

    logger.info("")
    logger.info(f"--- Localizer {loc_index}: {localizer.task_name} ({localizer.type}) ---")

    if localizer.is_rest:
        from .mask_extraction import mask_extraction_rest

        logger.info("Using ICA-based (resting-state) mask extraction")
        ex = localizer.extraction
        network_files = mask_extraction_rest(
            participant_id=config.paths.participant_id,
            session_id=config.paths.session_id,
            root_dir=str(loc_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            ref_file=ref_file,
            networks=localizer.networks,
            network_atlas=ex.network_atlas,
            smooth_fwhm=ex.smooth_fwhm,
            highpass=ex.highpass,
            lowpass=ex.lowpass,
            overwrite=overwrite,
        )
    else:
        from .mask_extraction import mask_extraction_task

        logger.info("Using task-based (GLM) mask extraction")
        ex = localizer.extraction
        network_files = mask_extraction_task(
            participant_id=config.paths.participant_id,
            session_id=config.paths.session_id,
            root_dir=str(loc_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            events=localizer.events_file,
            contrasts=localizer.contrasts,
            tr=localizer.tr,
            threshold=ex.threshold,
            threshold_method=ex.threshold_method,
            atlas_mask=ex.atlas_mask,
            atlas_source=ex.atlas_source,
            atlas_regions=ex.atlas_regions,
            ref_file=ref_file,
            cluster_start_threshold=ex.cluster_start_threshold,
            cluster_step=ex.cluster_step,
            cluster_return_all=ex.cluster_return_all,
            min_cluster_size=ex.min_cluster_size,
            max_total_voxels=ex.max_total_voxels,
            stat_type=localizer.stat_type,
            smoothing_fwhm=ex.smooth_fwhm,
            high_pass=ex.highpass,
            drift_model=localizer.drift_model,
            hrf_model=localizer.hrf_model,
            include_reverse=localizer.include_reverse_contrasts,
            overwrite=overwrite,
        )

    # Copy masks to shared 03_masks/ so resolve_mask_source() finds them
    shared_mask_dir = config.paths.mask_extraction_dir / "03_masks"
    shared_mask_dir.mkdir(parents=True, exist_ok=True)
    shared_files = []
    for f in network_files:
        dest = shared_mask_dir / f.name
        shutil.copy2(f, dest)
        shared_files.append(dest)
        logger.info(f"  Copied {f.name} -> 03_masks/")

    return shared_files


def run_coregistration(
    config: SessionConfig,
    network_files: list[Path],
    ref_file: Path,
    reference_dicom_dir: Path | None = None,
    overwrite: bool = False,
    include_tissue: bool = True,
) -> list[Path]:
    """Coregister masks to feedback (reference) space.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    network_files : list[Path]
        Paths to network mask files
    ref_file : Path
        BOLD reference from preprocessing
    reference_dicom_dir : Path, optional
        Collected reference DICOM directory
    overwrite : bool
        Whether to rerun even if outputs exist
    include_tissue : bool
        Whether to include WM/CSF tissue masks in coregistration.
        Set to False when coregistering a localizer group whose
        ref_file differs from the one used to create tissue masks.

    Returns
    -------
    list[Path]
        Paths to coregistered mask files
    """
    from .mask_extraction import coregister_reference

    logger.info("")
    logger.info("=" * 60)
    logger.info("COREGISTER MASKS TO FEEDBACK SPACE")
    logger.info("=" * 60)

    if config.paths.mask_extraction_dir is None:
        raise RuntimeError("config.paths.mask_extraction_dir must be set before coregistration")
    reference_dir = reference_dicom_dir or config.resolve_simulator_source("reference_realtime")

    # Append tissue masks to the coregistration list if nuisance regression enabled
    all_files = list(network_files)
    n_network = len(network_files)
    tissue_indices: dict[str, int] = {}

    if include_tissue and any(
        frc.glm.include_wm_regressor or frc.glm.include_csf_regressor for frc in config.feedback_runs
    ):
        if config.paths.mask_extraction_dir is None:
            raise RuntimeError("config.paths.mask_extraction_dir must be set to locate tissue masks")
        tissue_dir = config.paths.mask_extraction_dir / "05_tissue"
        wm_src = tissue_dir / f"{config.paths.participant_id}_{config.paths.session_id}_tissue_wm_mask.nii.gz"
        csf_src = tissue_dir / f"{config.paths.participant_id}_{config.paths.session_id}_tissue_csf_mask.nii.gz"
        if wm_src.exists() and csf_src.exists():
            tissue_indices["wm"] = len(all_files)
            all_files.append(wm_src)
            tissue_indices["csf"] = len(all_files)
            all_files.append(csf_src)
            logger.info("Including WM/CSF tissue masks in reference coregistration")
        else:
            logger.warning(
                f"Nuisance regression enabled but tissue masks not found in {tissue_dir}. Run mask extraction first."
            )

    coreg_files = coregister_reference(
        participant_id=config.paths.participant_id,
        session_id=config.paths.session_id,
        root_dir=str(config.paths.mask_extraction_dir),
        input_dicom=str(reference_dir),
        network_files=all_files,
        ref_file=ref_file,
        overwrite=overwrite,
        output_subdir="04_reference",
    )

    # Split results: network masks returned to caller, tissue paths stored on runtime config
    network_coreg = coreg_files[:n_network]
    if tissue_indices:
        config._wm_mask_path = coreg_files[tissue_indices["wm"]]
        config._csf_mask_path = coreg_files[tissue_indices["csf"]]
        logger.info(f"Tissue mask paths set: WM={config._wm_mask_path.name}, CSF={config._csf_mask_path.name}")

    return network_coreg


def setup_session_with_reused_masks(
    config: SessionConfig,
    reference_dicoms: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Set up a session that reuses masks from a previous session.

    This is used when config.session.mask_source='reuse'. Only collects reference
    and coregisters masks from the source session - skips localizer, preprocessing,
    and mask extraction.

    Parameters
    ----------
    config : SessionConfig
        Session configuration with session.mask_source='reuse'
    reference_dicoms : Path
        Path to collected reference DICOMs for this session
    overwrite : bool
        Whether to overwrite existing coregistration outputs

    Returns
    -------
    list[Path]
        Coregistered mask files in current session's feedback/01_reference/masks/
    """
    from .mask_extraction import coregister_reference

    logger.info("")
    logger.info("=" * 60)
    logger.info("MASK REUSE: Coregistering masks from previous session")
    logger.info("=" * 60)

    if config.session is None or config.session.mask_source != "reuse":
        raise ValueError("setup_session_with_reused_masks requires config.session.mask_source='reuse'")

    if config.session.mask_source_session is None:
        raise ValueError("config.session.mask_source_session must be set when mask_source='reuse'")
    source_session = config.session.mask_source_session
    logger.info(f"Source session: {source_session}")

    # Resolve source masks and reference
    source_masks, source_ref = resolve_mask_source(config)

    if not source_masks:
        source_base = config.paths.base_dir / config.paths.participant_id / source_session / "mask_extraction"
        raise FileNotFoundError(
            f"No masks found in source session {source_session}. Expected masks in: {source_base / '03_masks'}"
        )

    if not source_ref.exists():
        raise FileNotFoundError(f"Reference file not found in source session {source_session}: {source_ref}")

    logger.info(f"Found {len(source_masks)} masks from {source_session}")
    logger.info(f"Reference file: {source_ref}")

    # Coregister source masks to current session's reference
    # Output goes to feedback_dir instead of mask_extraction_dir
    if config.paths.feedback_dir is None:
        raise RuntimeError("config.paths.feedback_dir must be set before coregistration")
    coreg_masks = coregister_reference(
        participant_id=config.paths.participant_id,
        session_id=config.paths.session_id,
        root_dir=str(config.paths.feedback_dir),  # feedback/ directory
        input_dicom=str(reference_dicoms),
        network_files=source_masks,
        ref_file=source_ref,
        output_subdir="01_reference",
        overwrite=overwrite,
    )

    # Copy mask_selections.json from source session for reference
    source_selections = source_masks[0].parent / "mask_selections.json"
    dest_dir = config.paths.feedback_dir / "01_reference"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source_selections.exists():
        dest_selections = dest_dir / "mask_selections.json"
        shutil.copy2(source_selections, dest_selections)
        logger.info("Copied mask selections from source session")

    logger.info("")
    logger.info("Mask reuse complete! Coregistered masks ready for feedback:")
    for f in coreg_masks:
        logger.info(f"  {f}")

    return coreg_masks


def setup_session(
    config: SessionConfig,
    overwrite: bool = False,
    use_simulator: bool = True,
    skip_collection: bool = False,
) -> list[Path]:
    """Run all setup steps: collection, preprocessing, mask extraction, coregistration.

    This should be run once per participant/session before the feedback tasks.
    For sessions with mask_source='reuse', skips localizer/preprocessing/extraction
    and only runs reference collection + coregistration.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    overwrite : bool
        If True, rerun all steps even if outputs exist
    use_simulator : bool
        If True, use VolumeSimulator for DICOM collection (testing mode)
        If False, watch for real scanner DICOMs (real-time mode)
    skip_collection : bool
        If True, skip DICOM collection and use source DICOMs directly

    Returns
    -------
    list[Path]
        Paths to coregistered mask files ready for feedback
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION SETUP")
    logger.info("=" * 60)

    # Check if this is a mask-reuse session
    if config.session and config.session.mask_source == "reuse":
        logger.info(f"Session {config.session.session_id}: Reusing masks from {config.session.mask_source_session}")

        # Only need reference collection for mask reuse
        reference_source = config.resolve_simulator_source("reference_realtime")
        if skip_collection:
            logger.info("Skipping reference collection (using source DICOMs directly)")
            assert reference_source is not None, "reference_realtime source required for mask reuse"
            reference_dicoms = reference_source
        else:
            reference_list = run_dicom_collection(
                config,
                use_simulator=use_simulator,
                task_name="reference",
                source_dirs=[reference_source] if reference_source else [],
                n_volumes_list=[config.reference.initial_n_volumes],
                start_run=1,
                max_runs=1,
            )
            reference_dicoms = reference_list[0]

        return setup_session_with_reused_masks(config, reference_dicoms, overwrite=overwrite)

    # Manual ROI import: skip localizer + preprocessing, just import + coregister
    if config.localizers and config.localizers[0].type == "manual":
        if config.session:
            logger.info(f"Session {config.session.session_id}: Manual ROI import")

        # Step 0: Initial reference (still needed for coregistration)
        initial_reference_source = config.resolve_simulator_source("reference_initial")
        if skip_collection:
            logger.info("Skipping initial reference collection (using source DICOMs directly)")
            assert initial_reference_source is not None, "reference_initial source required"
            initial_reference: list[Path] = [initial_reference_source]
        else:
            initial_reference = run_dicom_collection(
                config,
                use_simulator=use_simulator,
                task_name="reference",
                source_dirs=[initial_reference_source] if initial_reference_source else [],
                n_volumes_list=[config.reference.initial_n_volumes],
                start_run=None,
                max_runs=int(config.reference.initial_enabled) + int(config.reference.realtime_enabled),
            )

        # Step 1: Import manual masks (no localizer or preprocessing needed)
        assert config.localizers[0].manual_ref_file is not None, "manual_ref_file required for manual localizer"
        ref_file = config.localizers[0].manual_ref_file
        network_files = run_mask_extraction(config, None, None, ref_file, overwrite)

        # Step 2: Coregistration to initial reference
        coreg_mask_files = run_coregistration(
            config, network_files, ref_file, reference_dicom_dir=initial_reference[0], overwrite=overwrite
        )

        logger.info("")
        logger.info("Setup complete! Masks ready for feedback:")
        for f in coreg_mask_files:
            logger.info(f"  {f}")

        return coreg_mask_files

    # Standard setup: full pipeline (compute masks)
    if config.session:
        logger.info(f"Session {config.session.session_id}: Computing new masks")

    # Step 0: Initial reference (pre-scan positioning check)
    initial_reference_source = config.resolve_simulator_source("reference_initial")
    if skip_collection:
        logger.info("Skipping initial reference collection (using source DICOMs directly)")
        assert initial_reference_source is not None, "reference_initial source required"
        initial_reference = [initial_reference_source]
    else:
        initial_reference = run_dicom_collection(
            config,
            use_simulator=use_simulator,
            task_name="reference",
            source_dirs=[initial_reference_source] if initial_reference_source else [],
            n_volumes_list=[config.reference.initial_n_volumes],
            start_run=None,
            max_runs=int(config.reference.initial_enabled) + int(config.reference.realtime_enabled),
        )

    # Multi-localizer pipeline: collect, preprocess, extract for each localizer
    if config.localizers:
        # Track per-localizer (ref_file, network_files) so each group is
        # coregistered with the correct BOLD reference.
        localizer_groups: list[tuple[Path, list[Path]]] = []
        last_ref_file: Path | None = None

        for i, localizer in enumerate(config.localizers):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"LOCALIZER {i}: {localizer.task_name} ({localizer.type})")
            logger.info("=" * 60)

            # Collection
            loc_sources = list(localizer.source_dicoms)
            n_volumes_list = [localizer.n_volumes] * localizer.n_runs
            if skip_collection:
                logger.info(f"Skipping {localizer.task_name} collection (using source DICOMs directly)")
                loc_dicoms = loc_sources
            else:
                loc_dicoms = run_dicom_collection(
                    config,
                    use_simulator=use_simulator,
                    task_name=localizer.task_name,
                    source_dirs=loc_sources,
                    n_volumes_list=n_volumes_list,
                    start_run=1,
                    max_runs=localizer.n_runs,
                )

            # Preprocessing (shared preprocessing_dir, files differentiated by task_name)
            preproc_file, mask_file, ref_file = run_preprocessing(
                config,
                rest_dicoms=loc_dicoms,
                overwrite=overwrite,
                task_name=localizer.task_name,
            )
            last_ref_file = ref_file

            # Mask extraction (each localizer gets its own subdirectory)
            network_files = run_localizer_extraction(
                config,
                localizer,
                i,
                preproc_file,
                mask_file,
                ref_file,
                overwrite,
            )
            localizer_groups.append((ref_file, network_files))

        # Create WM/CSF tissue masks (shared, using last ref_file)
        assert last_ref_file is not None
        if any(frc.glm.include_wm_regressor or frc.glm.include_csf_regressor for frc in config.feedback_runs):
            from .mask_extraction import create_tissue_masks

            assert config.paths.mask_extraction_dir is not None
            logger.info("")
            logger.info("Creating WM/CSF tissue masks for nuisance regression...")
            tissue_dir = config.paths.mask_extraction_dir / "05_tissue"

            inv_xfm = None
            # Try to find MNI inverse transforms from any localizer's extraction
            for j in range(len(config.localizers)):
                loc_mnireg = (
                    config.paths.mask_extraction_dir / f"loc_{j}_{config.localizers[j].task_name}" / "02_mnireg"
                )
                if loc_mnireg.exists():
                    inv_files = sorted(loc_mnireg.glob("*_inv_*"))
                    if inv_files:
                        inv_xfm = [str(f) for f in inv_files]
                        logger.info(f"Reusing MNI inverse transforms from {loc_mnireg}")
                        break

            create_tissue_masks(
                ref_file=last_ref_file,
                output_dir=tissue_dir,
                participant_id=config.paths.participant_id,
                session_id=config.paths.session_id,
                inv_transform_files=inv_xfm,
                overwrite=overwrite,
            )

        # Coregistration — each localizer's masks use their own BOLD reference
        # so the boldref→reference transform is correct for each set of masks.
        # Tissue masks (WM/CSF) are included only with the last group since
        # they are created using last_ref_file.
        coreg_mask_files = []
        for group_idx, (group_ref, group_masks) in enumerate(localizer_groups):
            is_last = group_idx == len(localizer_groups) - 1
            coreg = run_coregistration(
                config,
                group_masks,
                group_ref,
                reference_dicom_dir=initial_reference[0],
                overwrite=overwrite,
                include_tissue=is_last,
            )
            coreg_mask_files.extend(coreg)
    else:
        raise ValueError(
            "config.localizers must be non-empty for the standard setup pipeline. "
            "Mask reuse and manual import are handled earlier; there is no implicit "
            "single-localizer fallback."
        )

    logger.info("")
    logger.info("Setup complete! Masks ready for feedback:")
    for f in coreg_mask_files:
        logger.info(f"  {f}")

    return coreg_mask_files


# =============================================================================
# Run Functions (called for each run in session)
# =============================================================================


def run_single_task(
    config: SessionConfig,
    mask_files: list[Path],
    task_state: TaskState,
    run_type: str,
    use_simulator: bool = True,
    run_config: FeedbackRunConfig | None = None,
) -> dict[str, Any]:
    """Run a single task (transfer or feedback).

    Uses the task registry and factory to create the appropriate task based
    on the task type. When *run_config* is provided, its per-run settings
    override the flat ``config.feedback`` / ``config.task`` values.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    mask_files : list[Path]
        Paths to coregistered mask files
    task_state : TaskState
        Task state for multi-run continuity
    run_type : str
        'transfer' or 'feedback'
    use_simulator : bool
        If True, use DICOM simulator for testing
    run_config : FeedbackRunConfig, optional
        Per-feedback-run configuration. When provided, takes priority over
        the flat ``config.feedback`` and ``config.task`` fields.

    Returns
    -------
    dict
        Run results (task-specific, always includes run_number, run_type,
        volumes_processed)
    """
    from .feedback import run_feedback_task
    from .tasks.factory import create_task
    from .tasks.task_registry import get_task_metadata

    if run_config is None:
        raise ValueError("run_config is required; build configs with `feedback_runs=[...]`.")

    eff_task_type = run_config.task
    eff_task_params = run_config.task_params
    eff_network_assignment = run_config.network_assignment
    eff_n_baseline = run_config.n_baseline
    eff_n_volumes = run_config.n_volumes
    eff_tr = run_config.tr
    eff_motion_correction = run_config.glm.motion_correction
    eff_include_drift = run_config.glm.include_drift
    eff_include_motion_regressors = run_config.glm.include_motion_regressors
    eff_include_wm_regressor = run_config.glm.include_wm_regressor
    eff_include_csf_regressor = run_config.glm.include_csf_regressor
    eff_smooth_fwhm = run_config.glm.smooth_fwhm
    eff_feedback_interval = run_config.intermittent.interval
    eff_connectivity_pairs = run_config.connectivity.pairs
    eff_connectivity_window = run_config.connectivity.window
    # Derive differential_pair at runtime from network_assignment + TaskMetadata.network_roles.
    # For 2-network tasks, [role[0]→positive, role[1]→negative]. None for other arities.
    _metadata_for_pair = get_task_metadata(eff_task_type)
    if len(_metadata_for_pair.network_roles) == 2 and eff_network_assignment:
        _pos = eff_network_assignment.get(_metadata_for_pair.network_roles[0].key)
        _neg = eff_network_assignment.get(_metadata_for_pair.network_roles[1].key)
        eff_differential_pair: list[str] | None = [_pos, _neg] if _pos and _neg else None
    else:
        eff_differential_pair = None

    run_number = task_state.next_run_number
    task_run_number = task_state.next_run_number_for_type(run_type)
    feedback_mode = run_type == "feedback"

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RUN {run_number}: {run_type.upper()} (task run {task_run_number})")
    logger.info("=" * 60)
    logger.info(f"Task: {eff_task_type}")
    logger.info(f"Mode: {'FEEDBACK' if feedback_mode else 'TRANSFER (no feedback)'}")
    logger.info(f"Task state: {task_state.task_state}")

    # Compute task regressors based on task metadata design_type
    metadata = get_task_metadata(eff_task_type)
    task_regressors = _compute_regressors_for_task(metadata, config, run_config)

    # Set up simulator parameters if using simulator
    simulator_kwargs: dict[str, Any] = {}
    if use_simulator:
        source_run_type = f"{run_type}_run-{task_run_number:02d}"
        simulator_source = config.resolve_simulator_source(source_run_type)
        simulator_kwargs = {
            "simulator_source_dir": simulator_source,
            "simulator_speed": config.simulator.feedback_speed,
            "simulator_clear_target": True,
            "simulator_cleanup": False,
            "simulator_trigger_mode": config.simulator.trigger_mode,
        }

    # Resolve mask names from network assignment
    task_mask_names = _resolve_mask_names_for_run(config, metadata, eff_network_assignment)

    if config.paths.feedback_output_dir is None:
        raise RuntimeError("config.paths.feedback_output_dir must be set before starting a feedback task")
    if config.paths.collection_dir is None:
        raise RuntimeError("config.paths.collection_dir must be set before starting a feedback task")
    # Start feedback task
    session_info = run_feedback_task(
        mask_paths=list(mask_files),
        subject=config.paths.participant_id,
        session=config.paths.session_id,
        task="feedback" if feedback_mode else "transfer",
        run=task_run_number,
        output_dir=config.paths.feedback_output_dir,
        collection_dir=config.paths.collection_dir,
        n_baseline=eff_n_baseline,
        n_volumes=eff_n_volumes,
        tr=eff_tr,
        task_regressors=task_regressors,
        mask_names=config.combined_mask_names,
        roi_combine_method=run_config.roi_combine_method,
        motion_correction=eff_motion_correction,
        include_drift=eff_include_drift,
        include_motion_regressors=eff_include_motion_regressors,
        include_wm_regressor=eff_include_wm_regressor,
        include_csf_regressor=eff_include_csf_regressor,
        smooth_fwhm=eff_smooth_fwhm,
        differential_pair=eff_differential_pair,
        feedback_interval=eff_feedback_interval,
        wm_mask_path=config._wm_mask_path,
        csf_mask_path=config._csf_mask_path,
        connectivity_pairs=eff_connectivity_pairs,
        connectivity_window=eff_connectivity_window,
        task_id=eff_task_type,
        **simulator_kwargs,
    )

    coordinator = session_info["coordinator"]

    # Determine if this is the first run of this type
    completed_of_type = sum(1 for r in task_state.completed_runs if r.run_type == run_type)
    _is_first_of_type = completed_of_type == 0

    # Build config for task factory
    factory_config = {
        **eff_task_params,
        "mask_names": task_mask_names,
        "feedback_mode": feedback_mode,
        "trigger_keys": config.scanner.trigger_keys,
    }

    # Create task via factory
    task = create_task(
        task_id=eff_task_type,
        coordinator=coordinator,
        config=factory_config,
        task_state=task_state,
    )

    # Register callback
    coordinator._activation_callback = task._update_data

    try:
        task.run()
    finally:
        logger.info("")
        logger.info("Cleaning up run...")
        coordinator.stop()
        if "thread" in session_info:
            session_info["thread"].join(timeout=5)
        coordinator.cleanup(force=True)

    # Extract results generically via task contract
    results = task.get_run_results()
    results["run_number"] = run_number
    results["run_type"] = run_type

    logger.info("")
    logger.info(f"Run {run_number} complete:")
    logger.info(f"  Results: {results}")

    return results


def _compute_regressors_for_task(
    metadata: Any,
    config: SessionConfig,
    run_config: FeedbackRunConfig | None = None,
) -> Any:
    """Compute task regressors based on metadata.design_type.

    Parameters
    ----------
    metadata : TaskMetadata
        Task metadata with design_type field
    config : SessionConfig
        Session configuration (fallback values)
    run_config : FeedbackRunConfig, optional
        Per-run overrides for n_baseline, n_volumes, tr.

    Returns
    -------
    numpy.ndarray or None
        Task regressors, or None for nuisance-only designs
    """
    from .utils import compute_boxcar_task_regressor

    if run_config is None:
        if not config.feedback_runs:
            raise ValueError("run_config or config.feedback_runs must be provided")
        run_config = config.feedback_runs[0]
    n_baseline = run_config.n_baseline
    n_volumes = run_config.n_volumes
    tr = run_config.tr

    if metadata.design_type == "nuisance_only":
        return None
    elif metadata.design_type == "block":
        regressors = compute_boxcar_task_regressor(
            n_baseline=n_baseline,
            n_volumes=n_volumes,
            tr=tr,
        )
        logger.info(f"Computed boxcar task regressor (baseline={n_baseline}, volumes={n_volumes})")
        return regressors
    else:
        logger.warning(f"Unknown design_type '{metadata.design_type}', using no task regressors")
        return None


def _resolve_mask_names(config: SessionConfig, metadata: Any) -> list[str]:
    """Resolve ordered mask names from the first feedback_run's network_assignment.

    Parameters
    ----------
    config : SessionConfig
        Session configuration. The first ``feedback_runs[0].network_assignment``
        (if any) supplies the role-to-mask mapping.
    metadata : TaskMetadata
        Task metadata with network_roles and n_networks.
    """
    network_assignment = config.feedback_runs[0].network_assignment if config.feedback_runs else {}
    return _resolve_mask_names_for_run(config, metadata, network_assignment)


def _resolve_mask_names_for_run(
    config: SessionConfig,
    metadata: Any,
    network_assignment: dict[str, str],
) -> list[str]:
    """Resolve ordered mask names using an explicit network assignment.

    Falls back to the combined mask-names list (union of every localizer's
    mask names) when ``network_assignment`` doesn't cover every role.
    """
    fallback = config.combined_mask_names
    if network_assignment and metadata.network_roles:
        mask_names = []
        for i, role in enumerate(metadata.network_roles):
            name = network_assignment.get(role.key)
            if name:
                mask_names.append(name)
            elif i < len(fallback):
                mask_names.append(fallback[i])
        if len(mask_names) == metadata.n_networks:
            return mask_names
    # Fall back to first N mask names from the combined union
    return fallback[: metadata.n_networks]


def coregister_before_first_transfer(
    config: SessionConfig,
    task_state: TaskState,
    mask_files: list[Path],
    use_simulator: bool = True,
) -> list[Path]:
    """Acquire a second reference and re-coregister masks right before the first transfer.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    task_state : TaskState
        Task state (used to check if any runs completed)
    mask_files : list[Path]
        Current mask files
    use_simulator : bool
        If True, use DICOM simulator for testing

    Returns
    -------
    list[Path]
        Updated coregistered mask files
    """
    if task_state.completed_runs:
        return mask_files

    logger.info("")
    logger.info("=" * 60)
    logger.info("REAL-TIME REFERENCE + COREG (pre-transfer)")
    logger.info("=" * 60)

    network_files = [Path(f) for f in mask_files]

    ref_file = ref_file_path(config)
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found at {ref_file}; run preprocessing before coregistration.")

    reference_realtime_source = config.resolve_simulator_source("reference_realtime")
    reference_dicoms = run_dicom_collection(
        config,
        use_simulator=use_simulator,
        task_name="reference",
        source_dirs=[reference_realtime_source] if reference_realtime_source else [],
        n_volumes_list=[config.reference.realtime_n_volumes],
        start_run=None,
        max_runs=int(config.reference.initial_enabled) + int(config.reference.realtime_enabled),
    )

    coreg_mask_files = run_coregistration(
        config,
        network_files,
        ref_file,
        reference_dicom_dir=reference_dicoms[0],
        overwrite=True,
    )

    # Generate reference QC overlays (only at this pre-transfer step)
    try:
        from .atlas_registry import ATLAS_SOURCES
        from .mask_extraction import generate_reference_qc_overlays

        assert config.paths.mask_extraction_dir is not None
        _run_match = re.search(r"run-(\d+)", coreg_mask_files[0].stem) if coreg_mask_files else None
        run_str = _run_match.group(1) if _run_match else "01"
        _ref_atlas = config.localizers[0].extraction.network_atlas if config.localizers else "yeo_17"
        generate_reference_qc_overlays(
            participant_id=config.paths.participant_id,
            session_id=config.paths.session_id,
            session_dir=config.paths.mask_extraction_dir,  # reference outputs are in mask_extraction/04_reference
            ref_file=ref_file,
            mask_extraction_dir=config.paths.mask_extraction_dir,
            network_coreg_files=coreg_mask_files,
            run_str=run_str,
            n_networks=ATLAS_SOURCES.get(_ref_atlas, {}).get("n_networks", 17),
            overwrite=True,
            reference_subdir="04_reference",  # Match the output_subdir used in coregister_reference
        )
    except Exception as e:
        logger.warning(f"Reference QC overlay generation skipped: {e}")

    return [Path(f) for f in coreg_mask_files]


# =============================================================================
# Session Workflow
# =============================================================================


def _build_run_sequence(config: SessionConfig) -> list[tuple[str, int]]:
    """Build the run sequence from ``config.feedback_runs``.

    Each entry is ``(run_type, frc_index)`` where *frc_index* is the index
    into ``config.feedback_runs`` that owns this step.

    Parameters
    ----------
    config : SessionConfig
        Session configuration; the sequence is derived from ``feedback_runs``.

    Returns
    -------
    list[tuple[str, int]]
        Ordered ``(run_type, frc_index)`` pairs.
    """
    sequence: list[tuple[str, int]] = []
    for idx, frc in enumerate(config.feedback_runs):
        n_runs = frc.max_runs
        use_transfers = frc.use_transfer_runs
        if use_transfers:
            sequence.append(("transfer", idx))
        for _ in range(n_runs):
            sequence.append(("feedback", idx))
        if use_transfers:
            sequence.append(("transfer", idx))
    return sequence


def run_full_session(
    config: SessionConfig,
    mask_files: list[Path],
    reset_session: bool = False,
    use_simulator: bool = True,
) -> list[dict[str, Any]]:
    """Run a complete multi-run session.

    Workflow depends on task config: with transfer runs it's
    Transfer -> Feedback x N -> Transfer; without it's just Feedback x N.
    The run count comes from each ``feedback_runs[*].max_runs``.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    mask_files : list[Path]
        Paths to coregistered mask files
    reset_session : bool
        If True, reset task state before starting
    use_simulator : bool
        If True, use DICOM simulator for testing

    Returns
    -------
    list[dict]
        Results from each run
    """
    from .tasks.state import TaskState

    run_sequence = _build_run_sequence(config)
    # Derive total feedback run count from sequence
    total_feedback = sum(1 for rt, _ in run_sequence if rt == "feedback")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING MULTI-RUN SESSION")
    logger.info("=" * 60)
    if not config.feedback_runs:
        raise ValueError("config.feedback_runs must be non-empty.")
    primary_task = config.feedback_runs[0].task
    task_types = [frc.task for frc in config.feedback_runs]
    logger.info(f"Tasks: {', '.join(task_types)}")
    workflow_labels = [rt.capitalize() for rt, _ in run_sequence]
    logger.info(f"Workflow: {' -> '.join(workflow_labels)}")

    if config.paths.feedback_dir is None:
        raise RuntimeError("config.paths.feedback_dir must be set before running a session")
    feedback_dir = config.paths.feedback_dir
    if reset_session:
        task_state = TaskState.reset(feedback_dir, task_id=primary_task)
        task_state.max_feedback_runs = total_feedback
        task_state.save()
    else:
        task_state = TaskState.load(feedback_dir, task_id=primary_task)
        if task_state.max_feedback_runs != total_feedback:
            task_state.max_feedback_runs = total_feedback
            task_state.save()

    mask_files = [Path(f) for f in mask_files]

    all_results = []

    completed_count = len(task_state.completed_runs)
    total_runs = len(run_sequence)

    logger.info(f"Completed runs: {completed_count}")
    remaining_labels = [rt for rt, _ in run_sequence[completed_count:]]
    logger.info(f"Remaining runs: {remaining_labels}")

    for i, (run_type, frc_idx) in enumerate(run_sequence[completed_count:], start=completed_count):
        logger.info("")
        logger.info(f">>> Starting run {i + 1}/{total_runs} ({run_type})")

        if run_type == "transfer" and not task_state.completed_runs:
            mask_files = coregister_before_first_transfer(config, task_state, mask_files, use_simulator=use_simulator)

        frc = config.feedback_runs[frc_idx] if config.feedback_runs else None
        results = run_single_task(
            config=config,
            mask_files=mask_files,
            task_state=task_state,
            run_type=run_type,
            use_simulator=use_simulator,
            run_config=frc,
        )
        all_results.append(results)

        task_state = TaskState.load(feedback_dir, task_id=config.feedback_runs[frc_idx].task)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total runs: {len(task_state.completed_runs)}")
    logger.info(f"Final task state: {task_state.task_state}")

    return all_results


def run_next_task(
    config: SessionConfig,
    mask_files: list[Path],
    use_simulator: bool = True,
) -> dict[str, Any] | None:
    """Run the next task in the session sequence.

    Loads task state, determines the next run type, executes it, and exits.
    Call this repeatedly (separate script invocations) to progress through the session.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    mask_files : list[Path]
        Paths to coregistered mask files
    use_simulator : bool
        If True, use DICOM simulator for testing

    Returns
    -------
    dict or None
        Run results, or None if session is complete
    """
    from .tasks.state import TaskState

    if not config.feedback_runs:
        raise ValueError("config.feedback_runs must be non-empty")
    if config.paths.feedback_dir is None:
        raise RuntimeError("config.paths.feedback_dir must be set before running a task")
    feedback_dir = config.paths.feedback_dir
    task_state = TaskState.load(feedback_dir, task_id=config.feedback_runs[0].task)

    mask_files = [Path(f) for f in mask_files]

    run_sequence = _build_run_sequence(config)
    total_feedback = sum(1 for rt, _ in run_sequence if rt == "feedback")

    if task_state.max_feedback_runs != total_feedback:
        task_state.max_feedback_runs = total_feedback
        task_state.save()

    total_runs = len(run_sequence)
    completed_count = len(task_state.completed_runs)

    if completed_count >= total_runs:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SESSION ALREADY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total runs: {completed_count}")
        logger.info(f"Final task state: {task_state.task_state}")
        logger.info("Use reset_task_state() to start a new session.")
        return None

    run_type, frc_idx = run_sequence[completed_count]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"NEXT RUN: {completed_count + 1}/{total_runs} ({run_type.upper()})")
    logger.info("=" * 60)
    logger.info(f"Completed: {[r.run_type for r in task_state.completed_runs]}")
    remaining_labels = [rt for rt, _ in run_sequence[completed_count:]]
    logger.info(f"Remaining: {remaining_labels}")

    if run_type == "transfer" and not task_state.completed_runs:
        mask_files = coregister_before_first_transfer(config, task_state, mask_files, use_simulator=use_simulator)

    frc = config.feedback_runs[frc_idx] if config.feedback_runs else None
    results = run_single_task(
        config=config,
        mask_files=mask_files,
        task_state=task_state,
        run_type=run_type,
        use_simulator=use_simulator,
        run_config=frc,
    )

    return results


def reset_task_state(config: SessionConfig) -> TaskState:
    """Reset task state for a new session.

    Parameters
    ----------
    config : SessionConfig
        Session configuration

    Returns
    -------
    TaskState
        Fresh task state
    """
    from .tasks.state import TaskState

    if not config.feedback_runs:
        raise ValueError("config.feedback_runs must be non-empty")
    if config.paths.feedback_dir is None:
        raise RuntimeError("config.paths.feedback_dir must be set to reset task state")
    return TaskState.reset(config.paths.feedback_dir, task_id=config.feedback_runs[0].task)
