"""
Session configuration and orchestration for multi-run neurofeedback workflows.

Summary
-------
This module provides type-safe configuration dataclasses and high-level orchestration
functions for running neurofeedback sessions. It separates configuration from execution,
making it suitable for both scripted usage and future GUI integration.

Multi-Visit Support
------------------
The module supports longitudinal studies with multiple visits per participant:
- Single-visit mode: Traditional layout with all data in participant directory
- Multi-visit mode: Data organized in ses-XX subdirectories

For multi-visit studies, masks can be:
- Computed fresh each visit (mask_source='compute')
- Reused from a previous visit (mask_source='reuse')

Components
----------
Configuration Dataclasses:
- PathConfig: Paths for session directories (with optional visit_id)
- ScanConfig: Scanner/acquisition parameters
- MaskConfig: Mask extraction parameters
- FeedbackConfig: Feedback session parameters
- TaskConfig: Generic task parameters (task_type, task_params, network_assignment)
- VisitConfig: Visit-level configuration for multi-visit studies
- ShamConfig: Sham/control condition configuration
- SessionConfig: Top-level config combining all above

Orchestration Functions:
- setup_session(): Run all setup steps (collection, preprocessing, mask extraction)
- setup_visit_with_reused_masks(): Set up visit reusing masks from another visit
- run_single_task(): Run one transfer/feedback task
- run_full_session(): Run complete workflow (Transfer -> Feedback x N -> Transfer)
- run_next_task(): Run next task in sequence (for incremental progression)
- reset_session_state(): Reset session for new participant

Multi-Visit Support Functions:
- resolve_mask_source(): Get mask paths based on visit configuration
- detect_existing_visits(): List visits for a participant
- next_visit_id(): Get next visit ID (ses-01, ses-02, etc.)
- is_single_visit_data(): Check if data uses single-visit layout
- migrate_to_multi_visit(): Convert single-visit to multi-visit structure

Workflow
--------
Single-visit (traditional):
1. Create SessionConfig with PathConfig(visit_id=None)
2. Call setup_session() to prepare masks
3. Call run_next_task() repeatedly to progress through session

Multi-visit with fresh masks:
1. Create SessionConfig with PathConfig(visit_id='ses-01') and VisitConfig(mask_source='compute')
2. Call setup_session() for full pipeline
3. Run feedback session

Multi-visit with mask reuse:
1. Create SessionConfig with PathConfig(visit_id='ses-02') and
   VisitConfig(mask_source='reuse', mask_source_visit='ses-01')
2. Call setup_session() - only runs reference collection + coregistration
3. Run feedback session

Dependencies
------------
dataclasses, pathlib, json, shutil, datetime, yaml (optional), logging

Notes
-----
This module works alongside tasks/session.py which contains SessionState for
persisting task-level state (hits, scale factors) between runs.
Session state resets each visit (visits are independent).
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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from .tasks.session import SessionState

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
class VisitConfig:
    """Visit-level configuration for multi-visit studies.

    Parameters
    ----------
    visit_id : str
        Visit identifier (e.g., 'ses-01', 'ses-02')
    visit_number : int
        1-indexed visit number
    mask_source : str
        How masks are obtained: 'compute' (run localizer + extraction)
        or 'reuse' (reuse from another visit)
    mask_source_visit : str, optional
        If mask_source='reuse', the visit ID to copy masks from (e.g., 'ses-01')
    localizer_enabled : bool
        Whether to run localizer (resting-state) collection
    notes : str
        Optional notes about this visit
    created : str
        ISO timestamp of visit creation
    """

    visit_id: str
    visit_number: int
    mask_source: str = "compute"  # "compute" or "reuse"
    mask_source_visit: str | None = None
    localizer_enabled: bool = True
    notes: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.mask_source not in ("compute", "reuse"):
            raise ValueError(f"mask_source must be 'compute' or 'reuse', got '{self.mask_source}'")
        if self.mask_source == "reuse" and not self.mask_source_visit:
            raise ValueError("mask_source_visit is required when mask_source='reuse'")
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
    2. 'two_networks': Use two distinct Yeo networks (legacy behavior)

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
        Columns: subject_id, condition, matched_participant, matched_visit, notes
        For 'replay' method: matched_participant specifies which participant's
        feedbacklog to replay from (looked up in the study's base_dir).
    """

    enabled: bool = False
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
    visit_id : str, optional
        Visit identifier (e.g., 'ses-01'). If None, uses single-visit mode
        (backward compatible with existing data).
    session_dir : Path, optional
        Directory for session state and feedback outputs. Defaults to participant_visit_dir/feedback
    preprocessing_dir : Path, optional
        Directory for preprocessing outputs. Defaults to participant_visit_dir/preprocessing
    mask_extraction_dir : Path, optional
        Directory for mask extraction outputs. Defaults to participant_visit_dir/mask_extraction
    feedback_output_dir : Path, optional
        Directory for feedback run outputs. Defaults to session_dir/02_feedback_output
    collection_dir : Path, optional
        Directory for DICOM collection. Defaults to participant_visit_dir/collection
    mask_source_dir : Path, optional
        If reusing masks from another visit, the source mask_extraction directory
    """

    base_dir: Path
    participant_id: str
    visit_id: str | None = None  # None = single-visit mode (backward compat)
    session_dir: Path | None = None
    preprocessing_dir: Path | None = None
    mask_extraction_dir: Path | None = None
    feedback_output_dir: Path | None = None
    collection_dir: Path | None = None
    mask_source_dir: Path | None = None  # For mask reuse across visits

    def __post_init__(self) -> None:
        """Convert all paths to Path objects and set defaults for optional directories."""
        self.base_dir = Path(self.base_dir)

        # Validate visit_id format if provided
        if self.visit_id and not re.match(r"^ses-\d+$", self.visit_id):
            raise ValueError(f"visit_id must be in BIDS format 'ses-NN' (e.g., 'ses-01'), got '{self.visit_id}'")

        # Compute participant + visit directory
        if self.visit_id:
            # Multi-visit mode: base_dir/participant_id/visit_id/
            self._participant_visit_dir = self.base_dir / self.participant_id / self.visit_id
        else:
            # Single-visit mode (backward compatible): base_dir/participant_id/
            self._participant_visit_dir = self.base_dir / self.participant_id

        # Set default paths relative to participant_visit_dir if not provided
        if self.session_dir is None:
            self.session_dir = self._participant_visit_dir / "feedback"
        else:
            self.session_dir = Path(self.session_dir)

        if self.preprocessing_dir is None:
            self.preprocessing_dir = self._participant_visit_dir / "preprocessing"
        else:
            self.preprocessing_dir = Path(self.preprocessing_dir)

        if self.mask_extraction_dir is None:
            self.mask_extraction_dir = self._participant_visit_dir / "mask_extraction"
        else:
            self.mask_extraction_dir = Path(self.mask_extraction_dir)

        if self.feedback_output_dir is None:
            self.feedback_output_dir = self.session_dir / "02_feedback_output"
        else:
            self.feedback_output_dir = Path(self.feedback_output_dir)

        if self.collection_dir is None:
            self.collection_dir = self._participant_visit_dir / "collection"
        else:
            self.collection_dir = Path(self.collection_dir)

        if self.mask_source_dir is not None:
            self.mask_source_dir = Path(self.mask_source_dir)

    @property
    def participant_visit_dir(self) -> Path:
        """Return the participant + visit directory."""
        return cast(Path, self._participant_visit_dir)

    @property
    def is_multi_visit(self) -> bool:
        """Check if this is a multi-visit configuration."""
        return self.visit_id is not None


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
        (e.g. ``"rnt"``) for GLM-based extraction.
    tr : float
        Repetition time in seconds for this localizer.
    n_volumes : int
        Number of volumes per run.
    n_runs : int
        Number of runs for this localizer.
    source_dicoms : list[Path]
        DICOM source folders (for simulator).
    n_yeo : int
        Number of Yeo networks (7 or 17). Rest only.
    yeo_targets : list[str]
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
        ``"top_n"`` or ``"cluster"``.
    atlas_mask : str or None
        Binary atlas NIfTI path for cluster thresholding.
    atlas_name : str or None
        Curated atlas entry name for cluster thresholding.
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

    # Scan parameters
    tr: float = 1.2
    n_volumes: int = 200
    n_runs: int = 2
    source_dicoms: list[Path] = field(default_factory=list)

    # Rest-specific (ICA)
    n_yeo: int = 17
    yeo_targets: list[str] = field(default_factory=lambda: ["DefaultA", "ContA"])
    network_atlas: str = "yeo_17"
    smooth_fwhm: float = 5.0
    highpass: float = 0.01
    lowpass: float | None = None

    # Task-specific (GLM)
    contrasts: dict[str, str] = field(default_factory=dict)
    include_reverse_contrasts: bool = False
    events_file: Path | None = None
    stat_type: str = "t"
    hrf_model: str = "spm"
    drift_model: str = "cosine"

    # Mask extraction (shared)
    threshold: int = 2000
    threshold_method: str = "top_n"
    atlas_mask: str | None = None
    atlas_name: str | None = None
    atlas_source: str | None = None
    atlas_regions: list[str] | None = None
    cluster_start_threshold: float = 2.0
    cluster_step: float = 0.1

    @property
    def is_rest(self) -> bool:
        """Whether this is a resting-state localizer."""
        return self.type == "rest"

    @property
    def task_name(self) -> str:
        """Name used for file naming and directory structure."""
        return "rest" if self.is_rest else self.type

    @property
    def mask_names(self) -> list[str]:
        """Mask names this localizer produces."""
        if self.is_rest:
            return list(self.yeo_targets)
        names = list(self.contrasts.keys())
        if self.include_reverse_contrasts:
            names += [f"{k}_reverse" for k in self.contrasts]
        return names

    def __post_init__(self) -> None:
        self.source_dicoms = [Path(p) for p in self.source_dicoms]
        if self.events_file is not None:
            self.events_file = Path(self.events_file)
        if self.type != "rest" and not self.contrasts:
            raise ValueError(f"contrasts must be provided for task localizer type='{self.type}'")


@dataclass
class ScanConfig:
    """Scanner/acquisition parameters.

    Parameters
    ----------
    localizer_tr : float
        Repetition time in seconds for localizer runs
    localizer_n_volumes : int
        Number of volumes per localizer run
    localizer_n_runs : int
        Number of localizer runs
    localizer_task_type : str
        Task type for localizer: "rest" or a registered task identifier
    feedback_tr : float
        Repetition time in seconds for feedback runs
    feedback_n_volumes : int
        Number of volumes per feedback run
    localizer_source_dicoms : list[Path]
        Source DICOM folders for localizer runs (for simulator)
    reference_initial_enabled : bool
        Whether to collect initial reference scan (before localizer)
    reference_initial_n_volumes : int
        Number of volumes for initial reference acquisition
    reference_realtime_enabled : bool
        Whether to collect realtime reference scan (before feedback)
    reference_realtime_n_volumes : int
        Number of volumes for realtime reference acquisition
    reference_dicoms : Path, optional
        Default reference DICOM source
    reference_initial_dicoms : Path, optional
        Reference DICOM source for initial (pre-scan) acquisition
    reference_realtime_dicoms : Path, optional
        Reference DICOM source for real-time (day-of) acquisition
    feedback_dicoms : Path, optional
        Source DICOM folder for feedback runs (for simulator)
    simulator_source_mapping : Path, optional
        Path to CSV/TSV file mapping subjects to per-run DICOM source paths.
        When set, this takes priority over individual DICOM source fields.
    collection_speed : float
        Speed multiplier for DICOM collection simulation (localizer/setup runs)
    feedback_simulator_speed : float
        Speed multiplier for feedback DICOM simulation
    """

    # Localizer parameters
    localizer_tr: float = 1.2
    localizer_n_volumes: int = 200
    localizer_n_runs: int = 2
    localizer_task_type: str = "rest"

    # Feedback parameters
    feedback_tr: float = 1.2
    feedback_n_volumes: int = 150

    # Reference scan parameters (independently optional)
    reference_initial_enabled: bool = True
    reference_initial_n_volumes: int = 2
    reference_realtime_enabled: bool = True
    reference_realtime_n_volumes: int = 2

    # DICOM sources (for simulator)
    localizer_source_dicoms: list[Path] = field(default_factory=list)
    reference_dicoms: Path | None = None
    reference_initial_dicoms: Path | None = None
    reference_realtime_dicoms: Path | None = None
    feedback_dicoms: Path | None = None
    simulator_source_mapping: Path | None = None
    collection_speed: float = 10.0
    feedback_simulator_speed: float = 10.0

    # Legacy aliases
    @property
    def reference_n_volumes(self) -> int:
        """Alias for reference_initial_n_volumes (backward compatibility)."""
        return self.reference_initial_n_volumes

    @property
    def reference_max_runs(self) -> int:
        """Legacy alias - returns 2 if both reference enabled, 1 if only one, 0 if none."""
        return int(self.reference_initial_enabled) + int(self.reference_realtime_enabled)

    # Legacy aliases (for backward compatibility)
    @property
    def tr(self) -> float:
        """Alias for feedback_tr (backward compatibility)."""
        return self.feedback_tr

    @property
    def n_volumes(self) -> int:
        """Alias for feedback_n_volumes (backward compatibility)."""
        return self.feedback_n_volumes

    @property
    def rest_source_dicoms(self) -> list[Path]:
        """Alias for localizer_source_dicoms (backward compatibility)."""
        return self.localizer_source_dicoms

    @property
    def rest_n_volumes(self) -> list[int]:
        """Alias returning localizer volumes as list (backward compatibility)."""
        return [self.localizer_n_volumes] * self.localizer_n_runs

    @property
    def rest_max_runs(self) -> int:
        """Alias for localizer_n_runs (backward compatibility)."""
        return self.localizer_n_runs

    def __post_init__(self) -> None:
        """Convert all DICOM source paths to Path objects."""
        self.localizer_source_dicoms = [Path(p) for p in self.localizer_source_dicoms]
        if self.reference_dicoms is not None:
            self.reference_dicoms = Path(self.reference_dicoms)
        if self.reference_initial_dicoms is not None:
            self.reference_initial_dicoms = Path(self.reference_initial_dicoms)
        if self.reference_realtime_dicoms is not None:
            self.reference_realtime_dicoms = Path(self.reference_realtime_dicoms)
        if self.feedback_dicoms is not None:
            self.feedback_dicoms = Path(self.feedback_dicoms)
        if self.simulator_source_mapping is not None:
            self.simulator_source_mapping = Path(self.simulator_source_mapping)


@dataclass
class MaskConfig:
    """Mask extraction parameters.

    Parameters
    ----------
    n_yeo : int
        Number of Yeo networks (7 or 17). Used for ICA-based extraction.
        Ignored when ``network_atlas`` is explicitly set.
    yeo_targets : list[str]
        Target network names to extract. Used for ICA-based extraction.
    network_atlas : str
        Network atlas for ICA component identification in resting-state
        extraction. Options: ``"yeo_7"``, ``"yeo_17"``,
        ``"schaefer_100_7net"``, ``"schaefer_100_17net"``,
        ``"schaefer_400_7net"``, ``"schaefer_400_17net"``.
    smooth_fwhm : float
        Spatial smoothing FWHM in mm
    highpass : float
        High-pass filter cutoff in Hz
    lowpass : float, optional
        Low-pass filter cutoff in Hz. Used for ICA-based extraction.
    mask_names : list[str]
        Names for masks in feedback output
    localizer_type : str
        "rest" for ICA-based extraction, "task" for GLM-based extraction.
    contrasts : dict[str, str]
        For task-based extraction: mapping of mask name to contrast expression.
        Example: {"CEN": "nback - rest", "DMN": "rest - nback"}.
        Ignored when localizer_type="rest".
    include_reverse_contrasts : bool
        When True, automatically generate a reverse (negated) mask for each
        contrast. For example, if contrasts={"rnt": "rumination - distraction"},
        a second mask "rnt_reverse" is created from the negated stat map.
        This enables dual-network feedback from a single-contrast localizer.
    events_file : Path, optional
        For task-based extraction: path to BIDS events .tsv file.
        Ignored when localizer_type="rest".
    stat_type : str
        Statistical map type for task-based extraction: "t" or "z".
    hrf_model : str
        HRF model for task-based extraction: "spm", "spm + derivative", "glover".
    drift_model : str
        Drift model for task-based extraction: "cosine", "polynomial", or "none".
    threshold : int
        Number of voxels per mask.
    threshold_method : str
        ``"top_n"`` (default) selects the top N voxels by stat value.
        ``"cluster"`` uses atlas-guided iterative cluster thresholding.
    atlas_mask : str, optional
        Path to a binary atlas NIfTI in subject space. Used with
        ``threshold_method="cluster"``. Mutually exclusive with ``atlas_name``
        and ``atlas_source``.
    atlas_name : str, optional
        Legacy: name of a curated combined atlas entry (e.g., ``"harvard_oxford_stg"``).
        Used with ``threshold_method="cluster"``. Mutually exclusive with ``atlas_mask``
        and ``atlas_source``.
    atlas_source : str, optional
        Atlas source key (e.g., ``"harvard_oxford"``, ``"aal"``,
        ``"schaefer_400_7net"``). Used with ``atlas_regions`` for cluster
        thresholding. Mutually exclusive with ``atlas_mask`` and ``atlas_name``.
    atlas_regions : list[str], optional
        Region labels from the atlas to include in the cluster mask.
        Required when ``atlas_source`` is provided.
    cluster_start_threshold : float
        Starting stat threshold for cluster search (default: 2.0).
    cluster_step : float
        Threshold increment per iteration (default: 0.1).
    """

    # ICA-based extraction fields
    n_yeo: int = 17
    yeo_targets: list[str] = field(default_factory=lambda: ["DefaultA", "ContA"])
    network_atlas: str = "yeo_17"
    smooth_fwhm: float = 5.0
    highpass: float = 0.01
    lowpass: float | None = None
    mask_names: list[str] = field(default_factory=lambda: ["DefaultA", "ContA"])

    # Task-based extraction fields
    localizer_type: str = "rest"
    contrasts: dict[str, str] = field(default_factory=dict)
    include_reverse_contrasts: bool = False
    events_file: Path | None = None
    stat_type: str = "t"
    hrf_model: str = "spm"
    drift_model: str = "cosine"
    threshold: int = 2000

    # Cluster-based thresholding fields
    threshold_method: str = "top_n"
    atlas_mask: str | None = None
    atlas_name: str | None = None
    atlas_source: str | None = None
    atlas_regions: list[str] | None = None
    cluster_start_threshold: float = 2.0
    cluster_step: float = 0.1

    # Manual ROI import fields
    manual_mask_paths: list[Path] = field(default_factory=list)
    manual_ref_file: Path | None = None
    manual_anat_file: Path | None = None
    manual_mask_space: str = "native"

    def __post_init__(self) -> None:
        if self.localizer_type not in ("rest", "task", "manual"):
            raise ValueError(f"localizer_type must be 'rest', 'task', or 'manual', got '{self.localizer_type}'")
        if self.localizer_type == "task" and not self.contrasts:
            raise ValueError("contrasts must be provided when localizer_type='task'")
        if self.localizer_type == "manual":
            if not self.manual_mask_paths:
                raise ValueError("manual_mask_paths must be provided when localizer_type='manual'")
            if self.manual_mask_space not in ("native", "anat", "mni"):
                raise ValueError(
                    f"manual_mask_space must be 'native', 'anat', or 'mni', got '{self.manual_mask_space}'"
                )
            if self.manual_mask_space == "anat" and self.manual_anat_file is None:
                raise ValueError("manual_anat_file is required when manual_mask_space='anat'")
            self.manual_mask_paths = [Path(p) for p in self.manual_mask_paths]
            if self.manual_ref_file is not None:
                self.manual_ref_file = Path(self.manual_ref_file)
            if self.manual_anat_file is not None:
                self.manual_anat_file = Path(self.manual_anat_file)
        if self.threshold_method not in ("top_n", "cluster"):
            raise ValueError(f"threshold_method must be 'top_n' or 'cluster', got '{self.threshold_method}'")
        if self.events_file is not None:
            self.events_file = Path(self.events_file)


@dataclass
class FeedbackConfig:
    """Feedback session parameters.

    Parameters
    ----------
    n_baseline : int
        Number of baseline volumes before task onset
    n_volumes : int
        Total number of volumes per feedback run
    use_efficiency_weights : bool
        Whether to use efficiency weights in activation estimation
    motion_correction : str
        Real-time motion correction mode. Controls how volumes are corrected
        and how motion parameters are obtained:
        - ``"siemens_moco"``: Parse motion from Siemens MoCo JSON sidecar
          (``ImageComments`` field). Fast (~0 ms). Siemens scanners only.
        - ``"software"``: ANTsPy frame-to-reference rigid registration
          with warm-start. Works on any scanner (~300 ms per volume).
        - ``"none"``: No motion correction. No motion parameters available.
    include_drift : bool
        Include linear drift regressor in the GLM nuisance model.
    include_motion_regressors : bool
        Include 6 motion parameters (differential, arc-length converted)
        as nuisance regressors in the GLM. Requires motion_correction to
        be ``"siemens_moco"`` or ``"software"`` (ignored when ``"none"``).
        Motion parameters are still extracted for FD/QC even when this is
        False, as long as motion_correction is not ``"none"``.
    include_wm_regressor : bool
        Include white matter mean signal as a nuisance regressor in the GLM.
        Tissue mask created from MNI template priors (no T1 required).
    include_csf_regressor : bool
        Include CSF mean signal as a nuisance regressor in the GLM.
        Tissue mask created from MNI template priors (no T1 required).
    smooth_fwhm : float, optional
        Gaussian spatial smoothing FWHM in mm applied to each volume before
        voxel extraction. None (default) disables smoothing. Typical values
        are 5.0-6.0 mm. Uses nilearn.image.smooth_img().
    differential_pair : list of str, optional
        Ordered pair [positive_mask, negative_mask] for differential signal
        computation. The pipeline computes positive - negative and logs it
        as a column in feedbacklog.tsv. None (default) disables differential.
        Example: ["ContA", "DefaultA"] computes ContA - DefaultA.
    feedback_interval : int, optional
        Number of volumes between feedback deliveries. When set, activations
        are accumulated and the mean is delivered to the task every N volumes.
        None (default) delivers feedback every volume (continuous mode).
        Only applies after baseline period. Example: 5 delivers feedback
        every 5th volume.
    connectivity_pairs : list of list of str, optional
        Pairs of mask names for sliding-window Pearson correlation.
        Each pair is [mask_a, mask_b]. None (default) disables connectivity.
        Example: [["ContA", "DefaultA"]] computes correlation between
        ContA and DefaultA. Values logged as corr_{mask_a}_{mask_b} columns.
    connectivity_window : int
        Sliding window size in volumes for connectivity computation.
        Default is 15. Must be >= 3.
    feedback_mode : str
        Feedback signal type. "activation" (default) uses dual-network
        activation comparison. "connectivity" uses sliding-window Pearson
        correlation between the configured pair.
    connectivity_target : str
        Target direction for connectivity feedback. "increase" (default)
        means higher correlation = better. "decrease" means stronger
        anti-correlation = better (the correlation value is negated before
        delivery to the task).
    wm_mask_path : Path, optional
        Path to eroded WM binary mask. Resolved automatically during
        session setup (not set directly by user).
    csf_mask_path : Path, optional
        Path to eroded CSF binary mask. Resolved automatically during
        session setup (not set directly by user).
    """

    VALID_MOTION_CORRECTION = {"siemens_moco", "software", "none"}

    n_baseline: int = 25
    n_volumes: int = 150
    use_efficiency_weights: bool = True
    motion_correction: str = "siemens_moco"
    include_drift: bool = True
    include_motion_regressors: bool = True
    include_wm_regressor: bool = False
    include_csf_regressor: bool = False
    smooth_fwhm: float | None = None
    differential_pair: list[str] | None = None
    feedback_interval: int | None = None
    connectivity_pairs: list[list[str]] | None = None
    connectivity_window: int = 15
    feedback_mode: str = "activation"
    connectivity_target: str = "increase"
    wm_mask_path: Path | None = None
    csf_mask_path: Path | None = None

    def __post_init__(self) -> None:
        # Backward compatibility: migrate old config keys
        if hasattr(self, "moco_enabled"):
            self.motion_correction = "siemens_moco" if self.moco_enabled else "software"
            del self.moco_enabled
        if hasattr(self, "nuisance_regression"):
            if self.nuisance_regression:
                self.include_wm_regressor = True
                self.include_csf_regressor = True
            del self.nuisance_regression

        if self.motion_correction not in self.VALID_MOTION_CORRECTION:
            raise ValueError(
                f"motion_correction must be one of {self.VALID_MOTION_CORRECTION}, got '{self.motion_correction}'"
            )
        if self.wm_mask_path is not None:
            self.wm_mask_path = Path(self.wm_mask_path)
        if self.csf_mask_path is not None:
            self.csf_mask_path = Path(self.csf_mask_path)


@dataclass
class FeedbackRunConfig:
    """Configuration for a single feedback task type in a session.

    A session can have multiple feedback task types (e.g., triggered experience
    sampling + ball task), each with independent scan parameters, GLM settings,
    and feedback mode. Each ``FeedbackRunConfig`` is repeated for ``max_runs``
    iterations during the session.

    Parameters
    ----------
    task_type : str
        Registered task identifier (e.g., ``"ball"``, ``"thermometer"``).
    max_runs : int
        Number of feedback runs for this task type.
    tr : float
        Repetition time in seconds.
    n_volumes : int
        Total volumes per run.
    n_baseline : int
        Number of baseline volumes before task onset.
    motion_correction : str
        Real-time motion correction mode (``"siemens_moco"``, ``"software"``, ``"none"``).
    include_drift : bool
        Include linear drift regressor in GLM.
    include_motion_regressors : bool
        Include 6 motion parameters as GLM nuisance regressors.
    include_wm_regressor : bool
        Include white matter mean signal regressor.
    include_csf_regressor : bool
        Include CSF mean signal regressor.
    smooth_fwhm : float or None
        Gaussian spatial smoothing FWHM in mm. None disables.
    feedback_mode : str
        ``"activation"`` or ``"connectivity"``.
    target_mask : str or None
        Target mask name for single-network tasks.
    differential_pair : list of str or None
        [positive, negative] mask names for differential activation.
    connectivity_pairs : list of list of str or None
        Mask pairs for sliding-window correlation.
    connectivity_window : int
        Sliding window size in volumes.
    connectivity_target : str
        ``"increase"`` or ``"decrease"``.
    feedback_interval : int or None
        Volumes between feedback deliveries. None = continuous.
    fd_threshold : float
        Framewise displacement threshold.
    dvars_threshold : float
        DVARS threshold.
    tsnr_threshold : float
        tSNR threshold.
    use_transfer_runs : bool
        Whether to include transfer runs for this task.
    task_params : dict
        Task-specific parameters passed to ``create_task()``.
    network_assignment : dict
        Maps network role keys to mask names.
    sham : dict
        Sham configuration for this feedback task.
    source_dicoms : str
        Simulator DICOM source for feedback runs.
    transfer_source_dicoms : str
        Simulator DICOM source for transfer runs.
    reference_realtime_dicoms : str
        Simulator DICOM source for realtime reference.
    reference_realtime_enabled : bool
        Whether to collect realtime reference scan.
    reference_realtime_n_volumes : int
        Volumes for realtime reference acquisition.
    """

    task_type: str = "ball"
    max_runs: int = 1
    tr: float = 1.2
    n_volumes: int = 150
    n_baseline: int = 25

    # GLM settings
    motion_correction: str = "siemens_moco"
    include_drift: bool = True
    include_motion_regressors: bool = True
    include_wm_regressor: bool = False
    include_csf_regressor: bool = False
    smooth_fwhm: float | None = None

    # Feedback signal
    feedback_mode: str = "activation"
    target_mask: str | None = None
    differential_pair: list[str] | None = None
    connectivity_pairs: list[list[str]] | None = None
    connectivity_window: int = 15
    connectivity_target: str = "increase"
    feedback_interval: int | None = None

    # QC thresholds
    fd_threshold: float = 0.5
    dvars_threshold: float = 1.5
    tsnr_threshold: float = 20.0

    # Transfer runs
    use_transfer_runs: bool = True

    # Task-specific
    task_params: dict[str, Any] = field(default_factory=dict)
    network_assignment: dict[str, str] = field(default_factory=dict)

    # Sham
    sham: dict[str, Any] = field(default_factory=dict)

    # Simulator sources
    source_dicoms: str = ""
    transfer_source_dicoms: str = ""
    reference_realtime_dicoms: str = ""
    reference_realtime_enabled: bool = True
    reference_realtime_n_volumes: int = 2

    def __post_init__(self) -> None:
        valid_moco = {"siemens_moco", "software", "none"}
        if self.motion_correction not in valid_moco:
            raise ValueError(
                f"motion_correction must be one of {valid_moco}, got '{self.motion_correction}'"
            )


@dataclass
class TaskConfig:
    """Generic task configuration.

    Task-specific parameters are stored in task_params dict and passed to
    the task factory. Network assignments map role keys (declared in
    TaskMetadata.network_roles) to mask names.

    Parameters
    ----------
    task_type : str
        Registered task identifier (e.g., "ball", "thermometer")
    max_feedback_runs : int
        Number of feedback runs in session
    task_params : dict
        Task-specific parameters (passed to create_task())
    network_assignment : dict
        Maps network role keys to mask names.
        E.g., {"top_network": "ContA", "bottom_network": "DefaultA"}
    """

    task_type: str = "ball"
    max_feedback_runs: int = 2
    task_params: dict[str, Any] = field(default_factory=dict)
    network_assignment: dict[str, str] = field(default_factory=dict)


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
class SelectedAcquisitions:
    """Store user-selected run/acq combinations for reproducibility.

    This allows manual selection of which acquisition attempt to use for each run
    when multiple attempts exist (e.g., after interruptions/retries).
    """

    localizer: dict[int, int] | None = None  # run_num -> acq_num for localizer runs
    reference: dict[int, int] | None = None  # run_num -> acq_num for reference scans
    feedback: dict[str, dict[str, Any]] | None = None  # step_id -> {run, acq} for feedback runs

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
        Feedback session parameters
    task : TaskConfig
        BallTask parameters
    visit : VisitConfig, optional
        Visit-specific configuration for multi-visit studies.
        If None, uses single-visit mode (backward compatible).
    sham : ShamConfig, optional
        Sham/control condition configuration.
        If None or sham.enabled=False, all participants receive real feedback.

    Examples
    --------
    >>> # Single-visit mode (backward compatible)
    >>> config = SessionConfig(
    ...     paths=PathConfig(base_dir="/data/study", participant_id="sub-01"),
    ...     scan=ScanConfig(tr=1.2, feedback_dicoms="/data/dicoms/feedback"),
    ... )

    >>> # Multi-visit mode
    >>> config = SessionConfig(
    ...     paths=PathConfig(base_dir="/data/study", participant_id="sub-01", visit_id="ses-01"),
    ...     visit=VisitConfig(visit_id="ses-01", visit_number=1, mask_source="compute"),
    ... )

    >>> # Multi-visit with mask reuse
    >>> config = SessionConfig(
    ...     paths=PathConfig(base_dir="/data/study", participant_id="sub-01", visit_id="ses-02"),
    ...     visit=VisitConfig(visit_id="ses-02", visit_number=2, mask_source="reuse", mask_source_visit="ses-01"),
    ... )
    """

    paths: PathConfig
    scan: ScanConfig = field(default_factory=ScanConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    visit: VisitConfig | None = None
    sham: ShamConfig | None = None
    localizers: list[LocalizerConfig] = field(default_factory=list)
    feedback_runs: list[FeedbackRunConfig] = field(default_factory=list)

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

        scan_data = data.get("scan", {})
        scan = ScanConfig(**scan_data) if scan_data else ScanConfig()

        mask_data = data.get("mask", {})
        mask = MaskConfig(**mask_data) if mask_data else MaskConfig()

        feedback_data = data.get("feedback", {})
        feedback = FeedbackConfig(**feedback_data) if feedback_data else FeedbackConfig()

        task_data = data.get("task", {})
        task = TaskConfig(**task_data) if task_data else TaskConfig()

        visit_data = data.get("visit")
        visit = VisitConfig(**visit_data) if visit_data else None

        sham_data = data.get("sham")
        sham = ShamConfig(**sham_data) if sham_data else None

        # Parse localizers list
        localizers_data = data.get("localizers", [])
        localizers = [LocalizerConfig(**loc_data) for loc_data in localizers_data]

        # Parse feedback_runs list
        feedback_runs_data = data.get("feedback_runs", [])
        feedback_runs = [FeedbackRunConfig(**frc) for frc in feedback_runs_data]

        config = cls(
            paths=paths,
            scan=scan,
            mask=mask,
            feedback=feedback,
            task=task,
            visit=visit,
            sham=sham,
            localizers=localizers,
            feedback_runs=feedback_runs,
        )
        config.derive_combined_mask_names()
        return config

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
        result = {
            "paths": _path_to_str(asdict(self.paths)),
            "scan": _path_to_str(asdict(self.scan)),
            "mask": _path_to_str(asdict(self.mask)),
            "feedback": _path_to_str(asdict(self.feedback)),
            "task": _path_to_str(asdict(self.task)),
        }
        if self.visit is not None:
            result["visit"] = asdict(self.visit)
        if self.sham is not None:
            result["sham"] = _path_to_str(asdict(self.sham))
        if self.localizers:
            result["localizers"] = [_path_to_str(asdict(loc)) for loc in self.localizers]
        if self.feedback_runs:
            result["feedback_runs"] = [_path_to_str(asdict(frc)) for frc in self.feedback_runs]
        return result

    def derive_combined_mask_names(self) -> None:
        """Set ``mask.mask_names`` from the union of all localizer mask names.

        Called automatically when loading from dict/YAML/JSON. When building
        configs programmatically with a ``localizers`` list, call this after
        construction to keep ``mask.mask_names`` in sync.
        """
        if not self.localizers:
            return  # keep existing mask.mask_names (manual import or legacy)
        combined: list[str] = []
        for loc in self.localizers:
            combined.extend(loc.mask_names)
        self.mask.mask_names = combined

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

        if self.scan.tr <= 0:
            errors.append(f"scan.tr must be positive, got {self.scan.tr}")

        if self.scan.n_volumes <= 0:
            errors.append(f"scan.n_volumes must be positive, got {self.scan.n_volumes}")

        if self.mask.n_yeo not in (7, 17):
            errors.append(f"mask.n_yeo must be 7 or 17, got {self.mask.n_yeo}")

        if self.task.max_feedback_runs <= 0:
            errors.append(f"task.max_feedback_runs must be positive, got {self.task.max_feedback_runs}")

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
        if self.scan.simulator_source_mapping is not None:
            try:
                mapping = SimulatorSourceMapping.from_file(self.scan.simulator_source_mapping)
                source = mapping.get_source(self.paths.participant_id, run_type)
                if source is not None:
                    logger.debug(f"Resolved {run_type} for {self.paths.participant_id} from mapping: {source}")
                    return source
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load simulator source mapping: {e}")

        # Fall back to individual fields
        if run_type.startswith("rest"):
            # Extract run number (1-indexed)
            try:
                run_num = int(run_type[4:])
                if 0 < run_num <= len(self.scan.localizer_source_dicoms):
                    return self.scan.localizer_source_dicoms[run_num - 1]
            except (ValueError, IndexError):
                pass
        elif run_type == "reference_initial":
            return self.scan.reference_initial_dicoms or self.scan.reference_dicoms
        elif run_type == "reference_realtime":
            return self.scan.reference_realtime_dicoms or self.scan.reference_dicoms
        elif run_type.startswith("feedback_run-") or run_type.startswith("transfer_run-"):
            # All feedback/transfer runs fall back to feedback_dicoms
            return self.scan.feedback_dicoms

        return None

    def resolve_localizer_sources(self) -> list[Path]:
        """Resolve simulator DICOM sources for all localizer runs.

        This method resolves sources for rest1, rest2, ..., restN based on
        the configured number of localizer runs.

        Returns
        -------
        list[Path]
            List of resolved DICOM source paths for localizer runs
        """
        sources = []
        for i in range(self.scan.localizer_n_runs):
            source = self.resolve_simulator_source(f"rest{i + 1}")
            if source is not None:
                sources.append(source)
        return sources


# =============================================================================
# Helper Functions
# =============================================================================


def ref_file_path(config: SessionConfig, task: str = "rest", source_visit: str | None = None) -> Path:
    """Return the expected BOLD reference path based on config.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    task : str
        Task name (default: 'rest')
    source_visit : str, optional
        If specified, return ref path from this visit instead of current visit

    Returns
    -------
    Path
        Path to BOLD reference file
    """
    if source_visit:
        # Get ref from a different visit
        preprocessing_dir = config.paths.base_dir / config.paths.participant_id / source_visit / "preprocessing"
    else:
        assert config.paths.preprocessing_dir is not None
        preprocessing_dir = config.paths.preprocessing_dir
    return preprocessing_dir / f"{config.paths.participant_id}_task-{task}_boldref.nii.gz"


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
        pattern = re.compile(rf"{re.escape(task_name)}_run-(\d+)$")
        for entry in collection_dir.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    max_run = max(max_run, int(match.group(1)))
    return max_run + 1


def get_localizer_run_info(collection_dir: Path, task: str, run: int, expected_volumes: int) -> dict[str, Any]:
    """Get run info for localizer/reference with retry detection using acq entity.

    Scans for existing {task}_acq-*_run-{run:02d} folders and determines
    next acq number. Previous attempts are preserved in place.

    Parameters
    ----------
    collection_dir : Path
        Collection directory to scan
    task : str
        Task name (e.g., 'rest', 'reference', 'self_other')
    run : int
        Run number
    expected_volumes : int
        Expected number of volume files for a complete collection

    Returns
    -------
    dict
        Dictionary containing:
        - 'run': int - The run number (same as input)
        - 'acq': int - The acquisition number for this attempt (1, 2, 3, ...)
        - 'run_acq_string': str - Combined string like 'acq-01_run-01'
        - 'existing_acqs': list - List of existing acquisition numbers for this run

    Examples
    --------
    >>> info = get_localizer_run_info(collection_dir, 'task-rest', 1, 200)
    >>> info
    {'run': 1, 'acq': 1, 'run_acq_string': 'acq-01_run-01', 'existing_acqs': []}
    >>> # After one collection:
    >>> info = get_localizer_run_info(collection_dir, 'task-rest', 1, 200)
    >>> info
    {'run': 1, 'acq': 2, 'run_acq_string': 'acq-02_run-01', 'existing_acqs': [1]}
    """
    if not collection_dir.exists():
        return {
            "run": run,
            "acq": 1,
            "run_acq_string": f"acq-01_run-{run:02d}",
            "existing_acqs": [],
        }

    # Scan for existing collection folders for this run
    # Pattern: {task}_acq-*_run-{run:02d}/
    pattern = f"{task}_acq-*_run-{run:02d}"

    existing_acqs: list[int] = []

    # Check main collection directory
    for folder in collection_dir.glob(pattern):
        if folder.is_dir():
            match = re.search(r"acq-(\d+)_run-", folder.name)
            if match:
                existing_acqs.append(int(match.group(1)))

    # Remove duplicates and sort
    existing_acqs = sorted(set(existing_acqs))

    # Determine next acq number
    if not existing_acqs:
        acq = 1
    else:
        acq = max(existing_acqs) + 1

    return {
        "run": run,
        "acq": acq,
        "run_acq_string": f"acq-{acq:02d}_run-{run:02d}",
        "existing_acqs": existing_acqs,
    }


# =============================================================================
# Multi-Visit Support Functions
# =============================================================================


def resolve_mask_source(config: SessionConfig) -> tuple[list[Path], Path]:
    """Resolve mask file paths based on visit configuration.

    For visits that compute their own masks, returns paths from the current visit's
    mask_extraction directory. For visits that reuse masks from another visit,
    returns paths from the source visit.

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
    if config.visit is None or config.visit.mask_source == "compute":
        # Masks computed in this visit
        assert config.paths.mask_extraction_dir is not None
        mask_dir = config.paths.mask_extraction_dir / "03_masks"
        mask_files = sorted(mask_dir.glob("*_thr.nii.gz")) if mask_dir.exists() else []
        ref = ref_file_path(config)
        return (mask_files, ref)
    elif config.visit.mask_source == "reuse":
        # Masks from another visit
        assert config.visit.mask_source_visit is not None
        source_visit = config.visit.mask_source_visit
        source_base = config.paths.base_dir / config.paths.participant_id / source_visit / "mask_extraction"
        source_mask_dir = source_base / "03_masks"
        mask_files = sorted(source_mask_dir.glob("*_thr.nii.gz")) if source_mask_dir.exists() else []
        ref = ref_file_path(config, source_visit=source_visit)
        return (mask_files, ref)
    else:
        raise ValueError(f"Unknown mask_source: {config.visit.mask_source}")


def is_single_visit_data(participant_dir: Path) -> bool:
    """Check if participant data uses single-visit layout.

    Single-visit layout has mask_extraction/preprocessing directly in participant dir.
    Multi-visit layout has ses-* subdirectories.

    Parameters
    ----------
    participant_dir : Path
        Path to participant directory

    Returns
    -------
    bool
        True if single-visit layout, False if multi-visit
    """
    has_direct_masks = (participant_dir / "mask_extraction").exists()
    has_visit_dirs = (
        any(d.name.startswith("ses-") for d in participant_dir.iterdir() if d.is_dir())
        if participant_dir.exists()
        else False
    )
    return has_direct_masks and not has_visit_dirs


def detect_existing_visits(base_dir: Path, participant_id: str) -> list[dict[str, Any]]:
    """Detect existing visit directories for a participant.

    Parameters
    ----------
    base_dir : Path
        Study base directory
    participant_id : str
        Participant identifier

    Returns
    -------
    list[dict]
        List of visit info dicts with keys:
        - visit_id: str or None (None for single-visit mode)
        - has_masks: bool
        - has_feedback: bool
        - masks_approved: bool
    """
    participant_dir = base_dir / participant_id
    visits: list[dict[str, Any]] = []

    if not participant_dir.exists():
        return visits

    # Look for ses-* directories (multi-visit)
    for entry in sorted(participant_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("ses-"):
            me_dir = entry / "mask_extraction"
            visit_info = {
                "visit_id": entry.name,
                "has_masks": (me_dir / "03_masks").exists(),
                "has_feedback": (entry / "feedback" / "session_state.json").exists(),
                "masks_approved": (me_dir / ".masks_approved").exists(),
            }
            visits.append(visit_info)

    # Also check for single-visit mode (no ses-* directory)
    if is_single_visit_data(participant_dir):
        me_dir = participant_dir / "mask_extraction"
        visits.insert(
            0,
            {
                "visit_id": None,  # Single-visit mode
                "has_masks": (me_dir / "03_masks").exists(),
                "has_feedback": (participant_dir / "feedback" / "session_state.json").exists(),
                "masks_approved": (me_dir / ".masks_approved").exists(),
            },
        )

    return visits


def next_visit_id(base_dir: Path, participant_id: str) -> str:
    """Get the next visit ID for a participant.

    Parameters
    ----------
    base_dir : Path
        Study base directory
    participant_id : str
        Participant identifier

    Returns
    -------
    str
        Next visit ID (e.g., 'ses-01', 'ses-02')
    """
    visits = detect_existing_visits(base_dir, participant_id)
    # Filter to only multi-visit entries (those with actual visit_id)
    multi_visits = [v for v in visits if v["visit_id"] is not None]
    next_num = len(multi_visits) + 1
    return f"ses-{next_num:02d}"


def migrate_to_multi_visit(
    participant_dir: Path,
    target_visit_id: str = "ses-01",
) -> Path:
    """Migrate single-visit data to multi-visit structure.

    Moves existing data from participant directory into a visit subdirectory.

    Parameters
    ----------
    participant_dir : Path
        Path to participant directory with single-visit data
    target_visit_id : str
        Visit ID to migrate data to (default: 'ses-01')

    Returns
    -------
    Path
        Path to the new visit directory

    Raises
    ------
    ValueError
        If participant already uses multi-visit structure
    """
    participant_dir = Path(participant_dir)

    if not is_single_visit_data(participant_dir):
        raise ValueError("Participant already uses multi-visit structure or has no data")

    target_dir = participant_dir / target_visit_id
    target_dir.mkdir(exist_ok=True)

    # Move standard directories
    for dirname in ["collection", "preprocessing", "mask_extraction", "feedback", "qc"]:
        src = participant_dir / dirname
        if src.exists():
            shutil.move(str(src), str(target_dir / dirname))

    logger.info(f"Migrated {participant_dir} to {target_dir}")
    return target_dir


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
        If True, use DicomSimulator to feed DICOMs (for testing)
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
        n_volumes_list = config.scan.rest_n_volumes

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
        run_info = get_localizer_run_info(
            collection_dir=collection_dir, task=task_name, run=run_num, expected_volumes=n_volumes
        )
        acq = run_info["acq"]

        logger.info(f"Collecting {task_name} run {run_num} (acq-{acq:02d})...")
        if run_info["existing_acqs"]:
            logger.info(f"Previous attempts: {', '.join(f'acq-{a:02d}' for a in run_info['existing_acqs'])}")

        output_dir = collection_dir / f"{task_name}_acq-{acq:02d}_run-{run_num:02d}"

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

            qc_log_path = qc_log_dir / f"{task_name}_acq-{acq:02d}_run-{run_num:02d}_collectionlog.tsv"
            motion_mode = getattr(config.scan, "motion_correction", "siemens_moco")
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
                timeout=600.0,
                clear_watch_dir=True,
                clear_output_dir=True,
                simulator_source_dir=source_dir,
                simulator_speed=config.scan.collection_speed,
                qc_monitor=qc_mon,
            )
        else:
            watch_dir = collection_dir / "incoming"
            volume_folder = collect_volumes(
                watch_dir=watch_dir,
                output_dir=output_dir,
                n_volumes=n_volumes,
                timeout=600.0,
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
        ``"rnt"``). If None, inferred from ``config.scan.localizer_task_type``.

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
        task_type = config.scan.localizer_task_type
        task_name = task_type if task_type != "rest" else "rest"

    return preprocessing(
        participant_id=config.paths.participant_id,
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

    localizer_type = config.mask.localizer_type

    if localizer_type == "manual":
        from .mask_extraction import import_manual_masks

        logger.info("Using manual ROI import")
        manual_ref = config.mask.manual_ref_file or ref_file
        assert manual_ref is not None, "ref_file or manual_ref_file is required for manual import"
        network_files = import_manual_masks(
            participant_id=config.paths.participant_id,
            root_dir=str(config.paths.mask_extraction_dir),
            mask_paths=list(config.mask.manual_mask_paths),
            mask_names=config.mask.mask_names,
            ref_file=manual_ref,
            mask_space=config.mask.manual_mask_space,
            anat_file=config.mask.manual_anat_file,
            overwrite=overwrite,
        )
    elif localizer_type == "task":
        from .mask_extraction import mask_extraction_task

        assert preproc_file is not None and mask_file is not None and ref_file is not None
        logger.info("Using task-based (GLM) mask extraction")
        network_files = mask_extraction_task(
            participant_id=config.paths.participant_id,
            root_dir=str(config.paths.mask_extraction_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            events=config.mask.events_file,
            contrasts=config.mask.contrasts,
            tr=config.scan.localizer_tr,
            threshold=config.mask.threshold,
            threshold_method=config.mask.threshold_method,
            atlas_mask=config.mask.atlas_mask,
            atlas_name=config.mask.atlas_name,
            atlas_source=config.mask.atlas_source,
            atlas_regions=config.mask.atlas_regions,
            ref_file=ref_file,
            cluster_start_threshold=config.mask.cluster_start_threshold,
            cluster_step=config.mask.cluster_step,
            stat_type=config.mask.stat_type,
            smoothing_fwhm=config.mask.smooth_fwhm,
            high_pass=config.mask.highpass,
            drift_model=config.mask.drift_model,
            hrf_model=config.mask.hrf_model,
            include_reverse=config.mask.include_reverse_contrasts,
            overwrite=overwrite,
        )
    else:
        from .mask_extraction import mask_extraction_rest

        assert preproc_file is not None and mask_file is not None and ref_file is not None
        logger.info("Using ICA-based (resting-state) mask extraction")
        network_files = mask_extraction_rest(
            participant_id=config.paths.participant_id,
            root_dir=str(config.paths.mask_extraction_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            ref_file=ref_file,
            n_yeo=config.mask.n_yeo,
            yeo_targets=config.mask.yeo_targets,
            network_atlas=config.mask.network_atlas,
            smooth_fwhm=config.mask.smooth_fwhm,
            highpass=config.mask.highpass,
            lowpass=config.mask.lowpass,
            overwrite=overwrite,
        )

    # Create WM/CSF tissue masks for nuisance regression (if either is enabled)
    if config.feedback.include_wm_regressor or config.feedback.include_csf_regressor:
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
        network_files = mask_extraction_rest(
            participant_id=config.paths.participant_id,
            root_dir=str(loc_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            ref_file=ref_file,
            n_yeo=localizer.n_yeo,
            yeo_targets=localizer.yeo_targets,
            network_atlas=localizer.network_atlas,
            smooth_fwhm=localizer.smooth_fwhm,
            highpass=localizer.highpass,
            lowpass=localizer.lowpass,
            overwrite=overwrite,
        )
    else:
        from .mask_extraction import mask_extraction_task

        logger.info("Using task-based (GLM) mask extraction")
        network_files = mask_extraction_task(
            participant_id=config.paths.participant_id,
            root_dir=str(loc_dir),
            input_file=preproc_file,
            mask_file=mask_file,
            events=localizer.events_file,
            contrasts=localizer.contrasts,
            tr=localizer.tr,
            threshold=localizer.threshold,
            threshold_method=localizer.threshold_method,
            atlas_mask=localizer.atlas_mask,
            atlas_name=localizer.atlas_name,
            atlas_source=localizer.atlas_source,
            atlas_regions=localizer.atlas_regions,
            ref_file=ref_file,
            cluster_start_threshold=localizer.cluster_start_threshold,
            cluster_step=localizer.cluster_step,
            stat_type=localizer.stat_type,
            smoothing_fwhm=localizer.smooth_fwhm,
            high_pass=localizer.highpass,
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

    assert config.paths.mask_extraction_dir is not None
    reference_dir = reference_dicom_dir or config.resolve_simulator_source("reference_realtime")

    # Append tissue masks to the coregistration list if nuisance regression enabled
    all_files = list(network_files)
    n_network = len(network_files)
    tissue_indices: dict[str, int] = {}

    if config.feedback.include_wm_regressor or config.feedback.include_csf_regressor:
        assert config.paths.mask_extraction_dir is not None
        tissue_dir = config.paths.mask_extraction_dir / "05_tissue"
        wm_src = tissue_dir / f"{config.paths.participant_id}_tissue_wm_mask.nii.gz"
        csf_src = tissue_dir / f"{config.paths.participant_id}_tissue_csf_mask.nii.gz"
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
        root_dir=str(config.paths.mask_extraction_dir),
        input_dicom=str(reference_dir),
        network_files=all_files,
        ref_file=ref_file,
        overwrite=overwrite,
        output_subdir="04_reference",
    )

    # Split results: network masks returned to caller, tissue paths stored in config
    network_coreg = coreg_files[:n_network]
    if tissue_indices:
        config.feedback.wm_mask_path = coreg_files[tissue_indices["wm"]]
        config.feedback.csf_mask_path = coreg_files[tissue_indices["csf"]]
        logger.info(
            f"Tissue mask paths set: WM={config.feedback.wm_mask_path.name}, CSF={config.feedback.csf_mask_path.name}"
        )

    return network_coreg


def setup_visit_with_reused_masks(
    config: SessionConfig,
    reference_dicoms: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Set up a visit that reuses masks from a previous visit.

    This is used when config.visit.mask_source='reuse'. Only collects reference
    and coregisters masks from the source visit - skips localizer, preprocessing,
    and mask extraction.

    Parameters
    ----------
    config : SessionConfig
        Session configuration with visit.mask_source='reuse'
    reference_dicoms : Path
        Path to collected reference DICOMs for this visit
    overwrite : bool
        Whether to overwrite existing coregistration outputs

    Returns
    -------
    list[Path]
        Coregistered mask files in current visit's feedback/01_reference/masks/
    """
    from .mask_extraction import coregister_reference

    logger.info("")
    logger.info("=" * 60)
    logger.info("MASK REUSE: Coregistering masks from previous visit")
    logger.info("=" * 60)

    if config.visit is None or config.visit.mask_source != "reuse":
        raise ValueError("setup_visit_with_reused_masks requires config.visit.mask_source='reuse'")

    assert config.visit.mask_source_visit is not None
    source_visit = config.visit.mask_source_visit
    logger.info(f"Source visit: {source_visit}")

    # Resolve source masks and reference
    source_masks, source_ref = resolve_mask_source(config)

    if not source_masks:
        source_base = config.paths.base_dir / config.paths.participant_id / source_visit / "mask_extraction"
        raise FileNotFoundError(
            f"No masks found in source visit {source_visit}. Expected masks in: {source_base / '03_masks'}"
        )

    if not source_ref.exists():
        raise FileNotFoundError(f"Reference file not found in source visit {source_visit}: {source_ref}")

    logger.info(f"Found {len(source_masks)} masks from {source_visit}")
    logger.info(f"Reference file: {source_ref}")

    # Coregister source masks to current visit's reference
    # Output goes to session_dir (feedback/) instead of mask_extraction_dir
    assert config.paths.session_dir is not None
    coreg_masks = coregister_reference(
        participant_id=config.paths.participant_id,
        root_dir=str(config.paths.session_dir),  # feedback/ directory
        input_dicom=str(reference_dicoms),
        network_files=source_masks,
        ref_file=source_ref,
        output_subdir="01_reference",
        overwrite=overwrite,
    )

    # Copy mask_selections.json from source visit for reference
    source_selections = source_masks[0].parent / "mask_selections.json"
    dest_dir = config.paths.session_dir / "01_reference"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if source_selections.exists():
        dest_selections = dest_dir / "mask_selections.json"
        shutil.copy2(source_selections, dest_selections)
        logger.info("Copied mask selections from source visit")

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

    This should be run once per participant/visit before the feedback session.
    For visits with mask_source='reuse', skips localizer/preprocessing/extraction
    and only runs reference collection + coregistration.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    overwrite : bool
        If True, rerun all steps even if outputs exist
    use_simulator : bool
        If True, use DicomSimulator for DICOM collection (testing mode)
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

    # Check if this is a mask-reuse visit
    if config.visit and config.visit.mask_source == "reuse":
        logger.info(f"Visit {config.visit.visit_id}: Reusing masks from {config.visit.mask_source_visit}")

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
                n_volumes_list=[config.scan.reference_n_volumes],
                start_run=1,
                max_runs=1,
            )
            reference_dicoms = reference_list[0]

        return setup_visit_with_reused_masks(config, reference_dicoms, overwrite=overwrite)

    # Manual ROI import: skip localizer + preprocessing, just import + coregister
    if config.mask.localizer_type == "manual":
        if config.visit:
            logger.info(f"Visit {config.visit.visit_id}: Manual ROI import")

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
                n_volumes_list=[config.scan.reference_n_volumes],
                start_run=None,
                max_runs=config.scan.reference_max_runs,
            )

        # Step 1: Import manual masks (no localizer or preprocessing needed)
        assert config.mask.manual_ref_file is not None, "manual_ref_file required for manual localizer"
        ref_file = config.mask.manual_ref_file
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
    if config.visit:
        logger.info(f"Visit {config.visit.visit_id}: Computing new masks")

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
            n_volumes_list=[config.scan.reference_n_volumes],
            start_run=None,
            max_runs=config.scan.reference_max_runs,
        )

    # Multi-localizer pipeline: collect, preprocess, extract for each localizer
    if config.localizers:
        all_network_files: list[Path] = []
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
            all_network_files.extend(network_files)

        # Create WM/CSF tissue masks (shared, using last ref_file)
        assert last_ref_file is not None
        if config.feedback.include_wm_regressor or config.feedback.include_csf_regressor:
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
                inv_transform_files=inv_xfm,
                overwrite=overwrite,
            )

        # Coregistration (all masks together)
        coreg_mask_files = run_coregistration(
            config,
            all_network_files,
            last_ref_file,
            reference_dicom_dir=initial_reference[0],
            overwrite=overwrite,
        )
    else:
        # Legacy single-localizer path (no localizers list configured)
        localizer_sources = config.resolve_localizer_sources()
        if skip_collection:
            logger.info("Skipping localizer collection (using source DICOMs directly)")
            rest_dicoms = localizer_sources
        else:
            rest_dicoms = run_dicom_collection(
                config,
                use_simulator=use_simulator,
                task_name="rest",
                source_dirs=localizer_sources,
                n_volumes_list=config.scan.rest_n_volumes,
                start_run=1,
                max_runs=config.scan.rest_max_runs,
            )

        preproc_file, mask_file, ref_file = run_preprocessing(
            config,
            rest_dicoms=rest_dicoms,
            overwrite=overwrite,
        )
        network_files = run_mask_extraction(config, preproc_file, mask_file, ref_file, overwrite)

        coreg_mask_files = run_coregistration(
            config,
            network_files,
            ref_file,
            reference_dicom_dir=initial_reference[0],
            overwrite=overwrite,
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
    session_state: SessionState,
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
    session_state : SessionState
        Session state for multi-run continuity
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
    from .feedback import run_feedback_session
    from .tasks.factory import create_task
    from .tasks.task_registry import get_task_metadata

    # Resolve effective task type and parameters from run_config or flat config
    if run_config is not None:
        eff_task_type = run_config.task_type
        eff_task_params = run_config.task_params
        eff_network_assignment = run_config.network_assignment
        eff_n_baseline = run_config.n_baseline
        eff_n_volumes = run_config.n_volumes
        eff_tr = run_config.tr
        eff_motion_correction = run_config.motion_correction
        eff_include_drift = run_config.include_drift
        eff_include_motion_regressors = run_config.include_motion_regressors
        eff_include_wm_regressor = run_config.include_wm_regressor
        eff_include_csf_regressor = run_config.include_csf_regressor
        eff_smooth_fwhm = run_config.smooth_fwhm
        eff_differential_pair = run_config.differential_pair
        eff_feedback_interval = run_config.feedback_interval
        eff_connectivity_pairs = run_config.connectivity_pairs
        eff_connectivity_window = run_config.connectivity_window
    else:
        eff_task_type = config.task.task_type
        eff_task_params = config.task.task_params
        eff_network_assignment = config.task.network_assignment
        eff_n_baseline = config.feedback.n_baseline
        eff_n_volumes = config.feedback.n_volumes
        eff_tr = config.scan.feedback_tr
        eff_motion_correction = config.feedback.motion_correction
        eff_include_drift = config.feedback.include_drift
        eff_include_motion_regressors = config.feedback.include_motion_regressors
        eff_include_wm_regressor = config.feedback.include_wm_regressor
        eff_include_csf_regressor = config.feedback.include_csf_regressor
        eff_smooth_fwhm = config.feedback.smooth_fwhm
        eff_differential_pair = config.feedback.differential_pair
        eff_feedback_interval = config.feedback.feedback_interval
        eff_connectivity_pairs = config.feedback.connectivity_pairs
        eff_connectivity_window = config.feedback.connectivity_window

    run_number = session_state.next_run_number
    task_run_number = session_state.next_run_number_for_type(run_type)
    feedback_mode = run_type == "feedback"

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RUN {run_number}: {run_type.upper()} (task run {task_run_number})")
    logger.info("=" * 60)
    logger.info(f"Task: {eff_task_type}")
    logger.info(f"Mode: {'FEEDBACK' if feedback_mode else 'TRANSFER (no feedback)'}")
    logger.info(f"Task state: {session_state.task_state}")

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
            "simulator_speed": config.scan.feedback_simulator_speed,
            "simulator_clear_target": True,
            "simulator_cleanup": False,
        }

    # Resolve mask names from network assignment
    task_mask_names = _resolve_mask_names_for_run(
        config, metadata, eff_network_assignment
    )

    assert config.paths.feedback_output_dir is not None
    assert config.paths.collection_dir is not None
    # Start feedback session
    session_info = run_feedback_session(
        mask_paths=list(mask_files),
        subject=config.paths.participant_id,
        task="feedback" if feedback_mode else "transfer",
        run=task_run_number,
        output_dir=config.paths.feedback_output_dir,
        collection_dir=config.paths.collection_dir,
        n_baseline=eff_n_baseline,
        n_volumes=eff_n_volumes,
        tr=eff_tr,
        task_regressors=task_regressors,
        mask_names=config.mask.mask_names,
        use_efficiency_weights=config.feedback.use_efficiency_weights,
        motion_correction=eff_motion_correction,
        include_drift=eff_include_drift,
        include_motion_regressors=eff_include_motion_regressors,
        include_wm_regressor=eff_include_wm_regressor,
        include_csf_regressor=eff_include_csf_regressor,
        smooth_fwhm=eff_smooth_fwhm,
        differential_pair=eff_differential_pair,
        feedback_interval=eff_feedback_interval,
        wm_mask_path=config.feedback.wm_mask_path,
        csf_mask_path=config.feedback.csf_mask_path,
        connectivity_pairs=eff_connectivity_pairs,
        connectivity_window=eff_connectivity_window,
        **simulator_kwargs,
    )

    coordinator = session_info["coordinator"]

    # Determine if this is the first run of this type
    completed_of_type = sum(1 for r in session_state.completed_runs if r.run_type == run_type)
    _is_first_of_type = completed_of_type == 0

    # Build config for task factory
    factory_config = {
        **eff_task_params,
        "mask_names": task_mask_names,
        "feedback_mode": feedback_mode,
    }

    # Create task via factory
    task = create_task(
        task_id=eff_task_type,
        coordinator=coordinator,
        config=factory_config,
        session_state=session_state,
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

    n_baseline = run_config.n_baseline if run_config else config.feedback.n_baseline
    n_volumes = run_config.n_volumes if run_config else config.feedback.n_volumes
    tr = run_config.tr if run_config else config.scan.feedback_tr

    if metadata.design_type == "nuisance_only":
        return None
    elif metadata.design_type == "block":
        regressors = compute_boxcar_task_regressor(
            n_baseline=n_baseline,
            n_volumes=n_volumes,
            tr=tr,
        )
        logger.info(
            f"Computed boxcar task regressor (baseline={n_baseline}, volumes={n_volumes})"
        )
        return regressors
    else:
        logger.warning(f"Unknown design_type '{metadata.design_type}', using no task regressors")
        return None


def _resolve_mask_names(config: SessionConfig, metadata: Any) -> list[str]:
    """Resolve ordered mask names from network_assignment or defaults.

    Parameters
    ----------
    config : SessionConfig
        Session configuration with task.network_assignment and mask.mask_names
    metadata : TaskMetadata
        Task metadata with network_roles and n_networks

    Returns
    -------
    list[str]
        Ordered mask names for this task
    """
    if config.task.network_assignment and metadata.network_roles:
        mask_names = []
        for i, role in enumerate(metadata.network_roles):
            name = config.task.network_assignment.get(role.key)
            if name:
                mask_names.append(name)
            elif i < len(config.mask.mask_names):
                mask_names.append(config.mask.mask_names[i])
        if len(mask_names) == metadata.n_networks:
            return mask_names
    # Fall back to first N mask names from config
    return config.mask.mask_names[: metadata.n_networks]


def _resolve_mask_names_for_run(
    config: SessionConfig,
    metadata: Any,
    network_assignment: dict[str, str],
) -> list[str]:
    """Resolve ordered mask names using an explicit network assignment.

    Like :func:`_resolve_mask_names` but uses *network_assignment* directly
    instead of ``config.task.network_assignment``, so per-run overrides work.
    """
    if network_assignment and metadata.network_roles:
        mask_names = []
        for i, role in enumerate(metadata.network_roles):
            name = network_assignment.get(role.key)
            if name:
                mask_names.append(name)
            elif i < len(config.mask.mask_names):
                mask_names.append(config.mask.mask_names[i])
        if len(mask_names) == metadata.n_networks:
            return mask_names
    # Fall back to first N mask names from config
    return config.mask.mask_names[: metadata.n_networks]


def coregister_before_first_transfer(
    config: SessionConfig,
    session_state: SessionState,
    mask_files: list[Path],
    use_simulator: bool = True,
) -> list[Path]:
    """Acquire a second reference and re-coregister masks right before the first transfer.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    session_state : SessionState
        Session state (used to check if any runs completed)
    mask_files : list[Path]
        Current mask files
    use_simulator : bool
        If True, use DICOM simulator for testing

    Returns
    -------
    list[Path]
        Updated coregistered mask files
    """
    if session_state.completed_runs:
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
        n_volumes_list=[config.scan.reference_n_volumes],
        start_run=None,
        max_runs=config.scan.reference_max_runs,
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
        from .mask_extraction import generate_reference_qc_overlays

        assert config.paths.mask_extraction_dir is not None
        _run_match = re.search(r"run-(\d+)", coreg_mask_files[0].stem) if coreg_mask_files else None
        run_str = _run_match.group(1) if _run_match else "01"
        generate_reference_qc_overlays(
            participant_id=config.paths.participant_id,
            session_dir=config.paths.mask_extraction_dir,  # reference outputs are in mask_extraction/04_reference
            ref_file=ref_file,
            mask_extraction_dir=config.paths.mask_extraction_dir,
            network_coreg_files=coreg_mask_files,
            run_str=run_str,
            n_yeo=config.mask.n_yeo,
            overwrite=True,
            reference_subdir="04_reference",  # Match the output_subdir used in coregister_reference
        )
    except Exception as e:
        logger.warning(f"Reference QC overlay generation skipped: {e}")

    return [Path(f) for f in coreg_mask_files]


# =============================================================================
# Session Workflow
# =============================================================================


def _build_run_sequence(
    config: SessionConfig, n_feedback_runs: int
) -> list[tuple[str, int]]:
    """Build the run sequence from ``feedback_runs`` or legacy flat config.

    Each entry is ``(run_type, frc_index)`` where *frc_index* is the index
    into ``config.feedback_runs`` that owns this step.  For legacy configs
    with a single implicit feedback task, *frc_index* is always ``0``.

    Parameters
    ----------
    config : SessionConfig
        Session configuration (``feedback_runs`` is used when populated)
    n_feedback_runs : int
        Fallback number of feedback runs (used only when ``feedback_runs``
        is empty or has a single entry without ``max_runs`` override).

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
    n_feedback_runs: int = 2,
    reset_session: bool = False,
    use_simulator: bool = True,
) -> list[dict[str, Any]]:
    """Run a complete multi-run session.

    Workflow depends on task config: with transfer runs it's
    Transfer -> Feedback x N -> Transfer; without it's just Feedback x N.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    mask_files : list[Path]
        Paths to coregistered mask files
    n_feedback_runs : int
        Number of feedback runs between transfer runs
    reset_session : bool
        If True, reset session state before starting
    use_simulator : bool
        If True, use DICOM simulator for testing

    Returns
    -------
    list[dict]
        Results from each run
    """
    from .tasks.session import SessionState

    run_sequence = _build_run_sequence(config, n_feedback_runs)
    # Derive total feedback run count from sequence
    total_feedback = sum(1 for rt, _ in run_sequence if rt == "feedback")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING MULTI-RUN SESSION")
    logger.info("=" * 60)
    if config.feedback_runs:
        task_types = [frc.task_type for frc in config.feedback_runs]
        logger.info(f"Tasks: {', '.join(task_types)}")
    else:
        logger.info(f"Task: {config.task.task_type}")
    workflow_labels = [rt.capitalize() for rt, _ in run_sequence]
    logger.info(f"Workflow: {' -> '.join(workflow_labels)}")

    assert config.paths.session_dir is not None
    session_dir = config.paths.session_dir
    if reset_session:
        session_state = SessionState.reset(session_dir, task_id=config.task.task_type)
        session_state.max_feedback_runs = total_feedback
        session_state.save()
    else:
        session_state = SessionState.load(session_dir, task_id=config.task.task_type)
        if session_state.max_feedback_runs != total_feedback:
            session_state.max_feedback_runs = total_feedback
            session_state.save()

    mask_files = [Path(f) for f in mask_files]

    all_results = []

    completed_count = len(session_state.completed_runs)
    total_runs = len(run_sequence)

    logger.info(f"Completed runs: {completed_count}")
    remaining_labels = [rt for rt, _ in run_sequence[completed_count:]]
    logger.info(f"Remaining runs: {remaining_labels}")

    for i, (run_type, frc_idx) in enumerate(run_sequence[completed_count:], start=completed_count):
        logger.info("")
        logger.info(f">>> Starting run {i + 1}/{total_runs} ({run_type})")

        if run_type == "transfer" and not session_state.completed_runs:
            mask_files = coregister_before_first_transfer(
                config, session_state, mask_files, use_simulator=use_simulator
            )

        frc = config.feedback_runs[frc_idx] if config.feedback_runs else None
        results = run_single_task(
            config=config,
            mask_files=mask_files,
            session_state=session_state,
            run_type=run_type,
            use_simulator=use_simulator,
            run_config=frc,
        )
        all_results.append(results)

        session_state = SessionState.load(session_dir, task_id=config.task.task_type)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total runs: {len(session_state.completed_runs)}")
    logger.info(f"Final task state: {session_state.task_state}")

    return all_results


def run_next_task(
    config: SessionConfig,
    mask_files: list[Path],
    n_feedback_runs: int = 2,
    use_simulator: bool = True,
) -> dict[str, Any] | None:
    """Run the next task in the session sequence.

    Loads session state, determines the next run type, executes it, and exits.
    Call this repeatedly (separate script invocations) to progress through the session.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    mask_files : list[Path]
        Paths to coregistered mask files
    n_feedback_runs : int
        Number of feedback runs between transfer runs
    use_simulator : bool
        If True, use DICOM simulator for testing

    Returns
    -------
    dict or None
        Run results, or None if session is complete
    """
    from .tasks.session import SessionState

    assert config.paths.session_dir is not None
    session_dir = config.paths.session_dir
    session_state = SessionState.load(session_dir, task_id=config.task.task_type)

    mask_files = [Path(f) for f in mask_files]

    run_sequence = _build_run_sequence(config, n_feedback_runs)
    total_feedback = sum(1 for rt, _ in run_sequence if rt == "feedback")

    if session_state.max_feedback_runs != total_feedback:
        session_state.max_feedback_runs = total_feedback
        session_state.save()

    total_runs = len(run_sequence)
    completed_count = len(session_state.completed_runs)

    if completed_count >= total_runs:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SESSION ALREADY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total runs: {completed_count}")
        logger.info(f"Final task state: {session_state.task_state}")
        logger.info("Use reset_session_state() to start a new session.")
        return None

    run_type, frc_idx = run_sequence[completed_count]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"NEXT RUN: {completed_count + 1}/{total_runs} ({run_type.upper()})")
    logger.info("=" * 60)
    logger.info(f"Completed: {[r.run_type for r in session_state.completed_runs]}")
    remaining_labels = [rt for rt, _ in run_sequence[completed_count:]]
    logger.info(f"Remaining: {remaining_labels}")

    if run_type == "transfer" and not session_state.completed_runs:
        mask_files = coregister_before_first_transfer(config, session_state, mask_files, use_simulator=use_simulator)

    frc = config.feedback_runs[frc_idx] if config.feedback_runs else None
    results = run_single_task(
        config=config,
        mask_files=mask_files,
        session_state=session_state,
        run_type=run_type,
        use_simulator=use_simulator,
        run_config=frc,
    )

    return results


def reset_session_state(config: SessionConfig) -> SessionState:
    """Reset session state for a new session.

    Parameters
    ----------
    config : SessionConfig
        Session configuration

    Returns
    -------
    SessionState
        Fresh session state
    """
    from .tasks.session import SessionState

    assert config.paths.session_dir is not None
    return SessionState.reset(config.paths.session_dir, task_id=config.task.task_type)
