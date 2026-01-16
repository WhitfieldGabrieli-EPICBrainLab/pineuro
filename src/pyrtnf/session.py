"""
Session configuration and orchestration for multi-run neurofeedback workflows.

Summary
-------
This module provides type-safe configuration dataclasses and high-level orchestration
functions for running neurofeedback sessions. It separates configuration from execution,
making it suitable for both scripted usage and future GUI integration.

Components
----------
Configuration Dataclasses:
- PathConfig: Paths for session directories
- ScanConfig: Scanner/acquisition parameters
- MaskConfig: Mask extraction parameters
- FeedbackConfig: Feedback session parameters
- TaskConfig: BallTask parameters
- SessionConfig: Top-level config combining all above

Orchestration Functions:
- setup_session(): Run all setup steps (collection, preprocessing, mask extraction)
- run_single_task(): Run one transfer/feedback task
- run_full_session(): Run complete workflow (Transfer -> Feedback x N -> Transfer)
- run_next_task(): Run next task in sequence (for incremental progression)
- reset_session_state(): Reset session for new participant

Workflow
--------
1. Create or load SessionConfig
2. Call setup_session() to prepare masks
3. Call run_next_task() repeatedly to progress through session

Dependencies
------------
dataclasses, pathlib, json, yaml (optional), logging

Notes
-----
This module works alongside tasks/session.py which contains SessionState for
persisting task-level state (hits, scale factors) between runs.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class PathConfig:
    """Path configuration for session directories.

    Parameters
    ----------
    base_dir : Path
        Base directory for all session data
    participant_id : str
        Participant identifier (e.g., 'sub-01')
    session_dir : Path, optional
        Directory for session state and feedback outputs. Defaults to base_dir/feedback
    preprocessing_dir : Path, optional
        Directory for preprocessing outputs. Defaults to base_dir/preprocessing
    mask_extraction_dir : Path, optional
        Directory for mask extraction outputs. Defaults to base_dir/mask_extraction
    feedback_output_dir : Path, optional
        Directory for feedback run outputs. Defaults to session_dir/02_feedback_output
    collection_dir : Path, optional
        Directory for DICOM collection. Defaults to base_dir/collection
    """
    base_dir: Path
    participant_id: str
    session_dir: Optional[Path] = None
    preprocessing_dir: Optional[Path] = None
    mask_extraction_dir: Optional[Path] = None
    feedback_output_dir: Optional[Path] = None
    collection_dir: Optional[Path] = None

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        if self.session_dir is None:
            self.session_dir = self.base_dir / "feedback"
        else:
            self.session_dir = Path(self.session_dir)
        if self.preprocessing_dir is None:
            self.preprocessing_dir = self.base_dir / "preprocessing"
        else:
            self.preprocessing_dir = Path(self.preprocessing_dir)
        if self.mask_extraction_dir is None:
            self.mask_extraction_dir = self.base_dir / "mask_extraction"
        else:
            self.mask_extraction_dir = Path(self.mask_extraction_dir)
        if self.feedback_output_dir is None:
            self.feedback_output_dir = self.session_dir / "02_feedback_output"
        else:
            self.feedback_output_dir = Path(self.feedback_output_dir)
        if self.collection_dir is None:
            self.collection_dir = self.base_dir / "collection"
        else:
            self.collection_dir = Path(self.collection_dir)


@dataclass
class ScanConfig:
    """Scanner/acquisition parameters.

    Parameters
    ----------
    tr : float
        Repetition time in seconds
    n_volumes : int
        Number of volumes per feedback run
    n_baseline : int
        Number of baseline volumes before task
    rest_source_dicoms : list[Path]
        Source DICOM folders for resting-state runs (for simulator)
    rest_n_volumes : list[int]
        Number of volumes per resting-state run
    rest_max_runs : int
        Maximum number of resting-state runs
    twovol_dicoms : Path, optional
        Default 2vol DICOM source
    twovol_initial_dicoms : Path, optional
        2vol DICOM source for initial (pre-scan) acquisition
    twovol_realtime_dicoms : Path, optional
        2vol DICOM source for real-time (day-of) acquisition
    twovol_n_volumes : int
        Number of volumes in 2vol acquisition
    twovol_max_runs : int
        Maximum number of 2vol acquisitions
    feedback_dicoms : Path, optional
        Source DICOM folder for feedback runs (for simulator)
    collection_speed : float
        Speed multiplier for DICOM collection simulation
    simulator_speed : float
        Speed multiplier for feedback DICOM simulation
    """
    tr: float = 1.2
    n_volumes: int = 150
    n_baseline: int = 25
    rest_source_dicoms: list[Path] = field(default_factory=list)
    rest_n_volumes: list[int] = field(default_factory=lambda: [200, 200])
    rest_max_runs: int = 2
    twovol_dicoms: Optional[Path] = None
    twovol_initial_dicoms: Optional[Path] = None
    twovol_realtime_dicoms: Optional[Path] = None
    twovol_n_volumes: int = 2
    twovol_max_runs: int = 2
    feedback_dicoms: Optional[Path] = None
    collection_speed: float = 10.0
    simulator_speed: float = 1.0

    def __post_init__(self):
        self.rest_source_dicoms = [Path(p) for p in self.rest_source_dicoms]
        if self.twovol_dicoms is not None:
            self.twovol_dicoms = Path(self.twovol_dicoms)
        if self.twovol_initial_dicoms is not None:
            self.twovol_initial_dicoms = Path(self.twovol_initial_dicoms)
        if self.twovol_realtime_dicoms is not None:
            self.twovol_realtime_dicoms = Path(self.twovol_realtime_dicoms)
        if self.feedback_dicoms is not None:
            self.feedback_dicoms = Path(self.feedback_dicoms)


@dataclass
class MaskConfig:
    """Mask extraction parameters.

    Parameters
    ----------
    n_yeo : int
        Number of Yeo networks (7 or 17)
    yeo_targets : list[str]
        Target network names to extract
    smooth_fwhm : float
        Spatial smoothing FWHM in mm
    highpass : float
        High-pass filter cutoff in Hz
    lowpass : float, optional
        Low-pass filter cutoff in Hz
    mask_names : list[str]
        Names for masks in feedback output
    """
    n_yeo: int = 17
    yeo_targets: list[str] = field(default_factory=lambda: ["DefaultA", "ContA"])
    smooth_fwhm: float = 5.0
    highpass: float = 0.01
    lowpass: Optional[float] = None
    mask_names: list[str] = field(default_factory=lambda: ["DefaultA", "ContA"])


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
    """
    n_baseline: int = 25
    n_volumes: int = 150
    use_efficiency_weights: bool = True


@dataclass
class TaskConfig:
    """BallTask parameters.

    Parameters
    ----------
    initial_scale_factor : float
        Initial scale factor for ball movement
    fullscreen : bool
        Whether to run task in fullscreen mode
    trigger_keys : list[str]
        Scanner trigger key codes
    left_button : str
        Left slider button
    right_button : str
        Right slider button
    enter_button : str
        Confirm/enter button
    min_hits_threshold : int
        If total hits < this, increase scale_factor (ball too slow)
    max_hits_threshold : int
        If either network > this, decrease scale_factor (ball too fast)
    scale_adjust_factor : float
        Fraction to adjust scale_factor by (0.25 = 25%)
    max_feedback_runs : int
        Total number of feedback runs in session
    """
    initial_scale_factor: float = 10.0
    fullscreen: bool = True
    trigger_keys: list[str] = field(default_factory=lambda: ['t', '+', '5', 'equal'])
    left_button: str = '1'
    right_button: str = '2'
    enter_button: str = '0'
    min_hits_threshold: int = 3
    max_hits_threshold: int = 5
    scale_adjust_factor: float = 0.25
    max_feedback_runs: int = 2


def _path_to_str(obj):
    """Convert Path objects to strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_path_to_str(v) for v in obj]
    return obj


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

    Examples
    --------
    >>> config = SessionConfig(
    ...     paths=PathConfig(base_dir="/data/sub-01", participant_id="sub-01"),
    ...     scan=ScanConfig(tr=1.2, feedback_dicoms="/data/dicoms/feedback"),
    ... )
    >>> config.to_yaml("/data/sub-01/config.yaml")
    >>> config = SessionConfig.from_yaml("/data/sub-01/config.yaml")
    """
    paths: PathConfig
    scan: ScanConfig = field(default_factory=ScanConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "SessionConfig":
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: Path) -> "SessionConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SessionConfig":
        """Create config from dictionary."""
        paths_data = data.get('paths', {})
        paths = PathConfig(**paths_data)

        scan_data = data.get('scan', {})
        scan = ScanConfig(**scan_data) if scan_data else ScanConfig()

        mask_data = data.get('mask', {})
        mask = MaskConfig(**mask_data) if mask_data else MaskConfig()

        feedback_data = data.get('feedback', {})
        feedback = FeedbackConfig(**feedback_data) if feedback_data else FeedbackConfig()

        task_data = data.get('task', {})
        task = TaskConfig(**task_data) if task_data else TaskConfig()

        return cls(paths=paths, scan=scan, mask=mask, feedback=feedback, task=task)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML configs: pip install pyyaml")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)

    def _to_dict(self) -> dict:
        """Convert to dictionary with Path objects as strings."""
        return {
            'paths': _path_to_str(asdict(self.paths)),
            'scan': _path_to_str(asdict(self.scan)),
            'mask': _path_to_str(asdict(self.mask)),
            'feedback': _path_to_str(asdict(self.feedback)),
            'task': _path_to_str(asdict(self.task)),
        }

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


# =============================================================================
# Helper Functions
# =============================================================================

def ref_file_path(config: SessionConfig, task: str = "rest") -> Path:
    """Return the expected BOLD reference path based on config.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    task : str
        Task name (default: 'rest')

    Returns
    -------
    Path
        Path to BOLD reference file
    """
    return config.paths.preprocessing_dir / f"{config.paths.participant_id}_task-{task}_boldref.nii.gz"


def get_mask_run_number(mask_files: list[Path]) -> Optional[int]:
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
        Task name prefix (e.g., 'rest', '2vol')

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


# =============================================================================
# Setup Functions (run once per participant)
# =============================================================================

def run_dicom_collection(
    config: SessionConfig,
    use_simulator: bool = True,
    task_name: str = "rest",
    source_dirs: Optional[list[Path]] = None,
    n_volumes_list: Optional[list[int]] = None,
    start_run: Optional[int] = None,
    max_runs: Optional[int] = None,
) -> list[Path]:
    """Collect DICOMs for a task (rest, 2vol, etc.) using scanner or simulator.

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

    Returns
    -------
    list[Path]
        Paths to collected DICOM folders (one per run)
    """
    from .preprocessing import collect_dicoms
    from .utils import DICOM_EXTENSIONS

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"DICOM COLLECTION ({task_name})")
    logger.info("=" * 60)

    collection_dir = config.paths.collection_dir
    if source_dirs is None:
        source_dirs = config.scan.rest_source_dicoms
    if n_volumes_list is None:
        n_volumes_list = config.scan.rest_n_volumes

    if len(source_dirs) != len(n_volumes_list):
        raise ValueError(f"Mismatch: {len(source_dirs)} source dirs vs {len(n_volumes_list)} n_volumes entries")

    collected_folders = []

    start_run_candidate = next_collection_run_number(collection_dir, task_name) if start_run is None else start_run
    if max_runs is not None and start_run_candidate > max_runs:
        start_run_candidate = max_runs
    start_run = start_run_candidate

    for run_idx, (source_dir, n_volumes) in enumerate(zip(source_dirs, n_volumes_list)):
        run_num = start_run + run_idx
        if max_runs is not None and run_num > max_runs:
            logger.info(f"Stopping {task_name} collection at run {run_num}: exceeds max_runs={max_runs}")
            break

        logger.info(f"Collecting {task_name} run {run_num}...")

        output_dir = collection_dir / f"{task_name}_run-{run_num:02d}"

        # Skip if simulator outputs already present
        if use_simulator and output_dir.exists():
            existing_dicoms = []
            for ext in DICOM_EXTENSIONS:
                existing_dicoms.extend(output_dir.glob(ext))
            if len(existing_dicoms) >= n_volumes:
                logger.info(
                    f"Skipping simulator collection for run {run_num}: {len(existing_dicoms)} files already exist in {output_dir}"
                )
                collected_folders.append(output_dir)
                continue

        if use_simulator:
            watch_dir = collection_dir / "incoming"
            dicom_folder = collect_dicoms(
                watch_dir=watch_dir,
                output_dir=output_dir,
                n_volumes=n_volumes,
                timeout=600.0,
                clear_watch_dir=True,
                clear_output_dir=True,
                simulator_source_dir=source_dir,
                simulator_speed=config.scan.collection_speed,
            )
        else:
            watch_dir = collection_dir / "incoming"
            dicom_folder = collect_dicoms(
                watch_dir=watch_dir,
                output_dir=output_dir,
                n_volumes=n_volumes,
                timeout=600.0,
                clear_watch_dir=True,
                clear_output_dir=True,
            )

        collected_folders.append(dicom_folder)
        logger.info(f"  Collected {n_volumes} volumes to {dicom_folder}")

    logger.info(f"Collection complete: {len(collected_folders)} runs")
    return collected_folders


def run_preprocessing(
    config: SessionConfig,
    rest_dicoms: Optional[list[Path]] = None,
    overwrite: bool = False
) -> tuple[Path, Path, Path]:
    """Run preprocessing on rest scan.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    rest_dicoms : list[Path], optional
        Paths to DICOM folders. If None, uses config.scan.rest_source_dicoms
    overwrite : bool
        If True, rerun preprocessing even if outputs exist

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
        rest_dicoms = config.scan.rest_source_dicoms

    return preprocessing(
        participant_id=config.paths.participant_id,
        task="rest",
        root_dir=str(config.paths.preprocessing_dir),
        input_dicoms=[str(p) for p in rest_dicoms],
        overwrite=overwrite
    )


def run_mask_extraction(
    config: SessionConfig,
    preproc_file: Path,
    mask_file: Path,
    ref_file: Path,
    overwrite: bool = False
) -> list[Path]:
    """Run ICA-based mask extraction.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
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
        Paths to thresholded network mask files
    """
    from .mask_extraction import mask_extraction_rest

    logger.info("")
    logger.info("=" * 60)
    logger.info("MASK EXTRACTION")
    logger.info("=" * 60)

    return mask_extraction_rest(
        participant_id=config.paths.participant_id,
        root_dir=str(config.paths.mask_extraction_dir),
        input_file=preproc_file,
        mask_file=mask_file,
        ref_file=ref_file,
        n_yeo=config.mask.n_yeo,
        yeo_targets=config.mask.yeo_targets,
        smooth_fwhm=config.mask.smooth_fwhm,
        highpass=config.mask.highpass,
        lowpass=config.mask.lowpass,
        overwrite=overwrite
    )


def run_coregistration(
    config: SessionConfig,
    network_files: list[Path],
    ref_file: Path,
    twovol_dicom_dir: Optional[Path] = None,
    overwrite: bool = False
) -> list[Path]:
    """Coregister masks to feedback (2vol) space.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    network_files : list[Path]
        Paths to network mask files
    ref_file : Path
        BOLD reference from preprocessing
    twovol_dicom_dir : Path, optional
        Collected 2vol DICOM directory
    overwrite : bool
        Whether to rerun even if outputs exist

    Returns
    -------
    list[Path]
        Paths to coregistered mask files
    """
    from .mask_extraction import coregister_2vol

    logger.info("")
    logger.info("=" * 60)
    logger.info("COREGISTER MASKS TO FEEDBACK SPACE")
    logger.info("=" * 60)

    twovol_dir = twovol_dicom_dir or config.scan.twovol_dicoms

    return coregister_2vol(
        participant_id=config.paths.participant_id,
        root_dir=str(config.paths.session_dir),
        input_dicom=str(twovol_dir),
        network_files=network_files,
        ref_file=ref_file,
        overwrite=overwrite
    )


def setup_session(
    config: SessionConfig,
    overwrite: bool = False,
    use_simulator: bool = True,
    skip_collection: bool = False,
) -> list[Path]:
    """Run all setup steps: collection, preprocessing, mask extraction, coregistration.

    This should be run once per participant before the feedback session.

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

    # Step 0: Initial 2vol (pre-scan positioning check)
    if skip_collection:
        logger.info("Skipping initial 2vol collection (using source DICOMs directly)")
        initial_twovol = [config.scan.twovol_initial_dicoms or config.scan.twovol_dicoms]
    else:
        initial_twovol = run_dicom_collection(
            config,
            use_simulator=use_simulator,
            task_name="2vol",
            source_dirs=[config.scan.twovol_initial_dicoms or config.scan.twovol_dicoms],
            n_volumes_list=[config.scan.twovol_n_volumes],
            start_run=None,
            max_runs=config.scan.twovol_max_runs,
        )

    # Step 1: Resting-state collection
    if skip_collection:
        logger.info("Skipping resting-state collection (using source DICOMs directly)")
        rest_dicoms = config.scan.rest_source_dicoms
    else:
        rest_dicoms = run_dicom_collection(
            config,
            use_simulator=use_simulator,
            task_name="rest",
            source_dirs=config.scan.rest_source_dicoms,
            n_volumes_list=config.scan.rest_n_volumes,
            start_run=1,
            max_runs=config.scan.rest_max_runs,
        )

    # Step 2: Preprocessing
    preproc_file, mask_file, ref_file = run_preprocessing(config, rest_dicoms=rest_dicoms, overwrite=overwrite)

    # Step 3: Mask extraction
    network_files = run_mask_extraction(config, preproc_file, mask_file, ref_file, overwrite)

    # Step 4: Coregistration to initial 2vol
    coreg_mask_files = run_coregistration(
        config,
        network_files,
        ref_file,
        twovol_dicom_dir=initial_twovol[0],
        overwrite=overwrite
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
    session_state: "SessionState",
    run_type: str,
    use_simulator: bool = True,
) -> dict:
    """Run a single task (transfer or feedback).

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

    Returns
    -------
    dict
        Run results including hits, scale_factor, volumes_processed
    """
    from .feedback import run_feedback_session
    from .tasks.ball import BallTask

    run_number = session_state.next_run_number
    task_run_number = session_state.next_run_number_for_type(run_type)
    feedback_mode = (run_type == 'feedback')

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"RUN {run_number}: {run_type.upper()} (task run {task_run_number})")
    logger.info("=" * 60)
    logger.info(f"Mode: {'FEEDBACK' if feedback_mode else 'TRANSFER (no feedback)'}")
    logger.info(f"Scale factor: {session_state.scale_factor}")
    logger.info(f"Circle radii: {session_state.circle_radii}")

    # Set up simulator parameters if using simulator
    simulator_kwargs = {}
    if use_simulator:
        simulator_kwargs = {
            "simulator_source_dir": config.scan.feedback_dicoms,
            "simulator_speed": config.scan.simulator_speed,
            "simulator_clear_target": True,
            "simulator_cleanup": False,
        }

    # Start feedback session
    session_info = run_feedback_session(
        mask_paths=list(mask_files),
        subject=config.paths.participant_id,
        task="feedback" if feedback_mode else "transfer",
        run=task_run_number,
        output_dir=config.paths.feedback_output_dir,
        collection_dir=config.paths.collection_dir,
        n_baseline=config.feedback.n_baseline,
        n_volumes=config.feedback.n_volumes,
        tr=config.scan.tr,
        mask_names=config.mask.mask_names,
        use_efficiency_weights=config.feedback.use_efficiency_weights,
        **simulator_kwargs
    )

    coordinator = session_info['coordinator']

    # Determine if this is the first run of this type
    completed_of_type = sum(
        1 for r in session_state.completed_runs if r.run_type == run_type
    )
    is_first_of_type = (completed_of_type == 0)

    # Create task with session state
    task = BallTask(
        coordinator,
        mask_names=["ContA", "DefaultA"],  # ContA at top, DefaultA at bottom
        scale_factor=session_state.scale_factor,
        trigger_keys=config.task.trigger_keys,
        left_button=config.task.left_button,
        right_button=config.task.right_button,
        enter_button=config.task.enter_button,
        fullscreen=config.task.fullscreen,
        feedback_mode=feedback_mode,
        run_number=task_run_number,
        is_first_of_type=is_first_of_type,
        session_state=session_state,
        min_hits_threshold=config.task.min_hits_threshold,
        max_hits_threshold=config.task.max_hits_threshold,
        scale_adjust_factor=config.task.scale_adjust_factor,
    )

    # Register callback
    coordinator._activation_callback = task._update_data

    try:
        task.run()
    finally:
        logger.info("")
        logger.info("Cleaning up run...")
        coordinator.stop()
        if 'thread' in session_info:
            session_info['thread'].join(timeout=5)
        coordinator.cleanup(force=True)

    results = {
        "run_number": run_number,
        "run_type": run_type,
        "hits": task.hit_counters.copy(),
        "scale_factor": task.scale_factor,
        "circle_radii": task.circle_radii.copy(),
        "volumes_processed": task.current_volume,
    }

    logger.info("")
    logger.info(f"Run {run_number} complete:")
    logger.info(f"  Hits: {results['hits']}")
    logger.info(f"  Scale factor: {results['scale_factor']:.1f}")
    logger.info(f"  Volumes: {results['volumes_processed']}/{config.feedback.n_volumes}")

    return results


def coregister_before_first_transfer(
    config: SessionConfig,
    session_state: "SessionState",
    mask_files: list[Path],
    use_simulator: bool = True,
) -> list[Path]:
    """Acquire a second 2vol and re-coregister masks right before the first transfer.

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
    logger.info("REAL-TIME 2VOL + COREG (pre-transfer)")
    logger.info("=" * 60)

    network_files = [Path(f) for f in mask_files]

    ref_file = ref_file_path(config)
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found at {ref_file}; run preprocessing before coregistration.")

    twovol_dicoms = run_dicom_collection(
        config,
        use_simulator=use_simulator,
        task_name="2vol",
        source_dirs=[config.scan.twovol_realtime_dicoms or config.scan.twovol_dicoms],
        n_volumes_list=[config.scan.twovol_n_volumes],
        start_run=None,
        max_runs=config.scan.twovol_max_runs,
    )

    coreg_mask_files = run_coregistration(
        config,
        network_files,
        ref_file,
        twovol_dicom_dir=twovol_dicoms[0],
        overwrite=True,
    )

    return [Path(f) for f in coreg_mask_files]


# =============================================================================
# Session Workflow
# =============================================================================

def run_full_session(
    config: SessionConfig,
    mask_files: list[Path],
    n_feedback_runs: int = 2,
    reset_session: bool = False,
    use_simulator: bool = True,
) -> list[dict]:
    """Run a complete multi-run session.

    Workflow: Transfer -> Feedback x N -> Transfer

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

    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING MULTI-RUN SESSION")
    logger.info("=" * 60)
    logger.info(f"Workflow: Transfer -> Feedback x {n_feedback_runs} -> Transfer")

    session_dir = config.paths.session_dir
    if reset_session:
        session_state = SessionState.reset(session_dir)
        session_state.max_feedback_runs = n_feedback_runs
        session_state.save()
    else:
        session_state = SessionState.load(session_dir)
        if session_state.max_feedback_runs != n_feedback_runs:
            session_state.max_feedback_runs = n_feedback_runs
            session_state.save()

    mask_files = [Path(f) for f in mask_files]

    all_results = []

    completed_count = len(session_state.completed_runs)
    total_runs = 2 + n_feedback_runs

    run_sequence = ['transfer'] + ['feedback'] * n_feedback_runs + ['transfer']

    logger.info(f"Completed runs: {completed_count}")
    logger.info(f"Remaining runs: {run_sequence[completed_count:]}")

    for i, run_type in enumerate(run_sequence[completed_count:], start=completed_count):
        logger.info("")
        logger.info(f">>> Starting run {i + 1}/{total_runs} ({run_type})")

        if run_type == 'transfer' and not session_state.completed_runs:
            mask_files = coregister_before_first_transfer(config, session_state, mask_files, use_simulator=use_simulator)

        results = run_single_task(
            config=config,
            mask_files=mask_files,
            session_state=session_state,
            run_type=run_type,
            use_simulator=use_simulator,
        )
        all_results.append(results)

        session_state = SessionState.load(session_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total runs: {len(session_state.completed_runs)}")
    logger.info(f"Cumulative hits: {session_state.hit_counters}")
    logger.info(f"Final scale factor: {session_state.scale_factor:.1f}")
    logger.info(f"Final circle radii: {session_state.circle_radii}")

    return all_results


def run_next_task(
    config: SessionConfig,
    mask_files: list[Path],
    n_feedback_runs: int = 2,
    use_simulator: bool = True,
) -> Optional[dict]:
    """Run the next task in the session sequence.

    Loads session state, determines the next run type, executes it, and exits.
    Call this repeatedly (separate script invocations) to progress through the session.

    Workflow: Transfer -> Feedback x N -> Transfer

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

    session_dir = config.paths.session_dir
    session_state = SessionState.load(session_dir)

    mask_files = [Path(f) for f in mask_files]

    if session_state.max_feedback_runs != n_feedback_runs:
        session_state.max_feedback_runs = n_feedback_runs
        session_state.save()

    run_sequence = ['transfer'] + ['feedback'] * n_feedback_runs + ['transfer']
    total_runs = len(run_sequence)
    completed_count = len(session_state.completed_runs)

    if completed_count >= total_runs:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SESSION ALREADY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total runs: {completed_count}")
        logger.info(f"Cumulative hits: {session_state.hit_counters}")
        logger.info(f"Final scale factor: {session_state.scale_factor:.1f}")
        logger.info("Use reset_session_state() to start a new session.")
        return None

    run_type = run_sequence[completed_count]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"NEXT RUN: {completed_count + 1}/{total_runs} ({run_type.upper()})")
    logger.info("=" * 60)
    logger.info(f"Completed: {[r.run_type for r in session_state.completed_runs]}")
    logger.info(f"Remaining: {run_sequence[completed_count:]}")

    if run_type == 'transfer' and not session_state.completed_runs:
        mask_files = coregister_before_first_transfer(config, session_state, mask_files, use_simulator=use_simulator)

    results = run_single_task(
        config=config,
        mask_files=mask_files,
        session_state=session_state,
        run_type=run_type,
        use_simulator=use_simulator,
    )

    return results


def reset_session_state(config: SessionConfig) -> "SessionState":
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
    return SessionState.reset(config.paths.session_dir)
