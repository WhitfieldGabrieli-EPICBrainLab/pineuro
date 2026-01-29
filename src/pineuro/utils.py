"""
Utilities and command execution framework for pineuro.

This module provides the foundational utilities used throughout the pineuro
library. It centralizes external command execution, ensuring consistent
error handling, logging, and overwrite behavior across all pipeline stages.

The key design principle is that ALL external commands (FSL, dcm2niix, etc.)
route through run_command(), giving a single point of control for:
- Input/output validation
- Overwrite logic (skip if outputs exist)
- Performance timing
- Error reporting
- Logging

Key Components
--------------
Command Execution:
    run_command : Execute shell commands with validation
    parallel_run_command : Run multiple commands in parallel

DICOM/NIfTI Operations:
    dicom_to_nifti : Convert DICOM to NIfTI via dcm2niix
    is_valid_dicom, is_valid_volume : File validation helpers
    get_tr_from_volume : Extract TR from metadata

Registration (ANTs-backed):
    compute_xfm : Compute transformation matrix between volumes
    apply_xfm : Apply transformation to move volumes
    compute_xfm_to_mni : Register to MNI space
    compute_mean : Compute temporal mean of 4D volume

Masking Operations:
    apply_mask : Apply binary mask to volume
    erode_mask : Morphological erosion of mask

Atlas Helpers:
    get_yeo_labels : Get Yeo network names (7 or 17)
    get_mni_template : Fetch MNI152 template via templateflow

Timing Infrastructure:
    Timer : Context manager for performance profiling
    enable_timing, get_timing_data : Timing control

Why Centralized Command Execution?
----------------------------------
Centralizing command execution in run_command() provides several benefits:

1. **Consistent overwrite behavior**: All functions respect the overwrite flag,
   enabling efficient pipeline re-runs.

2. **Unified logging**: All external commands log their inputs/outputs in a
   consistent format for debugging.

3. **Performance tracking**: When timing is enabled, all operations are
   automatically profiled.

4. **Error handling**: Failed commands raise informative exceptions with
   context about what went wrong.

5. **Command availability checking**: Verifies commands exist before trying
   to run them.

Dependencies
------------
External:
    - FSL : mcflirt, bet, fslmaths, etc.
    - dcm2niix : DICOM to NIfTI conversion
    - ANTsPy : Python bindings for ANTs registration

Python:
    - numpy : Array operations
    - nibabel : NIfTI I/O
    - pydicom : DICOM header reading
    - scipy : Morphological operations
    - templateflow : Standard template fetching

Notes
-----
**ANTs vs FSL for Registration:**
Registration functions use ANTsPy (Python bindings) for speed and to avoid
spawning subprocesses. This provides ~2x speedup compared to shell-based
ANTs calls while maintaining identical results.

**Overwrite Semantics:**
All functions that produce outputs support `overwrite=False` (the default).
When False, the function checks if outputs exist and skips computation.
This enables efficient pipeline re-runs - only missing outputs are computed.

See Also
--------
preprocessing : Uses these utilities for motion correction and brain extraction
mask_extraction : Uses these utilities for ICA and registration
feedback : Uses volume utilities for real-time processing

References
----------
.. [1] Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith,
       S. M. (2012). FSL. NeuroImage, 62(2), 782-790.

.. [2] Li, X., Morgan, P. S., Ashburner, J., Smith, J., & Rorden, C. (2016).
       The first step for neuroimaging data analysis: DICOM to NIfTI conversion.
       Journal of Neuroscience Methods, 264, 47-56.

.. [3] Avants, B. B., Tustison, N., & Song, G. (2009). Advanced normalization
       tools (ANTS). Insight Journal, 2(365), 1-35.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# --- Standard Library ---
import shutil
import subprocess
import tempfile
import time
import json
import logging
from typing import Union, List, Optional, Dict, Any, cast
from pathlib import Path
from contextlib import contextmanager

# --- Third-Party ---
import numpy as np
import nibabel as nib
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
from templateflow import api as tflow  # Standard neuroimaging templates
import pydicom                          # DICOM header reading
from scipy.ndimage import binary_erosion, generate_binary_structure

# --- Module Logger ---
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: PERFORMANCE TIMING INFRASTRUCTURE
# =============================================================================
# These utilities provide optional performance profiling for pipeline operations.
# When timing is enabled, all operations wrapped with Timer() are tracked.
# This is useful for identifying bottlenecks and optimizing pipeline performance.

# Global timing storage - collects timing data when enabled
_timing_data: Dict[str, Any] = {}
_timing_enabled = False


def enable_timing() -> None:
    """Enable performance timing for all operations."""
    global _timing_enabled
    _timing_enabled = True
    _timing_data.clear()
    logger.info("Performance timing enabled")


def disable_timing() -> None:
    """Disable performance timing."""
    global _timing_enabled
    _timing_enabled = False


def get_timing_data() -> Dict[str, Any]:
    """
    Get collected timing data.

    Returns
    -------
    dict
        Timing data with operation names as keys and timing info as values.
    """
    return _timing_data.copy()


def save_timing_report(output_path: Union[str, Path]) -> None:
    """
    Save timing report to JSON file.

    Parameters
    ----------
    output_path : str or Path
        Path to save timing report (JSON format)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(_timing_data, f, indent=2)

    logger.info(f"Timing report saved to {output_path}")


@contextmanager
def Timer(name: str, category: str = "general"):
    """
    Context manager for timing code blocks.

    Parameters
    ----------
    name : str
        Name of the operation being timed
    category : str, default="general"
        Category for grouping operations (e.g., "preprocessing", "mask_extraction")

    Yields
    ------
    dict
        Timing information dictionary that gets populated

    Examples
    --------
    >>> with Timer("Motion correction", "preprocessing"):
    ...     # Code to time
    ...     pass
    """
    if not _timing_enabled:
        yield {}
        return

    start_time = time.time()
    timing_info = {
        "name": name,
        "category": category,
        "start_time": start_time
    }

    try:
        yield timing_info
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        timing_info["end_time"] = end_time
        timing_info["elapsed_seconds"] = elapsed

        # Store in global timing data
        if category not in _timing_data:
            _timing_data[category] = []
        _timing_data[category].append(timing_info)

        logger.info(f"[TIMING] {name}: {elapsed:.2f}s")

# ==============================================================
# Platform Compatibility Check
# ==============================================================

def check_platform_requirements() -> list[str]:
    """Check platform requirements and return list of warnings/errors.

    This function checks for required external tools (FSL, dcm2niix) and
    platform-specific limitations that may affect pineuro functionality.

    Returns
    -------
    list[str]
        List of warning messages. Empty list if all requirements are met.

    Examples
    --------
    >>> from pineuro.utils import check_platform_requirements
    >>> warnings = check_platform_requirements()
    >>> for warning in warnings:
    ...     print(f"Warning: {warning}")
    """
    from . import config

    warnings = []
    platform = config.get_platform()

    # Check FSL availability (required for preprocessing)
    fsl_commands = ['mcflirt', 'bet', 'fslmaths']
    fsl_missing = [cmd for cmd in fsl_commands if shutil.which(cmd) is None]
    if fsl_missing:
        if platform == 'windows':
            warnings.append(
                f"FSL not available on native Windows. Use WSL2 for preprocessing. "
                f"Missing: {', '.join(fsl_missing)}"
            )
        else:
            warnings.append(
                f"FSL commands not found in PATH: {', '.join(fsl_missing)}. "
                f"Preprocessing requires FSL. See: https://fsl.fmrib.ox.ac.uk/fsl/"
            )

    # Check dcm2niix availability (required for DICOM conversion)
    if shutil.which('dcm2niix') is None:
        warnings.append(
            "dcm2niix not found in PATH. DICOM conversion requires dcm2niix. "
            "Install via: apt install dcm2niix (Linux) or brew install dcm2niix (macOS)"
        )

    # Platform-specific warnings
    if platform == 'wsl':
        warnings.append(
            "Running under WSL2: Multi-monitor PsychoPy display may not work correctly. "
            "Single-monitor mode is recommended (pineuro.config.PSYCHOPY_SCREEN = 0)."
        )

    if platform == 'linux' and config.is_wayland():
        warnings.append(
            "Wayland display server detected. Some X11-based features (window positioning) "
            "may not work. Consider using an X11/Xorg session for full compatibility."
        )

    if platform == 'windows':
        warnings.append(
            "Native Windows is not supported. FSL and ANTsPy are not available. "
            "Please use WSL2 (Windows Subsystem for Linux) instead."
        )

    return warnings


def outputs_exist(
    outputs: Union[Path, List[Path], str, List[str]],
    overwrite: bool = False
) -> bool:
    """Check if all outputs exist and overwrite is False.

    This is a utility to standardize the common pattern of skipping
    computation when outputs already exist.

    Parameters
    ----------
    outputs : Path, str, or list of Path/str
        Output file(s) to check for existence
    overwrite : bool, default False
        If True, always return False (outputs should be regenerated)

    Returns
    -------
    bool
        True if all outputs exist and overwrite=False, False otherwise

    Examples
    --------
    >>> from pineuro.utils import outputs_exist
    >>> if outputs_exist(output_path, overwrite=False):
    ...     logger.info("Skipping - output exists")
    ...     return output_path
    """
    if overwrite:
        return False

    if isinstance(outputs, (str, Path)):
        outputs = [Path(outputs)]
    else:
        outputs = [Path(o) for o in outputs]

    return all(p.exists() for p in outputs)


class ProgressEmitter:
    """Utility for emitting progress to both logger and optional callback.

    This standardizes the common pattern of logging progress and optionally
    calling a callback function for GUI updates.

    Parameters
    ----------
    callback : callable, optional
        Callback function with signature (step: str, detail: str) -> None
    log : logging.Logger, optional
        Logger to use. Defaults to module logger if not provided.

    Examples
    --------
    >>> from pineuro.utils import ProgressEmitter
    >>> emitter = ProgressEmitter(progress_callback, logger)
    >>> emitter.emit("Step 1", "Processing files...")
    >>> emitter.emit("Step 2")  # detail is optional
    """

    def __init__(
        self,
        callback: Optional[callable] = None,
        log: Optional[logging.Logger] = None
    ):
        self._callback = callback
        self._logger = log or logger

    def emit(self, step: str, detail: str = "") -> None:
        """Emit progress message to logger and callback.

        Parameters
        ----------
        step : str
            Step name or description
        detail : str, optional
            Additional detail about the step
        """
        if detail:
            self._logger.info(f"{step}: {detail}")
        else:
            self._logger.info(step)

        if self._callback:
            self._callback(step, detail)


# ==============================================================
# HRF and Task Regressor Computation
# ==============================================================

def compute_hrf(tr: float, duration: float = 32.0) -> np.ndarray:
    """Compute canonical double-gamma HRF (SPM-style).

    Parameters
    ----------
    tr : float
        Repetition time in seconds.
    duration : float, optional
        Duration of HRF in seconds. Default is 32s.

    Returns
    -------
    hrf : ndarray
        HRF sampled at TR intervals, normalized to peak at 1.0.

    Notes
    -----
    Uses SPM canonical HRF parameters:
    - Positive gamma: shape=6, scale=1 (peak ~5s)
    - Negative gamma: shape=16, scale=1 (peak ~16s)
    - Undershoot ratio: 1/6

    Examples
    --------
    >>> from pineuro.utils import compute_hrf
    >>> hrf = compute_hrf(tr=1.2)
    >>> print(hrf.shape)  # (27,) for 32s duration at TR=1.2
    """
    from scipy.stats import gamma as gamma_dist

    # Time points at TR intervals
    t = np.arange(0, duration, tr)

    # SPM canonical HRF parameters
    # Positive response: gamma(6, 1) peaks at ~5s
    # Undershoot: gamma(16, 1) peaks at ~16s
    a1, b1 = 6.0, 1.0  # shape, scale for positive response
    a2, b2 = 16.0, 1.0  # shape, scale for undershoot
    c = 1.0 / 6.0  # undershoot ratio

    # Double-gamma HRF
    hrf = gamma_dist.pdf(t, a1, scale=b1) - c * gamma_dist.pdf(t, a2, scale=b2)

    # Normalize to peak at 1.0
    hrf = hrf / np.max(hrf)

    return hrf.astype(np.float64)


def compute_boxcar_task_regressor(
    n_baseline: int,
    n_volumes: int,
    tr: float,
) -> np.ndarray:
    """Compute HRF-convolved boxcar task regressor.

    Creates a task regressor where:
    - Baseline period (0 to n_baseline-1): task = 0
    - Active period (n_baseline onwards): task = 1 convolved with HRF

    This is useful for block designs with a single baseline/active structure.

    Parameters
    ----------
    n_baseline : int
        Number of baseline volumes (task = 0).
    n_volumes : int
        Total number of volumes.
    tr : float
        Repetition time in seconds.

    Returns
    -------
    regressor : ndarray, shape (n_volumes,)
        Task regressor values for each volume.

    Notes
    -----
    The convolution produces the characteristic HRF response shape:
    - Stays at 0 during baseline
    - Ramps up after baseline onset following HRF shape
    - Reaches steady state (~1.0) after ~25-30s

    Examples
    --------
    >>> from pineuro.utils import compute_boxcar_task_regressor
    >>> regressor = compute_boxcar_task_regressor(n_baseline=25, n_volumes=150, tr=1.2)
    >>> print(regressor.shape)  # (150,)
    """
    # Create boxcar: 0 during baseline, 1 during active period
    boxcar = np.zeros(n_volumes, dtype=np.float64)
    boxcar[n_baseline:] = 1.0

    # Compute HRF kernel
    hrf = compute_hrf(tr)

    # Convolve boxcar with HRF
    convolved = np.convolve(boxcar, hrf, mode='full')

    # Trim to original length
    regressor = convolved[:n_volumes]

    # Normalize so steady-state is ~1.0 (divide by sum of HRF)
    # This ensures the regressor reaches 1.0 after the HRF response settles
    hrf_sum = np.sum(hrf)
    if hrf_sum > 0:
        regressor = regressor / hrf_sum

    return regressor.astype(np.float64)


# ==============================================================
# Installation Check
# ==============================================================

class CommandNotFoundError(Exception):
    """Raised when an FSL command is not installed or found in PATH."""

def _check_command_installation(cmd: Union[str, Path] = 'fsl') -> None:
    """
    Check if command is installed and available in PATH.

    Parameters
    ----------
    cmd
        Command to look for
    
    Raises
    ------
    CommandNotFoundError
        If command is not found
        
    """

    if shutil.which(cmd) is None:
        raise CommandNotFoundError(
            f"{cmd} is not found in PATH."
        )
    
    logger.debug(f"{cmd} installation verified")

# ==============================================================
#  DICOM Utilities
# ==============================================================

# DICOM file extensions to search for
DICOM_EXTENSIONS = ['*.dcm', '*.DCM', '*.ima', '*.IMA']


def get_dicom_files(directory: Path) -> List[Path]:
    """
    Get all DICOM files in a directory, sorted by filename.

    Returns all files in the directory, assuming they are valid DICOM files.
    This supports DICOM files without extensions (common with some scanner exports).

    Parameters
    ----------
    directory : Path
        Directory to search for DICOM files

    Returns
    -------
    List[Path]
        Sorted list of DICOM file paths

    """
    # Get all files in directory (supports extensionless DICOMs)
    dicom_files = [f for f in directory.iterdir() if f.is_file()]
    dicom_files.sort()
    return dicom_files


def is_valid_dicom(file_path: Path) -> bool:
    """
    Check if a file is a valid DICOM by attempting to read its header.

    Parameters
    ----------
    file_path : Path
        Path to file to check

    Returns
    -------
    bool
        True if file is a valid DICOM, False otherwise

    """
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False


def get_tr_from_dicom(dicom_path: Path, default: float = 2.0) -> float:
    """
    Extract repetition time (TR) from DICOM header.

    Parameters
    ----------
    dicom_path : Path
        Path to DICOM file
    default : float, default=2.0
        Default TR value to return if extraction fails

    Returns
    -------
    float
        TR in seconds. Returns default value if TR cannot be read.

    """
    try:
        dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)

        # First check top-level (standard DICOM)
        if hasattr(dcm, 'RepetitionTime') and dcm.RepetitionTime is not None:
            return float(dcm.RepetitionTime) / 1000.0

        # Search in nested sequences (enhanced/multiframe DICOM)
        for elem in dcm.iterall():
            if elem.keyword == 'RepetitionTime' and elem.value is not None:
                return float(elem.value) / 1000.0

        logger.warning(f"TR not found in DICOM, using default {default}s")
        return default
    except Exception as e:
        logger.warning(f"Error reading TR: {e}, using default {default}s")
        return default


# ==============================================================
#  NIfTI Utilities
# ==============================================================

# NIfTI file extensions to search for
NIFTI_EXTENSIONS = ['*.nii', '*.nii.gz', '*.NII', '*.NII.GZ']

# Combined extensions for volume files (DICOM + NIfTI)
VOLUME_EXTENSIONS = DICOM_EXTENSIONS + NIFTI_EXTENSIONS


def get_nifti_files(directory: Path) -> List[Path]:
    """
    Get all NIfTI files in a directory, sorted by filename.

    Parameters
    ----------
    directory : Path
        Directory to search for NIfTI files

    Returns
    -------
    List[Path]
        Sorted list of NIfTI file paths

    """
    nifti_files = []
    for ext in NIFTI_EXTENSIONS:
        nifti_files.extend(directory.glob(ext))

    # Filter to only files and sort by name
    valid_niftis = [f for f in nifti_files if f.is_file()]
    valid_niftis.sort()
    return valid_niftis


def get_volume_files(directory: Path) -> List[Path]:
    """
    Get all volume files in a directory, sorted by filename.

    Returns all files in the directory, assuming they are valid volume files.
    This supports DICOM files without extensions (common with some scanner exports).

    Parameters
    ----------
    directory : Path
        Directory to search for volume files

    Returns
    -------
    List[Path]
        Sorted list of volume file paths

    """
    # Get all files in directory (supports extensionless DICOMs)
    volume_files = [f for f in directory.iterdir() if f.is_file()]
    volume_files.sort()
    return volume_files


def is_nifti_file(file_path: Path) -> bool:
    """
    Check if a file is a NIfTI file based on extension.

    Parameters
    ----------
    file_path : Path
        Path to file to check

    Returns
    -------
    bool
        True if file has NIfTI extension, False otherwise

    """
    name = str(file_path).lower()
    return name.endswith('.nii') or name.endswith('.nii.gz')


def is_valid_nifti(file_path: Path) -> bool:
    """
    Check if a file is a valid NIfTI by attempting to read its header.

    Parameters
    ----------
    file_path : Path
        Path to file to check

    Returns
    -------
    bool
        True if file is a valid NIfTI, False otherwise

    """
    try:
        nib.load(str(file_path))
        return True
    except Exception:
        return False


def is_valid_volume(file_path: Path) -> bool:
    """
    Check if a file is a valid volume (DICOM or NIfTI).

    Auto-detects file type and validates accordingly.

    Parameters
    ----------
    file_path : Path
        Path to file to check

    Returns
    -------
    bool
        True if file is a valid DICOM or NIfTI, False otherwise

    """
    if is_nifti_file(file_path):
        return is_valid_nifti(file_path)
    else:
        return is_valid_dicom(file_path)


def get_json_sidecar_path(nifti_path: Path) -> Path:
    """
    Get the expected JSON sidecar path for a NIfTI file.

    Follows the naming convention: vol_001.nii -> vol_001.json
                                   vol_001.nii.gz -> vol_001.json

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file

    Returns
    -------
    Path
        Path to corresponding JSON sidecar file

    """
    nifti_path = Path(nifti_path)
    name = nifti_path.name
    # Handle both .nii and .nii.gz
    if name.lower().endswith('.nii.gz'):
        base_name = name[:-7]  # Remove .nii.gz
    elif name.lower().endswith('.nii'):
        base_name = name[:-4]  # Remove .nii
    else:
        base_name = nifti_path.stem

    return nifti_path.parent / f"{base_name}.json"


def get_tr_from_json_sidecar(json_path: Path, default: float = 2.0) -> float:
    """
    Extract repetition time (TR) from JSON sidecar file.

    Parameters
    ----------
    json_path : Path
        Path to JSON sidecar file
    default : float, default=2.0
        Default TR value to return if extraction fails

    Returns
    -------
    float
        TR in seconds. Returns default value if TR cannot be read.

    """
    try:
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Check for RepetitionTime field (standard BIDS)
        if 'RepetitionTime' in metadata:
            return float(metadata['RepetitionTime'])

        logger.warning(f"TR not found in JSON sidecar, using default {default}s")
        return default
    except Exception as e:
        logger.warning(f"Error reading TR from JSON: {e}, using default {default}s")
        return default


def get_tr_from_volume(file_path: Path, default: float = 2.0) -> float:
    """
    Extract repetition time (TR) from a volume file (DICOM or NIfTI sidecar).

    Auto-detects file type and extracts TR accordingly:
    - For DICOM: reads from DICOM header
    - For NIfTI: reads from JSON sidecar (if exists)

    Parameters
    ----------
    file_path : Path
        Path to volume file (DICOM or NIfTI)
    default : float, default=2.0
        Default TR value to return if extraction fails

    Returns
    -------
    float
        TR in seconds. Returns default value if TR cannot be read.

    """
    file_path = Path(file_path)
    if is_nifti_file(file_path):
        json_path = get_json_sidecar_path(file_path)
        if json_path.exists():
            return get_tr_from_json_sidecar(json_path, default)
        else:
            logger.warning(f"JSON sidecar not found for {file_path}, using default {default}s")
            return default
    else:
        return get_tr_from_dicom(file_path, default)


# =============================================================================
# SECTION 2: COMMAND EXECUTION
# =============================================================================
# These functions handle external command execution with consistent validation,
# logging, and error handling. run_command() is the foundation that all
# external tool invocations use.


def run_command(
        cmd_name: str,
        cmd: List[str],
        description: str,
        inputs: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        outputs: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        overwrite: bool = False,
        verbose: bool = True,
        timing_category: str = "commands"
) -> None:
    """
    Run a shell command with error handling and logging

    Parameters
    ----------
    cmd_name : str
        Name of command to execute
    cmd : list of str
        Command and arguments to execute
    description : str
        Human readable description of what the command does (for logging)
    inputs : list of str or Path, optional
        Paths to files/folders that should exist before command starts
    outputs : list of str or Path, optional
        Paths to files/folders that should exist after command completes
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    verbose : bool, default=True
        If False, suppresses informational logging messages.
    timing_category : str, default="commands"
        Category for timing data (e.g., "preprocessing", "mask_extraction")

    Raises
    ------
    FileNotFoundError
        If input files are not found
    RuntimeError
        If command fails or if expected output files are not created

    """

    # Log the command
    if verbose:
        logger.info(f"{description}...")

    # Convert outputs to list if single file provided
    if isinstance(outputs, (str, Path)):
        outputs = [outputs]

    # Check if outputs exist and skip if overwrite=False
    if not overwrite and outputs is not None:
        output_paths_check = [Path(f) for f in outputs]
        if all(p.exists() for p in output_paths_check):
            if verbose:
                logger.info(f"All output files exist and overwrite=False. Skipping {description}.")
            return

    # Log the command
    if verbose:
        logger.info(f"Running command: {' '.join(cmd)}")

    # Check if command is installed
    _check_command_installation(cmd_name)

    # Convert inputs to list if single file provided
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    # Convert to Path objects and validate
    input_paths: list[Path] = []
    if inputs is not None:
        if verbose:
            logger.info(f"Input files:")
        for f in inputs:
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            input_paths.append(path)
            if verbose:
                logger.info(f"  {f}")
    output_paths: list[Path] = []
    if outputs is not None:
        if verbose:
            logger.info(f"Output files:")
        for f in outputs:
            path = Path(f)
            (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)
            output_paths.append(path)
            if verbose:
                logger.info(f"  {f}")

    # Run command with timing
    with Timer(f"{cmd_name}: {description}", timing_category):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Log output
            if result.stdout and verbose:
                logger.debug(f"stdout:\n{result.stdout}")
            if result.stderr and verbose:
                logger.debug(f"stderr:\n{result.stderr}")

            if verbose:
                logger.info(f"{description} completed successfully")

        except subprocess.CalledProcessError as e:
            error_msg = (
                f"{description} failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {e.returncode}\n"
                f"stdout: {e.stdout}"
                f"stderr: {e.stderr}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # Verify output files if specified
    if outputs is not None:
        missing_files: List[Path] = []
        for output_path in output_paths:
            if not output_path.exists():
                missing_files.append(output_path)
        if len(missing_files) > 0:
            error_msg = (
                f"{description} completed but expected output file(s) not found:\n"
                f"{missing_files}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

# Module-level helper for parallel_run_command (must be at module level for pickling)
def _run_job_worker(job_spec: dict) -> dict:
    """
    Worker function for parallel command execution.
    Must be at module level to be picklable by multiprocessing.

    Parameters
    ----------
    job_spec : dict
        Job specification containing cmd_name, cmd, description, inputs, outputs, etc.

    Returns
    -------
    dict
        Result dictionary with 'success', 'description', and optionally 'error'
    """
    try:
        run_command(
            cmd_name=job_spec.get('cmd_name'),
            cmd=job_spec.get('cmd'),
            description=job_spec.get('description'),
            inputs=job_spec.get('inputs'),
            outputs=job_spec.get('outputs'),
            overwrite=job_spec.get('overwrite', False),
            verbose=job_spec.get('verbose', True),
            timing_category=job_spec.get('timing_category', 'parallel')
        )
        return {'success': True, 'description': job_spec.get('description')}
    except Exception as e:
        return {'success': False, 'description': job_spec.get('description'), 'error': str(e)}

def parallel_run_command(
        jobs: List[dict],
        n_jobs: int = -1,
        timing_category: str = "parallel"
) -> None:
    """
    Run multiple commands in parallel using multiprocessing.

    Parameters
    ----------
    jobs : list of dict
        List of job specifications. Each dict should contain the same
        parameters as run_command (cmd_name, cmd, description, inputs, outputs, overwrite, verbose)
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available CPUs.
    timing_category : str, default="parallel"
        Category for timing data

    Examples
    --------
    >>> jobs = [
    ...     {
    ...         'cmd_name': 'mcflirt',
    ...         'cmd': ['mcflirt', '-in', 'run1.nii.gz', '-out', 'run1_mc.nii.gz'],
    ...         'description': 'Motion correction run 1',
    ...         'inputs': 'run1.nii.gz',
    ...         'outputs': 'run1_mc.nii.gz',
    ...         'overwrite': False
    ...     },
    ...     {
    ...         'cmd_name': 'mcflirt',
    ...         'cmd': ['mcflirt', '-in', 'run2.nii.gz', '-out', 'run2_mc.nii.gz'],
    ...         'description': 'Motion correction run 2',
    ...         'inputs': 'run2.nii.gz',
    ...         'outputs': 'run2_mc.nii.gz',
    ...         'overwrite': False
    ...     }
    ... ]
    >>> parallel_run_command(jobs, n_jobs=2)
    """
    import multiprocessing as mp

    if not jobs:
        return

    # Determine number of workers
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    else:
        n_workers = min(n_jobs, len(jobs), mp.cpu_count())

    logger.info(f"Running {len(jobs)} jobs in parallel with {n_workers} workers")

    # Add timing_category to each job spec
    jobs_with_category = []
    for job in jobs:
        job_copy = job.copy()
        job_copy['timing_category'] = timing_category
        jobs_with_category.append(job_copy)

    # Run jobs in parallel
    with Timer(f"Parallel execution ({len(jobs)} jobs)", timing_category):
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_run_job_worker, jobs_with_category)

    # Check for failures
    failures = [r for r in results if not r['success']]
    if failures:
        error_msgs = [f"{r['description']}: {r['error']}" for r in failures]
        raise RuntimeError(f"Parallel execution failed for {len(failures)} jobs:\n" + "\n".join(error_msgs))

    logger.info(f"All {len(jobs)} parallel jobs completed successfully")


# =============================================================================
# SECTION 3: DICOM/NIFTI OPERATIONS
# =============================================================================
# These functions handle DICOM to NIfTI conversion and related file operations.
# dcm2niix is the standard tool for DICOM conversion in neuroimaging.


def dicom_to_nifti(
        dicom_dir: Union[str, Path],
        output_dir: Union[str, Path],
        output_name: str,
        overwrite: bool = False,
        verbose: bool = True,
        compress: bool = True
) -> Path:
    """
    Convert DICOM file(s) to NIfTI format using dcm2niix.

    Parameters
    ----------
    dicom_dir : str or Path
        Directory containing DICOM files or path to a single DICOM file to convert.
    output_dir : str or Path
        Directory where the NIfTI file will be saved. Will be created if it doesn't exist
    output_name : str
        Base name for output file (without extension)
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    verbose : bool, default=True
        If False, suppresses dcm2niix output messages.
    compress : bool, default=True
        If True, output gzip-compressed NIfTI (.nii.gz).
        If False, output uncompressed NIfTI (.nii) for faster writes.

    Returns
    -------
    nifti_path : Path
        Path to the created NIfTI file

    Notes
    -----
    Set compress=False for real-time processing where speed is critical.
    Uncompressed output saves ~50-100ms per volume but produces larger files.

    """

    dicom_path = Path(dicom_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add quiet flag if not verbose
    quiet_flag = [] if verbose else ['-q']

    # Compression flag: -z y for gzip, -z n for uncompressed
    compress_flag = 'y' if compress else 'n'
    ext = '.nii.gz' if compress else '.nii'
    nifti_path = output_dir / f"{output_name}{ext}"

    # Handle single file vs directory
    # If a single file is given, copy it to a temp directory to avoid converting
    # all DICOMs in the same directory
    temp_dir = None
    if dicom_path.is_file():
        temp_dir = tempfile.mkdtemp(prefix='dcm2niix_')
        temp_dicom = Path(temp_dir) / dicom_path.name
        shutil.copy2(dicom_path, temp_dicom)
        conversion_path = Path(temp_dir)
    else:
        conversion_path = dicom_path

    try:
        # Build command
        cmd = ['dcm2niix'] + quiet_flag + ['-z', compress_flag, '-o', str(output_dir), '-f', str(output_name), str(conversion_path)]

        # Run dcm2niix
        run_command(
            cmd_name='dcm2niix',
            cmd=cmd,
            description='Converting DICOM to NIfTI',
            inputs=dicom_path,
            outputs=nifti_path,
            overwrite=overwrite,
            verbose=verbose
        )
    finally:
        # Clean up temp directory if created
        if temp_dir is not None and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    return nifti_path

def get_yeo_labels(
        n_networks: int
) -> List[str]:
    """
    Function to get the Yeo network labels.

    Parameters
    ---------
    n_networks : int
        7 or 17 network parcellation
    
    Returns
    -------
    labels : List of str
        List of network labels

    Raises
    ------
    ValueError
        If n_networks parameter is invalid

    """

    # Validate n_networks
    if n_networks != 7 and n_networks != 17:
        raise ValueError(
            f"Invalid number of networks: {n_networks}. "
            f"Valid options are 7 or 17"
        )
    
    # Assign labels
    if n_networks == 7:
        labels = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    else:
        labels = ["VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB",
        "SalVentAttnA", "SalVentAttnB", "LimbicA", "LimbicB", "ContC", "ContA", "ContB",
        "TempPar", "DefaultC", "DefaultA", "DefaultB"]

    return labels

# =============================================================================
# SECTION 4: REGISTRATION AND TRANSFORMATION (ANTs-backed)
# =============================================================================
# These functions handle image registration and transformation using ANTs
# (via ANTsPy). Registration aligns images from different sessions or to
# standard templates (like MNI152).
#
# Why ANTs?
# ---------
# ANTs provides state-of-the-art registration accuracy, particularly for
# nonlinear (SyN) registration. Using ANTsPy (Python bindings) avoids the
# overhead of spawning subprocesses while maintaining identical results.

def compute_mean(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Compute temporal mean of fMRI data using numpy.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (should be 4D)
    output_file: str or Path
        Path to save output mean image
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping compute mean.")
        return
    
    logger.info("Computing mean...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load 4D image and compute mean along time axis
    img = nib.load(str(input_file))
    data = img.get_fdata()
    mean_data = data.mean(axis=-1)
    
    # Save output
    out_img = Nifti1Image(mean_data, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}") 

def compute_xfm(
        moving: Union[str, Path],
        fixed: Union[str, Path],
        output_prefix: Union[str, Path],
        type_of_transform: str = 'Rigid',
        overwrite: bool = False
) -> tuple[Path, list[str]]:
    """
    Compute a transformation using ANTsPy registration.

    Parameters
    ----------
    moving : str or Path
        Path to moving image (image to be transformed)
    fixed : str or Path
        Path to fixed/reference image (target space)
    output_prefix : str or Path
        Base name for output files (without extension)
    type_of_transform : str, default='Rigid'
        ANTs transform type: 'Rigid', 'Affine', 'SyN', etc.
    overwrite : bool, default=False
        If True, always run and overwrite existing outputs.
        If False, skip if outputs exist.

    Returns
    -------
    warped_file : Path
        Path to warped moving image in fixed space
    transform_files : list[str]
        Transform files (moving → fixed). Use with invert=True in
        apply_xfm to go fixed → moving.
    """
    import ants

    output_file = Path(output_prefix).with_suffix(".nii.gz")
    xfm_dir = output_file.parent
    xfm_dir.mkdir(parents=True, exist_ok=True)

    # Check if outputs exist
    if not overwrite and output_file.exists():
        existing_xfm = sorted(xfm_dir.glob(f"{Path(output_prefix).stem}_*.mat"))
        if existing_xfm:
            logger.info("Output files exist and overwrite=False. Skipping registration.")
            return output_file, [str(t) for t in existing_xfm]

    logger.info(f"Computing transformation (ANTs {type_of_transform})...")
    logger.info(f"Moving: {moving}")
    logger.info(f"Fixed: {fixed}")

    with Timer(f"ANTs registration ({type_of_transform})", "registration"):
        fixed_img = ants.image_read(str(fixed))
        moving_img = ants.image_read(str(moving))

        registration = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform=type_of_transform,
            verbose=False
        )

        ants.image_write(registration['warpedmovout'], str(output_file))
        logger.info(f"Output: {output_file}")

        # Copy transform files to output directory
        transform_files: list[str] = []
        for i, xfm in enumerate(registration['fwdtransforms']):
            suffix = Path(xfm).suffix
            new_path = xfm_dir / f"{Path(output_prefix).stem}_{i}{suffix}"
            shutil.copy(xfm, new_path)
            transform_files.append(str(new_path))
            logger.info(f"Transform: {new_path}")

    return output_file, transform_files

def apply_xfm(
        moving: Union[str, Path],
        fixed: Union[str, Path],
        transform_files: Union[str, Path, list[str], tuple[str, ...]],
        output_file: Union[str, Path],
        interpolator: str = 'nearestNeighbor',
        invert: bool = False,
        overwrite: bool = False
) -> None:
    """
    Apply a transformation using ANTsPy.

    Parameters
    ----------
    moving : str or Path
        Path to image to be transformed
    fixed : str or Path
        Path to reference image (target space)
    transform_files : str or list
        Transform file(s) from compute_xfm
    output_file : str or Path
        Path for output file
    interpolator : str, default='nearestNeighbor'
        ANTs interpolator: 'nearestNeighbor', 'linear', 'bSpline', 'lanczosWindowedSinc'
    invert : bool, default=False
        If True, invert the transform (go fixed → moving direction)
    overwrite : bool, default=False
        If True, always run and overwrite existing outputs.
        If False, skip if output exists.
    """
    import ants

    output_path = Path(output_file)
    if not overwrite and output_path.exists():
        logger.info("Output file exists and overwrite=False. Skipping apply transform.")
        return

    valid_interps = {'nearestNeighbor', 'linear', 'bSpline', 'lanczosWindowedSinc'}
    if interpolator not in valid_interps:
        raise ValueError(f"Invalid interpolator: {interpolator}. Valid: {', '.join(sorted(valid_interps))}")

    if isinstance(transform_files, (str, Path)):
        transform_list = [str(transform_files)]
    else:
        transform_list = [str(t) for t in transform_files]

    if not transform_list:
        raise ValueError("At least one transform file is required.")

    logger.info("Applying transformation (ANTs)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_img = ants.image_read(str(fixed))
    moving_img = ants.image_read(str(moving))
    imagetype = 3 if moving_img.dimension > 3 else 0

    whichtoinvert = [invert] * len(transform_list) if invert else None

    transformed = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_img,
        transformlist=transform_list,
        interpolator=interpolator,
        imagetype=imagetype,
        whichtoinvert=whichtoinvert
    )

    ants.image_write(transformed, str(output_path))
    logger.info(f"Output: {output_path}")

def apply_xfm_sform_to_mni(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        interp: str = 'nearestneighbour',
        overwrite: bool = False
) -> None:
    """
    Apply transformation to MNI space using sform/qform matrices from NIfTI headers

    This function applies the transformation encoded in the sform/qform matrices
    to transform the input file to MNI152 space using the templateflow BOLD reference.
    No explicit transformation matrix file is required.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (should have sform/qform set to MNI space)
    output_file : str or Path
        Path for output file in MNI space
    interp : str, default='nearestneighbour'
        Interpolation method (per FSL, e.g., trilinear, nearestneighbour)
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Raises
    ------
    ValueError
        If interp parameter is invalid

    Notes
    -----
    This function uses FLIRT's -applyxfm -usesqform flags to apply the
    transformation encoded in the NIfTI headers rather than using an
    external transformation matrix file. The reference template is the
    MNI152NLin2009cAsym BOLD reference from templateflow.

    """

    # Validate interp
    valid_interps = ['nearestneighbour', 'trilinear', 'sinc', 'spline', 'nn']
    if interp not in valid_interps:
        raise ValueError(
            f"Invalid interpolation: {interp}. "
            f"Valid options are: {', '.join(valid_interps)}"
        )

    # Load MNI152 BOLD reference template from templateflow
    # type: ignore - templateflow.get() returns str, not Path, but we cast it
    epi_tpl_path = cast(Path, tflow.get(
        "MNI152NLin2009cAsym",
        resolution=2,          # 2 mm resolution
        desc="fMRIPrep",       # fMRIPrep-style boldref
        suffix="boldref"
    )) # type: ignore

    # Build flirt command
    cmd = [
        'flirt',
        '-in', str(input_file),
        '-ref', str(epi_tpl_path),
        '-out', str(output_file),
        '-applyxfm',
        '-usesqform',
        '-interp', interp
    ]

    run_command(
        cmd_name='flirt',
        cmd=cmd,
        description='Applying sform/qform transformation to MNI',
        inputs=input_file,
        outputs=output_file,
        overwrite=overwrite
    )

def compute_xfm_to_mni(
        input_file: Union[str, Path],
        output_prefix: Union[str, Path],
        type_of_transform: str = 'Rigid',
        overwrite: bool = False
) -> tuple[Path, list[str]]:
    """
    Compute transformation to MNI space using ANTs.

    Registers input to the fMRIPrep BOLD reference template (MNI152NLin2009cAsym).

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (subject space)
    output_prefix : str or Path
        Base name for output files (without extension)
    type_of_transform : str, default='Rigid'
        ANTs transform type: 'Rigid', 'Affine', 'SyN', etc.
    overwrite : bool, default=False
        If True, always run and overwrite existing outputs.
        If False, skip if outputs exist.

    Returns
    -------
    warped_file : Path
        Path to warped image in MNI space
    transform_files : list[str]
        Transform files (subject → MNI). Use with invert=True in
        apply_xfm to go MNI → subject.
    """
    sub2mni_file = Path(output_prefix).with_suffix(".nii.gz")
    sub2mni_dir = sub2mni_file.parent
    existing_xfm = sorted(sub2mni_dir.glob(f"{Path(output_prefix).stem}_*.mat"))
    if not overwrite and sub2mni_file.exists() and existing_xfm:
        logger.info("Output files exist and overwrite=False. Skipping MNI registration.")
        return sub2mni_file, [str(p) for p in existing_xfm]

    # Load MNI152 BOLD reference template from templateflow
    # type: ignore - templateflow.get() returns str, not Path, but we cast it
    epi_tpl_path = cast(Path, tflow.get(
        "MNI152NLin2009cAsym",
        resolution=2,
        desc="fMRIPrep",
        suffix="boldref"
    ))  # type: ignore

    return compute_xfm(
        moving=input_file,
        fixed=epi_tpl_path,
        output_prefix=output_prefix,
        type_of_transform=type_of_transform,
        overwrite=overwrite
    )

def apply_mask(
        input_file: Union[str, Path],
        mask_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Apply a binary mask to a NIfTI image using numpy.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    mask_file : str or Path
        Path to mask NIfTI file
    output_file : str or Path
        Path to output NIfTI file
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping apply_mask.")
        return
    
    logger.info("Applying mask...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load images
    img = nib.load(str(input_file))
    mask = nib.load(str(mask_file))
    
    # Get data
    data = img.get_fdata()
    mask_data = mask.get_fdata()

    # Align mask for broadcasting onto the data (e.g., 3D mask over 4D time series)
    if mask_data.shape == data.shape:
        mask_aligned = mask_data
    elif mask_data.ndim == data.ndim - 1 and data.shape[:mask_data.ndim] == mask_data.shape:
        mask_aligned = mask_data[..., np.newaxis]
    else:
        raise ValueError(
            f"Mask shape {mask_data.shape} is incompatible with data shape {data.shape}"
        )
    
    # Apply mask (multiply)
    masked = data * (mask_aligned > 0).astype(data.dtype)
    
    # Save output
    out_img = Nifti1Image(masked, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")


def erode_mask(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    overwrite: bool = False
) -> Path:
    """Apply binary erosion (2 voxels) to a mask.

    Uses 6-connected structuring element (face neighbors only) for conservative
    erosion. Removes two voxel layers from the mask boundary.

    Parameters
    ----------
    input_file : str or Path
        Path to input binary mask NIfTI
    output_file : str or Path
        Path to save eroded mask
    overwrite : bool, default=False
        If True, overwrite existing output

    Returns
    -------
    Path
        Path to eroded mask file
    """
    output_path = Path(output_file)

    if not overwrite and output_path.exists():
        logger.info(f"Output exists and overwrite=False. Skipping erosion: {output_path}")
        return output_path

    logger.info(f"Eroding mask by 2 voxels: {input_file}")

    # Load mask
    img = nib.load(str(input_file))
    data = img.get_fdata() > 0  # Ensure binary

    # 6-connected structuring element (face neighbors only)
    structure = generate_binary_structure(3, 1)

    # Apply erosion
    eroded = binary_erosion(data, structure=structure, iterations=2)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eroded_img = Nifti1Image(eroded.astype("uint8"), img.affine, img.header)
    save(eroded_img, output_path)

    logger.info(f"Eroded mask: {int(data.sum())} -> {int(eroded.sum())} voxels")
    return output_path


def parse_murfi_design(
    design_path: Union[str, Path],
) -> dict:
    """Parse a MURFI design matrix file.

    MURFI (mu-rt feedback imaging) design files contain pre-computed design
    matrices with task regressors and motion parameters. This function parses
    these files for use in pineuro's feedback pipeline.

    File Format
    -----------
    The first line is a header comment starting with '#' that contains column
    names in the format: "# ColName(contrast) ColName(contrast) ..."

    Subsequent lines contain space-separated numeric values, one row per volume.

    Standard columns:
    - Regulation or similar task names: HRF-convolved task regressor
    - trend_order_0: Intercept (constant 1)
    - trend_order_1: Linear drift (volume number)
    - motion_diff_tx/ty/tz: Translation differences (mm)
    - motion_diff_rx/ry/rz: Rotation differences (radians)

    Parameters
    ----------
    design_path : str or Path
        Path to MURFI design .mat file.

    Returns
    -------
    dict
        Dictionary containing:
        - 'task_regressors': ndarray, shape (n_volumes, n_task_cols)
            Task regressor columns (HRF-convolved). Empty array if no task columns.
        - 'motion_diff': ndarray, shape (n_volumes, 6)
            Motion difference parameters [tx, ty, tz, rx, ry, rz].
            Translations in mm, rotations in radians (as stored in MURFI).
        - 'intercept': ndarray, shape (n_volumes,)
            Intercept column.
        - 'drift': ndarray, shape (n_volumes,)
            Linear drift column.
        - 'column_names': list of str
            Original column names from header.
        - 'n_volumes': int
            Number of volumes (rows) in the design.

    Raises
    ------
    FileNotFoundError
        If design_path does not exist.
    ValueError
        If file format is invalid or required columns are missing.

    Notes
    -----
    MURFI stores rotation parameters in radians. The returned motion_diff
    preserves these units. Consumers may need to convert rotations to degrees
    if their pipeline expects that format.

    Task regressors are identified as columns that are NOT one of:
    trend_order_0, trend_order_1, motion_diff_*.

    Examples
    --------
    >>> design = parse_murfi_design('/path/to/design-00017.mat')
    >>> print(design['n_volumes'])
    150
    >>> print(design['task_regressors'].shape)
    (150, 1)
    >>> print(design['motion_diff'].shape)
    (150, 6)
    """
    design_path = Path(design_path)
    if not design_path.exists():
        raise FileNotFoundError(f"MURFI design file not found: {design_path}")

    # Read all lines
    with open(design_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"MURFI design file is empty: {design_path}")

    # Parse header (first line starting with #)
    header_line = lines[0].strip()
    if not header_line.startswith('#'):
        raise ValueError(
            f"MURFI design file must start with header comment: {design_path}"
        )

    # Extract column names from header
    # Format: "# ColName(contrast) ColName(contrast) ..."
    header_content = header_line[1:].strip()  # Remove leading #
    # Split on whitespace but preserve column names with parentheses
    import re
    column_pattern = re.compile(r'(\S+\(\d+\))')
    column_matches = column_pattern.findall(header_content)

    if not column_matches:
        raise ValueError(
            f"Could not parse column names from header: {header_line}"
        )

    # Extract base column names (without the contrast suffix)
    column_names = []
    for col in column_matches:
        # Extract name before parentheses
        base_name = col.split('(')[0]
        column_names.append(base_name)

    logger.debug(f"MURFI design columns: {column_names}")

    # Parse data rows
    data_rows = []
    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            values = [float(v) for v in line.split()]
            data_rows.append(values)
        except ValueError as e:
            raise ValueError(
                f"Invalid numeric data at line {i}: {line}"
            ) from e

    if not data_rows:
        raise ValueError(f"No data rows found in MURFI design file: {design_path}")

    # Convert to numpy array
    data = np.array(data_rows, dtype=np.float64)
    n_volumes = data.shape[0]
    n_cols = data.shape[1]

    if n_cols != len(column_names):
        raise ValueError(
            f"Column count mismatch: header has {len(column_names)} columns, "
            f"data has {n_cols} columns"
        )

    # Identify column indices
    motion_cols = ['motion_diff_tx', 'motion_diff_ty', 'motion_diff_tz',
                   'motion_diff_rx', 'motion_diff_ry', 'motion_diff_rz']
    nuisance_cols = ['trend_order_0', 'trend_order_1'] + motion_cols

    # Find motion parameter columns
    motion_indices = []
    for mcol in motion_cols:
        if mcol in column_names:
            motion_indices.append(column_names.index(mcol))
        else:
            raise ValueError(f"Required motion column '{mcol}' not found in design")

    # Find intercept and drift columns
    if 'trend_order_0' in column_names:
        intercept_idx = column_names.index('trend_order_0')
        intercept = data[:, intercept_idx]
    else:
        logger.warning("No trend_order_0 (intercept) column found, using ones")
        intercept = np.ones(n_volumes, dtype=np.float64)

    if 'trend_order_1' in column_names:
        drift_idx = column_names.index('trend_order_1')
        drift = data[:, drift_idx]
    else:
        logger.warning("No trend_order_1 (drift) column found, using 1:n_volumes")
        drift = np.arange(1, n_volumes + 1, dtype=np.float64)

    # Extract motion parameters
    motion_diff = data[:, motion_indices]

    # Identify task regressors (columns that are not nuisance)
    task_indices = []
    for i, name in enumerate(column_names):
        if name not in nuisance_cols:
            task_indices.append(i)

    if task_indices:
        task_regressors = data[:, task_indices]
        task_names = [column_names[i] for i in task_indices]
        logger.info(f"Found {len(task_indices)} task regressor(s): {task_names}")
    else:
        task_regressors = np.empty((n_volumes, 0), dtype=np.float64)
        logger.info("No task regressors found (nuisance-only design)")

    logger.info(
        f"Parsed MURFI design: {n_volumes} volumes, "
        f"{len(task_indices)} task regressor(s), 6 motion parameters"
    )

    return {
        'task_regressors': task_regressors,
        'motion_diff': motion_diff,
        'intercept': intercept,
        'drift': drift,
        'column_names': column_names,
        'n_volumes': n_volumes,
    }
