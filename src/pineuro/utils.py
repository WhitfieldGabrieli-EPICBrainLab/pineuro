"""
Utilities and command execution framework for pineuro.

Summary
-------
Centralized command execution, registration helpers, and general utilities used across
the real-time pipeline.

Functions
---------
- run_command: Shell execution with validation and overwrite control
- dicom_to_nifti: Convert DICOM files to NIfTI via dcm2niix
- get_yeo_labels: Return Yeo network labels (7 or 17 networks)
- compute_mean, compute_xfm, apply_xfm, compute_xfm_to_mni, apply_mask:
  registration and masking utilities (ANTs-backed for registration)
- apply_xfm_sform_to_mni: Resample to MNI grid using header information

Workflow
--------
- Validate inputs/outputs and optionally skip when outputs already exist
- Execute external commands with logging and error handling
- Provide small FSL/dcm2niix/templateflow helpers used by preprocessing/mask extraction

Dependencies
------------
FSL, ANTsPy, dcm2niix, templateflow, numpy, pathlib

Notes
-----
All external commands route through run_command() for consistent logging,
input/output validation, and overwrite handling. ANTs-based registration
uses in-process calls for speed while maintaining consistent logging. FSL
commands continue to use standard flags (-noresample, -noresampblur) for
reproducible behavior. This module is the gateway for anything that shells
out, so callers get uniform error reporting and overwrite semantics. Further
reading: see docs/references.md.

References
----------
[1] Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith,
    S. M. (2012). FSL. NeuroImage, 62(2), 782-790.
[2] Li, X., Morgan, P. S., Ashburner, J., Smith, J., & Rorden, C. (2016).
    The first step for neuroimaging data analysis: DICOM to NIfTI conversion.
    Journal of Neuroscience Methods, 264, 47-56.
"""

# Standard library imports
import shutil
from typing import Union, List, Optional, cast
from pathlib import Path
import subprocess
import tempfile

# Third-party imports
import numpy as np
import nibabel as nib
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
from templateflow import api as tflow
import pydicom
from scipy.ndimage import binary_erosion, generate_binary_structure

# Local imports

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

# Performance timing infrastructure
import time
import json
from typing import Dict, Any
from contextlib import contextmanager

# Global timing storage
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

    Parameters
    ----------
    directory : Path
        Directory to search for DICOM files

    Returns
    -------
    List[Path]
        Sorted list of DICOM file paths

    """
    dicom_files = []
    for ext in DICOM_EXTENSIONS:
        dicom_files.extend(directory.glob(ext))

    # Filter to only files and sort by name
    valid_dicoms = [f for f in dicom_files if f.is_file()]
    valid_dicoms.sort()
    return valid_dicoms


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
        if hasattr(dcm, 'RepetitionTime'):
            return float(dcm.RepetitionTime) / 1000.0
        logger.warning(f"TR not found in DICOM, using default {default}s")
        return default
    except Exception as e:
        logger.warning(f"Error reading TR: {e}, using default {default}s")
        return default


# ==============================================================
#  Helper Functions
# ==============================================================
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

# ==============================================================
#  FSL FUNCTIONS
# ==============================================================

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
