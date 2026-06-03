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
    - dcm2niix : DICOM to NIfTI conversion

Python:
    - ANTsPy : Registration, motion correction, and brain extraction
    - nilearn : MNI resampling (resample_to_img)

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
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Union, cast

import nibabel as nib

# --- Third-Party ---
import numpy as np
import pydicom  # DICOM header reading
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
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
_timing_data: dict[str, Any] = {}
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


def get_timing_data() -> dict[str, Any]:
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

    with open(output_path, "w") as f:
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
    timing_info = {"name": name, "category": category, "start_time": start_time}

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

    # FSL is no longer required. Motion correction, brain extraction, and
    # MNI alignment are all handled by ANTsPy and nilearn.

    # Check dcm2niix availability (required for DICOM conversion)
    if shutil.which("dcm2niix") is None:
        warnings.append(
            "dcm2niix not found in PATH. DICOM conversion requires dcm2niix. "
            "Install via: apt install dcm2niix (Linux), brew install dcm2niix (macOS), "
            "or winget install dcm2niix (Windows)"
        )

    # Platform-specific warnings
    if platform == "linux" and config.is_wayland():
        warnings.append(
            "Wayland display server detected. Some X11-based features (window positioning) "
            "may not work. Consider using an X11/Xorg session for full compatibility."
        )

    return warnings


def outputs_exist(outputs: Union[Path, list[Path], str, list[str]], overwrite: bool = False) -> bool:
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

    def __init__(self, callback: Callable[[str, str], None] | None = None, log: logging.Logger | None = None):
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


def _gamma_pdf_rate(t: np.ndarray, shape: float, rate: float) -> np.ndarray:
    """Gamma PDF with rate parameterization.

    Computes: rate^shape / Gamma(shape) * t^(shape-1) * exp(-rate*t)

    Parameters
    ----------
    t : ndarray
        Time points.
    shape : float
        Shape parameter (alpha).
    rate : float
        Rate parameter (beta = 1/scale).

    Returns
    -------
    pdf : ndarray
        Gamma PDF values.
    """
    from scipy.special import gamma as gamma_func

    # Avoid division by zero at t=0
    t_safe = np.maximum(t, 1e-10)

    pdf = np.power(rate, shape) / gamma_func(shape) * np.power(t_safe, shape - 1) * np.exp(-rate * t_safe)
    # Set t=0 to 0 (limit of t^(a-1) as t->0 for a>1)
    pdf[t == 0] = 0.0

    return np.asarray(pdf)


def _compute_hrf_highres(tr: float, duration: float = 32.0) -> np.ndarray:
    """Compute normalized high-resolution (16 Hz) double-gamma HRF.

    Parameters
    ----------
    tr : float
        Repetition time in seconds (used for rate parameter).
    duration : float, optional
        Duration of HRF in seconds. Default is 32s.

    Returns
    -------
    hrf_hires : ndarray
        High-resolution HRF at 16 Hz, normalized to sum to 1.0.
    """
    hires_dt = 1.0 / 16.0
    shape_pos = 6.0
    shape_neg = 16.0
    undershoot_ratio = 6.0
    rate = hires_dt

    n_samples = int(duration / hires_dt)
    t = np.arange(n_samples, dtype=np.float64)
    hrf_hires = _gamma_pdf_rate(t, shape_pos, rate) - _gamma_pdf_rate(t, shape_neg, rate) / undershoot_ratio

    hrf_sum = np.sum(hrf_hires)
    if hrf_sum > 0:
        hrf_hires = hrf_hires / hrf_sum

    return hrf_hires


def compute_hrf(tr: float, duration: float = 32.0) -> np.ndarray:
    """Compute canonical double-gamma HRF.

    Parameters
    ----------
    tr : float
        Repetition time in seconds.
    duration : float, optional
        Duration of HRF in seconds. Default is 32s.

    Returns
    -------
    hrf : ndarray
        HRF sampled at TR intervals, normalized to sum to 1.0.

    Notes
    -----
    Uses high-resolution sampling for accuracy:
    - Samples at 16 samples per TR internally, then downsamples to TR
    - Positive gamma: shape=6
    - Negative gamma: shape=16
    - Undershoot ratio: 1/6
    - Rate parameter: 1/16
    - Normalized to sum to 1.0 for proper convolution scaling

    Examples
    --------
    >>> from pineuro.utils import compute_hrf
    >>> hrf = compute_hrf(tr=1.2)
    >>> print(hrf.shape)  # (27,) for 32s duration at TR=1.2
    """
    hires_dt = 1.0 / 16.0
    hrf_hires = _compute_hrf_highres(tr, duration)

    # Downsample to TR resolution
    samples_per_tr = int(tr / hires_dt)
    n_tr_samples = int(duration / tr)
    hrf = np.zeros(n_tr_samples, dtype=np.float64)

    for i in range(n_tr_samples):
        hires_idx = i * samples_per_tr
        if hires_idx < len(hrf_hires):
            hrf[i] = hrf_hires[hires_idx]

    return hrf


def compute_boxcar_task_regressor(
    n_baseline: int,
    n_volumes: int,
    tr: float,
    hrf_duration: float = 32.0,
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
    hrf_duration : float, optional
        Duration of HRF in seconds. Default is 32s.

    Returns
    -------
    regressor : ndarray, shape (n_volumes,)
        Task regressor values for each volume.

    Notes
    -----
    The convolution is performed at high temporal resolution (16 Hz) for
    accuracy, then downsampled to TR. This produces the characteristic
    HRF response shape:
    - Stays at 0 during baseline
    - Ramps up after baseline onset following HRF shape
    - Reaches steady state (~1.0) after ~25-30s

    Examples
    --------
    >>> from pineuro.utils import compute_boxcar_task_regressor
    >>> regressor = compute_boxcar_task_regressor(n_baseline=25, n_volumes=150, tr=1.2)
    >>> print(regressor.shape)  # (150,)
    """
    hrf_hires = _compute_hrf_highres(tr, hrf_duration)
    hires_dt = 1.0 / 16.0
    samples_per_tr = int(tr / hires_dt)

    # Create high-resolution boxcar
    n_hires_samples = n_volumes * samples_per_tr + len(hrf_hires)
    onset_hires = n_baseline * samples_per_tr

    boxcar_hires = np.zeros(n_hires_samples, dtype=np.float64)
    boxcar_hires[onset_hires:] = 1.0

    # Convolve at high resolution
    convolved_hires = np.convolve(boxcar_hires, hrf_hires, mode="full")

    # Downsample back to volume resolution
    regressor = np.zeros(n_volumes, dtype=np.float64)
    for i in range(n_volumes):
        hires_idx = i * samples_per_tr
        if hires_idx < len(convolved_hires):
            regressor[i] = convolved_hires[hires_idx]

    return regressor


# ==============================================================
# Motion Quality Control (FD, DVARS, Siemens MoCo parsing)
# ==============================================================
# Shared head-motion and image-stability metrics used by both the real-time
# feedback estimator (feedback.py) and the live collection-QC monitor
# (collection_qc.py). Kept here as pure functions so the two paths cannot
# silently diverge.

# Assumed head radius (mm) for framewise-displacement arc-length conversion.
# Standard assumption from Power et al. (2012).
HEAD_RADIUS_MM = 50.0

# Pre-compiled regex for Siemens MoCo motion params in DICOM/JSON ImageComments.
_MOTION_PATTERN = re.compile(r"Motion:\s*([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)")


def compute_fd(
    motion_params: NDArray[np.float64],
    prev_motion: NDArray[np.float64] | None,
) -> float:
    """Compute framewise displacement between two motion parameter vectors.

    FD (Power et al., 2012) is the sum of absolute frame-to-frame differences
    across all 6 motion parameters. Rotations (indices 3:6, in degrees) are
    converted to mm of arc at HEAD_RADIUS_MM.

    Parameters
    ----------
    motion_params : ndarray, shape (6,)
        Current motion params [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z].
        Translations in mm, rotations in degrees.
    prev_motion : ndarray, shape (6,) or None
        Previous volume's motion params. Returns 0.0 when None (first volume).

    Returns
    -------
    float
        Framewise displacement in mm.
    """
    if prev_motion is None:
        return 0.0
    # Translations (0-2) are already in mm; rotations (3-5) become arc length:
    # arc = radius * angle_radians = HEAD_RADIUS_MM * (angle_degrees * pi/180).
    diff_mm = (motion_params - prev_motion).copy()
    diff_mm[3:6] *= HEAD_RADIUS_MM * np.pi / 180.0
    return float(np.sum(np.abs(diff_mm)))


def compute_dvars(
    voxel_values: NDArray[np.float64],
    prev_voxel_values: NDArray[np.float64] | None,
) -> float:
    """Compute DVARS: the spatial root mean square of the temporal derivative.

    DVARS (Power et al., 2012) is sqrt(mean((I_t - I_{t-1})^2)) across the
    supplied voxels.

    Parameters
    ----------
    voxel_values : ndarray
        Voxel intensities for the current volume.
    prev_voxel_values : ndarray or None
        Previous volume's voxel intensities. Returns 0.0 when None (first
        volume).

    Returns
    -------
    float
        DVARS value for this volume.
    """
    if prev_voxel_values is None:
        return 0.0
    diff = voxel_values - prev_voxel_values
    return float(np.sqrt(np.mean(diff**2)))


def standardize_dvars(raw_dvars: float, prior_raw_dvars: Sequence[float]) -> float:
    """Standardize a raw DVARS value by the running median of prior raw DVARS.

    Raw DVARS (:func:`compute_dvars`) is in scanner-intensity units, so its absolute
    scale is arbitrary and acquisition-dependent (typically tens of units for BOLD EPI).
    A fixed threshold like 1.5 is meaningless against it. Dividing by the running median
    of prior raw DVARS yields a scale-free metric where a typical (clean) volume ≈ 1.0 and
    a spike is >1, so a fixed threshold means "this many times the run's typical
    volume-to-volume change" and transfers across protocols. The median is robust, so
    spike volumes already in the history do not inflate the scale.

    The first real value (no prior history) is defined as 1.0 — it *is* the reference.

    Parameters
    ----------
    raw_dvars : float
        Raw DVARS for the current volume (0.0 on the first volume).
    prior_raw_dvars : sequence of float
        Raw DVARS values from earlier volumes (excluding the first-volume 0.0).

    Returns
    -------
    float
        Standardized DVARS (≈1.0 for a clean volume), or 0.0 when ``raw_dvars`` is 0.
    """
    if raw_dvars <= 0:
        return 0.0
    if len(prior_raw_dvars) == 0:
        return 1.0
    scale = float(np.median(prior_raw_dvars))
    return raw_dvars / scale if scale > 0 else 1.0


def parse_siemens_motion_comment(comments: str) -> NDArray[np.float64] | None:
    """Parse Siemens MoCo motion params from an ImageComments string.

    The dcm2niix JSON sidecar and the DICOM ImageComments field carry motion
    params as either:
    - "Reference volume for motion correction." for reference volumes (zeros)
    - "Motion: -0.01,-0.00,-0.05,0.01,0.05,0.00" for motion-corrected volumes

    Parameters
    ----------
    comments : str
        The ImageComments text.

    Returns
    -------
    ndarray, shape (6,) or None
        zeros(6) for the reference-volume sentinel, the 6 parsed floats on a
        ``Motion:`` match, or None when neither matches. Callers apply their
        own no-match policy.
    """
    if "Reference volume for motion correction" in comments:
        return np.zeros(6, dtype=np.float64)
    match = _MOTION_PATTERN.search(comments)
    if match:
        return np.array([float(m) for m in match.groups()], dtype=np.float64)
    return None


# ==============================================================
# Installation Check
# ==============================================================


class CommandNotFoundError(Exception):
    """Raised when an FSL command is not installed or found in PATH."""


def _check_command_installation(cmd: Union[str, Path] = "fsl") -> None:
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
        raise CommandNotFoundError(f"{cmd} is not found in PATH.")


# ==============================================================
#  DICOM Utilities
# ==============================================================

# Basenames that are never volume files (OS metadata). Matched case-insensitively.
_JUNK_BASENAMES = frozenset({"thumbs.db", "desktop.ini"})


def _is_obvious_junk(path: Path) -> bool:
    """Return True for files that are never DICOM/NIfTI volumes, detected cheaply.

    Catches OS metadata without a header read: dot-prefixed files (``.DS_Store``,
    AppleDouble ``._*``, atomic-copy ``.*.tmp``) and known junk basenames
    (``Thumbs.db``, ``desktop.ini``). Extensionless files are NOT junk — some
    scanners export DICOMs without an extension.
    """
    return path.name.startswith(".") or path.name.lower() in _JUNK_BASENAMES


def _filter_volume_files(directory: Path, validator: Callable[[Path], bool], *, kind: str) -> list[Path]:
    """List files in ``directory`` accepted by ``validator``, sorted by path.

    Skips obvious junk (see :func:`_is_obvious_junk`) and ``.json`` sidecars
    before the header read, then validates the rest. Emits a single warning
    naming the non-volume files that were ignored. ``kind`` labels the volume
    type in log messages (e.g. ``"DICOM/NIfTI"``).
    """
    valid: list[Path] = []
    skipped: list[Path] = []
    for f in directory.iterdir():
        if not f.is_file():
            continue
        if f.name.lower().endswith(".json"):
            # NIfTI sidecar — an expected companion, not a volume; exclude silently.
            continue
        if _is_obvious_junk(f) or not validator(f):
            skipped.append(f)
        else:
            valid.append(f)
    valid.sort()

    if skipped:
        names = sorted(f.name for f in skipped)
        shown = ", ".join(names[:20])
        if len(names) > 20:
            shown += f", and {len(names) - 20} more"
        if valid:
            logger.warning(f"Ignored {len(skipped)} non-volume file(s) in {directory}: {shown}")
        else:
            logger.warning(
                f"No valid {kind} volumes in {directory}; ignored {len(skipped)} non-volume file(s): {shown}"
            )
    return valid


# ==============================================================
#  Collection completion
# ==============================================================

# Sentinel written into an attempt directory by ``collect_volumes`` when a
# variable-duration task signals it has finished (``task_done_event``). Its
# presence means the collection ran to its natural end even though fewer
# volumes were collected than the worst-case ceiling. A dotfile so it is
# excluded from volume counts by :func:`_is_obvious_junk`.
COLLECTION_COMPLETE_MARKER = ".collection_complete"


def count_volume_files(folder: Path) -> int:
    """Count collected volume files in an attempt directory.

    Excludes the completion marker and OS junk (dotfiles), ``.json`` NIfTI
    sidecars, and the ``collectionlog.tsv`` QC log — none of which are volumes.
    """
    if not folder.exists():
        return 0
    return sum(
        1
        for f in folder.iterdir()
        if f.is_file()
        and not _is_obvious_junk(f)
        and not f.name.lower().endswith(".json")
        and f.name != "collectionlog.tsv"
    )


def collection_is_complete(folder: Path, expected_volumes: int) -> bool:
    """Whether an attempt directory represents a complete collection.

    Complete if the task signaled done (the :data:`COLLECTION_COMPLETE_MARKER`
    is present — the authoritative signal for variable-duration tasks that stop
    early) OR enough volume files were collected to meet ``expected_volumes``
    (the path for fixed-length tasks, which never write the marker).
    """
    if (folder / COLLECTION_COMPLETE_MARKER).exists():
        return True
    return count_volume_files(folder) >= expected_volumes


def is_valid_dicom(file_path: Path, validate_pixels: bool = False) -> bool:
    """
    Check if a file is a valid DICOM by attempting to read its header.

    Parameters
    ----------
    file_path : Path
        Path to file to check
    validate_pixels : bool
        If True, also verify that pixel data can be read (catches truncated
        files from incomplete network transfers). Adds ~10-50ms per file.

    Returns
    -------
    bool
        True if file is a valid DICOM, False otherwise

    """
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=not validate_pixels)
        if validate_pixels:
            # Force pixel data read — raises if file is truncated
            arr = dcm.pixel_array
            if arr.size == 0:
                return False
        return True
    except Exception:  # pydicom raises many exception types for corrupt/non-DICOM files
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
        if hasattr(dcm, "RepetitionTime") and dcm.RepetitionTime is not None:
            return float(dcm.RepetitionTime) / 1000.0

        # Search in nested sequences (enhanced/multiframe DICOM)
        for elem in dcm.iterall():
            if elem.keyword == "RepetitionTime" and elem.value is not None:
                return float(elem.value) / 1000.0

        logger.warning(f"TR not found in DICOM, using default {default}s")
        return default
    except Exception as e:
        logger.warning(f"Error reading TR: {e}, using default {default}s")
        return default


# ==============================================================
#  NIfTI Utilities
# ==============================================================


def get_volume_files(directory: Path, validate: bool = True) -> list[Path]:
    """
    Get all volume files (DICOM or NIfTI) in a directory, sorted by filename.

    By default, filters out non-volume files (OS metadata like ``.DS_Store``,
    text files, JSON sidecars, scanner junk) by reading each file's header, so
    stray files never enter the pipeline. Validation is header-only and supports
    extensionless DICOMs (common with some scanner exports).

    Parameters
    ----------
    directory : Path
        Directory to search for volume files
    validate : bool, default=True
        If True, return only files that validate as DICOM or NIfTI and log a
        warning naming any non-volume files that were ignored. If False, return
        every file unfiltered (escape hatch for callers that must list raw
        contents, e.g. to distinguish an empty directory from one full of junk).

    Returns
    -------
    List[Path]
        Sorted list of volume file paths

    """
    if not validate:
        volume_files = [f for f in directory.iterdir() if f.is_file()]
        volume_files.sort()
        return volume_files
    return _filter_volume_files(directory, is_valid_volume, kind="DICOM/NIfTI")


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
    return name.endswith(".nii") or name.endswith(".nii.gz")


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
    except Exception:  # nibabel raises various errors for corrupt/non-NIfTI files
        return False


def is_valid_volume(file_path: Path, validate_pixels: bool = False) -> bool:
    """
    Check if a file is a valid volume (DICOM or NIfTI).

    Auto-detects file type and validates accordingly. Ignores temporary files
    (used by simulator for atomic copy).

    Parameters
    ----------
    file_path : Path
        Path to file to check
    validate_pixels : bool
        If True, verify that pixel/voxel data can be read (DICOM only).
        Catches truncated files from incomplete network transfers.

    Returns
    -------
    bool
        True if file is a valid DICOM or NIfTI, False otherwise

    """
    # Skip temporary files from atomic copy (e.g., .IM-0001.dcm.tmp)
    if file_path.name.startswith(".") and file_path.name.endswith(".tmp"):
        return False

    if is_nifti_file(file_path):
        return is_valid_nifti(file_path)
    else:
        return is_valid_dicom(file_path, validate_pixels=validate_pixels)


def is_moco_series(file_path: Path) -> bool | None:
    """Check if a DICOM file belongs to a Siemens MoCo-derived series.

    Siemens scanners with motion correction (MoCo) enabled produce two
    series per acquisition: the original and a corrected "MoCoSeries".
    This function distinguishes them by checking SeriesDescription.

    Parameters
    ----------
    file_path : Path
        Path to a DICOM file.

    Returns
    -------
    bool or None
        True if from a MoCo series, False if from the original series,
        None if the file is not a DICOM or SeriesDescription is missing.
    """
    if is_nifti_file(file_path):
        return None
    try:
        dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
        desc = getattr(dcm, "SeriesDescription", None)
        if desc is None:
            return None
        return bool(desc == "MoCoSeries")
    except Exception as e:
        logger.warning(f"Could not check MoCo series: {e}")
        return None


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
    if name.lower().endswith(".nii.gz"):
        base_name = name[:-7]  # Remove .nii.gz
    elif name.lower().endswith(".nii"):
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
        with open(json_path) as f:
            metadata = json.load(f)

        # Check for RepetitionTime field (standard BIDS)
        if "RepetitionTime" in metadata:
            return float(metadata["RepetitionTime"])

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
    cmd: list[str],
    description: str,
    inputs: Union[str, Path, list[Union[str, Path]]] | None = None,
    outputs: Union[str, Path, list[Union[str, Path]]] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
    timing_category: str = "commands",
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
            logger.info("Input files:")
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
            logger.info("Output files:")
        for f in outputs:
            path = Path(f)
            (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)
            output_paths.append(path)
            if verbose:
                logger.info(f"  {f}")

    # Run command with timing
    with Timer(f"{cmd_name}: {description}", timing_category):
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            if verbose:
                logger.info(f"{description} completed successfully")

        except subprocess.CalledProcessError as e:
            error_msg = (
                f"{description} failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {e.returncode}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # Verify output files if specified
    if outputs is not None:
        missing_files: list[Path] = []
        for output_path in output_paths:
            if not output_path.exists():
                missing_files.append(output_path)
        if len(missing_files) > 0:
            error_msg = f"{description} completed but expected output file(s) not found:\n{missing_files}"
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
            cmd_name=job_spec["cmd_name"],
            cmd=job_spec["cmd"],
            description=job_spec["description"],
            inputs=job_spec.get("inputs"),
            outputs=job_spec.get("outputs"),
            overwrite=job_spec.get("overwrite", False),
            verbose=job_spec.get("verbose", True),
            timing_category=job_spec.get("timing_category", "parallel"),
        )
        return {"success": True, "description": job_spec.get("description")}
    except Exception as e:
        logger.exception("Parallel job failed: %s", job_spec.get("description"))
        return {
            "success": False,
            "description": job_spec.get("description"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def parallel_run_command(jobs: list[dict], n_jobs: int = -1, timing_category: str = "parallel") -> None:
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
        job_copy["timing_category"] = timing_category
        jobs_with_category.append(job_copy)

    # Run jobs in parallel
    with Timer(f"Parallel execution ({len(jobs)} jobs)", timing_category), mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_job_worker, jobs_with_category)

    # Check for failures
    failures = [r for r in results if not r["success"]]
    if failures:
        error_msgs = [f"{r['description']}:\n{r.get('traceback') or r['error']}" for r in failures]
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
    compress: bool = True,
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
        If True, delete any prior dcm2niix outputs for ``output_name`` in
        ``output_dir`` (primary file, collision-suffixed variants like
        ``outputa.nii.gz``, and multi-echo / phase splits like ``output_e2.nii.gz``)
        before re-running. Necessary because dcm2niix's default ``-w 2`` appends
        a letter on collision instead of overwriting, which otherwise leaves
        stale files behind.
        If False, skip command if the primary output file already exists.
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

    # Compression flag: -z y for gzip, -z n for uncompressed
    compress_flag = "y" if compress else "n"
    ext = ".nii.gz" if compress else ".nii"
    nifti_path = output_dir / f"{output_name}{ext}"

    # When overwriting, remove prior dcm2niix artifacts for this output_name first.
    # dcm2niix's default (-w 2) appends a letter suffix (a, b, ...) on filename
    # collision instead of overwriting, and multi-echo / phase series produce
    # _e1/_e2/_ph/_ADC variants. Without cleanup, stale files persist and the
    # primary nifti_path may end up holding old data while fresh output sits
    # next to it with a suffix.
    if overwrite:
        sidecar_exts = ("nii.gz", "nii", "json", "bval", "bvec")
        for ext_ in sidecar_exts:
            for pattern in (
                f"{output_name}.{ext_}",  # primary
                f"{output_name}[a-z].{ext_}",  # collision suffix (a, b, ...)
                f"{output_name}_*.{ext_}",  # multi-echo / phase variants (_e1, _ph, ...)
            ):
                for stale in output_dir.glob(pattern):
                    try:
                        stale.unlink()
                    except OSError as e:
                        logger.warning(f"Could not remove stale dcm2niix output {stale}: {e}")

    # Handle single file vs directory
    # If a single file is given, copy it to a temp directory to avoid converting
    # all DICOMs in the same directory
    temp_dir = None
    if dicom_path.is_file():
        temp_dir = tempfile.mkdtemp(prefix="dcm2niix_")
        temp_dicom = Path(temp_dir) / dicom_path.name
        shutil.copy2(dicom_path, temp_dicom)
        conversion_path = Path(temp_dir)
    else:
        conversion_path = dicom_path

    try:
        # Build command
        cmd = ["dcm2niix"] + ["-z", compress_flag, "-o", str(output_dir), "-f", str(output_name), str(conversion_path)]

        # Run dcm2niix
        run_command(
            cmd_name="dcm2niix",
            cmd=cmd,
            description="Converting DICOM to NIfTI",
            inputs=dicom_path,
            outputs=nifti_path,
            overwrite=overwrite,
            verbose=verbose,
        )
    finally:
        # Clean up temp directory if created
        if temp_dir is not None and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    return nifti_path


def get_yeo_labels(n_networks: int) -> list[str]:
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
        raise ValueError(f"Invalid number of networks: {n_networks}. Valid options are 7 or 17")

    # Assign labels
    if n_networks == 7:
        labels = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    else:
        labels = [
            "VisCent",
            "VisPeri",
            "SomMotA",
            "SomMotB",
            "DorsAttnA",
            "DorsAttnB",
            "SalVentAttnA",
            "SalVentAttnB",
            "LimbicA",
            "LimbicB",
            "ContC",
            "ContA",
            "ContB",
            "TempPar",
            "DefaultC",
            "DefaultA",
            "DefaultB",
        ]

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


def compute_mean(input_file: Union[str, Path], output_file: Union[str, Path], overwrite: bool = False) -> None:
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
        logger.info("Output file exists and overwrite=False. Skipping compute mean.")
        return

    logger.info("Computing mean...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load 4D image and compute mean along time axis
    img = cast(Nifti1Image, nib.load(str(input_file)))
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
    type_of_transform: str = "Rigid",
    overwrite: bool = False,
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
        apply_xfm to go fixed → moving (for Rigid/Affine only).
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
            fixed=fixed_img, moving=moving_img, type_of_transform=type_of_transform, verbose=False
        )

        ants.image_write(registration["warpedmovout"], str(output_file))
        logger.info(f"Output: {output_file}")

        # Copy transform files to output directory
        transform_files: list[str] = []
        for i, xfm in enumerate(registration["fwdtransforms"]):
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
    interpolator: str = "nearestNeighbor",
    invert: bool = False,
    overwrite: bool = False,
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

    valid_interps = {"nearestNeighbor", "linear", "bSpline", "lanczosWindowedSinc"}
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
        whichtoinvert=whichtoinvert,
    )

    ants.image_write(transformed, str(output_path))
    logger.info(f"Output: {output_path}")


def _get_mni_epi_template() -> Path:
    """Get MNI152 BOLD reference template path from templateflow."""
    import templateflow.api as tflow

    return cast(Path, tflow.get("MNI152NLin2009cAsym", resolution=2, desc="fMRIPrep", suffix="boldref"))  # type: ignore


def apply_xfm_sform_to_mni(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    interp: str = "nearest",
    overwrite: bool = False,
) -> None:
    """
    Resample image to MNI space using sform/qform matrices from NIfTI headers.

    Uses the affine transforms stored in the NIfTI headers to resample the
    input image onto the MNI152NLin2009cAsym BOLD reference grid from
    templateflow. No explicit transformation matrix file is required.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (should have sform/qform set to MNI space)
    output_file : str or Path
        Path for output file in MNI space
    interp : str, default='nearest'
        nilearn interpolation method ('nearest', 'linear', or 'continuous').
    overwrite : bool, default=False
        If True, always run and overwrite existing outputs.
        If False, skip if output file already exists.
    """
    from nilearn.image import resample_to_img

    output_file = Path(output_file)

    if not overwrite and output_file.exists():
        logger.info(f"Skipping sform-to-MNI resampling (output exists): {output_file.name}")
        return

    epi_tpl_path = _get_mni_epi_template()

    logger.info(f"Resampling to MNI space: {Path(input_file).name}")
    source_img = cast(Nifti1Image, nib.load(str(input_file)))
    target_img = cast(Nifti1Image, nib.load(str(epi_tpl_path)))

    resampled = resample_to_img(source_img, target_img, interpolation=interp, force_resample=True, copy_header=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(resampled, str(output_file))


def compute_xfm_to_mni(
    input_file: Union[str, Path],
    output_prefix: Union[str, Path],
    type_of_transform: str = "SyN",
    overwrite: bool = False,
) -> tuple[Path, list[str], list[str]]:
    """
    Compute transformation to MNI space using ANTs.

    Registers input to the fMRIPrep BOLD reference template (MNI152NLin2009cAsym).
    Unlike compute_xfm, this function saves both forward and inverse transforms,
    which is necessary for SyN (nonlinear) registration where warp fields cannot
    be inverted on-the-fly.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file (subject space)
    output_prefix : str or Path
        Base name for output files (without extension)
    type_of_transform : str, default='SyN'
        ANTs transform type: 'Rigid', 'Affine', 'SyN', etc.
    overwrite : bool, default=False
        If True, always run and overwrite existing outputs.
        If False, skip if outputs exist.

    Returns
    -------
    warped_file : Path
        Path to warped image in MNI space
    fwd_transform_files : list[str]
        Forward transform files (subject → MNI).
    inv_transform_files : list[str]
        Inverse transform files (MNI → subject). Use these directly
        with apply_xfm (invert=False) to go MNI → subject.
    """
    import ants

    sub2mni_file = Path(output_prefix).with_suffix(".nii.gz")
    sub2mni_dir = sub2mni_file.parent
    sub2mni_dir.mkdir(parents=True, exist_ok=True)
    prefix_stem = Path(output_prefix).stem

    # Check if outputs exist
    existing_fwd = sorted(sub2mni_dir.glob(f"{prefix_stem}_fwd_*"))
    existing_inv = sorted(sub2mni_dir.glob(f"{prefix_stem}_inv_*"))
    if not overwrite and sub2mni_file.exists() and existing_fwd and existing_inv:
        logger.info("Output files exist and overwrite=False. Skipping MNI registration.")
        return sub2mni_file, [str(p) for p in existing_fwd], [str(p) for p in existing_inv]

    epi_tpl_path = _get_mni_epi_template()

    logger.info(f"Computing MNI transformation (ANTs {type_of_transform})...")
    logger.info(f"Input: {input_file}")
    logger.info(f"Template: {epi_tpl_path}")

    with Timer(f"ANTs MNI registration ({type_of_transform})", "registration"):
        fixed_img = ants.image_read(str(epi_tpl_path))
        moving_img = ants.image_read(str(input_file))

        registration = ants.registration(
            fixed=fixed_img, moving=moving_img, type_of_transform=type_of_transform, verbose=False
        )

        ants.image_write(registration["warpedmovout"], str(sub2mni_file))
        logger.info(f"Output: {sub2mni_file}")

        # Copy forward transform files (subject → MNI)
        fwd_transform_files: list[str] = []
        for i, xfm in enumerate(registration["fwdtransforms"]):
            # Handle .nii.gz (two suffixes) vs .mat (one suffix)
            suffixes = "".join(Path(xfm).suffixes)
            new_path = sub2mni_dir / f"{prefix_stem}_fwd_{i}{suffixes}"
            shutil.copy(xfm, new_path)
            fwd_transform_files.append(str(new_path))
            logger.info(f"Forward transform: {new_path}")

        # Copy inverse transform files (MNI → subject)
        inv_transform_files: list[str] = []
        for i, xfm in enumerate(registration["invtransforms"]):
            # Handle .nii.gz (two suffixes) vs .mat (one suffix)
            suffixes = "".join(Path(xfm).suffixes)
            new_path = sub2mni_dir / f"{prefix_stem}_inv_{i}{suffixes}"
            shutil.copy(xfm, new_path)
            inv_transform_files.append(str(new_path))
            logger.info(f"Inverse transform: {new_path}")

    return sub2mni_file, fwd_transform_files, inv_transform_files


def apply_mask(
    input_file: Union[str, Path], mask_file: Union[str, Path], output_file: Union[str, Path], overwrite: bool = False
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
        logger.info("Output file exists and overwrite=False. Skipping apply_mask.")
        return

    logger.info("Applying mask...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load images
    img = cast(Nifti1Image, nib.load(str(input_file)))
    mask = cast(Nifti1Image, nib.load(str(mask_file)))

    # Get data
    data = img.get_fdata()
    mask_data = mask.get_fdata()

    # Align mask for broadcasting onto the data (e.g., 3D mask over 4D time series)
    if mask_data.shape == data.shape:
        mask_aligned = mask_data
    elif mask_data.ndim == data.ndim - 1 and data.shape[: mask_data.ndim] == mask_data.shape:
        mask_aligned = mask_data[..., np.newaxis]
    else:
        raise ValueError(f"Mask shape {mask_data.shape} is incompatible with data shape {data.shape}")

    # Apply mask (multiply)
    masked = data * (mask_aligned > 0).astype(data.dtype)

    # Save output
    out_img = Nifti1Image(masked, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")


def erode_mask(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    n_erode: int = 2,
    overwrite: bool = False,
) -> Path:
    """Apply binary erosion to a mask.

    Uses 6-connected structuring element (face neighbors only) for conservative
    erosion. Removes ``n_erode`` voxel layers from the mask boundary.

    Parameters
    ----------
    input_file : str or Path
        Path to input binary mask NIfTI
    output_file : str or Path
        Path to save eroded mask
    n_erode : int, default=2
        Number of erosion iterations (voxel layers to remove)
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

    logger.info(f"Eroding mask by {n_erode} voxels: {input_file}")

    # Load mask
    img = cast(Nifti1Image, nib.load(str(input_file)))
    data = img.get_fdata() > 0  # Ensure binary

    # 6-connected structuring element (face neighbors only)
    structure = generate_binary_structure(3, 1)

    # Apply erosion
    eroded = binary_erosion(data, structure=structure, iterations=n_erode)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eroded_img = Nifti1Image(eroded.astype("uint8"), img.affine, img.header)
    save(eroded_img, output_path)

    logger.info(f"Eroded mask: {int(data.sum())} -> {int(eroded.sum())} voxels")
    return output_path
