"""
Real-time feedback estimation for fMRI neurofeedback.

This module provides the core algorithms for real-time neurofeedback signal
estimation. It implements an incremental GLM that updates in O(p²) time per
volume, enabling sub-second latency between volume acquisition and feedback
delivery.

The key innovation is using Gentleman's algorithm for incremental QR decomposition,
which avoids expensive full-matrix recomputation while maintaining numerical
stability. Combined with continuous normalization (rather than post-hoc z-scoring),
this provides moment-to-moment activation estimates suitable for real-time feedback.

Key Components
--------------
IncrementalGLM : Core algorithm
    Online least-squares solver using Gentleman's algorithm. Updates the QR
    decomposition incrementally as new observations arrive.

FeedbackEstimator : Signal processing
    Processes fMRI volumes, extracts motion parameters, builds GLM design
    matrices, and computes per-mask activation estimates.

NeurofeedbackCoordinator : Pipeline orchestration
    Watches for incoming volumes (DICOM or NIfTI), handles conversion,
    coordinates timing, and streams results to display tasks.

run_feedback_task : High-level API
    One-call function that sets up and runs a complete feedback task,
    handling both real-time and simulation modes.

Workflow
--------
The feedback pipeline operates as follows:

1. **Setup Phase**
   - Load network masks and build voxel index mapping
   - Compute HRF-convolved task regressor (or use custom design)
   - Initialize incremental GLM with appropriate dimensions

2. **Baseline Period** (volumes 0 to n_baseline-1)
   - Task regressor = 0 (no task effect)
   - Accumulate baseline statistics for normalization
   - Freeze variance estimates at end of baseline

3. **Feedback Period** (volumes n_baseline onwards)
   - For each incoming volume:
     a. Extract motion parameters from JSON sidecar
     b. Build design row: [task_regressors, intercept, drift, motion]
     c. Extract voxel values from masked regions
     d. Update GLM and compute normalized activation
     e. Aggregate voxel activations per mask (mean or efficiency-weighted)
     f. Stream activation to display task

GLM Design Matrix
-----------------
The design matrix has the following column structure::

    [task_columns] [intercept] [drift] [motion_x6]
    ↑              ↑                   ↑
    Signal of     Nuisance regressors (removed from signal)
    interest

- Task columns (0 to n_task-1): NOT removed from error signal
- Nuisance columns: Removed to isolate task-related signal
- Motion: 6 rigid-body parameters (3 translation, 3 rotation)

Normalization
-------------
Activation is computed using continuous normalization (Hinds et al., 2011)::

    error = y - X_nuisance @ beta_nuisance   # Remove only nuisance
    deviation = sqrt(RSS / t)                 # Running standard deviation
    activation = error / deviation            # Normalized activation

After baseline, the deviation is frozen to prevent drift during feedback.

Dependencies
------------
External:
    - dcm2niix (optional): For DICOM-to-NIfTI conversion during real-time

Python:
    - numpy : Array operations and linear algebra
    - nibabel : NIfTI I/O
    - pydicom : DICOM header reading
    - scipy : HRF computation (gamma distribution)
    - watchdog : Filesystem monitoring for incoming volumes

Notes
-----
**Performance:**
The incremental GLM achieves O(p²) complexity per update (where p = number
of regressors), compared to O(np²) for full recomputation. For typical
neurofeedback designs (p ≈ 10, n ≈ 500), this is ~50x faster.

**Numerical Stability:**
Gentleman's algorithm uses Givens rotations rather than Householder
reflections. While slightly slower, Givens rotations better handle the
rank-deficient case that occurs during baseline (when task regressor = 0).

**Input Formats:**
The coordinator accepts both DICOM and NIfTI inputs. For NIfTI files,
motion parameters should be in the JSON sidecar's ImageComments field
(format: "Motion: tx,ty,tz,rx,ry,rz").

See Also
--------
preprocessing : Prepares fMRI data for feedback tasks
mask_extraction : Creates network masks for feedback regions
tasks : Display tasks that receive and visualize activations

References
----------
.. [1] Gentleman, W.M. (1974). Algorithm AS 75: Basic procedures for large,
       sparse or weighted linear least problems. Journal of the Royal
       Statistical Society. Series C (Applied Statistics), 23(3), 448-454.

.. [2] Hinds, O., et al. (2011). Computing moment-to-moment BOLD activation
       for real-time neurofeedback. NeuroImage, 54(1), 361-368.

.. [3] Koush, Y., et al. (2012). Signal quality and Bayesian signal processing
       in neurofeedback based on real-time fMRI. NeuroImage, 59(1), 478-489.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# --- Standard Library ---
import contextlib  # Suppress exceptions cleanly
import csv  # TSV logging for feedback results
import heapq  # Priority queue for ordered volume processing
import json  # JSON sidecar parsing for NIfTI metadata
import logging  # Structured logging
import re  # Regex for motion parameter extraction
import shutil  # File operations during collection
import threading  # Multi-threading for volume watching
import time  # Timing for volume collection
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Optional, cast

# Type-checking only imports (avoid circular dependencies)
if TYPE_CHECKING:
    from .sham import ShamActivationProvider

# --- Third-Party: Neuroimaging ---
# NOTE: ANTsPy (ants) is imported lazily inside _ants_rigid_registration()
# to avoid its ~9-second import cost at module load time. The GUI imports
# feedback.py at startup but doesn't need ANTsPy until a pipeline runs.
import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray
from watchdog.events import DirCreatedEvent, FileCreatedEvent, FileSystemEventHandler

# --- Third-Party: Filesystem Monitoring ---
# Watchdog provides cross-platform filesystem event monitoring for detecting
# new volume files as they arrive from the scanner
from watchdog.observers import Observer

# --- Local Imports ---
from .utils import (
    HEAD_RADIUS_MM,  # Head radius for FD arc-length conversion
    compute_dvars,  # Shared DVARS computation
    compute_fd,  # Shared framewise-displacement computation
    get_json_sidecar_path,  # Get path to JSON sidecar
    get_tr_from_volume,  # Extract TR from volume metadata
    is_moco_series,  # Detect Siemens MoCo-derived DICOM series
    is_nifti_file,  # Check if file is NIfTI format
    is_valid_volume,  # Validate DICOM/NIfTI files
    parse_siemens_motion_comment,  # Parse Siemens MoCo motion string
)

# --- Module Logger ---
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Units of the activation values delivered to display tasks.
# Activations are continuous z-scores computed via Hinds et al. (2011)
# normalization: error / running_std, where error excludes nuisance regressors
# (motion, drift, intercept) but retains the task regressor contribution.
# After the baseline period the standard deviation is frozen to prevent drift.
ACTIVATION_UNITS = "z-score"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _rotation_matrix_to_euler(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert 3x3 rotation matrix to Euler angles (XYZ convention).

    Uses scipy's Rotation class for robust conversion.
    Convention matches FSL mcflirt: rotations around X, Y, Z axes.

    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    angles : ndarray, shape (3,)
        Euler angles [rot_x, rot_y, rot_z] in radians.
    """
    from scipy.spatial.transform import Rotation

    # Use scipy for robust conversion
    rot = Rotation.from_matrix(R)
    # XYZ intrinsic = xyz extrinsic, matches FSL convention
    return np.asarray(rot.as_euler("xyz"))


def _ants_register_volume(
    ref_ants: Any,
    mov_vol: NDArray[np.float64],
    spacing: list[float],
    *,
    initial_transform: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    """Frame-to-reference rigid registration with warm-start using ANTsPy.

    Registers the moving volume to a cached reference ANTs image using a
    single-level strategy (3 iterations at 2x shrink) with optional warm-start
    from the previous frame's transform. Benchmarked at ~300ms per volume.

    Parameters
    ----------
    ref_ants : ants.ANTsImage
        Pre-built ANTs image of the reference volume (cached by caller).
    mov_vol : ndarray, shape (X, Y, Z)
        Moving volume to register to reference.
    spacing : list of float
        Voxel spacing [sx, sy, sz] in mm for the moving volume.
    initial_transform : str or None
        Path to ANTs transform file from previous registration, used as
        warm-start initialization. None for first volume (no warm-start).

    Returns
    -------
    corrected_vol : ndarray, shape (X, Y, Z)
        Motion-corrected volume aligned to reference.
    motion_params : ndarray, shape (6,)
        Motion parameters in standard format:
        [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        Translations in mm, rotations in degrees.
    transform_path : str
        Path to the forward transform file, to be passed as
        initial_transform on the next call for warm-starting.
    """
    import ants  # Lazy import: ~9s cold start, only needed when running pipeline

    # Convert moving volume to ANTs image with proper spacing
    mov_ants = ants.from_numpy(mov_vol.astype(np.float32), spacing=spacing)

    # Single-level rigid registration with warm-start
    # 2 iterations at 2x shrink with sigma=1 smoothing (A8_warm_2iter_2x)
    kwargs: dict[str, Any] = {
        "fixed": ref_ants,
        "moving": mov_ants,
        "type_of_transform": "Rigid",
        "aff_iterations": (2,),
        "aff_shrink_factors": (2,),
        "aff_smoothing_sigmas": (1,),
        "verbose": False,
    }
    if initial_transform is not None:
        kwargs["initial_transform"] = initial_transform

    result = ants.registration(**kwargs)

    corrected_vol = result["warpedmovout"].numpy()

    # Extract motion parameters from transform
    tx_path = result["fwdtransforms"][0]
    tx = ants.read_transform(tx_path)

    params = np.array(tx.parameters)
    fixed_params = (
        np.array(tx.fixed_parameters)
        if tx.fixed_parameters is not None and len(tx.fixed_parameters) > 0
        else np.zeros(3)
    )

    # ANTs Rigid returns Euler3DTransform with 6 parameters:
    # [angleX, angleY, angleZ, transX, transY, transZ]
    if len(params) == 6:
        euler_rad = params[:3]
        translation = params[3:6]
        euler_deg = np.degrees(euler_rad)
        motion_params = np.array(
            [
                translation[0],
                translation[1],
                translation[2],
                euler_deg[0],
                euler_deg[1],
                euler_deg[2],
            ],
            dtype=np.float64,
        )

    elif len(params) == 12:
        # Affine/MatrixOffsetTransform format: [M00..M22, tx, ty, tz]
        matrix = params[:9].reshape(3, 3)
        offset = params[9:12]

        # Compute effective translation accounting for rotation center
        if len(fixed_params) == 3:
            center = fixed_params
            effective_translation = offset + center - matrix @ center
        else:
            effective_translation = offset

        euler_rad = _rotation_matrix_to_euler(matrix)
        euler_deg = np.degrees(euler_rad)
        motion_params = np.array(
            [
                effective_translation[0],
                effective_translation[1],
                effective_translation[2],
                euler_deg[0],
                euler_deg[1],
                euler_deg[2],
            ],
            dtype=np.float64,
        )

    else:
        logger.warning(f"Unexpected ANTs transform params length: {len(params)}")
        motion_params = np.zeros(6, dtype=np.float64)

    return corrected_vol, motion_params, tx_path


class _DicomGeometryCache:
    """Cached DICOM geometry for fast subsequent reads.

    After the first volume, stores the affine, spacing, and orientation
    transform so that subsequent volumes only need to read pixel_array.
    For Siemens mosaic DICOMs, also caches the mosaic grid parameters
    needed to unpack the 2D mosaic into a 3D volume on each read.
    """

    __slots__ = ("affine", "spacing", "ornt_transform", "mosaic_params")

    def __init__(
        self,
        affine: NDArray[np.float64],
        spacing: list[float],
        ornt_transform: NDArray[np.float64] | None,
        mosaic_params: tuple[int, int, int, int] | None = None,
    ):
        self.affine = affine
        self.spacing = spacing
        self.ornt_transform = ornt_transform
        # (n_slices, matrix_rows, matrix_cols, n_grid_cols) or None
        self.mosaic_params = mosaic_params


def _unpack_mosaic(
    pixel_array_2d: NDArray, n_slices: int, matrix_rows: int, matrix_cols: int, n_grid_cols: int
) -> NDArray:
    """Unpack a Siemens mosaic 2D image into a 3D volume.

    In mosaic format, all slices are tiled into a single 2D image in
    row-major order (left-to-right, top-to-bottom).

    Returns data in (n_slices, matrix_rows, matrix_cols) axis order,
    matching the enhanced multiframe (n_frames, rows, cols) convention.
    """
    data = np.zeros((n_slices, matrix_rows, matrix_cols), dtype=pixel_array_2d.dtype)
    for s in range(n_slices):
        grid_row = s // n_grid_cols
        grid_col = s % n_grid_cols
        r0 = grid_row * matrix_rows
        c0 = grid_col * matrix_cols
        data[s] = pixel_array_2d[r0 : r0 + matrix_rows, c0 : c0 + matrix_cols]
    return data


def _read_dicom_volume(
    dicom_path: Path,
    target_ornt: NDArray[np.float64] | None = None,
    geometry_cache: _DicomGeometryCache | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float], _DicomGeometryCache]:
    """Read an enhanced (multiframe) DICOM volume directly with pydicom.

    Builds a proper affine from ImageOrientationPatient and ImagePositionPatient,
    then reorients the data to match a target orientation (e.g. the mask orientation).
    Only supports Enhanced MR Image Storage (one file = one 3D volume).

    On the first call, pass ``geometry_cache=None`` to parse the full DICOM
    header and build the affine.  The returned cache object can be passed on
    subsequent calls to skip header parsing entirely.

    Parameters
    ----------
    dicom_path : Path
        Path to enhanced/multiframe DICOM file.
    target_ornt : ndarray, shape (N, 2), optional
        Target orientation from ``nibabel.io_orientation(mask_affine)``.
        If provided, reorients the volume to match this orientation.
        If None, returns data in the native DICOM orientation after
        LPS-to-RAS conversion.
    geometry_cache : _DicomGeometryCache or None
        Cached geometry from a previous call.  When provided, skips all
        header parsing and affine construction.

    Returns
    -------
    data : ndarray, shape (X, Y, Z)
        Volume data as float64, reoriented to match target.
    affine : ndarray, shape (4, 4)
        Affine matrix mapping voxel indices to RAS coordinates.
    spacing : list of float
        Voxel spacing [sx, sy, sz] in mm, in output axis order.
    cache : _DicomGeometryCache
        Geometry cache to pass on subsequent calls.

    Raises
    ------
    ValueError
        If the DICOM is not an enhanced multiframe file or required
        header fields are missing.
    """
    import pydicom
    from nibabel.orientations import apply_orientation, inv_ornt_aff, ornt_transform

    # --- Fast path: use cached geometry ---
    if geometry_cache is not None:
        dcm = pydicom.dcmread(dicom_path)
        data = dcm.pixel_array.astype(np.float64)
        if geometry_cache.mosaic_params is not None:
            n_sl, m_rows, m_cols, n_gc = geometry_cache.mosaic_params
            data = _unpack_mosaic(data, n_sl, m_rows, m_cols, n_gc)
        if geometry_cache.ornt_transform is not None:
            data = apply_orientation(data, geometry_cache.ornt_transform)
        return data, geometry_cache.affine, geometry_cache.spacing, geometry_cache

    # --- First volume: full header parse ---
    dcm = pydicom.dcmread(dicom_path)

    # Detect Siemens mosaic DICOM (all slices tiled in one 2D image)
    is_mosaic = hasattr(dcm, "ImageType") and "MOSAIC" in dcm.ImageType and (0x0019, 0x100A) in dcm
    mosaic_cache_params: tuple[int, int, int, int] | None = None

    if is_mosaic:
        # --- Siemens mosaic path ---
        n_slices = int(dcm[0x0019, 0x100A].value)
        acq = getattr(dcm, "AcquisitionMatrix", None)
        if acq is None:
            raise ValueError(f"Missing AcquisitionMatrix in mosaic DICOM: {dicom_path.name}")
        matrix_rows = max(int(acq[0]), int(acq[2]))
        matrix_cols = max(int(acq[1]), int(acq[3]))
        n_grid_cols = int(dcm.Columns) // matrix_cols

        mosaic_cache_params = (n_slices, matrix_rows, matrix_cols, n_grid_cols)
        data = _unpack_mosaic(dcm.pixel_array.astype(np.float64), n_slices, matrix_rows, matrix_cols, n_grid_cols)

        # Geometry from top-level tags (mosaic uses classic DICOM tags)
        pixel_spacing = [float(x) for x in dcm.PixelSpacing]
        iop = [float(x) for x in dcm.ImageOrientationPatient]
        ipp_raw = [float(x) for x in dcm.ImagePositionPatient]

        row_cosine = np.array(iop[:3])
        col_cosine = np.array(iop[3:])

        # Correct origin: mosaic IPP refers to the full mosaic grid corner,
        # not the actual first slice. Shift inward by the mosaic padding.
        origin = np.array(ipp_raw) + (
            row_cosine * pixel_spacing[1] * (int(dcm.Columns) - matrix_cols) / 2.0
            + col_cosine * pixel_spacing[0] * (int(dcm.Rows) - matrix_rows) / 2.0
        )

        # Slice direction from cross product + SpacingBetweenSlices
        slice_vec = np.cross(row_cosine, col_cosine)
        sbs = float(getattr(dcm, "SpacingBetweenSlices", None) or getattr(dcm, "SliceThickness", 1.0))  # type: ignore[arg-type]
        slice_vec = slice_vec / np.linalg.norm(slice_vec) * sbs

    elif hasattr(dcm, "NumberOfFrames"):
        # --- Enhanced multiframe path ---
        data = dcm.pixel_array.astype(np.float64)
        n_slices = int(dcm.NumberOfFrames)

        # Extract geometry from functional group sequences
        pixel_spacing = None
        iop = None
        ipp_first = None
        ipp_last = None

        for seq_attr in ("PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence"):
            seq = getattr(dcm, seq_attr, None)
            if not seq:
                continue
            f0 = seq[0]

            if pixel_spacing is None and hasattr(f0, "PixelMeasuresSequence"):
                pm = f0.PixelMeasuresSequence[0]
                pixel_spacing = [float(x) for x in pm.PixelSpacing]

            if iop is None and hasattr(f0, "PlaneOrientationSequence"):
                iop = [float(x) for x in f0.PlaneOrientationSequence[0].ImageOrientationPatient]

            if ipp_first is None and hasattr(f0, "PlanePositionSequence"):
                ipp_first = [float(x) for x in f0.PlanePositionSequence[0].ImagePositionPatient]

        # Position of last frame (for slice direction) — always from PerFrame
        pffgs = getattr(dcm, "PerFrameFunctionalGroupsSequence", None)
        if pffgs and hasattr(pffgs[-1], "PlanePositionSequence"):
            ipp_last = [float(x) for x in pffgs[-1].PlanePositionSequence[0].ImagePositionPatient]

        if pixel_spacing is None:
            if hasattr(dcm, "PixelSpacing"):
                pixel_spacing = [float(x) for x in dcm.PixelSpacing]
            else:
                raise ValueError(f"Cannot extract PixelSpacing from {dicom_path.name}")

        if iop is None:
            raise ValueError(f"Cannot extract ImageOrientationPatient from {dicom_path.name}")
        if ipp_first is None:
            raise ValueError(f"Cannot extract ImagePositionPatient from {dicom_path.name}")

        row_cosine = np.array(iop[:3])
        col_cosine = np.array(iop[3:])

        # Slice direction from actual frame positions (handles oblique acquisitions)
        if ipp_last is not None and n_slices > 1:
            slice_vec = (np.array(ipp_last) - np.array(ipp_first)) / (n_slices - 1)
        else:
            slice_vec = np.cross(row_cosine, col_cosine)
            sbs = float(getattr(dcm, "SpacingBetweenSlices", None) or getattr(dcm, "SliceThickness", 1.0))  # type: ignore[arg-type]
            slice_vec = slice_vec / np.linalg.norm(slice_vec) * sbs

        origin = np.array(ipp_first)
    else:
        raise ValueError(
            f"Unsupported DICOM format: {dicom_path.name}. Expected enhanced multiframe or Siemens mosaic DICOM."
        )

    # pydicom pixel_array is (n_frames, rows, cols)
    # Affine maps (frame_idx, row_idx, col_idx) -> (x, y, z) in LPS
    # axis 0 = frame/slice: slice direction
    # axis 1 = row: col_cosine direction, step = pixel_spacing[0]
    # axis 2 = col: row_cosine direction, step = pixel_spacing[1]
    assert pixel_spacing is not None  # guaranteed by branches above
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = slice_vec
    affine_lps[:3, 1] = col_cosine * pixel_spacing[0]
    affine_lps[:3, 2] = row_cosine * pixel_spacing[1]
    affine_lps[:3, 3] = origin

    # Convert LPS -> RAS (negate x and y rows)
    affine = affine_lps.copy()
    affine[0, :] *= -1
    affine[1, :] *= -1

    # --- Reorient to match target orientation ---
    cached_ornt_transform = None
    if target_ornt is not None:
        current_ornt = nib.io_orientation(affine)
        cached_ornt_transform = ornt_transform(current_ornt, target_ornt)
        data = apply_orientation(data, cached_ornt_transform)
        affine = affine @ inv_ornt_aff(cached_ornt_transform, data.shape)

    # Compute spacing in output axis order
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).tolist()

    cache = _DicomGeometryCache(affine, spacing, cached_ornt_transform, mosaic_cache_params)
    return data, affine, spacing, cache


# =============================================================================
# SECTION 1: INCREMENTAL GLM
# =============================================================================
# This section implements the core algorithm for real-time GLM estimation.
#
# Why Incremental?
# ----------------
# Traditional GLM fitting requires O(np²) operations where n = observations
# and p = regressors. For real-time use, we can't afford to recompute the
# entire solution every TR. Gentleman's algorithm achieves O(p²) per update
# by incrementally updating the QR decomposition.
#
# How It Works:
# -------------
# 1. Maintain the R matrix from QR decomposition (X = QR)
# 2. When a new observation arrives, apply Givens rotations to incorporate it
# 3. Solve for beta via back-substitution on the updated R
# 4. Compute normalized activation from residuals
#
# The key insight is that Givens rotations can update R in-place, and the
# back-substitution only depends on R (not Q), so we never need to store
# or manipulate the full Q matrix.


class IncrementalGLM:
    """Incremental least squares solver using Gentleman's algorithm.

    Implements online GLM fitting with continuous normalization for real-time
    applications. The algorithm maintains an R matrix (upper triangular from
    QR decomposition) that can be updated incrementally as new observations
    arrive, avoiding full matrix recomputation.

    Parameters
    ----------
    n_regressors : int
        Number of regressors (columns) in the design matrix, including intercept
        if used. The GLM cannot produce estimates until at least this many
        observations have been received.
    n_nuisance : int
        Number of nuisance regressors (first n_nuisance columns of design matrix).
        The error signal is computed by removing only nuisance contributions.
    n_outputs : int, optional
        Number of output variables (e.g., voxels) to fit simultaneously.
        Default is 1.

    Attributes
    ----------
    n_obs : int
        Current number of observations received.

    Notes
    -----
    Gentleman's algorithm incrementally updates the QR decomposition by
    applying Givens rotations to incorporate each new observation. This
    maintains numerical stability while achieving O(p²) complexity per
    update instead of O(np²) for full recomputation.

    The normalization approach:
    - Error signal = y - X_nuisance @ beta_nuisance (removing only nuisance)
    - Deviation = sqrt(RSS / t) where RSS is residual sum of squares from
      the full model and t is the current timepoint (no df correction)
    - Activation = error / deviation

    This provides continuous normalization throughout the session, not just
    post-baseline z-scoring.

    References
    ----------
    Gentleman, W.M. (1974). Algorithm AS 75: Basic procedures for large, sparse
        or weighted linear least problems. Journal of the Royal Statistical
        Society. Series C (Applied Statistics), 23(3), 448-454.

    Hinds, O., et al. (2011). Computing moment-to-moment BOLD activation for
        real-time neurofeedback. NeuroImage, 54(1), 361-368.

    Examples
    --------
    >>> glm = IncrementalGLM(n_regressors=8, n_nuisance=8, n_outputs=100)
    >>> for t in range(50):
    ...     X_t = np.array([1.0, t, m1, m2, m3, m4, m5, m6])  # design row
    ...     y_t = bold_data[t]  # shape (100,)
    ...     activation = glm.update(X_t, y_t)
    ...     if activation is not None:
    ...         # Use activation for feedback
    ...         pass
    """

    def __init__(
        self,
        n_regressors: int,
        n_nuisance: int,
        n_outputs: int = 1,
    ) -> None:
        if n_regressors < 1:
            raise ValueError("n_regressors must be at least 1")
        if n_nuisance < 1 or n_nuisance > n_regressors:
            raise ValueError(f"n_nuisance ({n_nuisance}) must be between 1 and n_regressors ({n_regressors})")

        self.n_regressors = n_regressors
        self.n_nuisance = n_nuisance
        self.n_outputs = n_outputs

        self.n_obs = 0

        # R matrix from QR decomposition: upper triangular, augmented with y cross-product
        self._R: NDArray[np.float64] = np.zeros((n_regressors, n_regressors + n_outputs), dtype=np.float64)

        # Track d[] for debugging (approximated from R[j,j]^2 in normal operation)
        # This tracks what d[] would be if we computed it properly
        self._d_debug: NDArray[np.float64] = np.zeros(n_regressors, dtype=np.float64)

        # Residual sum of squares and mean squared error for each output
        self._rss: NDArray[np.float64] = np.zeros(n_outputs, dtype=np.float64)
        self._residual_mse: NDArray[np.float64] = np.zeros(n_outputs, dtype=np.float64)

        # Debug: store last weight for comparison
        self._last_weight: float = 0.0

        # Steady-state statistics (frozen after baseline for stable normalization)
        self._steady_state_rss: NDArray[np.float64] | None = None
        self._steady_state_residual_mse: NDArray[np.float64] | None = None
        self._steady_state_timepoint: int | None = None

        # Guard to emit rank-deficiency warning at most once per session
        self._rank_deficient_warned = False

    def _apply_givens_rotation(
        self,
        j: int,
        aug_row: NDArray[np.float64],
        weight: float,
    ) -> float:
        """Apply Givens rotation to incorporate new observation at column j.

        Updates self._R[j, j:] in place and modifies aug_row to zero out position j.
        Also updates the weight following MURFI's Gentleman algorithm.

        Parameters
        ----------
        j : int
            Column index for rotation (0 to n_regressors-1).
        aug_row : ndarray, shape (n_regressors + n_outputs,)
            Augmented row [X, y] being incorporated. Modified in place.
        weight : float
            Current weight (scalar, same for all outputs since X is shared).

        Returns
        -------
        weight : float
            Updated weight after this rotation.
        """
        r_jj = self._R[j, j]
        x_j = aug_row[j]

        if x_j == 0:
            return weight

        # Compute rotation parameters
        if abs(r_jj) >= abs(x_j):
            t = x_j / r_jj
            c = 1.0 / np.sqrt(1.0 + t * t)
            s = c * t
        else:
            t = r_jj / x_j
            s = 1.0 / np.sqrt(1.0 + t * t)
            c = s * t

        # Update weight using MURFI's formula: weight = cbar * weight
        # where cbar = d[j] / (d[j] + weight * x_j^2)
        # Use R[j,j]^2 as approximation of d[j]
        d_j = r_jj * r_jj
        x_j_sq = x_j * x_j
        denom = d_j + weight * x_j_sq
        if denom > 0:
            cbar = d_j / denom
        else:
            cbar = 0.0

        # Track what proper d[] would be (for debugging)
        # MURFI: d[j] = d[j] + weight * x_j^2 (before weight update)
        self._d_debug[j] = self._d_debug[j] + weight * x_j_sq

        weight = cbar * weight

        # Update R[j, j:] (vectorized for speed with n_outputs >> 1)
        self._R[j, j] = c * r_jj + s * x_j
        r_tail = self._R[j, j + 1 :].copy()
        x_tail = aug_row[j + 1 :].copy()
        self._R[j, j + 1 :] = c * r_tail + s * x_tail
        aug_row[j + 1 :] = -s * r_tail + c * x_tail

        return weight

    def _solve_via_backsubstitution(
        self,
        R: NDArray[np.float64],
        r: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve R @ beta = r via back-substitution.

        Parameters
        ----------
        R : ndarray, shape (n_regressors, n_regressors)
            Upper triangular matrix from QR decomposition.
        r : ndarray, shape (n_regressors, n_outputs)
            Right-hand side (cross-product with y).

        Returns
        -------
        beta : ndarray, shape (n_regressors, n_outputs)
            Parameter estimates for all outputs.
        """
        p = self.n_regressors
        beta = np.zeros((p, self.n_outputs), dtype=np.float64)

        for i in range(p - 1, -1, -1):
            beta[i, :] = r[i, :]
            for j in range(i + 1, p):
                beta[i, :] -= R[i, j] * beta[j, :]
            # Handle zero diagonal (rank-deficient column, e.g., task regressor during baseline)
            if abs(R[i, i]) > 1e-10:
                beta[i, :] /= R[i, i]
            else:
                # Column is effectively zero - set beta to zero
                beta[i, :] = 0.0
                if not self._rank_deficient_warned:
                    logger.warning(
                        f"Near-zero pivot at regressor index {i} (|R[{i},{i}]|={abs(R[i, i]):.2e}): "
                        "design matrix may be rank-deficient (expected during baseline); "
                        "setting beta to zero. This warning is emitted once per session."
                    )
                    self._rank_deficient_warned = True

        return beta

    def _compute_normalized_activation(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        beta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute normalized activation.

        Computes error by removing only nuisance regressors, then normalizes
        by the deviation (sqrt(RSS / t)).

        Parameters
        ----------
        X : ndarray, shape (n_regressors,)
            Design matrix row for current observation.
        y : ndarray, shape (n_outputs,)
            Response values for current observation.
        beta : ndarray, shape (n_regressors, n_outputs)
            Current parameter estimates.

        Returns
        -------
        activation : ndarray, shape (n_outputs,)
            Normalized activation (error / deviation).
        """
        p = self.n_regressors

        # Compute error by removing only nuisance regressors
        # This preserves signal of interest while removing nuisance
        # Nuisance columns are the LAST n_nuisance columns (task columns come first)
        nuisance_start = p - self.n_nuisance
        nuisance_beta = beta[nuisance_start:, :]
        nuisance_X = X[nuisance_start:]
        error = y - (nuisance_X @ nuisance_beta).ravel()

        # Compute deviation as sqrt(RSS / t)
        # Use frozen steady-state values if available (after baseline completes)
        if self._steady_state_rss is not None and self._steady_state_timepoint is not None:
            rss_to_use = self._steady_state_rss
            timepoint_to_use = self._steady_state_timepoint
        else:
            rss_to_use = self._rss
            timepoint_to_use = self.n_obs

        # Use timepoint directly (no degrees of freedom correction)
        # After freezing, uses frozen timepoint for stable normalization
        deviation = np.sqrt(rss_to_use / timepoint_to_use)

        # Avoid division by zero
        deviation = np.where(deviation > 0, deviation, 1.0)

        # Compute normalized activation (error / deviation)
        activation = error / deviation

        return np.asarray(activation)

    def update(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Incorporate a new observation and return normalized activation.

        Parameters
        ----------
        X : ndarray, shape (n_regressors,)
            Design matrix row for the new observation.
        y : ndarray, shape (n_outputs,) or scalar
            Response value(s) for the new observation.

        Returns
        -------
        activation : ndarray, shape (n_outputs,) or None
            Normalized activation (error / deviation),
            or None if not enough observations to estimate parameters yet.
        """
        # Validate and prepare inputs
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        p = self.n_regressors

        if X.shape[0] != self.n_regressors:
            raise ValueError(f"X has {X.shape[0]} elements, expected {self.n_regressors}")
        if y.shape[0] != self.n_outputs:
            raise ValueError(f"y has {y.shape[0]} elements, expected {self.n_outputs}")

        # Increment observation counter
        self.n_obs += 1

        # Apply Givens rotations to update R matrix (Gentleman's algorithm)
        # Track weight as in MURFI: weight starts at 1.0 and is reduced by cbar at each step
        # Weight is scalar since it only depends on X and d (same for all voxels)
        aug_row = np.concatenate([X, y])
        weight = 1.0
        for j in range(p):
            weight = self._apply_givens_rotation(j, aug_row, weight)

        # After all rotations, aug_row[p:] contains the transformed y values
        # Although MURFI's code shows sserr += weight * y * y, in practice MURFI's
        # weight converges to ~1.0 as d[] accumulates over observations.
        # To match MURFI's behavior, we accumulate RSS without the weight factor.
        transformed_y = aug_row[p:]
        self._rss += transformed_y**2

        # Store last weight for debugging
        self._last_weight = weight

        # Update residual MSE (RSS / n_obs) to monitor variance over time
        self._residual_mse = self._rss / self.n_obs

        # Cannot compute activation until we have enough observations
        if self.n_obs < self.n_regressors:
            return None

        # Solve for parameter estimates via back-substitution
        R = self._R[:, :p]
        r = self._R[:, p:]
        beta = self._solve_via_backsubstitution(R, r)

        # Compute normalized activation
        activation = self._compute_normalized_activation(X, y, beta)

        return activation

    def get_residual_mse(self) -> NDArray[np.float64]:
        """Get current residual mean squared error for each output.

        Returns the residual MSE (RSS / n_obs). After steady state is reached,
        returns the frozen steady-state values for stable normalization.

        Returns
        -------
        residual_mse : ndarray, shape (n_outputs,)
            Current (or frozen steady-state) residual MSE for each output (voxel).
        """
        # Return frozen steady-state MSE if available
        if self._steady_state_residual_mse is not None:
            return self._steady_state_residual_mse.copy()
        return self._residual_mse.copy()

    def get_rss(self) -> NDArray[np.float64]:
        """Get current RSS (residual sum of squares) for each output.

        Always returns the current (not frozen) RSS values.
        Used for dynamic weighting in ROI averaging to match MURFI behavior.

        Returns
        -------
        rss : ndarray, shape (n_outputs,)
            Current RSS for each output (voxel).
        """
        return self._rss.copy()

    def freeze_steady_state(self) -> None:
        """Freeze RSS, residual MSE, and timepoint for steady-state.

        Called by FeedbackEstimator when baseline completes. After freezing,
        deviation calculations use the frozen values for stable normalization
        throughout the session.
        """
        if self._steady_state_residual_mse is None:
            self._steady_state_rss = self._rss.copy()
            self._steady_state_residual_mse = self._residual_mse.copy()
            self._steady_state_timepoint = self.n_obs
            logger.info(f"Steady state frozen at observation {self.n_obs}")

    def is_frozen(self) -> bool:
        """Return True if steady-state has already been frozen."""
        return self._steady_state_residual_mse is not None

    def dump_algo_vars(self, voxel_indices: list[int] | None = None) -> dict:
        """Dump algorithm variables for debugging/comparison with MURFI.

        Parameters
        ----------
        voxel_indices : list of int, optional
            Specific voxel indices to dump. If None, dumps first 5 voxels.

        Returns
        -------
        dict : Algorithm state variables
            - n_obs: current observation count
            - d_approx: R[j,j]^2 values (what we use as d approximation)
            - d_debug: proper d[] tracking (what MURFI uses)
            - weight: final weight after all columns
            - rss: residual sum of squares per voxel
            - R_diag: diagonal of R matrix
        """
        p = self.n_regressors
        if voxel_indices is None:
            voxel_indices = list(range(min(5, self.n_outputs)))

        return {
            "n_obs": self.n_obs,
            "d_approx": [self._R[j, j] ** 2 for j in range(p)],
            "d_debug": self._d_debug.tolist(),
            "weight": self._last_weight,
            "rss": self._rss[voxel_indices].tolist(),
            "R_diag": [self._R[j, j] for j in range(p)],
        }


# =============================================================================
# SECTION 2: FEEDBACK ESTIMATOR
# =============================================================================
# The FeedbackEstimator orchestrates real-time BOLD activation estimation.
# It brings together:
# - Mask loading and voxel indexing
# - Design matrix construction (task + nuisance regressors)
# - Incremental GLM updates
# - Per-mask activation aggregation
#
# Key Design Decisions:
# ---------------------
# 1. Single GLM for all voxels: Rather than fitting separate GLMs per voxel,
#    we fit one GLM with n_outputs = n_voxels. This is more efficient because
#    all voxels share the same design matrix.
#
# 2. Continuous normalization: Instead of z-scoring after baseline, we
#    normalize continuously using running variance estimates. After baseline
#    completes, we "freeze" the variance to prevent drift.
#
# 3. Configurable ROI combining: per-voxel activations are reduced to one
#    feedback value per mask by _combine_roi -- an unweighted mean or the
#    Hinds et al. 2011 efficiency weighting (w = 1/sigma).


ROI_COMBINE_METHODS = ("mean", "inverse_sd")


def _combine_roi(
    activations: NDArray[np.float64],
    rss: NDArray[np.float64],
    method: str,
) -> float:
    """Combine per-voxel ROI activations into a single feedback value.

    Parameters
    ----------
    activations : ndarray
        Per-voxel normalized activations (z-scored error) for one ROI.
    rss : ndarray
        Per-voxel residual sum of squares for the same voxels. RSS is
        proportional to the noise variance (sigma**2).
    method : str
        ``"mean"`` -- unweighted mean of the per-voxel activations.
        ``"inverse_sd"`` -- Hinds et al. 2011 voxel efficiency weighting:
        weight ``w_i = 1 / sigma_i`` with ``sigma_i = sqrt(RSS_i)``, giving
        the weighted average ``sum(w_i * z_i) / sum(w_i)``. Noisier voxels
        (larger sigma) contribute less.

    Returns
    -------
    float
        The combined ROI feedback value.
    """
    if activations.size == 0:
        return 0.0
    if method == "mean":
        return float(np.mean(activations))
    if method != "inverse_sd":
        raise ValueError(f"roi_combine_method must be one of {ROI_COMBINE_METHODS}, got '{method}'")
    sigma = np.sqrt(rss)
    ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(activations)
    if not np.any(ok):
        # No usable noise estimate yet (e.g. before the GLM has data) --
        # fall back to an unweighted mean.
        return float(np.mean(activations))
    weights = 1.0 / sigma[ok]
    return float(np.sum(weights * activations[ok]) / np.sum(weights))


class FeedbackEstimator:
    """Real-time neurofeedback estimator using incremental GLM.

    Processes fMRI volumes in real-time, fitting a GLM to remove nuisance
    signals and estimate BOLD activation. Supports multiple masks with
    optional voxel efficiency weighting.

    Parameters
    ----------
    mask_paths : list of Path or str
        Paths to NIfTI mask files. Each mask defines a region of interest.
    n_baseline : int
        Number of initial volumes for baseline period. Default is 25.
    n_volumes : int
        Total expected number of volumes. Used to validate task regressors length.
    tr : float
        Repetition time in seconds.
    task_regressors : ndarray or TaskDesign, optional
        Task regressors for the GLM design matrix. Can be:
        - ndarray, shape (n_volumes, n_task_regressors): Pre-computed regressors
        - TaskDesign object: Will compute regressors using n_volumes and tr
        - None: No task regressors (nuisance-only model)
    mask_names : list of str, optional
        Names for each mask. If not provided, uses mask filenames.
    roi_combine_method : str, optional
        How per-voxel activations are combined into one ROI value:
        ``"inverse_sd"`` (Hinds et al. 2011 efficiency weighting, w = 1/sigma;
        the default) or ``"mean"`` (unweighted average).

    Attributes
    ----------
    n_volumes : int
        Number of volumes processed so far.
    timeseries : dict of str to list of float
        Mean activation timeseries for each mask.

    Notes
    -----
    The GLM design matrix structure:
    - Columns 0 to n_task_regressors-1: Task regressors (of interest, NOT subtracted from error)
    - Remaining columns: Nuisance regressors (configurable subset of:
      intercept [always], drift, 6 motion parameters, WM mean, CSF mean)

    Activation is computed using continuous normalization (Hinds et al., 2011):
    - Error = y - X_nuisance @ beta_nuisance (removing nuisance only)
    - Deviation = sqrt(RSS / t) where RSS is residual sum of squares
    - Activation = error / deviation

    Motion parameters are extracted from dcm2niix JSON sidecar files
    (ImageComments field) for Siemens MoCo, or computed via ANTsPy
    rigid registration for other scanners.
    """

    def __init__(
        self,
        mask_paths: Sequence[Path | str],
        n_baseline: int,
        n_volumes: int,
        tr: float,
        task_regressors: NDArray[np.float64] | None = None,
        mask_names: list[str] | None = None,
        roi_combine_method: str = "inverse_sd",
        sham_provider: Optional["ShamActivationProvider"] = None,
        dump_algo_vars_path: Path | str | None = None,
        smooth_fwhm: float | None = None,
        motion_correction: str = "siemens_moco",
        include_drift: bool = True,
        include_motion_regressors: bool = True,
        include_wm_regressor: bool = False,
        include_csf_regressor: bool = False,
        wm_mask_path: Path | str | None = None,
        csf_mask_path: Path | str | None = None,
    ) -> None:
        if len(mask_paths) == 0:
            raise ValueError("At least one mask path required")
        if n_baseline < 1:
            raise ValueError("n_baseline must be at least 1")
        if n_volumes < n_baseline:
            raise ValueError("n_volumes must be >= n_baseline")
        if tr <= 0:
            raise ValueError("tr must be positive")

        self.n_baseline = n_baseline
        self._n_volumes_expected = n_volumes
        self._tr = tr
        self.n_volumes = 0
        self.roi_combine_method = roi_combine_method
        self._smooth_fwhm = smooth_fwhm
        self._dump_algo_vars_path = Path(dump_algo_vars_path) if dump_algo_vars_path else None
        self._algo_dump_fh: IO[Any] | None = None
        self._closed = False

        # Motion correction mode (decoupled from GLM regressors)
        self._motion_correction = motion_correction

        # GLM nuisance regressor configuration (individually toggleable)
        self._include_drift = include_drift
        # Motion regressors require motion params to be available
        self._include_motion = include_motion_regressors and motion_correction != "none"
        if include_motion_regressors and motion_correction == "none":
            logger.warning(
                "include_motion_regressors=True but motion_correction='none' — "
                "no motion parameters available, disabling motion regressors in GLM"
            )
        self._include_wm = include_wm_regressor
        self._include_csf = include_csf_regressor

        # Compute dynamic nuisance regressor count
        # Design row layout: [task_cols...] [intercept] [drift?] [motion_x6?] [wm?] [csf?]
        n_nuisance = 1  # intercept always
        self._drift_offset: int | None = None
        self._motion_offset: int | None = None
        self._wm_offset: int | None = None
        self._csf_offset: int | None = None
        offset = 1  # next available slot after intercept
        if self._include_drift:
            self._drift_offset = offset
            offset += 1
            n_nuisance += 1
        if self._include_motion:
            self._motion_offset = offset
            offset += 6
            n_nuisance += 6
        if self._include_wm:
            self._wm_offset = offset
            offset += 1
            n_nuisance += 1
        if self._include_csf:
            self._csf_offset = offset
            offset += 1
            n_nuisance += 1
        self.N_NUISANCE_REGRESSORS = n_nuisance

        # WM/CSF tissue mask paths
        self._wm_mask_path: Path | None = Path(wm_mask_path) if wm_mask_path else None
        self._csf_mask_path: Path | None = Path(csf_mask_path) if csf_mask_path else None
        self._latest_wm_mean: float = 0.0
        self._latest_csf_mean: float = 0.0

        # Log GLM configuration
        regressor_parts = ["intercept"]
        if self._include_drift:
            regressor_parts.append("drift")
        if self._include_motion:
            regressor_parts.append("motion(6)")
        if self._include_wm:
            regressor_parts.append("WM")
        if self._include_csf:
            regressor_parts.append("CSF")
        logger.info(f"GLM nuisance model: {' + '.join(regressor_parts)} ({n_nuisance} regressors)")

        # Validate and store task regressors
        if task_regressors is not None:
            # Check if it's a TaskDesign object (has compute_regressors method)
            if hasattr(task_regressors, "compute_regressors"):
                # It's a TaskDesign - compute regressors from it
                task_regressors = task_regressors.compute_regressors(n_volumes, tr)

            task_regressors = np.asarray(task_regressors, dtype=np.float64)
            if task_regressors.ndim == 1:
                # Single task regressor provided as 1D array
                task_regressors = task_regressors.reshape(-1, 1)
            if task_regressors.shape[0] != n_volumes:
                raise ValueError(
                    f"task_regressors has {task_regressors.shape[0]} rows, expected {n_volumes} to match n_volumes"
                )
            self._task_regressors: NDArray[np.float64] | None = task_regressors
            self._n_task_regressors = task_regressors.shape[1]
        else:
            # No task regressors - nuisance-only model
            self._task_regressors = None
            self._n_task_regressors = 0

        # Compute total number of regressors
        self._n_regressors = self._n_task_regressors + self.N_NUISANCE_REGRESSORS
        self._n_nuisance = self.N_NUISANCE_REGRESSORS  # only nuisance columns
        self._nuisance_start_idx = self._n_task_regressors  # nuisance starts after task columns

        # Set up mask names
        mask_paths_resolved: list[Path] = [Path(p) for p in mask_paths]
        if mask_names is None:
            mask_names = [p.stem for p in mask_paths_resolved]
        if len(mask_names) != len(mask_paths_resolved):
            raise ValueError("mask_names must match length of mask_paths")
        self.mask_names = mask_names

        # Load masks and build voxel index mapping
        self._load_masks(mask_paths_resolved)

        # Load tissue masks for WM/CSF mean signal extraction (separate from ROI masks)
        if self._include_wm:
            if self._wm_mask_path is None:
                raise ValueError("wm_mask_path required when include_wm_regressor=True")
            self._wm_indices = self._load_tissue_mask(self._wm_mask_path)
            logger.info(f"WM nuisance regressor: {len(self._wm_indices[0])} voxels")
        if self._include_csf:
            if self._csf_mask_path is None:
                raise ValueError("csf_mask_path required when include_csf_regressor=True")
            self._csf_indices = self._load_tissue_mask(self._csf_mask_path)
            logger.info(f"CSF nuisance regressor: {len(self._csf_indices[0])} voxels")

        # Initialize single GLM for all voxels (shared model across voxels)
        self._glm = IncrementalGLM(
            n_regressors=self._n_regressors,
            n_nuisance=self._n_nuisance,
            n_outputs=self._n_voxels,
        )

        # Motion parameter tracking for differential motion
        self._prev_motion: NDArray[np.float64] | None = None

        # Framewise displacement tracking (computed each volume)
        self._latest_fd: float = 0.0

        # DVARS tracking (per-volume metric: RMS of temporal derivative across voxels)
        self._latest_dvars: float = 0.0
        self._prev_voxel_values: NDArray[np.float64] | None = None

        # tSNR tracking (running aggregate via Welford's online algorithm)
        self._latest_tsnr: float = 0.0
        self._welford_count: int = 0
        self._welford_mean: NDArray[np.float64] | None = None
        self._welford_m2: NDArray[np.float64] | None = None

        # Timeseries storage
        self.timeseries: dict[str, list[float]] = {name: [] for name in mask_names}

        # Sham support: when sham_provider is set, we return sham activations
        # but still compute and store real activations for logging
        self._sham_provider = sham_provider
        self._last_real_activations: dict[str, float] = {}

        sham_status = "SHAM MODE" if sham_provider is not None else "disabled"
        logger.info(
            f"FeedbackEstimator initialized with {len(mask_paths)} masks, "
            f"{self._n_voxels} total voxels, "
            f"n_baseline={n_baseline}, n_volumes={n_volumes}, tr={tr:.3f}s, "
            f"task_regressors={self._n_task_regressors}, "
            f"roi_combine={roi_combine_method}, "
            f"sham={sham_status}"
        )

    def close(self) -> None:
        """Close open resources (e.g., algo dump file handle)."""
        if self._closed:
            return
        self._closed = True
        if self._algo_dump_fh is not None:
            self._algo_dump_fh.close()
            self._algo_dump_fh = None

    def _load_single_mask(self, path: Path, name: str) -> tuple[NDArray[np.intp], set[tuple[int, int, int]]]:
        """Load a single mask file and extract voxel coordinates.

        Parameters
        ----------
        path : Path
            Path to mask NIfTI file.
        name : str
            Name of the mask.

        Returns
        -------
        flat_indices : ndarray
            Flattened indices for voxels in this mask.
        coords : set of tuple
            Set of (i, j, k) coordinates where mask > 0.
        """
        img = cast(Nifti1Image, nib.loadsave.load(path))
        data = np.asarray(img.dataobj)

        # Verify compatible shape
        if data.shape[:3] != self._shape:
            raise ValueError(f"Mask {name} shape {data.shape[:3]} != reference {self._shape}")

        # Get indices where mask > 0
        indices = np.where(data > 0)
        coords = set(zip(indices[0], indices[1], indices[2], strict=False))
        if not coords:
            raise ValueError(f"Mask '{name}' is empty (no voxels > 0): {path}")

        # Store flat indices for this mask
        flat_indices = np.ravel_multi_index(indices, self._shape)

        logger.debug(f"Mask '{name}': {len(coords)} voxels")

        return flat_indices, coords

    def _build_master_voxel_index(
        self, all_coords: set[tuple[int, int, int]]
    ) -> tuple[int, tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]]:
        """Build sorted master voxel index from all mask coordinates.

        Parameters
        ----------
        all_coords : set of tuple
            Set of all (i, j, k) coordinates across all masks.

        Returns
        -------
        n_voxels : int
            Total number of unique voxels across all masks.
        voxel_indices : tuple of ndarray
            Tuple of (i_indices, j_indices, k_indices) for all voxels.
        """
        # Build master index array (sorted for consistent ordering)
        all_coords_list = sorted(all_coords)
        n_voxels = len(all_coords_list)

        # Convert to arrays for fast indexing
        coords_array = np.array(all_coords_list)
        voxel_indices = (
            coords_array[:, 0],
            coords_array[:, 1],
            coords_array[:, 2],
        )

        return n_voxels, voxel_indices

    def _map_masks_to_master(self, mask_indices: dict[str, NDArray[np.intp]]) -> dict[str, NDArray[np.intp]]:
        """Map each mask's flat indices to positions in master voxel array.

        Parameters
        ----------
        mask_indices : dict of str to ndarray
            Dictionary mapping mask names to flat indices.

        Returns
        -------
        mask_voxel_idx : dict of str to ndarray
            Dictionary mapping mask names to indices into master voxel array.
        """
        # Build mapping from flat index to master index position
        master_flat = np.ravel_multi_index(self._voxel_indices, self._shape)
        flat_to_master = {flat: i for i, flat in enumerate(master_flat)}

        # Map each mask's flat indices to master array positions
        mask_voxel_idx: dict[str, NDArray[np.intp]] = {}
        for name, flat_indices in mask_indices.items():
            master_idx = np.array([flat_to_master[f] for f in flat_indices], dtype=np.intp)
            mask_voxel_idx[name] = master_idx

        return mask_voxel_idx

    def _load_masks(self, mask_paths: list[Path]) -> None:
        """Load masks and build voxel-to-mask index mapping.

        This orchestrates loading all masks, building a master voxel index,
        and creating mappings from each mask to the master index.
        """
        # Load first mask to get reference shape and orientation for validation
        first_img = cast(Nifti1Image, nib.loadsave.load(mask_paths[0]))
        self._shape = first_img.shape[:3]
        self._mask_ornt = nib.io_orientation(first_img.affine)

        # Load all masks and collect coordinates
        mask_indices: dict[str, NDArray[np.intp]] = {}
        all_coords: set[tuple[int, int, int]] = set()

        for path, name in zip(mask_paths, self.mask_names, strict=True):
            flat_indices, coords = self._load_single_mask(path, name)
            mask_indices[name] = flat_indices
            all_coords.update(coords)

        # Build master voxel index
        self._n_voxels, self._voxel_indices = self._build_master_voxel_index(all_coords)

        # Map masks to master index
        self._mask_voxel_idx = self._map_masks_to_master(mask_indices)

    def _load_tissue_mask(self, mask_path: Path) -> tuple[np.ndarray, ...]:
        """Load a tissue mask and return 3D fancy indices for extraction.

        Parameters
        ----------
        mask_path : Path
            Path to binary tissue mask NIfTI.

        Returns
        -------
        tuple of (i, j, k) index arrays for direct fancy indexing into volumes.

        Raises
        ------
        ValueError
            If mask shape doesn't match ROI masks or mask is empty.
        """
        img = cast(Nifti1Image, nib.loadsave.load(mask_path))
        mask_data = np.asarray(img.dataobj) > 0
        if mask_data.shape[:3] != self._shape:
            raise ValueError(f"Tissue mask shape {mask_data.shape[:3]} != ROI mask shape {self._shape}")
        indices = np.where(mask_data)
        if len(indices[0]) == 0:
            raise ValueError(f"Tissue mask is empty after loading: {mask_path}")
        return indices

    def _extract_motion_params_from_json(self, json_path: Path) -> NDArray[np.float64]:
        """Extract motion parameters from dcm2niix JSON sidecar.

        Reads motion parameters from the ImageComments field in dcm2niix output:
        - "Motion: -0.01,-0.00,-0.05,0.01,0.05,0.00" for motion corrected volumes
        - "Reference volume for motion correction." for reference volumes (all zeros)

        Parameters
        ----------
        json_path : Path
            Path to dcm2niix JSON sidecar file.

        Returns
        -------
        motion : ndarray, shape (6,)
            Motion parameters [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z].
        """

        try:
            with open(json_path) as f:
                metadata = json.load(f)

            image_comments = metadata.get("ImageComments", "")

            # Reference-volume sentinel -> zeros; "Motion: ..." -> 6 params.
            parsed = parse_siemens_motion_comment(image_comments)
            if parsed is not None:
                return parsed

            logger.warning(f"No motion parameters found in {json_path}, using zeros")
            return np.zeros(6, dtype=np.float64)

        except Exception as e:
            logger.error(f"Error reading JSON {json_path}: {e}")
            return np.zeros(6, dtype=np.float64)

    def _compute_fd(self, motion_params: NDArray[np.float64]) -> float:
        """Compute framewise displacement from current motion parameters.

        FD is the sum of absolute frame-to-frame differences across all 6 motion
        parameters. Rotations are converted to mm using arc projection at 50mm
        head radius.

        Parameters
        ----------
        motion_params : ndarray, shape (6,)
            Motion parameters: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z].
            Translations in mm, rotations in degrees.

        Returns
        -------
        fd : float
            Framewise displacement in mm.
        """
        return compute_fd(motion_params, self._prev_motion)

    def _compute_dvars(self, voxel_values: NDArray[np.float64]) -> float:
        """Compute DVARS from current and previous volume's voxel values.

        DVARS (Power et al., 2012) is the spatial root mean square of the
        temporal derivative: sqrt(mean((I_t - I_{t-1})^2)) across all brain
        mask voxels. Returns 0.0 for the first volume.

        Parameters
        ----------
        voxel_values : ndarray
            Voxel intensities for current volume (all mask voxels).

        Returns
        -------
        float
            DVARS value for this volume.
        """
        return compute_dvars(voxel_values, self._prev_voxel_values)

    def _update_tsnr(self, voxel_values: NDArray[np.float64]) -> float:
        """Update running tSNR estimate using Welford's online algorithm.

        tSNR = median(mean_i / std_i) across voxels, where mean and std are
        computed incrementally (Welford, 1962). Returns 0.0 until at least
        2 volumes have been processed.

        Parameters
        ----------
        voxel_values : ndarray
            Voxel intensities for current volume (all mask voxels).

        Returns
        -------
        float
            Current median tSNR estimate across voxels.
        """
        self._welford_count += 1
        if self._welford_mean is None:
            self._welford_mean = voxel_values.copy()
            self._welford_m2 = np.zeros_like(voxel_values)
            return 0.0

        assert self._welford_m2 is not None  # initialized alongside _welford_mean
        delta = voxel_values - self._welford_mean
        self._welford_mean += delta / self._welford_count
        delta2 = voxel_values - self._welford_mean
        self._welford_m2 += delta * delta2

        if self._welford_count < 2:
            return 0.0

        variance = self._welford_m2 / (self._welford_count - 1)
        std = np.sqrt(variance)
        valid = std > 0
        if not np.any(valid):
            return 0.0
        voxel_tsnr = self._welford_mean[valid] / std[valid]
        return float(np.median(voxel_tsnr))

    def _build_design_row(self, motion_params: np.ndarray) -> NDArray[np.float64]:
        """Build design matrix row for current volume.

        Parameters
        ----------
        motion_params : ndarray, shape (6,)
            Raw motion parameters for current volume (absolute values).
            Translations in mm, rotations in degrees.
            Ignored if motion regressors are disabled.

        Returns
        -------
        X : ndarray, shape (n_regressors,)
            Design row: [task_cols..., intercept, drift?, motion_diff?, wm?, csf?]
            Motion parameters are differential (frame-to-frame) with rotations
            converted to arc length (mm) at 50mm radius.
        """
        vol_idx = self.n_volumes  # Current volume index (0-based)

        # Compute motion diff (needed for GLM motion regressors)
        if self._include_motion:
            # Compute frame-to-frame motion from absolute parameters
            motion_diff: np.ndarray
            if self.n_volumes == 0:
                motion_diff = np.zeros(6, dtype=np.float64)
            else:
                prev = self._prev_motion if self._prev_motion is not None else np.zeros(6, dtype=np.float64)
                motion_diff = np.asarray(motion_params - prev, dtype=np.float64)

            # Convert rotations from degrees to arc length (mm) at HEAD_RADIUS_MM
            # arc_length = radius * angle_radians = HEAD_RADIUS_MM * (angle_degrees * pi/180)
            motion_diff[3:6] *= HEAD_RADIUS_MM * np.pi / 180.0

        self._prev_motion = motion_params.copy()

        # Build design row
        X = np.zeros(self._n_regressors, dtype=np.float64)

        # Fill in task regressors (if any)
        if self._task_regressors is not None:
            if vol_idx < len(self._task_regressors):
                X[: self._n_task_regressors] = self._task_regressors[vol_idx, :]
            else:
                X[: self._n_task_regressors] = self._task_regressors[-1, :]

        # Fill in nuisance regressors (after task columns, using pre-computed offsets)
        nui = self._nuisance_start_idx
        X[nui] = 1.0  # intercept (always at offset 0)
        if self._drift_offset is not None:
            X[nui + self._drift_offset] = float(vol_idx + 1)  # linear drift (1-indexed)
        if self._motion_offset is not None:
            X[nui + self._motion_offset : nui + self._motion_offset + 6] = motion_diff
        if self._wm_offset is not None:
            X[nui + self._wm_offset] = self._latest_wm_mean
        if self._csf_offset is not None:
            X[nui + self._csf_offset] = self._latest_csf_mean

        return X

    def _dump_algo_vars_to_file(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        activation: NDArray[np.float64] | None,
    ) -> None:
        """Dump algorithm variables to file for debugging/comparison with MURFI."""
        if self._algo_dump_fh is None:
            assert self._dump_algo_vars_path is not None  # caller checks _dump_algo_vars_path
            self._algo_dump_fh = open(self._dump_algo_vars_path, "w")  # noqa: SIM115
            self._algo_dump_fh.write("# pineuro algo dump\n")

        algo_vars = self._glm.dump_algo_vars(list(range(min(10, self._n_voxels))))
        assert self._algo_dump_fh is not None
        fh = self._algo_dump_fh

        fh.write(f"\n=== Volume {self.n_volumes} ===\n")
        fh.write(f"n_obs: {algo_vars['n_obs']}\n")
        fh.write(f"X: {X.tolist()}\n")
        fh.write(f"weight (final): {algo_vars['weight']:.10e}\n")
        fh.write(f"d_approx (R[j,j]^2): {algo_vars['d_approx']}\n")
        fh.write(f"d_debug (proper): {algo_vars['d_debug']}\n")
        fh.write(f"R_diag: {algo_vars['R_diag']}\n")
        fh.write(f"rss (first 10 voxels): {algo_vars['rss']}\n")
        fh.write(f"y (first 10 voxels): {y[:10].tolist()}\n")
        if activation is not None:
            fh.write(f"activation (first 10 voxels): {activation[:10].tolist()}\n")
        fh.flush()

    def process_volume(
        self,
        nifti_path: Path | str,
        motion_source: Path | str | NDArray[np.float64] | None = None,
        *,
        volume_data: NDArray[np.float64] | None = None,
        volume_affine: NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        """Process a new fMRI volume and return activation estimates.

        Parameters
        ----------
        nifti_path : Path or str
            Path to NIfTI volume file. Used as fallback when volume_data is
            not provided (e.g. NIfTI input).
        motion_source : Path, str, ndarray, or None, optional
            Source of motion parameters. Can be:
            - Path to JSON file (dcm2niix sidecar): extracts motion from ImageComments field
            - ndarray of shape (6,): use provided motion parameters directly
            - None: use zeros for motion parameters
        volume_data : ndarray or None, optional
            Pre-loaded volume data to use instead of loading from nifti_path.
            When provided, skips the disk load entirely.
        volume_affine : ndarray or None, optional
            Affine matrix for volume_data. Required when volume_data is provided
            and spatial smoothing is configured.

        Returns
        -------
        activations : dict of str to float
            Weighted mean activation (z-score after baseline) for each mask.
            During baseline period, returns weighted mean residuals.
        """
        nifti_path = Path(nifti_path)

        # ===== 1. Load volume data =====
        if volume_data is not None:
            # Fast path: use in-memory data directly
            data = np.asarray(volume_data, dtype=np.float64)
            if self._smooth_fwhm is not None:
                from nilearn.image import smooth_img

                if volume_affine is None:
                    raise ValueError("volume_affine is required when smoothing is enabled with in-memory data")
                img = nib.Nifti1Image(data.astype(np.float32), volume_affine)
                img = smooth_img(img, fwhm=self._smooth_fwhm)
                data = np.asarray(img.dataobj, dtype=np.float64)
        else:
            # Standard path: load from disk
            img = cast(Nifti1Image, nib.loadsave.load(nifti_path))

            # Apply spatial smoothing if configured
            if self._smooth_fwhm is not None:
                from nilearn.image import smooth_img

                img = smooth_img(img, fwhm=self._smooth_fwhm)

            data = np.asarray(img.dataobj, dtype=np.float64)

        # Verify shape matches masks
        if data.shape[:3] != self._shape:
            raise ValueError(f"Volume shape {data.shape[:3]} != mask shape {self._shape}")

        # Extract voxel values using direct fancy indexing (faster than reshape + ravel)
        # self._voxel_indices is a tuple of (i, j, k) arrays
        voxel_values = data[self._voxel_indices]

        # Extract tissue mean signals for nuisance regression (if enabled)
        if self._include_wm:
            self._latest_wm_mean = float(np.mean(data[self._wm_indices]))
        if self._include_csf:
            self._latest_csf_mean = float(np.mean(data[self._csf_indices]))

        # ===== 2. Extract motion parameters =====
        # Motion params are extracted whenever available (for FD/QC), even if not in GLM
        if self._motion_correction == "none" or motion_source is None:
            motion_params: np.ndarray = np.zeros(6, dtype=np.float64)
        elif isinstance(motion_source, (np.ndarray, list)):
            motion_params = np.array(motion_source, dtype=np.float64)
            if motion_params.shape != (6,):
                raise ValueError(f"Motion parameters must have shape (6,), got {motion_params.shape}")
        else:
            motion_path = Path(motion_source)
            # Treat as JSON file path
            motion_params = self._extract_motion_params_from_json(motion_path)

        # ===== 3. Compute QC metrics and build design matrix =====
        # FD must be computed before _build_design_row (which updates _prev_motion)
        # FD is NaN when motion correction is disabled (no motion params available)
        if self._motion_correction == "none":
            self._latest_fd = float("nan")
        else:
            self._latest_fd = self._compute_fd(motion_params)

        # DVARS: RMS of temporal derivative across mask voxels
        self._latest_dvars = self._compute_dvars(voxel_values)
        self._prev_voxel_values = voxel_values.copy()

        # tSNR: running median(mean/std) across voxels (Welford's algorithm)
        self._latest_tsnr = self._update_tsnr(voxel_values)

        X = self._build_design_row(motion_params)
        self._last_design_row = X
        result = self._glm.update(X, voxel_values)
        self.n_volumes += 1

        # Dump algo vars for debugging (if enabled)
        if self._dump_algo_vars_path is not None:
            self._dump_algo_vars_to_file(X, voxel_values, result)

        # Handle case where GLM can't compute activation yet (not enough observations)
        if result is None:
            activations = {name: 0.0 for name in self.mask_names}
            self._last_real_activations = activations.copy()
            for name in self.mask_names:
                self.timeseries[name].append(0.0)
            # If sham mode, still return zeros (consistent behavior)
            return activations

        # Freeze RSS for stable deviation normalization when baseline completes
        # Note: Freeze at n_baseline - 1 to match MURFI behavior (off-by-one)
        # MURFI uses `if (timepoint < numDataPointsForErrEst)` so at timepoint=25
        # it uses the saved residual from timepoint=24
        # Use == with a guard for n_baseline <= 1 (where n_baseline - 1 == 0 would
        # never match after the n_volumes += 1 above)
        if self.n_volumes == self.n_baseline - 1 or (
            self.n_baseline <= 1 and self.n_volumes == 1 and not self._glm.is_frozen()
        ):
            # Freeze RSS, MSE, and timepoint for stable normalization
            self._glm.freeze_steady_state()

        # ===== 4. Combine per-voxel activations into one value per mask =====
        # Per-voxel RSS is read fresh each volume so inverse-SD weighting
        # tracks the current noise estimate; see _combine_roi.
        real_activations: dict[str, float] = {}
        current_rss = self._glm.get_rss()

        for name, idx in self._mask_voxel_idx.items():
            combined = _combine_roi(result[idx], current_rss[idx], self.roi_combine_method)
            real_activations[name] = combined
            self.timeseries[name].append(combined)

        # Store real activations for logging (used by coordinator)
        self._last_real_activations = real_activations.copy()

        # ===== 5. Handle sham mode =====
        if self._sham_provider is not None:
            # Get sham activations to deliver to task
            elapsed_time = self.n_volumes * self._tr
            sham_activations = self._sham_provider.get_activation(self.n_volumes, elapsed_time)
            return sham_activations

        return real_activations


class ConnectivityEstimator:
    """Sliding-window Pearson correlation between ROI pairs.

    Computes functional connectivity as Pearson r between the mean
    activation timeseries of configured ROI pairs. Uses a sliding
    window of configurable length. Returns 0.0 during the warm-up
    period (before the window is full).

    Parameters
    ----------
    pairs : list of tuple[str, str]
        Pairs of mask names to compute correlation between.
    window : int
        Sliding window size in volumes. Default is 15.

    Notes
    -----
    This class does not store timeseries itself. It operates on
    externally-provided timeseries data (from FeedbackEstimator.timeseries).
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        window: int = 15,
    ) -> None:
        if window < 3:
            raise ValueError(f"connectivity_window must be >= 3 (got {window})")
        if not pairs:
            raise ValueError("At least one connectivity pair required")
        self.pairs = pairs
        self.window = window
        self.names: list[str] = [f"corr_{m0}_{m1}" for m0, m1 in pairs]

    def compute(
        self,
        timeseries: dict[str, list[float]],
    ) -> dict[str, float]:
        """Compute sliding-window Pearson correlation for all pairs.

        Parameters
        ----------
        timeseries : dict of str to list of float
            Per-mask mean activation timeseries from FeedbackEstimator.

        Returns
        -------
        correlations : dict of str to float
            Correlation value for each pair. Keys are ``corr_{m0}_{m1}``.
            Returns 0.0 for pairs where the window is not yet full.
        """
        results: dict[str, float] = {}
        for (m0, m1), name in zip(self.pairs, self.names, strict=True):
            ts0 = timeseries.get(m0, [])
            ts1 = timeseries.get(m1, [])
            n = min(len(ts0), len(ts1))
            if n < self.window:
                results[name] = 0.0
                continue
            x = np.array(ts0[-self.window :])
            y = np.array(ts1[-self.window :])
            std_x = np.std(x)
            std_y = np.std(y)
            if std_x < 1e-10 or std_y < 1e-10:
                results[name] = 0.0
            else:
                r = np.corrcoef(x, y)[0, 1]
                results[name] = float(r) if np.isfinite(r) else 0.0
        return results


class _DicomEventHandler(FileSystemEventHandler):
    """Watchdog event handler for volume file (DICOM or NIfTI) detection."""

    def __init__(self, callback: Callable[[Path], None]) -> None:
        super().__init__()
        self._callback = callback

    @staticmethod
    def _wait_for_stable_file(path: Path, interval: float = 0.1, stable_count: int = 5) -> None:
        """Poll file size until stable, indicating the writer has finished.

        Needed for non-atomic writers (e.g., real scanners pushing over SMB)
        where the file creation event fires before writing is complete.
        Uses conservative defaults (500ms stability window) to handle
        brief network write pauses on SMB shares.
        """
        prev_size = -1
        count = 0
        for _ in range(100):  # Max ~10 seconds of polling
            try:
                size = path.stat().st_size
            except OSError:
                time.sleep(interval)
                continue
            if size == prev_size and size > 0:
                count += 1
                if count >= stable_count:
                    return
            else:
                count = 0
                prev_size = size
            time.sleep(interval)

    def on_created(self, event: FileCreatedEvent | DirCreatedEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            file_path = Path(src_path.decode())
        elif isinstance(src_path, str):
            file_path = Path(src_path)
        else:
            return
        self._wait_for_stable_file(file_path)
        self._callback(file_path)


# ==============================================================================
# Interrupted Run Handling - Detection, Archiving, and Run Number Management
# ==============================================================================


def _detect_interrupted_run(
    output_dir: Path,
    subject: str,
    session: str,
    task: str,
    run: int,
    n_volumes: int,
    acq: int,
) -> dict | None:
    """Check if a previous interrupted run exists for this task/run combination.

    A run is considered interrupted if:
    - feedbacklog file exists with > 0 but < n_volumes data rows

    Parameters
    ----------
    output_dir : Path
        Base output directory for feedback tasks
    subject : str
        Subject identifier (e.g., 'sub-01')
    task : str
        Task name (e.g., 'feedback', 'transfer')
    run : int
        Run number to check
    n_volumes : int
        Expected number of volumes for a complete run
    acq : int
        Acquisition number (1, 2, 3, ...) for retry tracking.

    Returns
    -------
    dict or None
        If interrupted run found, returns dict with:
        - feedbacklog_path: Path to feedbacklog TSV
        - design_path: Path to design matrix TSV
        - nifti_dir: Path to NIfTI directory
        - volumes_completed: Number of volumes in feedbacklog
        Returns None if no incomplete run found.
    """
    run_str = f"{run:02d}"
    acq_str = f"{acq:02d}"
    feedbacklog_path = output_dir / f"{subject}_{session}_task-{task}_acq-{acq_str}_run-{run_str}_feedbacklog.tsv"
    design_path = output_dir / f"{subject}_{session}_task-{task}_acq-{acq_str}_run-{run_str}_design.tsv"
    nifti_dir = output_dir / f"nifti_{session}_task-{task}_acq-{acq_str}_run-{run_str}"

    if not feedbacklog_path.exists():
        return None

    # Count data rows (excluding header)
    try:
        with open(feedbacklog_path) as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header
    except Exception as e:
        logger.debug(f"Could not read feedbacklog {feedbacklog_path}: {e}")
        return None

    # Complete if row count matches expected volumes
    if row_count >= n_volumes:
        return None

    # Interrupted if has some data but not all
    if row_count > 0:
        return {
            "feedbacklog_path": feedbacklog_path,
            "design_path": design_path,
            "nifti_dir": nifti_dir,
            "volumes_completed": row_count,
        }

    return None


def _archive_interrupted_run(
    output_dir: Path,
    subject: str,
    session: str,
    task: str,
    run: int,
    acq: int,
) -> Path:
    """Move interrupted run files to an archive subfolder with timestamp.

    Creates: output_dir/interrupted_YYYYMMDD_HHMMSS/
    Moves: feedbacklog, design matrix, nifti directory, qc directory

    Parameters
    ----------
    output_dir : Path
        Base output directory containing the interrupted run files
    subject : str
        Subject identifier
    task : str
        Task name
    run : int
        Run number
    acq : int
        Acquisition number (1, 2, 3, ...) for retry tracking.

    Returns
    -------
    Path
        Path to the created archive directory
    """
    run_str = f"{run:02d}"
    acq_str = f"{acq:02d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / f"interrupted_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    items_to_move = [
        output_dir / f"{subject}_{session}_task-{task}_acq-{acq_str}_run-{run_str}_feedbacklog.tsv",
        output_dir / f"{subject}_{session}_task-{task}_acq-{acq_str}_run-{run_str}_design.tsv",
        output_dir / f"nifti_{session}_task-{task}_acq-{acq_str}_run-{run_str}",
    ]

    # Also check for QC directory
    qc_dir = output_dir / "qc"
    if qc_dir.exists():
        items_to_move.append(qc_dir)

    for item in items_to_move:
        if item.exists():
            dest = archive_dir / item.name
            shutil.move(str(item), str(dest))
            logger.debug(f"Archived {item.name} to {archive_dir}")

    logger.info(f"Archived interrupted run to: {archive_dir}")
    return archive_dir


def _generate_partial_qc(
    output_dir: Path,
    subject: str,
    session: str,
    task: str,
    run: int,
    acq: int,
    task_id: str | None = None,
) -> Path | None:
    """Generate QC report for interrupted session before archiving.

    Uses existing QC infrastructure with current partial data.

    Parameters
    ----------
    output_dir : Path
        Base output directory
    subject : str
        Subject identifier
    task : str
        Task name
    run : int
        Run number
    acq : int
        Acquisition number (1, 2, 3, ...) for retry tracking.
    task_id : str, optional
        Registered task identifier (e.g., "ball", "gauge"). When provided, the
        QC report surfaces the task's declared live metrics in its summary.

    Returns
    -------
    Path or None
        Path to generated QC report, or None if generation failed
    """
    try:
        from pineuro import qc
        from pineuro.tasks.task_registry import get_task_metadata

        qc_output_dir = output_dir / "qc"
        qc_output_dir.mkdir(parents=True, exist_ok=True)

        live_metrics = None
        if task_id:
            try:
                live_metrics = list(get_task_metadata(task_id).live_metrics)
            except (KeyError, ValueError):
                live_metrics = None

        report_path = qc.generate_feedback_report(
            feedback_dir=output_dir,
            subject=subject,
            session=session,
            task=task,
            run=run,
            acq=acq,
            output_dir=qc_output_dir,
            open_browser=False,
            live_metrics=live_metrics,
        )
        logger.info(f"Generated partial QC report: {report_path}")
        return report_path
    except Exception as e:
        logger.warning(f"Failed to generate partial QC report: {e}")
        return None


def get_next_run_number(
    output_dir: Path,
    subject: str,
    session: str,
    task: str,
) -> int:
    """Scan output directory for existing runs and return next available number.

    Looks for feedbacklog files matching the pattern:
    {subject}_{session}_task-{task}_run-{NN}_feedbacklog.tsv

    Returns the next sequential run number after any existing runs.
    Also considers archived interrupted runs.

    Parameters
    ----------
    output_dir : Path
        Base output directory for feedback tasks
    subject : str
        Subject identifier (e.g., 'sub-01')
    session : str
        Session identifier (e.g., 'ses-01')
    task : str
        Task name (e.g., 'feedback', 'transfer')

    Returns
    -------
    int
        Next available run number (1 if no existing runs found)
    """
    if not output_dir.exists():
        return 1

    pattern = f"{subject}_{session}_task-{task}_run-*_feedbacklog.tsv"
    existing_runs: set[int] = set()

    # Check main output directory
    for f in output_dir.glob(pattern):
        match = re.search(r"run-(\d+)_feedbacklog", f.name)
        if match:
            existing_runs.add(int(match.group(1)))

    # Also check archived directories for runs
    for archive_dir in output_dir.glob("interrupted_*"):
        if archive_dir.is_dir():
            for f in archive_dir.glob(pattern):
                match = re.search(r"run-(\d+)_feedbacklog", f.name)
                if match:
                    existing_runs.add(int(match.group(1)))

    if not existing_runs:
        return 1

    return max(existing_runs) + 1


def parse_step_run_number(step_id: str) -> int:
    """Extract run number from a workflow step_id.

    Parameters
    ----------
    step_id : str
        Workflow step identifier (e.g., 'feedback_3', 'transfer_1')

    Returns
    -------
    int
        The run number extracted from the step_id.

    Raises
    ------
    ValueError
        If step_id is not in the expected format ``{type}_{number}``.
    """
    parts = step_id.rsplit("_", 1)
    if len(parts) == 2 and parts[0] in ("feedback", "transfer"):
        try:
            return int(parts[1])
        except ValueError:
            pass
    raise ValueError(
        f"Invalid step_id: {step_id!r}. Expected format 'feedback_N' or 'transfer_N' (e.g., 'feedback_3')."
    )


def get_slot_run_info(
    output_dir: Path,
    subject: str,
    session: str,
    task: str,
    step_id: str,
) -> dict:
    """Get run info based on workflow slot with BIDS acq entity for retries.

    Uses a slot-based run numbering system where:
    - The run number is derived from the step_id suffix (e.g., feedback_3 -> run-03)
    - acq-a/acq-b/acq-c corresponds to attempt number (first, second, third...)

    Parameters
    ----------
    output_dir : Path
        Base output directory for feedback tasks
    subject : str
        Subject identifier (e.g., 'sub-01')
    task : str
        Task name (e.g., 'feedback', 'transfer')
    step_id : str
        Workflow step identifier (e.g., 'transfer_1', 'feedback_3')

    Returns
    -------
    dict
        Dictionary containing:
        - 'run': int - The run number for this slot
        - 'acq': int - The acquisition number for this attempt (1, 2, 3, ...)
        - 'run_acq_string': str - Combined string like 'acq-01_run-03'
        - 'existing_acqs': list - List of existing acquisition numbers for this slot

    Examples
    --------
    >>> info = get_slot_run_info(output_dir, 'sub-01', 'transfer', 'transfer_1')
    >>> info
    {'run': 1, 'acq': 1, 'run_acq_string': 'acq-01_run-01', 'existing_acqs': []}
    >>> # After one interrupted run:
    >>> info = get_slot_run_info(output_dir, 'sub-01', 'transfer', 'transfer_1')
    >>> info
    {'run': 1, 'acq': 2, 'run_acq_string': 'acq-02_run-01', 'existing_acqs': [1]}
    """
    run_num = parse_step_run_number(step_id)

    if not output_dir.exists():
        return {
            "run": run_num,
            "acq": 1,
            "run_acq_string": f"acq-01_run-{run_num:02d}",
            "existing_acqs": [],
        }

    # Scan for existing attempts at this run number
    # Pattern: {subject}_{session}_task-{task}_acq-*_run-{run:02d}_feedbacklog.tsv
    pattern = f"{subject}_{session}_task-{task}_acq-*_run-{run_num:02d}_feedbacklog.tsv"

    existing_acqs: list[int] = []

    # Check main output directory
    for f in output_dir.glob(pattern):
        match = re.search(r"acq-(\d+)_run-", f.name)
        if match:
            existing_acqs.append(int(match.group(1)))

    # Also check archived directories for runs
    for archive_dir in output_dir.glob("interrupted_*"):
        if archive_dir.is_dir():
            for f in archive_dir.glob(pattern):
                match = re.search(r"acq-(\d+)_run-", f.name)
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
        "run": run_num,
        "acq": acq,
        "run_acq_string": f"acq-{acq:02d}_run-{run_num:02d}",
        "existing_acqs": existing_acqs,
    }


# =============================================================================
# SECTION 4: NEUROFEEDBACK COORDINATOR
# =============================================================================
# The NeurofeedbackCoordinator is the top-level orchestrator for real-time
# neurofeedback tasks. It handles:
#
# 1. **Volume Watching**: Monitors a directory for incoming files using
#    filesystem events (watchdog) plus periodic polling for reliability.
#
# 2. **DICOM Handling**: Converts DICOMs to NIfTI on-the-fly using dcm2niix.
#    Extracts motion parameters from DICOM headers for GLM nuisance modeling.
#
# 3. **Processing Coordination**: Feeds volumes to FeedbackEstimator, collects
#    activations, and makes them available to display tasks.
#
# 4. **Logging**: Records all activations, motion parameters, and design
#    matrix rows to TSV files for post-hoc analysis.
#
# 5. **Timing**: Manages acquisition timing, start triggers, and volume
#    counting to keep feedback synchronized with scanner acquisition.
#
# Threading Model:
# ----------------
# The coordinator runs in a background thread, watching for volumes and
# processing them as they arrive. The main thread (running the display task)
# polls for new activations without blocking the feedback loop.


class NeurofeedbackCoordinator:
    """Coordinates real-time neurofeedback by watching for volume files.

    Monitors a directory for incoming volume files (DICOM or NIfTI), converts
    them to NIfTI if needed, processes through FeedbackEstimator, and logs
    results to CSV.

    **Timing Model:**

    Volumes are processed strictly in order (by filename). If volume N+1 arrives
    before volume N, it waits in a priority queue until volume N is processed.
    This ensures deterministic output regardless of filesystem event ordering.

    Processing time per volume is logged for diagnostics. If processing
    consistently takes longer than TR, a warning is issued (queue depth > 0).

    **NIfTI Support:**

    For NIfTI input files (.nii, .nii.gz), a JSON sidecar should be present
    alongside each volume containing motion parameters (ImageComments field)
    and TR (RepetitionTime field). The dcm2niix conversion step is skipped.

    Parameters
    ----------
    watch_dir : Path or str
        Directory to watch for incoming volume files (DICOM or NIfTI).
    output_dir : Path or str
        Directory for NIfTI outputs and CSV log.
    mask_paths : list of Path or str
        Paths to NIfTI mask files for feedback estimation.
    subject : str
        Subject identifier (e.g., 'sub-01').
    task : str
        Task name (e.g., 'rest', 'nback').
    run : int
        Run number (e.g., 1 for run-01).
    n_baseline : int
        Number of initial volumes for baseline.
    n_volumes : int
        Total number of volumes to process.
    tr : float
        Repetition time in seconds.
    task_regressors : ndarray, shape (n_volumes, n_task_regressors), optional
        Pre-computed task regressors to include in the GLM. If None, uses
        nuisance-only model. Task-specific designs should be computed
        externally and passed here.
    mask_names : list of str, optional
        Names for each mask. If not provided, uses mask filenames.
    roi_combine_method : str, optional
        How per-voxel activations are combined into one ROI value:
        ``"inverse_sd"`` (Hinds et al. 2011 efficiency weighting, w = 1/sigma;
        the default) or ``"mean"`` (unweighted average).

    Attributes
    ----------
    n_processed : int
        Number of volumes processed so far.

    Notes
    -----
    Uses watchdog for efficient filesystem event detection. Each new volume
    triggers processing through FeedbackEstimator (with dcm2niix conversion
    for DICOM input). Results are logged to CSV after each volume.

    Task regressors should be pre-computed and passed to the constructor.
    If not provided, a nuisance-only model is used (intercept + drift + motion).

    Examples
    --------
    >>> coordinator = NeurofeedbackCoordinator(
    ...     watch_dir='/path/to/incoming',
    ...     output_dir='/path/to/output',
    ...     mask_paths=['/path/to/dmn_mask.nii.gz'],
    ...     subject='sub-01',
    ...     task='rest',
    ...     run=1,
    ...     n_baseline=25,
    ...     n_volumes=100,
    ...     tr=1.2
    ... )
    >>> coordinator.run()
    """

    def __init__(
        self,
        watch_dir: Path | str,
        output_dir: Path | str,
        dicom_collection_dir: Path | str,
        mask_paths: list[Path | str],
        subject: str,
        session: str,
        task: str,
        run: int,
        acq: int,
        n_baseline: int,
        n_volumes: int,
        tr: float,
        task_regressors: NDArray[np.float64] | None = None,
        mask_names: list[str] | None = None,
        roi_combine_method: str = "inverse_sd",
        activation_callback: Callable[[dict[str, float], int, float, bool, bool], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
        qc_enabled: bool = True,  # Generate QC report after session completes
        fd_threshold: float = 0.5,  # FD threshold in mm for QC plots
        sham_provider: Optional["ShamActivationProvider"] = None,  # Sham activation source
        condition: str = "real",  # Participant condition: "real" or "sham"
        motion_correction: str = "siemens_moco",  # "siemens_moco", "software", "none"
        include_drift: bool = True,
        include_motion_regressors: bool = True,
        include_wm_regressor: bool = False,
        include_csf_regressor: bool = False,
        dump_algo_vars_path: Path | str | None = None,  # Path to dump GLM algo vars for debugging
        smooth_fwhm: float | None = None,  # Gaussian spatial smoothing FWHM in mm per volume
        differential_pair: list[str] | None = None,  # Ordered pair [positive, negative] for differential signal
        feedback_interval: int | None = None,  # Intermittent mode: deliver mean every N volumes
        connectivity_pairs: list[list[str]] | None = None,  # ROI pairs for sliding-window correlation
        connectivity_window: int = 15,  # Sliding window size in volumes
        connectivity_target: str = "increase",  # "increase" or "decrease" (negate r)
        wm_mask_path: Path | str | None = None,  # WM tissue mask for nuisance regression
        csf_mask_path: Path | str | None = None,  # CSF tissue mask for nuisance regression
        task_id: str | None = None,  # Registered task_id for live-metric QC summary lookup
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.subject = subject
        self.session = session
        self.task = task
        self.run_number = run
        self.acq = acq  # Acquisition number for retry tracking (1, 2, 3, ...)
        self._task_id = task_id
        self.n_baseline = n_baseline
        self.n_volumes = n_volumes
        self._tr = tr
        self._qc_enabled = qc_enabled
        self._fd_threshold = fd_threshold

        # Debug: dump algo vars for comparison with MURFI
        self._dump_algo_vars_path = Path(dump_algo_vars_path) if dump_algo_vars_path else None

        # Motion correction mode
        self._motion_correction = motion_correction
        # When MoCo is enabled on the scanner, both original and MoCo series
        # land in the watched folder. Keep MoCo series only in siemens_moco mode
        # (we need the embedded motion params); otherwise keep originals.
        self._keep_moco = motion_correction == "siemens_moco"
        self._moco_reference_vol: NDArray[np.float64] | None = None
        self._moco_reference_affine: NDArray[np.float64] | None = None
        self._moco_reference_ants: Any | None = None  # Cached ANTs image of vol 0
        self._moco_spacing: list[float] | None = None  # Cached voxel spacing
        self._moco_prev_transform: str | None = None  # Warm-start transform path
        self._dicom_geometry_cache: _DicomGeometryCache | None = None
        logger.info(f"Motion correction mode: {motion_correction}")
        if motion_correction == "software":
            # Eagerly import ANTs to pay the ~3-4s cold start now (during setup),
            # not on the first volume when latency matters
            import ants  # noqa: F811

            logger.info(f"ANTs {ants.__version__} loaded")
            logger.info("Will use ANTsPy frame-to-reference rigid registration with warm-start (~300ms/volume)")

        # GLM regressor configuration (passed through to FeedbackEstimator)
        self._include_drift = include_drift
        self._include_motion_regressors = include_motion_regressors
        self._include_wm_regressor = include_wm_regressor
        self._include_csf_regressor = include_csf_regressor

        # Validate watch directory
        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Watch directory not found: {self.watch_dir}")
        if not self.watch_dir.is_dir():
            raise ValueError(f"Watch path is not a directory: {self.watch_dir}")

        # Create output directories using acq-based naming
        self._run_acq_str = f"acq-{self.acq:02d}_run-{self.run_number:02d}"
        self._bids_subject_prefix = f"{self.subject}_{self.session}"
        self.nifti_dir = self.output_dir / f"nifti_{self.session}_task-{self.task}_{self._run_acq_str}"
        self.nifti_dir.mkdir(parents=True, exist_ok=True)
        # Multi-run handling: Current implementation processes a single continuous
        # acquisition. For multi-run sessions, run separate NeurofeedbackCoordinator
        # instances or enhance to detect run boundaries via metadata

        # Set up mask names
        mask_paths_resolved = [Path(p) for p in mask_paths]
        if mask_names is None:
            mask_names = [p.stem for p in mask_paths_resolved]
        self.mask_names = mask_names

        # Differential signal: explicit pair, or auto from first two masks
        self._differential_name: str | None = None
        self._differential_pair: tuple[str, str] | None = None
        if differential_pair is not None and len(differential_pair) == 2:
            m0, m1 = differential_pair
            if m0 in self.mask_names and m1 in self.mask_names:
                self._differential_pair = (m0, m1)
                self._differential_name = f"{m0}-{m1}"
            else:
                logger.warning(
                    f"differential_pair {differential_pair} references unknown mask names "
                    f"(available: {self.mask_names}). Differential disabled."
                )
        elif len(self.mask_names) >= 2:
            # Auto-compute from first two masks when no explicit pair given
            m0, m1 = self.mask_names[0], self.mask_names[1]
            self._differential_pair = (m0, m1)
            self._differential_name = f"{m0}-{m1}"

        # Intermittent feedback mode
        self._feedback_interval = feedback_interval
        self._intermittent_buffer: list[dict[str, float]] = []
        if feedback_interval is not None:
            if feedback_interval < 2:
                raise ValueError(f"feedback_interval must be >= 2 (got {feedback_interval})")
            logger.info(f"Intermittent feedback: delivering mean every {feedback_interval} volumes")

        # Connectivity feedback mode (sliding-window Pearson correlation)
        self._connectivity_estimator: ConnectivityEstimator | None = None
        self._connectivity_names: list[str] = []
        self._connectivity_target = connectivity_target
        if connectivity_pairs is not None and len(connectivity_pairs) > 0:
            validated_pairs: list[tuple[str, str]] = []
            for pair in connectivity_pairs:
                if len(pair) == 2 and pair[0] in self.mask_names and pair[1] in self.mask_names:
                    validated_pairs.append((pair[0], pair[1]))
                else:
                    logger.warning(
                        f"connectivity_pair {pair} references unknown mask names "
                        f"(available: {self.mask_names}). Skipping."
                    )
            if validated_pairs:
                self._connectivity_estimator = ConnectivityEstimator(
                    pairs=validated_pairs,
                    window=connectivity_window,
                )
                self._connectivity_names = self._connectivity_estimator.names
                logger.info(
                    f"Connectivity feedback: {len(validated_pairs)} pair(s), "
                    f"window={connectivity_window} volumes, target={connectivity_target}"
                )

        # Sham mode configuration
        self._sham_provider = sham_provider
        self._condition = condition
        self._is_sham_mode = sham_provider is not None
        # Sham timeseries: accumulates replayed mask activations for connectivity computation
        self._sham_timeseries: dict[str, list[float]] = (
            {name: [] for name in self.mask_names} if self._is_sham_mode else {}
        )

        # Initialize FeedbackEstimator (pass sham provider for activation delivery)
        self._estimator = FeedbackEstimator(
            mask_paths=list(mask_paths_resolved),
            n_baseline=n_baseline,
            n_volumes=n_volumes,
            tr=tr,
            task_regressors=task_regressors,
            mask_names=mask_names,
            roi_combine_method=roi_combine_method,
            sham_provider=sham_provider,
            dump_algo_vars_path=self._dump_algo_vars_path,
            smooth_fwhm=smooth_fwhm,
            motion_correction=motion_correction,
            include_drift=include_drift,
            include_motion_regressors=include_motion_regressors,
            include_wm_regressor=include_wm_regressor,
            include_csf_regressor=include_csf_regressor,
            wm_mask_path=wm_mask_path,
            csf_mask_path=csf_mask_path,
        )

        # Processing state
        self.n_processed = 0
        self._processed_files: set[str] = set()
        self._pending_files: list[Path] = []
        self._pending_heap: list[tuple[str, Path]] = []  # Priority queue: (filename, path)
        self._retry_counts: dict[str, int] = {}  # Track validation retries per file
        self._MAX_RETRIES = 5
        self._next_expected_filename: str | None = None  # Filename we're waiting for
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # Signal when coordinator is ready
        self._trigger_event = threading.Event()  # Signal when trigger received (starts acquisition)
        self._error: Exception | None = None  # Store any error that occurs
        self._start_time: float | None = None
        self.design_matrix: list[NDArray[np.float64]] = []

        # Timing diagnostics
        self._timing_stats: dict[str, list[float]] = {
            "dcm2niix_ms": [],
            "glm_ms": [],
            "registration_ms": [],
            "estimator_ms": [],
            "total_ms": [],
            "queue_depth": [],
        }

        # DICOM collection directory (for unified collection with rest/reference)
        self._dicom_collection_dir = Path(dicom_collection_dir)
        self._dicom_collection_dir.mkdir(parents=True, exist_ok=True)

        # Cleanup options
        self._cleanup_incoming: bool = False
        self._cleanup_nifti: bool = False

        # Shared state for PsychoPy task communication (thread-safe)
        self._latest_activations: dict[str, float] = {name: 0.0 for name in mask_names}
        self._latest_volume: int = 0
        self._latest_time: float = 0.0
        self._latest_is_baseline: bool = True
        self._latest_is_outlier: bool = False
        self._activation_callback = activation_callback
        self._error_callback = error_callback

        # CSV log setup with BIDS-style naming (use acq entity if provided)
        self._csv_path = (
            self.output_dir / f"{self._bids_subject_prefix}_task-{self.task}_{self._run_acq_str}_feedbacklog.tsv"
        )
        self._design_matrix_path = (
            self.output_dir / f"{self._bids_subject_prefix}_task-{self.task}_{self._run_acq_str}_design.tsv"
        )
        self._design_file: IO[Any] | None = None  # File handle for incremental design matrix writes

        logger.info(f"NeurofeedbackCoordinator initialized: watching {self.watch_dir}, {n_volumes} volumes expected")

    def _init_csv(self) -> None:
        """Initialize CSV log file and design matrix file with headers.

        Simplified format: one column per mask with just the mask name.
        All masks are computed and logged regardless of condition.
        The 'condition' column indicates real vs sham for analysis.

        In sham mode, additional {mask}_real columns are added to log
        the actual computed activations alongside the delivered (sham) values.
        """
        self._csv_file: IO[Any] | None = open(self._csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._csv_writer: Any = csv.writer(self._csv_file, delimiter="\t")

        # Simple header: one column per mask (no _delivered/_real suffix)
        header = ["volume", "time", "is_baseline", "is_outlier", "fd", "dvars", "tsnr", "condition"]
        # WM/CSF tissue signal columns (independently toggled)
        if self._estimator._include_wm:
            header.append("wm_mean")
        if self._estimator._include_csf:
            header.append("csf_mean")
        header.extend(self.mask_names)
        if self._differential_name:
            header.append(self._differential_name)
        # Connectivity columns
        header.extend(self._connectivity_names)

        # In sham mode, add {mask}_real columns for the actual computed activations
        if self._is_sham_mode:
            header.extend([f"{name}_real" for name in self.mask_names])
            if self._differential_name:
                header.append(f"{self._differential_name}_real")
            header.extend([f"{name}_real" for name in self._connectivity_names])

        self._csv_writer.writerow(header)
        self._csv_file.flush()
        # Initialize incremental design matrix file
        self._init_design_matrix()

    def _close_csv(self) -> None:
        """Close CSV log file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _build_design_column_names(self) -> list[str]:
        """Build column names for the design matrix based on active regressors."""
        column_names = []
        n_task = self._estimator._n_task_regressors if self._estimator else 0
        if n_task == 1:
            column_names.append("task")
        elif n_task > 1:
            column_names.extend([f"task_{i}" for i in range(n_task)])

        # Nuisance columns (must match _build_design_row layout)
        column_names.append("intercept")
        if self._estimator._include_drift:
            column_names.append("drift")
        if self._estimator._include_motion:
            column_names.extend(["motion_x", "motion_y", "motion_z", "motion_rx", "motion_ry", "motion_rz"])
        if self._estimator._include_wm:
            column_names.append("wm_mean")
        if self._estimator._include_csf:
            column_names.append("csf_mean")
        return column_names

    def _save_design_matrix(self) -> None:
        """Save design matrix to TSV file."""
        if not self.design_matrix:
            logger.warning("No design matrix data to save")
            return

        column_names = self._build_design_column_names()

        # Convert design matrix list to numpy array
        design_array = np.array(self.design_matrix)

        # Save to TSV file
        with open(self._design_matrix_path, "w") as f:
            # Write header
            f.write("\t".join(column_names) + "\n")

            # Write data rows
            for row in design_array:
                row_str = "\t".join(f"{val:.6f}" for val in row)
                f.write(row_str + "\n")

    def _init_design_matrix(self) -> None:
        """Initialize design matrix TSV file with header for incremental writes."""
        self._design_file = open(self._design_matrix_path, "w", newline="", encoding="utf-8")  # noqa: SIM115

        column_names = self._build_design_column_names()

        self._design_file.write("\t".join(column_names) + "\n")
        self._design_file.flush()

    def _append_design_row(self, row: NDArray[np.float64]) -> None:
        """Append a single design matrix row to TSV file.

        Parameters
        ----------
        row : NDArray
            Design matrix row to append (9 values: task, intercept, drift, 6 motion params)
        """
        if self._design_file is not None:
            row_str = "\t".join(f"{val:.6f}" for val in row)
            self._design_file.write(row_str + "\n")
            self._design_file.flush()

    def _close_design_matrix(self) -> None:
        """Close design matrix file handle."""
        if self._design_file is not None:
            self._design_file.close()
            self._design_file = None

    @property
    def tr(self) -> float:
        """
        Get the TR (repetition time) in seconds.

        Returns
        -------
        float
            TR in seconds

        Raises
        ------
        ValueError
            If TR has not been set and cannot be determined from simulator

        Notes
        -----
        In simulation mode, TR is extracted from the simulator.
        In real-time mode, TR must be provided to the constructor or will be
        extracted from the first DICOM file.
        """
        if self._tr is not None:
            return self._tr

        raise ValueError(
            "TR not available. Provide tr parameter to NeurofeedbackCoordinator "
            "constructor or wait for first DICOM to be processed."
        )

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """
        Wait until the coordinator is ready to process volumes.

        This method blocks until the coordinator's watchdog Observer has started
        and is ready to detect new DICOM files, or until the timeout is reached.

        Parameters
        ----------
        timeout : float, default=5.0
            Maximum time to wait in seconds

        Returns
        -------
        bool
            True if coordinator is ready, False if timeout occurred

        Raises
        ------
        TimeoutError
            If coordinator is not ready within timeout period

        Examples
        --------
        >>> coordinator = NeurofeedbackCoordinator(...)
        >>> coordinator.wait_until_ready(timeout=5)
        True
        """
        is_ready = self._ready_event.wait(timeout=timeout)
        if not is_ready:
            raise TimeoutError(
                f"Coordinator not ready within {timeout}s. "
                "Ensure coordinator.run() has been called in a background thread."
            )
        return is_ready

    def send_trigger(self) -> None:
        """
        Signal that acquisition should begin.

        This method is called by the task (e.g., after displaying "waiting for scanner"
        and receiving a trigger key press) to signal the coordinator to start processing.

        In simulation mode, the trigger is sent automatically. In real-time mode, this
        should be called after a scanner trigger is received (e.g., keyboard press).

        Examples
        --------
        >>> coordinator = NeurofeedbackCoordinator(...)
        >>> coordinator.wait_until_ready()
        >>> coordinator.send_trigger()  # Start acquisition
        """
        self._trigger_event.set()
        logger.info("Trigger received - acquisition will begin")

    def check_error(self) -> None:
        """
        Check if an error occurred in the coordinator or simulator thread.

        This method should be called periodically (e.g., from the task loop)
        to detect and propagate errors that occurred in background threads.

        Raises
        ------
        Exception
            Re-raises the stored exception if one occurred

        Examples
        --------
        >>> coordinator = NeurofeedbackCoordinator(...)
        >>> # In task loop:
        >>> coordinator.check_error()  # Raises if background error occurred
        """
        if self._error is not None:
            raise self._error

    def report_outlier(self, volume: int) -> None:
        """
        Report that the current volume is an outlier.

        This method should be called by tasks (e.g., BallTask) when they detect
        an outlier activation that exceeds their threshold. The outlier flag will
        be included in the feedback log TSV for the specified volume.

        Parameters
        ----------
        volume : int
            Volume number that is an outlier

        Notes
        -----
        This method is typically called from the task's activation callback
        after processing activations and detecting outliers.
        """
        if volume == self._latest_volume:
            self._latest_is_outlier = True

    def is_complete(self) -> bool:
        """
        Check if the feedback task is complete or stopped.

        Returns True if any of these conditions are met:
        - All expected volumes have been processed
        - Stop event has been set (via stop() method)
        - An error has occurred

        Returns
        -------
        bool
            True if session is complete, False otherwise

        Examples
        --------
        >>> coordinator = NeurofeedbackCoordinator(...)
        >>> while not coordinator.is_complete():
        ...     # Process task updates
        ...     pass
        """
        return self.n_processed >= self.n_volumes or self._stop_event.is_set() or self._error is not None

    def was_stopped(self) -> bool:
        """
        Check if the session was stopped via stop() method.

        This differs from is_complete() in that it specifically checks
        if the stop() method was called, rather than natural completion.

        Returns
        -------
        bool
            True if stop() was called, False otherwise

        Examples
        --------
        >>> coordinator = NeurofeedbackCoordinator(...)
        >>> # ... run session ...
        >>> if coordinator.was_stopped():
        ...     print("Session was interrupted by user")
        ... else:
        ...     print("Session completed naturally")
        """
        return self._stop_event.is_set()

    def _on_new_file(self, file_path: Path) -> None:
        """Callback when a new file is detected.

        Adds file to priority queue (heap) sorted by full path.
        Uses full path as key to avoid collisions when scanner creates
        per-sequence subfolders (e.g., Siemens).
        """
        with self._lock:
            key = str(file_path)
            if key not in self._processed_files:
                heapq.heappush(self._pending_heap, (key, file_path))

    def _scan_for_existing_files(self) -> None:
        """Scan watch directory recursively for volume files not yet processed.

        This handles files that may have been missed by watchdog events,
        particularly when the simulator runs faster than processing.
        Recursive scan supports Siemens-style per-sequence subfolders.
        """
        # Get all files OUTSIDE the lock to avoid blocking other threads during I/O
        # Supports extensionless DICOMs from some scanner exports
        found_files = [f for f in self.watch_dir.rglob("*") if f.is_file()]

        # Only hold lock for the quick heap operations
        with self._lock:
            existing_keys = {item[0] for item in self._pending_heap}
            for file_path in found_files:
                key = str(file_path)
                if key not in self._processed_files and key not in existing_keys:
                    heapq.heappush(self._pending_heap, (key, file_path))
                    existing_keys.add(key)

    def _get_next_file_to_process(self) -> Path | None:
        """Get the next file to process in strict order, with retry for truncated files.

        Also filters MoCo vs original DICOM series when scanner MoCo is enabled.
        In siemens_moco mode, keeps only MoCo series; otherwise keeps originals.
        """
        with self._lock:
            while self._pending_heap:
                key, file_path = self._pending_heap[0]

                if key in self._processed_files:
                    heapq.heappop(self._pending_heap)
                    continue

                if is_valid_volume(file_path, validate_pixels=True):
                    # Filter MoCo vs original series
                    moco = is_moco_series(file_path)
                    if moco is not None:
                        if self._keep_moco and not moco:
                            heapq.heappop(self._pending_heap)
                            self._processed_files.add(key)
                            logger.debug(f"Skipping original volume (using MoCo): {file_path.name}")
                            continue
                        if not self._keep_moco and moco:
                            heapq.heappop(self._pending_heap)
                            self._processed_files.add(key)
                            logger.debug(f"Skipping MoCo-derived volume: {file_path.name}")
                            continue

                    heapq.heappop(self._pending_heap)
                    self._processed_files.add(key)
                    self._retry_counts.pop(key, None)
                    return file_path
                else:
                    heapq.heappop(self._pending_heap)
                    retries = self._retry_counts.get(key, 0) + 1
                    self._retry_counts[key] = retries
                    if retries >= self._MAX_RETRIES:
                        self._processed_files.add(key)
                        logger.warning(f"Skipping volume after {retries} failed validations: {file_path.name}")
                    else:
                        logger.debug(
                            f"Validation failed for {file_path.name} "
                            f"(attempt {retries}/{self._MAX_RETRIES}), will retry"
                        )

            return None

    def _get_queue_depth(self) -> int:
        """Get current queue depth (files waiting to be processed)."""
        with self._lock:
            return len(self._pending_heap)

    def _process_volume(self, volume_path: Path) -> None:
        """Convert volume (if DICOM) and process through estimator, tracking timing.

        Handles both DICOM and NIfTI input files. For DICOM, reads pixel data
        directly with pydicom (~15ms) instead of spawning dcm2niix (~206ms).
        For NIfTI, copies directly to the output directory along with JSON sidecar.
        """
        process_start = time.monotonic()

        # Extract TR from first volume if not yet set
        if self._tr is None and self.n_processed == 0:
            self._tr = get_tr_from_volume(volume_path)
            logger.info(f"Extracted TR from volume: {self._tr}s")

        # Copy volume to collection directory (for unified collection with rest/reference)
        try:
            import shutil

            dest_path = self._dicom_collection_dir / volume_path.name
            shutil.copy2(volume_path, dest_path)
            # For NIfTI, also copy JSON sidecar
            if is_nifti_file(volume_path):
                json_src = get_json_sidecar_path(volume_path)
                if json_src.exists():
                    json_dst = get_json_sidecar_path(dest_path)
                    shutil.copy2(json_src, json_dst)
        except Exception as e:
            logger.warning(f"Failed to copy volume to collection dir: {e}")

        # Read volume data (timed for diagnostics)
        volume_num = self.n_processed + 1
        output_name = f"vol_{volume_num:04d}"

        dcm2niix_start = time.monotonic()

        # In-memory data from DICOM read (None for NIfTI input)
        vol_data_mem: NDArray[np.float64] | None = None
        vol_affine_mem: NDArray[np.float64] | None = None
        dicom_image_comments: str | None = None

        if is_nifti_file(volume_path):
            # Input is already NIfTI - copy directly to output directory
            # Determine extension (.nii or .nii.gz)
            if str(volume_path).lower().endswith(".nii.gz"):
                ext = ".nii.gz"
            else:
                ext = ".nii"
            nifti_path = self.nifti_dir / f"{output_name}{ext}"
            try:
                shutil.copy2(volume_path, nifti_path)
                # Also copy JSON sidecar if present
                json_src = get_json_sidecar_path(volume_path)
                json_dest = self.nifti_dir / f"{output_name}.json"
                json_path: Path | None = json_dest
                if json_src.exists():
                    shutil.copy2(json_src, json_dest)
                else:
                    json_path = None  # No sidecar available
                    if self._motion_correction == "siemens_moco":
                        logger.warning(
                            f"No JSON sidecar found for {volume_path.name}; "
                            "motion parameters will be set to zero for this volume."
                        )
            except Exception as e:
                logger.error(f"Failed to copy NIfTI {volume_path}: {e}")
                return
        else:
            # Input is DICOM — read directly with pydicom (bypasses dcm2niix)
            json_path = None
            nifti_path = self.nifti_dir / f"{output_name}.nii"
            try:
                vol_data_mem, vol_affine_mem, _spacing, self._dicom_geometry_cache = _read_dicom_volume(
                    volume_path,
                    target_ornt=self._estimator._mask_ornt,
                    geometry_cache=self._dicom_geometry_cache,
                )

                # For siemens_moco, extract ImageComments from DICOM header
                # (already loaded by _read_dicom_volume, but we need the header field)
                if self._motion_correction == "siemens_moco":
                    import pydicom

                    dcm = pydicom.dcmread(volume_path, stop_before_pixels=True)
                    dicom_image_comments = str(getattr(dcm, "ImageComments", ""))
            except Exception as e:
                logger.error(f"Failed to read DICOM {volume_path}: {e}")
                return

        dcm2niix_duration = (time.monotonic() - dcm2niix_start) * 1000  # ms

        # Process through estimator (timed for diagnostics)
        glm_start = time.monotonic()
        _reg_ms = 0.0
        _est_ms = 0.0
        try:
            # Determine motion source based on motion correction mode
            motion_source: Path | NDArray[np.float64] | None
            corrected_data: NDArray[np.float64] | None = None
            corrected_affine: NDArray[np.float64] | None = None
            if self._motion_correction == "siemens_moco":
                if json_path is not None:
                    # NIfTI input: use JSON sidecar (existing path)
                    motion_source = json_path
                elif dicom_image_comments is not None:
                    # DICOM input: parse motion params directly from ImageComments
                    parsed = parse_siemens_motion_comment(dicom_image_comments)
                    if parsed is not None:
                        motion_source = parsed
                    else:
                        logger.warning(
                            f"No motion parameters in DICOM ImageComments for volume {volume_num}, using zeros"
                        )
                        motion_source = np.zeros(6, dtype=np.float64)
                else:
                    motion_source = np.zeros(6, dtype=np.float64)

                if vol_data_mem is not None:
                    corrected_data = vol_data_mem
                    corrected_affine = vol_affine_mem

            elif self._motion_correction == "software":
                # Software (ANTsPy): frame-to-reference rigid registration with warm-start
                if vol_data_mem is not None:
                    # DICOM input: use in-memory data directly
                    assert vol_affine_mem is not None
                    vol_data = vol_data_mem
                    vol_affine = vol_affine_mem
                else:
                    # NIfTI input: load from disk
                    vol_img = cast(Nifti1Image, nib.load(nifti_path))
                    vol_data = np.asarray(vol_img.get_fdata(), dtype=np.float64)
                    vol_affine = np.asarray(vol_img.affine, dtype=np.float64)

                # Ensure we have a 3D volume, not 4D
                if vol_data.ndim == 4:
                    logger.warning(
                        f"Software MoCo: volume {volume_num} is 4D with shape {vol_data.shape}, extracting first volume"
                    )
                    vol_data = vol_data[..., 0]

                if self._moco_reference_vol is None:
                    # First volume becomes reference — cache ANTs image + spacing
                    import ants

                    self._moco_reference_vol = np.array(vol_data, dtype=np.float64)
                    self._moco_reference_affine = np.array(vol_affine, dtype=np.float64)
                    self._moco_spacing = np.sqrt(np.sum(vol_affine[:3, :3] ** 2, axis=0)).tolist()
                    self._moco_reference_ants = ants.from_numpy(vol_data.astype(np.float32), spacing=self._moco_spacing)
                    self._moco_prev_transform = None
                    motion_source = np.zeros(6, dtype=np.float64)
                    corrected_data = vol_data
                    corrected_affine = vol_affine
                    logger.info(f"Software MoCo: volume {volume_num} set as reference, shape={vol_data.shape}")
                else:
                    assert self._moco_reference_ants is not None
                    assert self._moco_reference_affine is not None
                    assert self._moco_spacing is not None
                    # Verify shapes match
                    if vol_data.shape != self._moco_reference_vol.shape:
                        logger.error(f"MoCo shape mismatch: ref={self._moco_reference_vol.shape}, mov={vol_data.shape}")

                    # Register to REFERENCE volume (frame-to-ref) with warm-start
                    _reg_start = time.monotonic()
                    corrected_vol, motion_params, tx_path = _ants_register_volume(
                        self._moco_reference_ants,
                        np.asarray(vol_data, dtype=np.float64),
                        self._moco_spacing,
                        initial_transform=self._moco_prev_transform,
                    )
                    _reg_ms = (time.monotonic() - _reg_start) * 1000
                    self._moco_prev_transform = tx_path
                    motion_source = motion_params
                    corrected_data = corrected_vol
                    corrected_affine = self._moco_reference_affine
            else:
                # None: no motion correction, no motion parameters
                motion_source = None
                if vol_data_mem is not None:
                    corrected_data = vol_data_mem
                    corrected_affine = vol_affine_mem

            _est_start = time.monotonic()
            activations = self._estimator.process_volume(
                nifti_path,
                motion_source,
                volume_data=corrected_data,
                volume_affine=corrected_affine,
            )
            _est_ms = (time.monotonic() - _est_start) * 1000
            # Compute differential signal (explicit pair)
            if self._differential_name is not None:
                assert self._differential_pair is not None  # set alongside _differential_name
                m0, m1 = self._differential_pair
                activations[self._differential_name] = activations.get(m0, 0.0) - activations.get(m1, 0.0)
            # Compute connectivity (sliding-window Pearson correlation)
            if self._connectivity_estimator is not None:
                # In sham mode: accumulate replayed mask activations and compute from those
                if self._is_sham_mode:
                    for name in self.mask_names:
                        self._sham_timeseries[name].append(activations.get(name, 0.0))
                    conn_values = self._connectivity_estimator.compute(self._sham_timeseries)
                else:
                    conn_values = self._connectivity_estimator.compute(self._estimator.timeseries)
                activations.update(conn_values)
            self.design_matrix.append(self._estimator._last_design_row)
            # Write design row incrementally (so it's not lost on interrupt)
            self._append_design_row(self._estimator._last_design_row)
        except Exception as e:
            logger.error(f"Failed to process volume {volume_num}: {e}")
            return
        glm_duration = (time.monotonic() - glm_start) * 1000  # ms

        total_duration = (time.monotonic() - process_start) * 1000  # ms
        queue_depth = self._get_queue_depth()

        self._timing_stats["dcm2niix_ms"].append(dcm2niix_duration)
        self._timing_stats["glm_ms"].append(glm_duration)
        self._timing_stats["registration_ms"].append(_reg_ms)
        self._timing_stats["estimator_ms"].append(_est_ms)
        self._timing_stats["total_ms"].append(total_duration)
        self._timing_stats["queue_depth"].append(queue_depth)

        self.n_processed += 1
        elapsed_time = time.monotonic() - self._start_time if self._start_time else 0.0
        is_baseline = self.n_processed <= self.n_baseline

        expected_time = self.n_processed * (self._tr if self._tr else 1.2)  # fallback TR
        timing_drift = elapsed_time - expected_time

        with self._lock:
            self._latest_activations = activations.copy()
            self._latest_volume = self.n_processed
            self._latest_time = elapsed_time
            self._latest_is_baseline = is_baseline
            self._latest_is_outlier = False  # Reset before callback

        if self._activation_callback is not None:
            try:
                if self._feedback_interval is None or is_baseline:
                    # Continuous mode or baseline: deliver every volume
                    self._activation_callback(activations, self.n_processed, elapsed_time, is_baseline, False)
                else:
                    # Intermittent mode: accumulate and deliver mean every N volumes
                    self._intermittent_buffer.append(activations.copy())
                    if len(self._intermittent_buffer) >= self._feedback_interval:
                        mean_activations = {}
                        for key in self._intermittent_buffer[0]:
                            mean_activations[key] = sum(buf.get(key, 0.0) for buf in self._intermittent_buffer) / len(
                                self._intermittent_buffer
                            )
                        self._intermittent_buffer.clear()
                        self._activation_callback(mean_activations, self.n_processed, elapsed_time, False, False)
            except Exception as e:
                logger.error(f"Error in activation callback: {e}")
                if self._error_callback is not None:
                    with contextlib.suppress(Exception):
                        self._error_callback(e)

        # Read outlier status (may have been set by callback via report_outlier())
        with self._lock:
            is_outlier = self._latest_is_outlier

        # Get QC metrics from estimator (computed during process_volume)
        fd = self._estimator._latest_fd
        dvars = self._estimator._latest_dvars
        tsnr = self._estimator._latest_tsnr

        # Build CSV row: one column per mask (simplified format)
        # All masks are computed and logged regardless of condition
        row = [
            self.n_processed,
            f"{elapsed_time:.2f}",
            is_baseline,
            is_outlier,
            f"{fd:.2f}",
            f"{dvars:.2f}",
            f"{tsnr:.2f}",
            self._condition,  # Log condition (real/sham) for each volume
        ]

        # WM/CSF tissue signal values (independently toggled)
        if self._estimator._include_wm:
            row.append(f"{self._estimator._latest_wm_mean:.4f}")
        if self._estimator._include_csf:
            row.append(f"{self._estimator._latest_csf_mean:.4f}")

        # Add activation for each mask (these are delivered values - sham or real)
        row.extend([f"{activations.get(name, 0.0):.4f}" for name in self.mask_names])
        if self._differential_name:
            row.append(f"{activations.get(self._differential_name, 0.0):.4f}")
        # Connectivity values
        for conn_name in self._connectivity_names:
            row.append(f"{activations.get(conn_name, 0.0):.4f}")

        # In sham mode, also log the real computed activations
        if self._is_sham_mode:
            real_activations = self._estimator._last_real_activations
            row.extend([f"{real_activations.get(name, 0.0):.4f}" for name in self.mask_names])
            if self._differential_name:
                assert self._differential_pair is not None  # set alongside _differential_name
                real_diff = real_activations.get(self._differential_pair[0], 0.0) - real_activations.get(
                    self._differential_pair[1], 0.0
                )
                row.append(f"{real_diff:.4f}")
            # Real connectivity values (timeseries always stores real values)
            if self._connectivity_estimator is not None:
                real_conn = self._connectivity_estimator.compute(self._estimator.timeseries)
                for conn_name in self._connectivity_names:
                    row.append(f"{real_conn.get(conn_name, 0.0):.4f}")

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
        if self._csv_file is not None:
            self._csv_file.flush()

        # Log concise timing: proc = dcm2niix+GLM for this volume; drift = (elapsed - expected)
        status = "BASELINE" if is_baseline else "ACTIVE"
        activation_str = "  ".join(f"{name}={activations[name]:+.3f}" for name in self.mask_names)
        if self._differential_name and self._differential_name in activations:
            activation_str += f"  diff={activations[self._differential_name]:+.3f}"
        for conn_name in self._connectivity_names:
            if conn_name in activations:
                activation_str += f"  {conn_name}={activations[conn_name]:+.3f}"

        queue_warning = f" [QUEUE:{queue_depth}]" if queue_depth > 0 else ""
        timing_drift_ms = timing_drift * 1000  # Convert to ms for consistency
        drift_str = f"+{timing_drift_ms:.0f}" if timing_drift_ms >= 0 else f"{timing_drift_ms:.0f}"
        outlier_indicator = " OUTLIER" if is_outlier else ""

        task_suffix = ""
        if (
            self._activation_callback is not None
            and hasattr(self._activation_callback, "__self__")
            and hasattr(self._activation_callback.__self__, "get_volume_log_suffix")
        ):
            task_suffix = self._activation_callback.__self__.get_volume_log_suffix()

        logger.info(
            f"[{elapsed_time:7.2f}s] Vol {self.n_processed:4d}/{self.n_volumes} "
            f"({status}): {activation_str} "
            f"[proc:{total_duration:.0f}ms drift:{drift_str}ms{outlier_indicator}]{queue_warning}" + task_suffix
        )

    def run(self) -> None:
        """Run the coordinator until n_volumes processed or interrupted.

        Blocks until all volumes are processed or KeyboardInterrupt.
        """
        observer = None
        try:
            # Initialize CSV
            self._init_csv()

            # Catalog existing files so we only process NEW arrivals.
            # This prevents picking up DICOMs from previous sequences when
            # scanners (e.g., Siemens) create per-sequence subfolders.
            for f in self.watch_dir.rglob("*"):
                if f.is_file():
                    self._processed_files.add(str(f))
            if self._processed_files:
                logger.debug(f"Cataloged {len(self._processed_files)} existing files in watch dir")

            # Set up watchdog observer (recursive=True for Siemens-style subfolders)
            event_handler = _DicomEventHandler(self._on_new_file)
            observer = Observer()
            observer.schedule(event_handler, str(self.watch_dir), recursive=True)
            observer.start()
            logger.info(f"Watching {self.watch_dir} for DICOMs (recursive)...")

            self._ready_event.set()
            logger.debug("Coordinator ready signal set")

            logger.info("Waiting for trigger...")
            self._trigger_event.wait()  # Block until trigger received or stop() sets it

            if self._stop_event.is_set():
                logger.info("Stop requested before trigger; exiting coordinator")
            else:
                logger.info("Trigger received - starting acquisition")

            self._start_time = time.monotonic()

            # Prune cataloged files that were deleted between cataloging and
            # trigger (e.g., simulator cleared the incoming directory).  Without
            # this, new files with the same names are silently skipped.
            before = len(self._processed_files)
            self._processed_files = {f for f in self._processed_files if Path(f).exists()}
            pruned = before - len(self._processed_files)
            if pruned:
                logger.info(f"Pruned {pruned} stale entries from processed-files set")

            while not self._stop_event.is_set() and self.n_processed < self.n_volumes:
                # If stop was requested before trigger, skip processing entirely
                if self._stop_event.is_set():
                    break

                # Get next file in strict order (smallest filename first)
                file_to_process = self._get_next_file_to_process()

                if file_to_process:
                    self._process_volume(file_to_process)
                else:
                    # Watchdog missed something or queue empty; rescan and back off
                    self._scan_for_existing_files()
                    time.sleep(0.05)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception(f"Coordinator error: {e}")
            self._error = e
            self._stop_event.set()
        finally:
            if observer is not None:
                observer.stop()
                observer.join()
            self._close_csv()
            self._close_design_matrix()  # Design matrix was written incrementally

            elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
            logger.info(f"Session complete: {self.n_processed} volumes in {elapsed:.1f}s")
            logger.info(f"Log saved to: {self._csv_path}")
            logger.info(f"Design matrix saved to: {self._design_matrix_path}")

            # Log timing statistics
            self._log_timing_summary()

            # Generate QC report if requested
            if self._qc_enabled:
                self._generate_qc_report()

            # Cleanup temporary directories if requested
            self._cleanup_directories()

    def _log_timing_summary(self) -> None:
        """Log summary of processing timing statistics."""
        if not self._timing_stats["total_ms"]:
            return

        total_ms = np.array(self._timing_stats["total_ms"])
        dcm2niix_ms = np.array(self._timing_stats["dcm2niix_ms"])
        glm_ms = np.array(self._timing_stats["glm_ms"])
        queue_depth = np.array(self._timing_stats["queue_depth"])

        tr_ms = (self._tr or 1.2) * 1000  # TR in ms

        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"TR: {self._tr:.3f}s ({tr_ms:.0f}ms)")
        logger.info(f"Volumes processed: {len(total_ms)}")
        logger.info("-" * 40)
        logger.info(
            f"Total processing:  mean={total_ms.mean():.1f}ms, std={total_ms.std():.1f}ms, max={total_ms.max():.1f}ms"
        )
        logger.info(
            f"  dcm2niix:        mean={dcm2niix_ms.mean():.1f}ms, "
            f"std={dcm2niix_ms.std():.1f}ms, max={dcm2niix_ms.max():.1f}ms"
        )
        logger.info(
            f"  GLM update:      mean={glm_ms.mean():.1f}ms, std={glm_ms.std():.1f}ms, max={glm_ms.max():.1f}ms"
        )
        reg_ms = np.array(self._timing_stats["registration_ms"])
        est_ms = np.array(self._timing_stats["estimator_ms"])
        if reg_ms.sum() > 0:
            logger.info(
                f"    registration:  mean={reg_ms.mean():.1f}ms, std={reg_ms.std():.1f}ms, max={reg_ms.max():.1f}ms"
            )
        if est_ms.sum() > 0:
            logger.info(
                f"    estimator:     mean={est_ms.mean():.1f}ms, std={est_ms.std():.1f}ms, max={est_ms.max():.1f}ms"
            )
        logger.info("-" * 40)

        # Check if processing typically exceeded TR
        pct_exceeds_tr = (total_ms > tr_ms).mean() * 100
        max_queue = queue_depth.max()

        if pct_exceeds_tr > 10:
            logger.warning(f"Processing exceeded TR in {pct_exceeds_tr:.1f}% of volumes!")
            logger.warning(f"Max queue depth: {max_queue} (volumes waiting)")
        else:
            logger.info(f"Processing within TR budget: {100 - pct_exceeds_tr:.1f}% of volumes")
            logger.info(f"Max queue depth: {max_queue}")

        logger.info("=" * 60)

    def _generate_qc_report(self) -> None:
        """Generate QC report after session completes."""
        try:
            from . import qc
            from .tasks.task_registry import get_task_metadata

            live_metrics = None
            if self._task_id:
                try:
                    live_metrics = list(get_task_metadata(self._task_id).live_metrics)
                except (KeyError, ValueError):
                    live_metrics = None

            logger.info("Generating feedback QC report...")
            report_path = qc.generate_feedback_report(
                feedback_dir=self.output_dir,
                subject=self.subject,
                session=self.session,
                task=self.task,
                run=self.run_number,
                acq=self.acq,
                output_dir=self.output_dir / "qc",
                overwrite=True,
                open_browser=False,
                expected_volumes=self.n_volumes,
                fd_threshold=self._fd_threshold,
                live_metrics=live_metrics,
            )
            logger.info(f"QC report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate QC report: {e}")
            logger.debug("QC report generation error", exc_info=True)

    def get_timing_stats(self) -> dict:
        """Get timing statistics for analysis.

        Returns
        -------
        dict
            Dictionary with timing arrays:
            - 'dcm2niix_ms': dcm2niix conversion time per volume (ms)
            - 'glm_ms': GLM update time per volume (ms)
            - 'total_ms': total processing time per volume (ms)
            - 'queue_depth': queue depth at each volume
        """
        return {k: np.array(v) for k, v in self._timing_stats.items()}

    def _cleanup_directories(self) -> None:
        """Clean up temporary directories after session completes."""
        import shutil

        if self._cleanup_incoming and self.watch_dir.exists():
            try:
                shutil.rmtree(self.watch_dir)
                logger.info(f"Cleaned up incoming directory: {self.watch_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up incoming directory: {e}")

        if self._cleanup_nifti and self.nifti_dir.exists():
            try:
                shutil.rmtree(self.nifti_dir)
                logger.info(f"Cleaned up nifti directory: {self.nifti_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up nifti directory: {e}")

    def cleanup(self, force: bool = False) -> None:
        """Clean up temporary directories created during the session.

        Use this method to explicitly clean up incoming DICOM and converted
        NIfTI directories after a feedback task completes. This is useful
        when automatic cleanup is disabled (e.g., for debugging) but you want
        to clean up directories after inspecting them.

        Parameters
        ----------
        force : bool, optional
            If True, clean up directories regardless of the `_cleanup_incoming`
            and `_cleanup_nifti` flags set during initialization. If False
            (default), only clean up directories if their respective cleanup
            flags are True.

        Notes
        -----
        This method is safe to call multiple times; it will only remove
        directories that exist. Directories removed:

        - ``watch_dir`` (incoming): Where DICOMs are copied/received
        - ``nifti_dir``: Where converted NIfTI volumes are stored

        Examples
        --------
        >>> # Clean up only if cleanup flags were set during init
        >>> coordinator.cleanup()
        >>>
        >>> # Force cleanup regardless of initial settings
        >>> coordinator.cleanup(force=True)
        """
        if force:
            self._cleanup_incoming = True
            self._cleanup_nifti = True
        self._cleanup_directories()

    def stop(self, show_questions: bool = False) -> None:
        """Signal the coordinator to stop.

        Parameters
        ----------
        show_questions : bool, optional
            If True, the task should still show post-run questions to the participant
            before exiting. Default is False (skip questions on stop).
        """
        self._show_questions_on_stop = show_questions
        self._stop_event.set()
        # Also release trigger wait so run() can exit cleanly even if task aborted pre-trigger
        self._trigger_event.set()
        # Close estimator resources (file handles, etc.)
        if hasattr(self, "_estimator"):
            self._estimator.close()

    def should_show_questions(self) -> bool:
        """Check if post-run questions should be shown.

        Returns True if either:
        - The session completed naturally (not stopped)
        - The session was stopped but show_questions was requested
        """
        if not self.was_stopped():
            return True
        return getattr(self, "_show_questions_on_stop", False)

    def get_latest_activations(self) -> dict:
        """Get latest activation values in thread-safe manner.

        Returns
        -------
        dict
            Dictionary containing:
            - 'activations': dict[str, float] - Activation per mask
            - 'volume': int - Current volume number
            - 'time': float - Elapsed time in seconds
            - 'is_baseline': bool - Whether in baseline period
        """
        with self._lock:
            return {
                "activations": self._latest_activations.copy(),
                "volume": self._latest_volume,
                "time": self._latest_time,
                "is_baseline": self._latest_is_baseline,
            }


# =============================================================================
# SECTION 5: HIGH-LEVEL SESSION ORCHESTRATION
# =============================================================================
# This section provides the user-facing API for running feedback tasks.
#
# run_feedback_task() is designed to be the primary entry point for most
# users. It handles all the complexity of:
# - Setting up the coordinator with appropriate parameters
# - Configuring simulation mode for offline testing
# - Managing the background thread that processes volumes
# - Setting up sham conditions for control groups
#
# For users who need more control, they can instantiate NeurofeedbackCoordinator
# directly. But for typical use cases, run_feedback_task() is recommended.


def run_feedback_task(
    mask_paths: list[Path | str],
    subject: str,
    session: str,
    task: str,
    run: int,
    output_dir: Path | str,
    collection_dir: Path | str,
    n_baseline: int,
    n_volumes: int,
    tr: float,
    task_regressors: NDArray[np.float64] | None = None,
    incoming_dir: Path | str | None = None,
    mask_names: list[str] | None = None,
    roi_combine_method: str = "inverse_sd",
    simulator_source_dir: Path | str | None = None,
    simulator_speed: float = 1.0,
    simulator_clear_target: bool = True,
    simulator_cleanup: bool = True,
    simulator_trigger_mode: str = "manual",
    activation_callback: Callable[[dict[str, float], int, float, bool, bool], None] | None = None,
    error_callback: Callable[[Exception], None] | None = None,
    qc_enabled: bool = True,
    fd_threshold: float = 0.5,
    step_id: str | None = None,
    sham_provider: Optional["ShamActivationProvider"] = None,
    condition: str = "real",
    motion_correction: str = "siemens_moco",
    include_drift: bool = True,
    include_motion_regressors: bool = True,
    include_wm_regressor: bool = False,
    include_csf_regressor: bool = False,
    dump_algo_vars_path: Path | str | None = None,
    smooth_fwhm: float | None = None,
    differential_pair: list[str] | None = None,
    feedback_interval: int | None = None,
    connectivity_pairs: list[list[str]] | None = None,
    connectivity_window: int = 15,
    connectivity_target: str = "increase",
    wm_mask_path: Path | str | None = None,
    csf_mask_path: Path | str | None = None,
    task_id: str | None = None,
) -> dict:
    """Run a complete feedback task with optional volume simulation.

    This is the main entry point for running neurofeedback tasks. It handles
    both simulated (offline) and real-time (scanner) acquisition modes. Supports
    both DICOM and NIfTI input files.

    Parameters
    ----------
    mask_paths : list of Path or str
        Paths to NIfTI mask files for feedback estimation.
    subject : str
        Subject identifier (e.g., 'sub-01').
    task : str
        Task name (e.g., 'rest', 'nback').
    run : int
        Run number (e.g., 1 for run-01).
    output_dir : Path or str
        Base directory for all outputs (NIfTI conversions, logs, etc.).
    collection_dir : Path or str
        Base directory for incoming volumes. When incoming_dir is not specified,
        defaults to collection_dir/incoming as the watch directory.
    n_baseline : int
        Number of initial volumes for baseline.
    n_volumes : int
        Total number of volumes to process.
    tr : float
        Repetition time in seconds.
    task_regressors : ndarray, shape (n_volumes, n_task_regressors), optional
        Pre-computed task regressors to include in the GLM. If None, uses
        nuisance-only model. Task-specific designs (e.g., boxcar+HRF for ball task)
        should be computed externally and passed here.
    incoming_dir : Path or str, optional
        Directory to watch for incoming volumes. If None, uses collection_dir/incoming.
        For scanner acquisition, this should be the directory where the scanner
        writes volumes.
    mask_names : list of str, optional
        Names for each mask. If not provided, uses mask filenames.
    roi_combine_method : str, optional
        How per-voxel activations are combined into one ROI value:
        ``"inverse_sd"`` (Hinds et al. 2011 efficiency weighting, w = 1/sigma;
        the default) or ``"mean"`` (unweighted average).
    simulator_source_dir : Path or str, optional
        If provided, runs in simulation mode. Directory containing source volume
        files (DICOM or NIfTI) to simulate scanner acquisition. If None, runs in
        real-time mode and watches incoming_dir for scanner-generated volumes.
    simulator_speed : float, optional
        Speed multiplier for simulation (e.g., 2.0 for 2x speed). Default is 1.0.
        Only used if simulator_source_dir is provided.
    simulator_clear_target : bool, optional
        Whether to clear incoming directory before simulation starts. Default is True.
        Only used if simulator_source_dir is provided.
    simulator_cleanup : bool, optional
        Whether to delete incoming directory after simulation completes. Default is True.
        Only used if simulator_source_dir is provided.
    activation_callback : callable, optional
        Function called after each volume with signature:
        callback(activations: dict, volume: int, elapsed: float, is_baseline: bool, is_outlier: bool).
        Use this to update PsychoPy displays immediately when new data arrives.
        Default is None.
    error_callback : callable, optional
        Function called if activation_callback raises an exception.
        Signature: callback(exception: Exception). Default is None.
    qc_enabled : bool, optional
        If True, generate a QC report after the session completes. The report
        includes activation timeseries, baseline distributions, and design matrix
        visualizations. Default is True.
    step_id : str, optional
        Workflow step identifier (e.g., 'transfer_1', 'feedback_2'). If provided,
        uses slot-based run numbering with BIDS acq entity for retry tracking.
        The step_id determines the run number (transfer_1/feedback_1 -> run-01,
        transfer_2/feedback_2 -> run-02) and an acquisition letter (a, b, c...)
        is appended for each attempt at the same slot. When step_id is provided,
        the `run` parameter is ignored and determined from the step_id mapping.
    sham_provider : ShamActivationProvider, optional
        Sham activation source for control conditions. When provided, the estimator
        returns sham activations (from provider) to the task while still computing
        real activations for logging. Both delivered and real values are logged
        to feedbacklog.tsv for post-hoc analysis. Default is None (real feedback).
    condition : str, optional
        Participant condition label to log in feedbacklog.tsv. Should be 'real'
        or 'sham'. This is logged for each volume but doesn't affect processing -
        the sham_provider determines what activations are delivered. Default is 'real'.
    motion_correction : str, optional
        Motion correction mode: "siemens_moco" (parse from JSON sidecar),
        "software" (ANTsPy registration, ~400-500ms/vol), or "none" (disabled).
    include_drift : bool, optional
        Include linear drift regressor in GLM. Default True.
    include_motion_regressors : bool, optional
        Include 6 motion parameters in GLM. Default True. Requires
        motion_correction != "none".
    include_wm_regressor : bool, optional
        Include WM mean signal in GLM. Default False.
    include_csf_regressor : bool, optional
        Include CSF mean signal in GLM. Default False.
    smooth_fwhm : float, optional
        Gaussian spatial smoothing FWHM in mm applied to each volume before
        voxel extraction. None (default) disables smoothing.
    differential_pair : list of str, optional
        Ordered pair [positive_mask, negative_mask] for differential signal.
        None (default) auto-computes from first two masks if 2+ masks given.
    feedback_interval : int, optional
        Number of volumes between feedback deliveries (intermittent mode).
        When set, activations are accumulated and the mean is delivered to
        the task every N volumes after baseline. None (default) delivers
        feedback every volume (continuous mode).
    connectivity_pairs : list of list of str, optional
        Pairs of mask names for sliding-window Pearson correlation.
        Each pair is [mask_a, mask_b]. None (default) disables connectivity.
        Example: [["ContA", "DefaultA"]] computes correlation between
        ContA and DefaultA, logged as corr_ContA_DefaultA.
    connectivity_window : int, optional
        Sliding window size in volumes for connectivity computation.
        Default is 15. Must be >= 3.
    connectivity_target : str, optional
        Target direction for connectivity. "increase" (default) means higher
        correlation = better. "decrease" negates the value so stronger
        anti-correlation = positive signal for the task.

    Returns
    -------
    session_info : dict
        Dictionary containing:
        - 'coordinator': NeurofeedbackCoordinator instance
        - 'thread': Background thread running the coordinator
        - 'mode': 'simulated' or 'real-time'
        - 'started': True (always, indicates session has started)

        Use coordinator.wait_until_ready() before accessing coordinator methods.
        Use coordinator.stop() and thread.join() to gracefully shut down.

    Examples
    --------
    Simulated (offline) mode:

    >>> results = run_feedback_task(
    ...     mask_paths=['/path/to/dmn_mask.nii.gz'],
    ...     subject='sub-01',
    ...     task='nback',
    ...     run=1,
    ...     output_dir='/path/to/output',
    ...     n_baseline=25,
    ...     n_volumes=150,
    ...     tr=1.2,
    ...     simulator_source_dir='/path/to/source_dicoms',
    ...     simulator_speed=10.0,  # 10x speed for testing
    ... )

    Real-time (scanner) mode:

    >>> results = run_feedback_task(
    ...     mask_paths=['/path/to/dmn_mask.nii.gz'],
    ...     subject='sub-01',
    ...     task='nback',
    ...     run=1,
    ...     output_dir='/path/to/output',
    ...     n_baseline=25,
    ...     n_volumes=200,
    ...     tr=1.2,
    ...     incoming_dir='/scanner/export/dicoms',
    ... )

    Notes
    -----
    **Event Ordering and Trigger Mechanism**

    The function implements a trigger-based workflow to synchronize task display
    with data acquisition:

    1. Coordinator starts in background thread
    2. Coordinator signals ready (via wait_until_ready())
    3. Task waits for trigger (keyboard or automatic)
    4. Trigger starts data acquisition
    5. Baseline period begins
    6. Feedback display starts after baseline

    In **simulation mode**, the trigger is sent automatically when the task calls
    coordinator.send_trigger(). This allows for seamless testing without manual
    intervention.

    In **real-time mode**, the coordinator waits for the task to call send_trigger()
    (typically after displaying "waiting for scanner" and receiving a keyboard
    trigger like 't'). This synchronizes with scanner pulse timing.

    **Threading Model**

    The function always runs the coordinator in a background thread and returns
    immediately. This enables integration with GUI frameworks like PsychoPy.

    In simulation mode, the coordinator starts a simulator thread to copy DICOMs
    at TR intervals. In real-time mode, it simply watches the incoming directory
    for scanner-generated DICOMs.

    Results are logged to CSV and the design matrix is saved to TSV.
    """
    import threading

    from .simulator import VolumeSimulator

    # Convert paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine run number and acquisition number based on step_id.
    # step_id is required: it selects the workflow slot ("feedback_1", "transfer_2", ...)
    # which `get_slot_run_info` uses to derive run-NN and the next acq-NN attempt.
    if step_id is None:
        raise ValueError("step_id is required: it selects the workflow slot for slot-based acq tracking.")
    run_info = get_slot_run_info(output_dir, subject, session, task, step_id)
    run = run_info["run"]  # Override run parameter with slot-determined value
    acq = run_info["acq"]
    logger.info(f"Using slot-based naming for {step_id}: run-{run:02d}, acq-{acq:02d}")

    # Check for and archive any interrupted run before starting fresh.
    # With acq-based naming, a new acq means a fresh attempt; we only archive if
    # there's a partial file already at the *same* acq (shouldn't normally happen).
    interrupted = _detect_interrupted_run(
        output_dir=output_dir,
        subject=subject,
        session=session,
        task=task,
        run=run,
        n_volumes=n_volumes,
        acq=acq,
    )
    if interrupted is not None:
        logger.warning(f"Found interrupted run with {interrupted['volumes_completed']} volumes (expected {n_volumes})")
        # Generate partial QC before archiving
        _generate_partial_qc(output_dir, subject, session, task, run, acq=acq, task_id=task_id)
        # Archive interrupted data
        archive_path = _archive_interrupted_run(output_dir, subject, session, task, run, acq=acq)
        logger.info(f"Archived interrupted run to: {archive_path}")

    # Create DICOM collection folder for this attempt (e.g., collection/feedback_acq-01_run-01)
    collection_dir = Path(collection_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)
    dicom_collection_dir = collection_dir / f"{task}_acq-{acq:02d}_run-{run:02d}"
    dicom_collection_dir.mkdir(parents=True, exist_ok=True)

    # Default incoming_dir to collection_dir/incoming if not specified
    if incoming_dir is None:
        incoming_dir = collection_dir / "incoming"

    # Determine mode and set up incoming directory
    if simulator_source_dir is not None:
        mode = "simulated"
        incoming_dir = Path(incoming_dir)
        incoming_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running in SIMULATION mode")
        logger.info(f"Source DICOMs: {simulator_source_dir}")
        logger.info(f"Target incoming: {incoming_dir}")
        logger.info(f"DICOM collection: {dicom_collection_dir}")
    else:
        mode = "real-time"
        incoming_dir = Path(incoming_dir)
        if not incoming_dir.exists():
            raise FileNotFoundError(
                f"Incoming directory not found: {incoming_dir}. "
                "For real-time mode, this should be the directory where the scanner writes DICOMs."
            )
        logger.info("Running in REAL-TIME mode")
        logger.info(f"Watching: {incoming_dir}")
        logger.info(f"DICOM collection: {dicom_collection_dir}")

    # Set up simulator if in simulation mode
    simulator = None
    if mode == "simulated":
        # Type checker: simulator_source_dir is guaranteed non-None here
        # because mode='simulated' only when simulator_source_dir is not None
        assert simulator_source_dir is not None

        # Create simulator instance
        # Note: cleanup_on_complete is set to False here because the coordinator
        # handles cleanup of both incoming and nifti directories after completion
        simulator = VolumeSimulator(
            source_dir=simulator_source_dir,
            target_dir=incoming_dir,
            tr=tr,
            speed=simulator_speed,
            clear_target=simulator_clear_target,
            cleanup_on_complete=False,  # Coordinator handles cleanup
            trigger_mode=simulator_trigger_mode,  # type: ignore[arg-type]
        )
        logger.info(f"Simulator configured (speed={simulator_speed}x)")

    # Log sham mode if active
    if sham_provider is not None:
        logger.info(f"SHAM MODE ACTIVE: condition='{condition}'")

    # Initialize coordinator with simulator and sham configuration
    coordinator = NeurofeedbackCoordinator(
        watch_dir=incoming_dir,
        output_dir=output_dir,
        dicom_collection_dir=dicom_collection_dir,
        mask_paths=mask_paths,
        subject=subject,
        session=session,
        task=task,
        run=run,
        n_baseline=n_baseline,
        n_volumes=n_volumes,
        tr=tr,
        task_regressors=task_regressors,
        mask_names=mask_names,
        roi_combine_method=roi_combine_method,
        activation_callback=activation_callback,
        error_callback=error_callback,
        qc_enabled=qc_enabled,
        fd_threshold=fd_threshold,
        acq=acq,
        sham_provider=sham_provider,
        condition=condition,
        motion_correction=motion_correction,
        include_drift=include_drift,
        include_motion_regressors=include_motion_regressors,
        include_wm_regressor=include_wm_regressor,
        include_csf_regressor=include_csf_regressor,
        dump_algo_vars_path=dump_algo_vars_path,
        smooth_fwhm=smooth_fwhm,
        differential_pair=differential_pair,
        feedback_interval=feedback_interval,
        connectivity_pairs=connectivity_pairs,
        connectivity_window=connectivity_window,
        connectivity_target=connectivity_target,
        wm_mask_path=wm_mask_path,
        csf_mask_path=csf_mask_path,
        task_id=task_id,
    )

    # Set cleanup flags for simulation mode
    if mode == "simulated":
        coordinator._cleanup_incoming = simulator_cleanup
        coordinator._cleanup_nifti = simulator_cleanup

    # Start coordinator in background thread (always)
    coordinator_thread = threading.Thread(target=coordinator.run, daemon=False, name="FeedbackCoordinator")
    coordinator_thread.start()
    logger.info("Coordinator thread started")

    return {
        "coordinator": coordinator,
        "thread": coordinator_thread,
        "mode": mode,
        "started": True,
        "simulator": simulator,
    }
