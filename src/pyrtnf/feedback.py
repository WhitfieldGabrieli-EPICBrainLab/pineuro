"""
Real-time feedback estimation for fMRI neurofeedback.

Summary
-------
Incremental GLM fitting for continuous BOLD activation estimates, plus a
coordinator that watches for incoming DICOMs and streams activations to
tasks.

Components
----------
- IncrementalGLM: Online least-squares solver using Gentleman's algorithm
- FeedbackEstimator: Per-volume activation estimation with nuisance removal
- NeurofeedbackCoordinator: Watches for DICOMs, converts to NIfTI, logs results

Workflow
--------
- Compute task regressor (HRF boxcar), load masks, and build voxel index map
- For each volume: extract motion, build design row, update incremental GLM
- Aggregate voxel activations per mask (optionally efficiency-weighted)
- Coordinator watches incoming DICOMs, converts to NIfTI, streams activations/logs

Dependencies
------------
numpy, nibabel, watchdog

Notes
-----
Incremental updates avoid recomputing the QR decomposition every TR, which
keeps latency low enough for real-time use.

References
----------
Gentleman, W.M. (1974). Algorithm AS 75: Basic procedures for large, sparse
    or weighted linear least problems. Journal of the Royal Statistical
    Society. Series C (Applied Statistics), 23(3), 448-454.

Hinds, O., et al. (2011). Computing moment-to-moment BOLD activation for
    real-time neurofeedback. NeuroImage, 54(1), 361-368.
"""

# Standard library imports
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import threading
import time
from typing import Callable, Optional, Sequence, TextIO
from collections import OrderedDict
import heapq
import re

# Third-party imports
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
from numpy.typing import NDArray
import pydicom
from scipy.stats import gamma as gamma_dist
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent

# Local imports
from .utils import DICOM_EXTENSIONS, is_valid_dicom, get_tr_from_dicom

# Configure module-level logger
logger = logging.getLogger(__name__)


# ==============================================================================
# INCREMENTAL GLM - Online least squares using Gentleman's algorithm
# ==============================================================================


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
            raise ValueError(
                f"n_nuisance ({n_nuisance}) must be between 1 and n_regressors ({n_regressors})"
            )

        self.n_regressors = n_regressors
        self.n_nuisance = n_nuisance
        self.n_outputs = n_outputs

        self.n_obs = 0

        # R matrix from QR decomposition: upper triangular, augmented with y cross-product
        self._R: NDArray[np.float64] = np.zeros(
            (n_regressors, n_regressors + n_outputs), dtype=np.float64
        )

        # Residual sum of squares and mean squared error for each output
        self._rss: NDArray[np.float64] = np.zeros(n_outputs, dtype=np.float64)
        self._residual_mse: NDArray[np.float64] = np.zeros(n_outputs, dtype=np.float64)

        # Steady-state statistics (frozen after baseline for stable normalization)
        self._steady_state_rss: Optional[NDArray[np.float64]] = None
        self._steady_state_residual_mse: Optional[NDArray[np.float64]] = None
        self._steady_state_timepoint: Optional[int] = None

    def _apply_givens_rotation(
        self,
        j: int,
        aug_row: NDArray[np.float64],
    ) -> None:
        """Apply Givens rotation to incorporate new observation at column j.

        Updates self._R[j, j:] in place and modifies aug_row to zero out position j.

        Parameters
        ----------
        j : int
            Column index for rotation (0 to n_regressors-1).
        aug_row : ndarray, shape (n_regressors + n_outputs,)
            Augmented row [X, y] being incorporated. Modified in place.
        """
        p = self.n_regressors
        r_jj = self._R[j, j]
        x_j = aug_row[j]

        if x_j == 0:
            return

        # Compute rotation parameters
        if abs(r_jj) >= abs(x_j):
            t = x_j / r_jj
            c = 1.0 / np.sqrt(1.0 + t * t)
            s = c * t
        else:
            t = r_jj / x_j
            s = 1.0 / np.sqrt(1.0 + t * t)
            c = s * t

        # Update R[j, j:]
        self._R[j, j] = c * r_jj + s * x_j
        for k in range(j + 1, p + self.n_outputs):
            r_jk = self._R[j, k]
            x_k = aug_row[k]
            self._R[j, k] = c * r_jk + s * x_k
            aug_row[k] = -s * r_jk + c * x_k

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

        return activation

    def update(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> Optional[NDArray[np.float64]]:
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
            raise ValueError(
                f"X has {X.shape[0]} elements, expected {self.n_regressors}"
            )
        if y.shape[0] != self.n_outputs:
            raise ValueError(
                f"y has {y.shape[0]} elements, expected {self.n_outputs}"
            )

        # Increment observation counter
        self.n_obs += 1

        # Apply Givens rotations to update R matrix (Gentleman's algorithm)
        # aug_row will contain the residual in the y positions after all rotations
        aug_row = np.concatenate([X, y])
        for j in range(p):
            self._apply_givens_rotation(j, aug_row)

        # After all rotations, aug_row[p:] contains the residual for this observation
        # Update RSS with squared residuals from the full model
        full_model_residual = aug_row[p:]
        self._rss += full_model_residual ** 2
        
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
            logger.info(
                f"Steady state frozen at observation {self.n_obs}"
            )


# ==============================================================================
# HRF and Task Regressor Computation
# ==============================================================================


def _compute_hrf(tr: float, duration: float = 32.0) -> NDArray[np.float64]:
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
    """
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


def _compute_task_regressor(
    n_baseline: int,
    n_volumes: int,
    tr: float,
) -> NDArray[np.float64]:
    """Compute HRF-convolved boxcar task regressor.

    Creates a task regressor where:
    - Baseline period (0 to n_baseline-1): task = 0
    - Feedback period (n_baseline onwards): task = 1 convolved with HRF

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
    """
    # Create boxcar: 0 during baseline, 1 during feedback
    boxcar = np.zeros(n_volumes, dtype=np.float64)
    boxcar[n_baseline:] = 1.0

    # Compute HRF kernel
    hrf = _compute_hrf(tr)

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


# ==============================================================================
# FeedbackEstimator - Real-time GLM fitting for neurofeedback
# ==============================================================================


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
    use_efficiency_weights : bool, optional
        If True, use inverse-variance weighting for ROI aggregation (computed
        from residual MSE). If False, use simple mean. Default is False.

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
    - Remaining columns: Nuisance regressors (intercept, drift, 6 motion parameters)

    Activation is computed using continuous normalization (Hinds et al., 2011):
    - Error = y - X_nuisance @ beta_nuisance (removing nuisance only)
    - Deviation = sqrt(RSS / t) where RSS is residual sum of squares
    - Activation = error / deviation

    Motion parameters are extracted from dcm2niix JSON sidecar files
    (ImageComments field).
    """

    N_NUISANCE_REGRESSORS = 8  # intercept + drift + 6 motion
    # Pre-compiled regex for motion parameter extraction (avoids re-compilation per volume)
    _MOTION_PATTERN = re.compile(r'Motion:\s*([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)')

    def __init__(
        self,
        mask_paths: Sequence[Path | str],
        n_baseline: int,
        n_volumes: int,
        tr: float,
        task_regressors: Optional[NDArray[np.float64]] = None,
        mask_names: Optional[list[str]] = None,
        use_efficiency_weights: bool = False,
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
        self.use_efficiency_weights = use_efficiency_weights

        # Validate and store task regressors
        if task_regressors is not None:
            # Check if it's a TaskDesign object (has compute_regressors method)
            if hasattr(task_regressors, 'compute_regressors'):
                # It's a TaskDesign - compute regressors from it
                task_regressors = task_regressors.compute_regressors(n_volumes, tr)

            task_regressors = np.asarray(task_regressors, dtype=np.float64)
            if task_regressors.ndim == 1:
                # Single task regressor provided as 1D array
                task_regressors = task_regressors.reshape(-1, 1)
            if task_regressors.shape[0] != n_volumes:
                raise ValueError(
                    f"task_regressors has {task_regressors.shape[0]} rows, "
                    f"expected {n_volumes} to match n_volumes"
                )
            self._task_regressors = task_regressors
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

        # Initialize single GLM for all voxels (shared model across voxels)
        self._glm = IncrementalGLM(
            n_regressors=self._n_regressors,
            n_nuisance=self._n_nuisance,
            n_outputs=self._n_voxels,
        )

        # Motion parameter tracking for differential motion
        self._prev_motion: Optional[NDArray[np.float64]] = None

        # Framewise displacement tracking (computed each volume)
        self._latest_fd: float = 0.0

        # Dict to store efficiency weights per mask: {mask_name: weights_array}
        # Weights are computed from residual MSE for inverse-variance weighting
        self._weights: dict[str, NDArray[np.float64]] = {}

        # Timeseries storage
        self.timeseries: dict[str, list[float]] = {name: [] for name in mask_names}

        logger.info(
            f"FeedbackEstimator initialized with {len(mask_paths)} masks, "
            f"{self._n_voxels} total voxels, "
            f"n_baseline={n_baseline}, n_volumes={n_volumes}, tr={tr:.3f}s, "
            f"task_regressors={self._n_task_regressors}, "
            f"efficiency_weights={'enabled' if use_efficiency_weights else 'disabled'}"
        )

    def _load_single_mask(
        self, path: Path, name: str
    ) -> tuple[NDArray[np.intp], set[tuple[int, int, int]]]:
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
        # type: ignore[assignment] - nibabel load() returns SpatialImage, not Nifti1Image
        img: Nifti1Image = nib.loadsave.load(path)  # type: ignore[assignment]
        data = np.asarray(img.dataobj)

        # Verify compatible shape
        if data.shape[:3] != self._shape:
            raise ValueError(
                f"Mask {name} shape {data.shape[:3]} != reference {self._shape}"
            )

        # Get indices where mask > 0
        indices = np.where(data > 0)
        coords = set(zip(indices[0], indices[1], indices[2]))

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

    def _map_masks_to_master(
        self, mask_indices: dict[str, NDArray[np.intp]]
    ) -> dict[str, NDArray[np.intp]]:
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
            master_idx = np.array(
                [flat_to_master[f] for f in flat_indices], dtype=np.intp
            )
            mask_voxel_idx[name] = master_idx

        return mask_voxel_idx

    def _load_masks(self, mask_paths: list[Path]) -> None:
        """Load masks and build voxel-to-mask index mapping.

        This orchestrates loading all masks, building a master voxel index,
        and creating mappings from each mask to the master index.
        """
        # Load first mask to get reference shape for validation
        # type: ignore[assignment] - nibabel load() returns SpatialImage, not Nifti1Image
        first_img: Nifti1Image = nib.loadsave.load(mask_paths[0])  # type: ignore[assignment]
        self._shape = first_img.shape[:3]

        # Load all masks and collect coordinates
        mask_indices: dict[str, NDArray[np.intp]] = {}
        all_coords: set[tuple[int, int, int]] = set()

        for path, name in zip(mask_paths, self.mask_names):
            flat_indices, coords = self._load_single_mask(path, name)
            mask_indices[name] = flat_indices
            all_coords.update(coords)

        # Build master voxel index
        self._n_voxels, self._voxel_indices = self._build_master_voxel_index(all_coords)

        # Map masks to master index
        self._mask_voxel_idx = self._map_masks_to_master(mask_indices)

    def _extract_motion_params_from_json(
        self, json_path: Path
    ) -> NDArray[np.float64]:
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
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            image_comments = metadata.get('ImageComments', '')

            # Check if this is the reference volume (all zeros)
            if 'Reference volume for motion correction' in image_comments:
                logger.debug(f"Reference volume detected in {json_path}")
                return np.zeros(6, dtype=np.float64)

            # Use pre-compiled pattern for motion extraction
            match = self._MOTION_PATTERN.search(image_comments)

            if match:
                values = [float(m) for m in match.groups()]
                return np.array(values, dtype=np.float64)

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
        if self._prev_motion is None:
            # First volume - no previous frame to compare
            return 0.0

        # Compute frame-to-frame difference
        diff = motion_params - self._prev_motion

        # Format: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        # Translations (indices 0-2) are already in mm
        # Rotations (indices 3-5) are in degrees - convert to mm arc length
        # Arc length = radius * angle_in_radians = 50 * (angle_degrees * pi/180)
        diff_mm = diff.copy()
        diff_mm[3:6] *= 50.0 * np.pi / 180.0

        # FD is sum of absolute values
        fd = float(np.sum(np.abs(diff_mm)))
        return fd

    def _build_design_row(
        self, motion_params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Build design matrix row for current volume.

        Parameters
        ----------
        motion_params : ndarray, shape (6,)
            Raw motion parameters for current volume (absolute values).

        Returns
        -------
        X : ndarray, shape (n_regressors,)
            Design row: [task_cols..., intercept, drift, motion_diff...]
        """
        # Use frame-to-frame motion
        if self.n_volumes == 0:
            motion_diff = np.zeros(6, dtype=np.float64)
        else:
            prev = self._prev_motion if self._prev_motion is not None else np.zeros(6, dtype=np.float64)
            motion_diff = motion_params - prev

        self._prev_motion = motion_params.copy()

        # Build design row
        X = np.zeros(self._n_regressors, dtype=np.float64)

        # Current volume index (0-based for array access)
        vol_idx = self.n_volumes  # This is the volume we're about to process

        # Fill in task regressors (if any)
        if self._task_regressors is not None:
            if vol_idx < len(self._task_regressors):
                X[:self._n_task_regressors] = self._task_regressors[vol_idx, :]
            else:
                # Extrapolate with last row if we exceed design length
                X[:self._n_task_regressors] = self._task_regressors[-1, :]

        # Fill in nuisance regressors (after task columns)
        nuisance_idx = self._nuisance_start_idx
        X[nuisance_idx] = 1.0  # intercept
        X[nuisance_idx + 1] = float(vol_idx + 1)  # linear drift (1-indexed volume number)
        X[nuisance_idx + 2:nuisance_idx + 8] = motion_diff  # differential motion

        return X

    def _compute_weights(self) -> None:
        """Compute voxel efficiency weights from residual MSE.

        Uses inverse-variance weighting based on the residual MSE from the GLM.
        """
        if not self.use_efficiency_weights:
            return

        # Get residual MSE from GLM
        residual_mse = self._glm.get_residual_mse()

        # Compute weights for each mask
        for name, idx in self._mask_voxel_idx.items():
            mask_mse = residual_mse[idx]

            # Inverse variance (MSE) weighting (handle zero MSE)
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / mask_mse
                weights[~np.isfinite(weights)] = 0.0

            # Normalize to sum to 1
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                # Fallback to uniform weights
                weights = np.ones_like(weights) / len(weights)
                logger.warning(f"Mask '{name}': using uniform weights (zero MSE)")

            self._weights[name] = weights

        logger.info("Voxel efficiency weights computed from residual MSE")

    def process_volume(
        self,
        nifti_path: Path | str,
        motion_source: Path | str | NDArray[np.float64] | None = None,
    ) -> dict[str, float]:
        """Process a new fMRI volume and return activation estimates.

        Parameters
        ----------
        nifti_path : Path or str
            Path to NIfTI volume file.
        motion_source : Path, str, ndarray, or None, optional
            Source of motion parameters. Can be:
            - Path to JSON file (dcm2niix sidecar): extracts motion from ImageComments field
            - ndarray of shape (6,): use provided motion parameters directly
            - None: use zeros for motion parameters

        Returns
        -------
        activations : dict of str to float
            Weighted mean activation (z-score after baseline) for each mask.
            During baseline period, returns weighted mean residuals.
        """
        nifti_path = Path(nifti_path)

        # ===== 1. Load volume data =====
        # type: ignore[assignment] - nibabel load() returns SpatialImage, not Nifti1Image
        img: Nifti1Image = nib.loadsave.load(nifti_path)  # type: ignore[assignment]
        data = np.asarray(img.dataobj, dtype=np.float64)

        # Verify shape matches masks
        if data.shape[:3] != self._shape:
            raise ValueError(
                f"Volume shape {data.shape[:3]} != mask shape {self._shape}"
            )

        # Extract voxel values using direct fancy indexing (faster than reshape + ravel)
        # self._voxel_indices is a tuple of (i, j, k) arrays
        voxel_values = data[self._voxel_indices]

        # ===== 2. Extract motion parameters =====
        if motion_source is None:
            motion_params = np.zeros(6, dtype=np.float64)
        elif isinstance(motion_source, (np.ndarray, list)):
            motion_params = np.array(motion_source, dtype=np.float64)
            if motion_params.shape != (6,):
                raise ValueError(f"Motion parameters must have shape (6,), got {motion_params.shape}")
        else:
            motion_path = Path(motion_source)
            # Treat as JSON file path
            motion_params = self._extract_motion_params_from_json(motion_path)

        # ===== 3. Compute FD and build design matrix =====
        # FD must be computed before _build_design_row (which updates _prev_motion)
        self._latest_fd = self._compute_fd(motion_params)

        X = self._build_design_row(motion_params)
        self._last_design_row = X
        result = self._glm.update(X, voxel_values)
        self.n_volumes += 1

        # Handle case where GLM can't compute activation yet (not enough observations)
        if result is None:
            activations = {name: 0.0 for name in self.mask_names}
            for name in self.mask_names:
                self.timeseries[name].append(0.0)
            return activations

        # Compute efficiency weights when baseline completes
        if self.n_volumes == self.n_baseline:
            # Freeze RSS, MSE, and timepoint for stable normalization
            self._glm.freeze_steady_state()

            # Compute weights using frozen residual MSE
            if self.use_efficiency_weights:
                self._compute_weights()

        # ===== 4. Compute weighted activations per mask =====
        activations: dict[str, float] = {}
        for name, idx in self._mask_voxel_idx.items():
            mask_values = result[idx]

            if self.use_efficiency_weights and name in self._weights:
                # Use efficiency weights if enabled and computed
                weighted_mean = float(np.sum(self._weights[name] * mask_values))
            else:
                # Use simple mean
                weighted_mean = float(np.mean(mask_values))

            activations[name] = weighted_mean
            self.timeseries[name].append(weighted_mean)

        return activations


class _DicomEventHandler(FileSystemEventHandler):
    """Watchdog event handler for DICOM file detection."""

    def __init__(self, callback: Callable[[Path], None]) -> None:
        super().__init__()
        self._callback = callback

    def on_created(self, event: FileCreatedEvent | DirCreatedEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            self._callback(Path(src_path.decode()))
        elif isinstance(src_path, str):
            self._callback(Path(src_path))


# ==============================================================================
# Interrupted Run Handling - Detection, Archiving, and Run Number Management
# ==============================================================================


def _detect_interrupted_run(
    output_dir: Path,
    subject: str,
    task: str,
    run: int,
    n_volumes: int,
    acq: str = None,
) -> Optional[dict]:
    """Check if a previous interrupted run exists for this task/run combination.

    A run is considered interrupted if:
    - feedbacklog file exists with > 0 but < n_volumes data rows

    Parameters
    ----------
    output_dir : Path
        Base output directory for feedback sessions
    subject : str
        Subject identifier (e.g., 'sub-01')
    task : str
        Task name (e.g., 'feedback', 'transfer')
    run : int
        Run number to check
    n_volumes : int
        Expected number of volumes for a complete run
    acq : str, optional
        Acquisition letter (e.g., 'a', 'b'). If provided, uses new acq-based naming.
        If None, uses legacy run-only naming for backward compatibility.

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

    # Use acq-based naming if acq provided, otherwise legacy naming
    if acq is not None:
        # Format acq as 2-digit number if int
        acq_str = f"{acq:02d}" if isinstance(acq, int) else acq
        feedbacklog_path = output_dir / f"{subject}_task-{task}_acq-{acq_str}_run-{run_str}_feedbacklog.tsv"
        design_path = output_dir / f"{subject}_task-{task}_acq-{acq_str}_run-{run_str}_design.tsv"
        nifti_dir = output_dir / f"nifti_task-{task}_acq-{acq_str}_run-{run_str}"
    else:
        feedbacklog_path = output_dir / f"{subject}_task-{task}_run-{run_str}_feedbacklog.tsv"
        design_path = output_dir / f"{subject}_task-{task}_run-{run_str}_design.tsv"
        nifti_dir = output_dir / f"nifti_task-{task}_run-{run_str}"

    if not feedbacklog_path.exists():
        return None

    # Count data rows (excluding header)
    try:
        with open(feedbacklog_path, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header
    except Exception:
        return None

    # Complete if row count matches expected volumes
    if row_count >= n_volumes:
        return None

    # Interrupted if has some data but not all
    if row_count > 0:
        return {
            'feedbacklog_path': feedbacklog_path,
            'design_path': design_path,
            'nifti_dir': nifti_dir,
            'volumes_completed': row_count,
        }

    return None


def _archive_interrupted_run(
    output_dir: Path,
    subject: str,
    task: str,
    run: int,
    acq: str = None,
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
    acq : str, optional
        Acquisition letter (e.g., 'a', 'b'). If provided, uses new acq-based naming.
        If None, uses legacy run-only naming for backward compatibility.

    Returns
    -------
    Path
        Path to the created archive directory
    """
    run_str = f"{run:02d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = output_dir / f"interrupted_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Files/directories to archive - use acq-based naming if acq provided
    if acq is not None:
        # Format acq as 2-digit number if int
        acq_str = f"{acq:02d}" if isinstance(acq, int) else acq
        items_to_move = [
            output_dir / f"{subject}_task-{task}_acq-{acq_str}_run-{run_str}_feedbacklog.tsv",
            output_dir / f"{subject}_task-{task}_acq-{acq_str}_run-{run_str}_design.tsv",
            output_dir / f"nifti_task-{task}_acq-{acq_str}_run-{run_str}",
        ]
    else:
        items_to_move = [
            output_dir / f"{subject}_task-{task}_run-{run_str}_feedbacklog.tsv",
            output_dir / f"{subject}_task-{task}_run-{run_str}_design.tsv",
            output_dir / f"nifti_task-{task}_run-{run_str}",
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
    task: str,
    run: int,
    acq: str = None,
) -> Optional[Path]:
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
    acq : str, optional
        Acquisition letter (e.g., 'a', 'b'). If provided, uses new acq-based naming.

    Returns
    -------
    Path or None
        Path to generated QC report, or None if generation failed
    """
    try:
        from pyrtnf import qc

        qc_output_dir = output_dir / "qc"
        qc_output_dir.mkdir(parents=True, exist_ok=True)

        report_path = qc.generate_feedback_report(
            feedback_dir=output_dir,
            subject=subject,
            task=task,
            run=run,
            acq=acq,
            output_dir=qc_output_dir,
            open_browser=False,
        )
        logger.info(f"Generated partial QC report: {report_path}")
        return report_path
    except Exception as e:
        logger.warning(f"Failed to generate partial QC report: {e}")
        return None


def get_next_run_number(
    output_dir: Path,
    subject: str,
    task: str,
) -> int:
    """Scan output directory for existing runs and return next available number.

    Looks for feedbacklog files matching the pattern:
    {subject}_task-{task}_run-{NN}_feedbacklog.tsv

    Returns the next sequential run number after any existing runs.
    Also considers archived interrupted runs.

    Parameters
    ----------
    output_dir : Path
        Base output directory for feedback sessions
    subject : str
        Subject identifier (e.g., 'sub-01')
    task : str
        Task name (e.g., 'feedback', 'transfer')

    Returns
    -------
    int
        Next available run number (1 if no existing runs found)
    """
    if not output_dir.exists():
        return 1

    pattern = f"{subject}_task-{task}_run-*_feedbacklog.tsv"
    existing_runs: set[int] = set()

    # Check main output directory
    for f in output_dir.glob(pattern):
        match = re.search(r'run-(\d+)_feedbacklog', f.name)
        if match:
            existing_runs.add(int(match.group(1)))

    # Also check archived directories for runs
    for archive_dir in output_dir.glob("interrupted_*"):
        if archive_dir.is_dir():
            for f in archive_dir.glob(pattern):
                match = re.search(r'run-(\d+)_feedbacklog', f.name)
                if match:
                    existing_runs.add(int(match.group(1)))

    if not existing_runs:
        return 1

    return max(existing_runs) + 1


# Mapping from workflow step_id to expected run number
# transfer_1 and feedback_1 are slot 1 (run-01)
# transfer_2 and feedback_2 are slot 2 (run-02)
STEP_TO_RUN_NUMBER = {
    "transfer_1": 1,
    "transfer_2": 2,
    "feedback_1": 1,
    "feedback_2": 2,
}


def get_slot_run_info(
    output_dir: Path,
    subject: str,
    task: str,
    step_id: str,
) -> dict:
    """Get run info based on workflow slot with BIDS acq entity for retries.

    Uses a slot-based run numbering system where:
    - run-01/run-02 corresponds to workflow slot (transfer_1/feedback_1 vs transfer_2/feedback_2)
    - acq-a/acq-b/acq-c corresponds to attempt number (first, second, third...)

    Parameters
    ----------
    output_dir : Path
        Base output directory for feedback sessions
    subject : str
        Subject identifier (e.g., 'sub-01')
    task : str
        Task name (e.g., 'feedback', 'transfer')
    step_id : str
        Workflow step identifier (e.g., 'transfer_1', 'feedback_2')

    Returns
    -------
    dict
        Dictionary containing:
        - 'run': int - The fixed run number for this slot (1 or 2)
        - 'acq': int - The acquisition number for this attempt (1, 2, 3, ...)
        - 'run_acq_string': str - Combined string like 'acq-01_run-01'
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
    if step_id not in STEP_TO_RUN_NUMBER:
        raise ValueError(f"Unknown step_id: {step_id}. Expected one of {list(STEP_TO_RUN_NUMBER.keys())}")

    run_num = STEP_TO_RUN_NUMBER[step_id]

    if not output_dir.exists():
        return {
            "run": run_num,
            "acq": 1,
            "run_acq_string": f"acq-01_run-{run_num:02d}",
            "existing_acqs": [],
        }

    # Scan for existing attempts at this run number
    # Pattern: {subject}_task-{task}_acq-*_run-{run:02d}_feedbacklog.tsv
    pattern = f"{subject}_task-{task}_acq-*_run-{run_num:02d}_feedbacklog.tsv"

    existing_acqs: list[int] = []

    # Check main output directory
    for f in output_dir.glob(pattern):
        match = re.search(r'acq-(\d+)_run-', f.name)
        if match:
            existing_acqs.append(int(match.group(1)))

    # Also check archived directories for runs
    for archive_dir in output_dir.glob("interrupted_*"):
        if archive_dir.is_dir():
            for f in archive_dir.glob(pattern):
                match = re.search(r'acq-(\d+)_run-', f.name)
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


# ==============================================================================
# NeurofeedbackCoordinator - Watch for DICOMs and coordinate processing
# ==============================================================================


class NeurofeedbackCoordinator:
    """Coordinates real-time neurofeedback by watching for DICOMs.

    Monitors a directory for incoming DICOM files, converts them to NIfTI,
    processes through FeedbackEstimator, and logs results to CSV.

    **Timing Model:**

    Volumes are processed strictly in order (by filename). If volume N+1 arrives
    before volume N, it waits in a priority queue until volume N is processed.
    This ensures deterministic output regardless of filesystem event ordering.

    Processing time per volume is logged for diagnostics. If processing
    consistently takes longer than TR, a warning is issued (queue depth > 0).

    Parameters
    ----------
    watch_dir : Path or str
        Directory to watch for incoming DICOM files.
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
    use_efficiency_weights : bool, optional
        If True, use inverse-variance weighting for ROI aggregation (computed
        from residual MSE). If False, use simple mean. Default is False.

    Attributes
    ----------
    n_processed : int
        Number of volumes processed so far.

    Notes
    -----
    Uses watchdog for efficient filesystem event detection. Each new DICOM
    triggers conversion via dcm2niix and processing through FeedbackEstimator.
    Results are logged to CSV after each volume.

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
        task: str,
        run: int,
        n_baseline: int,
        n_volumes: int,
        tr: float,
        task_regressors: Optional[NDArray[np.float64]] = None,
        mask_names: Optional[list[str]] = None,
        use_efficiency_weights: bool = False,
        activation_callback: Optional[Callable[[dict[str, float], int, float, bool, bool], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        simulator=None,  # DicomSimulator instance for simulation mode
        qc_enabled: bool = True,  # Generate QC report after session completes
        acq: str = None,  # Acquisition letter for BIDS-compatible naming (e.g., 'a', 'b')
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.subject = subject
        self.task = task
        self.run_number = run
        self.acq = acq  # Acquisition letter for retry tracking
        self.n_baseline = n_baseline
        self.n_volumes = n_volumes
        self._tr = tr
        self._qc_enabled = qc_enabled

        # Validate watch directory
        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Watch directory not found: {self.watch_dir}")
        if not self.watch_dir.is_dir():
            raise ValueError(f"Watch path is not a directory: {self.watch_dir}")

        # Create output directories - use acq-based naming if acq provided
        run_str = f"{self.run_number:02d}"
        if self.acq is not None:
            # acq can be int or str, format as 2-digit number if int
            acq_str = f"{self.acq:02d}" if isinstance(self.acq, int) else self.acq
            self._run_acq_str = f"acq-{acq_str}_run-{run_str}"
            self.nifti_dir = self.output_dir / f"nifti_task-{self.task}_{self._run_acq_str}"
        else:
            self._run_acq_str = f"run-{run_str}"
            self.nifti_dir = self.output_dir / f"nifti_task-{self.task}_run-{run_str}"
        self.nifti_dir.mkdir(parents=True, exist_ok=True)
        # Multi-run handling: Current implementation processes a single continuous
        # acquisition. For multi-run sessions, run separate NeurofeedbackCoordinator
        # instances or enhance to detect run boundaries via metadata

        # Set up mask names
        mask_paths_resolved = [Path(p) for p in mask_paths]
        if mask_names is None:
            mask_names = [p.stem for p in mask_paths_resolved]
        self.mask_names = mask_names

        # Initialize FeedbackEstimator
        self._estimator = FeedbackEstimator(
            mask_paths=list(mask_paths_resolved),
            n_baseline=n_baseline,
            n_volumes=n_volumes,
            tr=tr,
            task_regressors=task_regressors,
            mask_names=mask_names,
            use_efficiency_weights=use_efficiency_weights,
        )

        # Processing state
        self.n_processed = 0
        self._processed_files: set[str] = set()
        self._pending_files: list[Path] = []
        self._pending_heap: list[tuple[str, Path]] = []  # Priority queue: (filename, path)
        self._next_expected_filename: Optional[str] = None  # Filename we're waiting for
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # Signal when coordinator is ready
        self._trigger_event = threading.Event()  # Signal when trigger received (starts acquisition)
        self._error: Optional[Exception] = None  # Store any error that occurs
        self._start_time: Optional[float] = None
        self.design_matrix: list[NDArray[np.float64]] = []
        
        # Timing diagnostics
        self._timing_stats: dict[str, list[float]] = {
            'dcm2niix_ms': [],
            'glm_ms': [],
            'total_ms': [],
            'queue_depth': [],
        }

        # Simulator integration
        self._simulator = simulator
        self._simulator_thread: Optional[threading.Thread] = None

        # DICOM collection directory (for unified collection with rest/2vol)
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
        self._csv_path = self.output_dir / f"{self.subject}_task-{self.task}_{self._run_acq_str}_feedbacklog.tsv"
        self._design_matrix_path = self.output_dir / f"{self.subject}_task-{self.task}_{self._run_acq_str}_design.tsv"
        self._design_file: Optional[TextIO] = None  # File handle for incremental design matrix writes

        logger.info(
            f"NeurofeedbackCoordinator initialized: "
            f"watching {self.watch_dir}, {n_volumes} volumes expected"
        )

    def _init_csv(self) -> None:
        """Initialize CSV log file and design matrix file with headers."""
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file, delimiter='\t')
        header = ["volume", "time", "is_baseline", "is_outlier", "fd"] + self.mask_names
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

    def _save_design_matrix(self) -> None:
        """Save design matrix to TSV file.

        Saves the design matrix with column names:
        [task, intercept, drift, motion_x, motion_y, motion_z, motion_rx, motion_ry, motion_rz]
        """
        if not self.design_matrix:
            logger.warning("No design matrix data to save")
            return

        column_names = ['task', 'intercept', 'drift',
                       'motion_x', 'motion_y', 'motion_z',
                       'motion_rx', 'motion_ry', 'motion_rz']

        # Convert design matrix list to numpy array
        design_array = np.array(self.design_matrix)

        # Save to TSV file
        with open(self._design_matrix_path, 'w') as f:
            # Write header
            f.write('\t'.join(column_names) + '\n')

            # Write data rows
            for row in design_array:
                row_str = '\t'.join(f'{val:.6f}' for val in row)
                f.write(row_str + '\n')

    def _init_design_matrix(self) -> None:
        """Initialize design matrix TSV file with header for incremental writes."""
        self._design_file = open(self._design_matrix_path, "w", newline="")

        # Build column names based on actual number of task regressors
        column_names = []

        # Task regressor columns (if any)
        n_task = self._estimator._n_task_regressors if self._estimator else 0
        if n_task == 1:
            column_names.append('task')
        elif n_task > 1:
            column_names.extend([f'task_{i}' for i in range(n_task)])

        # Nuisance regressor columns
        column_names.extend([
            'intercept', 'drift',
            'motion_x', 'motion_y', 'motion_z',
            'motion_rx', 'motion_ry', 'motion_rz'
        ])

        self._design_file.write('\t'.join(column_names) + '\n')
        self._design_file.flush()

    def _append_design_row(self, row: NDArray[np.float64]) -> None:
        """Append a single design matrix row to TSV file.

        Parameters
        ----------
        row : NDArray
            Design matrix row to append (9 values: task, intercept, drift, 6 motion params)
        """
        if self._design_file is not None:
            row_str = '\t'.join(f'{val:.6f}' for val in row)
            self._design_file.write(row_str + '\n')
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

        # Try to get from simulator
        if self._simulator is not None:
            self._tr = self._simulator.tr
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
        Check if the feedback session is complete or stopped.

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
        return (
            self.n_processed >= self.n_volumes
            or self._stop_event.is_set()
            or self._error is not None
        )

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
        
        Adds file to priority queue (heap) sorted by filename.
        """
        with self._lock:
            if file_path.name not in self._processed_files:
                # Add to heap (priority queue) sorted by filename
                heapq.heappush(self._pending_heap, (file_path.name, file_path))

    def _scan_for_existing_files(self) -> None:
        """Scan watch directory for any DICOM files not yet processed.

        This handles files that may have been missed by watchdog events,
        particularly when the simulator runs faster than processing.
        """
        # Glob files OUTSIDE the lock to avoid blocking other threads during I/O
        found_files: list[Path] = []
        for ext in DICOM_EXTENSIONS:
            found_files.extend(self.watch_dir.glob(ext))

        # Only hold lock for the quick heap operations
        with self._lock:
            existing_names = {item[0] for item in self._pending_heap}
            for file_path in found_files:
                if file_path.name not in self._processed_files:
                    if file_path.name not in existing_names:
                        heapq.heappush(self._pending_heap, (file_path.name, file_path))
                        existing_names.add(file_path.name)
    
    def _get_next_file_to_process(self) -> Optional[Path]:
        """Get the next file to process in strict order, skipping invalid files."""
        with self._lock:
            while self._pending_heap:
                filename, file_path = self._pending_heap[0]

                if filename in self._processed_files:
                    heapq.heappop(self._pending_heap)
                    continue

                if is_valid_dicom(file_path):
                    heapq.heappop(self._pending_heap)
                    self._processed_files.add(filename)
                    return file_path
                else:
                    heapq.heappop(self._pending_heap)
                    logger.debug(f"Skipping invalid DICOM: {filename}")

            return None
    
    def _get_queue_depth(self) -> int:
        """Get current queue depth (files waiting to be processed)."""
        with self._lock:
            return len(self._pending_heap)

    def _process_volume(self, dicom_path: Path) -> None:
        """Convert DICOM and process through estimator, tracking timing."""
        import shutil
        from .utils import dicom_to_nifti

        process_start = time.time()

        # Extract TR from first DICOM if not yet set
        if self._tr is None and self.n_processed == 0:
            self._tr = get_tr_from_dicom(dicom_path)
            logger.info(f"Extracted TR from DICOM: {self._tr}s")

        # Copy DICOM to collection directory (for unified collection with rest/2vol)
        try:
            dest_path = self._dicom_collection_dir / dicom_path.name
            shutil.copy2(dicom_path, dest_path)
        except Exception as e:
            logger.warning(f"Failed to copy DICOM to collection: {e}")

        # Convert DICOM to NIfTI (timed for diagnostics)
        volume_num = self.n_processed + 1
        output_name = f"vol_{volume_num:04d}"
        
        dcm2niix_start = time.time()
        try:
            nifti_path = dicom_to_nifti(
                dicom_dir=dicom_path,
                output_dir=self.nifti_dir,
                output_name=output_name,
                overwrite=True,
                verbose=False,
                compress=False,  # Uncompressed for faster real-time processing
            )
        except Exception as e:
            logger.error(f"Failed to convert DICOM {dicom_path}: {e}")
            return
        dcm2niix_duration = (time.time() - dcm2niix_start) * 1000  # ms

        json_path = Path(str(nifti_path).replace('.nii.gz', '.json').replace('.nii', '.json'))

        # Process through estimator (timed for diagnostics)
        glm_start = time.time()
        try:
            activations = self._estimator.process_volume(nifti_path, json_path)
            self.design_matrix.append(self._estimator._last_design_row)
            # Write design row incrementally (so it's not lost on interrupt)
            self._append_design_row(self._estimator._last_design_row)
        except Exception as e:
            logger.error(f"Failed to process volume {volume_num}: {e}")
            return
        glm_duration = (time.time() - glm_start) * 1000  # ms
        
        total_duration = (time.time() - process_start) * 1000  # ms
        queue_depth = self._get_queue_depth()
        
        self._timing_stats['dcm2niix_ms'].append(dcm2niix_duration)
        self._timing_stats['glm_ms'].append(glm_duration)
        self._timing_stats['total_ms'].append(total_duration)
        self._timing_stats['queue_depth'].append(queue_depth)

        self.n_processed += 1
        elapsed_time = time.time() - self._start_time if self._start_time else 0.0
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
                self._activation_callback(activations, self.n_processed, elapsed_time, is_baseline, False)
            except Exception as e:
                logger.error(f"Error in activation callback: {e}")
                if self._error_callback is not None:
                    try:
                        self._error_callback(e)
                    except:
                        pass

        # Read outlier status (may have been set by callback via report_outlier())
        with self._lock:
            is_outlier = self._latest_is_outlier

        # Get FD from estimator (computed during process_volume)
        fd = self._estimator._latest_fd

        row = [self.n_processed, f"{elapsed_time:.2f}", is_baseline, is_outlier, f"{fd:.4f}"]
        row.extend([f"{activations[name]:.4f}" for name in self.mask_names])
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
        if self._csv_file is not None:
            self._csv_file.flush()

        # Log concise timing: proc = dcm2niix+GLM for this volume; drift = (elapsed - expected)
        status = "BASELINE" if is_baseline else "ACTIVE"
        activation_str = "  ".join(
            f"{name}={activations[name]:+.3f}" for name in self.mask_names
        )

        queue_warning = f" [QUEUE:{queue_depth}]" if queue_depth > 0 else ""
        timing_drift_ms = timing_drift * 1000  # Convert to ms for consistency
        drift_str = f"+{timing_drift_ms:.0f}" if timing_drift_ms >= 0 else f"{timing_drift_ms:.0f}"
        outlier_indicator = " OUTLIER" if is_outlier else ""

        logger.info(
            f"[{elapsed_time:7.2f}s] Vol {self.n_processed:4d}/{self.n_volumes} "
            f"({status}): {activation_str} "
            f"[proc:{total_duration:.0f}ms drift:{drift_str}ms{outlier_indicator}]{queue_warning}"
        )

    def _run_simulator(self) -> None:
        """Internal method to run simulator in background thread."""
        try:
            logger.info("Starting simulator thread")
            if self._simulator is None:
                logger.error("Simulator is not configured (self._simulator is None)")
                return
            for vol_num in self._simulator.copy_volumes_iter(n_volumes=self.n_volumes):
                if self._stop_event.is_set():
                    logger.info("Simulator stopped by stop event")
                    break
            logger.info("Simulator thread complete")
        except Exception as e:
            logger.exception(f"Simulator thread error: {e}")
            self._error = e
            self._stop_event.set()

    def run(self) -> None:
        """Run the coordinator until n_volumes processed or interrupted.

        Blocks until all volumes are processed or KeyboardInterrupt.
        """
        # Initialize CSV
        self._init_csv()

        if self._simulator is not None and self._simulator.clear_target:
            self._simulator._clear_target_directory()
            self._simulator.clear_target = False

        # Set up watchdog observer
        event_handler = _DicomEventHandler(self._on_new_file)
        observer = Observer()
        observer.schedule(event_handler, str(self.watch_dir), recursive=False)
        observer.start()

        logger.info(f"Watching {self.watch_dir} for DICOMs...")

        self._ready_event.set()
        logger.debug("Coordinator ready signal set")

        logger.info("Waiting for trigger...")
        self._trigger_event.wait()  # Block until trigger received or stop() sets it

        if self._stop_event.is_set():
            logger.info("Stop requested before trigger; exiting coordinator")
        else:
            logger.info("Trigger received - starting acquisition")

        self._start_time = time.time()

        if self._simulator is not None and not self._stop_event.is_set():
            self._simulator.start_time = self._start_time
            
            self._simulator_thread = threading.Thread(
                target=self._run_simulator,
                daemon=False,
                name="SimulatorThread"
            )
            self._simulator_thread.start()
            logger.info("Simulator thread started")

        try:
            while (
                not self._stop_event.is_set()
                and self.n_processed < self.n_volumes
                and not self._stop_event.is_set()
            ):
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
            # Stop simulator thread if running
            if self._simulator_thread is not None and self._simulator_thread.is_alive():
                logger.info("Stopping simulator thread")
                self._stop_event.set()
                self._simulator_thread.join(timeout=5)
                if self._simulator_thread.is_alive():
                    logger.warning("Simulator thread did not stop gracefully")

            observer.stop()
            observer.join()
            self._close_csv()
            self._close_design_matrix()  # Design matrix was written incrementally

            elapsed = time.time() - self._start_time if self._start_time else 0.0
            logger.info(
                f"Session complete: {self.n_processed} volumes in {elapsed:.1f}s"
            )
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
        if not self._timing_stats['total_ms']:
            return
        
        total_ms = np.array(self._timing_stats['total_ms'])
        dcm2niix_ms = np.array(self._timing_stats['dcm2niix_ms'])
        glm_ms = np.array(self._timing_stats['glm_ms'])
        queue_depth = np.array(self._timing_stats['queue_depth'])
        
        tr_ms = (self._tr or 1.2) * 1000  # TR in ms
        
        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"TR: {self._tr:.3f}s ({tr_ms:.0f}ms)")
        logger.info(f"Volumes processed: {len(total_ms)}")
        logger.info("-" * 40)
        logger.info(f"Total processing:  mean={total_ms.mean():.1f}ms, "
                   f"std={total_ms.std():.1f}ms, max={total_ms.max():.1f}ms")
        logger.info(f"  dcm2niix:        mean={dcm2niix_ms.mean():.1f}ms, "
                   f"std={dcm2niix_ms.std():.1f}ms, max={dcm2niix_ms.max():.1f}ms")
        logger.info(f"  GLM update:      mean={glm_ms.mean():.1f}ms, "
                   f"std={glm_ms.std():.1f}ms, max={glm_ms.max():.1f}ms")
        logger.info("-" * 40)
        
        # Check if processing typically exceeded TR
        pct_exceeds_tr = (total_ms > tr_ms).mean() * 100
        max_queue = queue_depth.max()
        
        if pct_exceeds_tr > 10:
            logger.warning(f"Processing exceeded TR in {pct_exceeds_tr:.1f}% of volumes!")
            logger.warning(f"Max queue depth: {max_queue} (volumes waiting)")
        else:
            logger.info(f"Processing within TR budget: {100-pct_exceeds_tr:.1f}% of volumes")
            logger.info(f"Max queue depth: {max_queue}")
        
        logger.info("=" * 60)

    def _generate_qc_report(self) -> None:
        """Generate QC report after session completes."""
        try:
            from . import qc

            logger.info("Generating feedback QC report...")
            report_path = qc.generate_feedback_report(
                feedback_dir=self.output_dir,
                subject=self.subject,
                task=self.task,
                run=self.run_number,
                acq=self.acq,
                output_dir=self.output_dir / 'qc',
                overwrite=True,
                open_browser=False
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
        NIfTI directories after a feedback session completes. This is useful
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

    def should_show_questions(self) -> bool:
        """Check if post-run questions should be shown.

        Returns True if either:
        - The session completed naturally (not stopped)
        - The session was stopped but show_questions was requested
        """
        if not self.was_stopped():
            return True
        return getattr(self, '_show_questions_on_stop', False)

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
                'activations': self._latest_activations.copy(),
                'volume': self._latest_volume,
                'time': self._latest_time,
                'is_baseline': self._latest_is_baseline,
            }


# ==============================================================================
# HIGH-LEVEL SESSION ORCHESTRATION - Run complete feedback sessions
# ==============================================================================


def run_feedback_session(
    mask_paths: list[Path | str],
    subject: str,
    task: str,
    run: int,
    output_dir: Path | str,
    collection_dir: Path | str,
    n_baseline: int,
    n_volumes: int,
    tr: float,
    task_regressors: Optional[NDArray[np.float64]] = None,
    incoming_dir: Optional[Path | str] = None,
    mask_names: Optional[list[str]] = None,
    use_efficiency_weights: bool = False,
    simulator_source_dir: Optional[Path | str] = None,
    simulator_speed: float = 1.0,
    simulator_clear_target: bool = True,
    simulator_cleanup: bool = True,
    activation_callback: Optional[Callable[[dict[str, float], int, float, bool, bool], None]] = None,
    error_callback: Optional[Callable[[Exception], None]] = None,
    qc_enabled: bool = True,
    step_id: Optional[str] = None,
) -> dict:
    """Run a complete feedback session with optional DICOM simulation.

    This is the main entry point for running neurofeedback sessions. It handles
    both simulated (offline) and real-time (scanner) acquisition modes.

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
        Directory for collecting raw DICOMs (consistent with rest/2vol collection).
        DICOMs are copied to collection_dir/{task}_run-{XX}/ and incoming_dir
        defaults to collection_dir/incoming. This provides a unified collection
        structure for all scan types.
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
        Directory to watch for incoming DICOMs. If None, uses collection_dir/incoming.
        For scanner acquisition, this should be the directory where the scanner
        writes DICOMs.
    mask_names : list of str, optional
        Names for each mask. If not provided, uses mask filenames.
    use_efficiency_weights : bool, optional
        If True, use inverse-variance weighting for ROI aggregation (computed
        from residual MSE). If False, use simple mean. Default is False.
    simulator_source_dir : Path or str, optional
        If provided, runs in simulation mode. Directory containing source DICOM
        files to simulate scanner acquisition. If None, runs in real-time mode
        and watches incoming_dir for scanner-generated DICOMs.
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

    >>> results = run_feedback_session(
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

    >>> results = run_feedback_session(
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
    from .simulator import DicomSimulator
    import threading

    # Convert paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine run number and acquisition letter based on step_id or run parameter
    if step_id is not None:
        # Use slot-based run numbering with acq entity for retries
        run_info = get_slot_run_info(output_dir, subject, task, step_id)
        run = run_info["run"]  # Override run parameter with slot-determined value
        acq = run_info["acq"]
        logger.info(f"Using slot-based naming for {step_id}: run-{run:02d}, acq-{acq}")
    else:
        # Legacy mode: no acq entity, just run number
        acq = None

    # Check for and archive any interrupted run before starting fresh
    # Note: with acq-based naming, we don't archive the *same* acq - the new acq letter
    # means this is a fresh attempt. But we should archive if there's an interrupted run
    # at the *same* acq (shouldn't normally happen, but handle edge cases).
    interrupted = _detect_interrupted_run(
        output_dir=output_dir,
        subject=subject,
        task=task,
        run=run,
        n_volumes=n_volumes,
        acq=acq,
    )
    if interrupted is not None:
        logger.warning(
            f"Found interrupted run with {interrupted['volumes_completed']} volumes "
            f"(expected {n_volumes})"
        )
        # Generate partial QC before archiving
        _generate_partial_qc(output_dir, subject, task, run, acq=acq)
        # Archive interrupted data
        archive_path = _archive_interrupted_run(output_dir, subject, task, run, acq=acq)
        logger.info(f"Archived interrupted run to: {archive_path}")

    # Set up collection directory (required)
    collection_dir = Path(collection_dir)
    collection_dir.mkdir(parents=True, exist_ok=True)
    # Create DICOM collection folder for this run (e.g., collection/feedback_run-01 or feedback_acq-01_run-01)
    run_str = f"{run:02d}"
    if acq is not None:
        # Format acq as 2-digit number if int
        acq_str = f"{acq:02d}" if isinstance(acq, int) else acq
        dicom_collection_dir = collection_dir / f"{task}_acq-{acq_str}_run-{run_str}"
    else:
        dicom_collection_dir = collection_dir / f"{task}_run-{run_str}"
    dicom_collection_dir.mkdir(parents=True, exist_ok=True)
    # Default incoming_dir to collection_dir/incoming if not specified
    if incoming_dir is None:
        incoming_dir = collection_dir / 'incoming'

    # Determine mode and set up incoming directory
    if simulator_source_dir is not None:
        mode = 'simulated'
        incoming_dir = Path(incoming_dir)
        incoming_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running in SIMULATION mode")
        logger.info(f"Source DICOMs: {simulator_source_dir}")
        logger.info(f"Target incoming: {incoming_dir}")
        logger.info(f"DICOM collection: {dicom_collection_dir}")
    else:
        mode = 'real-time'
        incoming_dir = Path(incoming_dir)
        if not incoming_dir.exists():
            raise FileNotFoundError(
                f"Incoming directory not found: {incoming_dir}. "
                "For real-time mode, this should be the directory where the scanner writes DICOMs."
            )
        logger.info(f"Running in REAL-TIME mode")
        logger.info(f"Watching: {incoming_dir}")
        logger.info(f"DICOM collection: {dicom_collection_dir}")

    # Set up simulator if in simulation mode
    simulator = None
    if mode == 'simulated':
        # Type checker: simulator_source_dir is guaranteed non-None here
        # because mode='simulated' only when simulator_source_dir is not None
        assert simulator_source_dir is not None

        # Create simulator instance
        # Note: cleanup_on_complete is set to False here because the coordinator
        # handles cleanup of both incoming and nifti directories after completion
        simulator = DicomSimulator(
            source_dir=simulator_source_dir,
            target_dir=incoming_dir,
            tr=tr,
            speed=simulator_speed,
            clear_target=simulator_clear_target,
            cleanup_on_complete=False,  # Coordinator handles cleanup
        )
        logger.info(f"Simulator configured (speed={simulator_speed}x)")

    # Initialize coordinator with simulator
    coordinator = NeurofeedbackCoordinator(
        watch_dir=incoming_dir,
        output_dir=output_dir,
        dicom_collection_dir=dicom_collection_dir,
        mask_paths=mask_paths,
        subject=subject,
        task=task,
        run=run,
        n_baseline=n_baseline,
        n_volumes=n_volumes,
        tr=tr,
        task_regressors=task_regressors,
        mask_names=mask_names,
        use_efficiency_weights=use_efficiency_weights,
        activation_callback=activation_callback,
        error_callback=error_callback,
        simulator=simulator,
        qc_enabled=qc_enabled,
        acq=acq,
    )

    # Set cleanup flags for simulation mode
    if mode == 'simulated':
        coordinator._cleanup_incoming = simulator_cleanup
        coordinator._cleanup_nifti = simulator_cleanup

    # Start coordinator in background thread (always)
    coordinator_thread = threading.Thread(
        target=coordinator.run,
        daemon=False,
        name="FeedbackCoordinator"
    )
    coordinator_thread.start()
    logger.info("Coordinator thread started")

    return {
        'coordinator': coordinator,
        'thread': coordinator_thread,
        'mode': mode,
        'started': True,
    }
