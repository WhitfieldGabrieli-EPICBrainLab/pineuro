"""
Real-time feedback estimation for fMRI neurofeedback.

Key Classes
-----------
IncrementalGLM : Incremental least squares solver using Gentleman's algorithm
FeedbackEstimator : Real-time neurofeedback estimator using incremental GLM
NeurofeedbackCoordinator : Watches for DICOMs and coordinates real-time processing

Dependencies
------------
numpy, nibabel, watchdog

Notes
-----
The IncrementalGLM class implements an efficient incremental least squares
solver that updates parameter estimates with new observations without
recomputing from scratch. This is essential for real-time fMRI processing
where each TR brings new data.

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
import logging
from pathlib import Path
import threading
import time
from typing import Callable, Optional, Sequence

# Third-party imports
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
from numpy.typing import NDArray
import pydicom
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, DirCreatedEvent
import json
import re

# Configure module-level logger
logger = logging.getLogger(__name__)

class IncrementalGLM:
    """Incremental least squares solver using Gentleman's algorithm.

    Implements online GLM fitting with MURFI-style normalization for real-time
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
    n_steady_state : int, optional
        Number of observations after which to freeze the residual variance
        estimate (for computational efficiency). Default is 200. Set to 0
        to never freeze.

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

    The normalization follows the MURFI approach (Hinds et al., 2011):
    - Error signal = y - X_nuisance @ beta_nuisance (removing only nuisance)
    - Deviation = sqrt(RSS / t) where RSS is residual sum of squares from
      the full model and t is the current timepoint
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
        n_steady_state: int = 200,
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
        self.n_steady_state = n_steady_state

        # Observation counter
        self.n_obs = 0

        # R matrix from QR decomposition: upper triangular (p x p)
        # Augmented with an extra column for the cross-product with y
        # Shape: (n_regressors, n_regressors + n_outputs)
        self._R: NDArray[np.float64] = np.zeros(
            (n_regressors, n_regressors + n_outputs), dtype=np.float64
        )

        # Residual sum of squares (RSS) for each output - tracks full model error
        # This is accumulated from the Givens rotation residuals
        self._rss: NDArray[np.float64] = np.zeros(n_outputs, dtype=np.float64)

        # Steady-state RSS (frozen after n_steady_state observations)
        self._steady_state_rss: Optional[NDArray[np.float64]] = None

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

    def _compute_murfi_activation(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        beta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute MURFI-style normalized activation.

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

        # Compute error by removing only nuisance regressors (MURFI approach)
        # This preserves signal of interest while removing nuisance
        # Nuisance columns are the LAST n_nuisance columns (task columns come first)
        nuisance_start = p - self.n_nuisance
        nuisance_beta = beta[nuisance_start:, :]
        nuisance_X = X[nuisance_start:]
        error = y - (nuisance_X @ nuisance_beta).ravel()

        # Compute deviation as sqrt(RSS / t) - MURFI approach
        # Use steady-state RSS if available (after n_steady_state observations)
        if self._steady_state_rss is not None:
            rss_to_use = self._steady_state_rss
        else:
            rss_to_use = self._rss
            # Freeze RSS at steady state for efficiency
            if self.n_steady_state > 0 and self.n_obs == self.n_steady_state:
                self._steady_state_rss = self._rss.copy()
                logger.info(
                    f"Steady state reached after {self.n_steady_state} observations"
                )

        # Use degrees of freedom to avoid near-zero deviation when n_obs ≈ n_regressors
        df = max(self.n_obs - self.n_regressors, 1)
        deviation = np.sqrt(rss_to_use / df)

        # Avoid division by zero
        deviation = np.where(deviation > 0, deviation, 1.0)

        # Compute normalized activation (MURFI: err / dev)
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
            Normalized activation (error / deviation) following MURFI approach,
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

        # Cannot compute activation until we have enough observations
        if self.n_obs < self.n_regressors:
            return None

        # Solve for parameter estimates via back-substitution
        R = self._R[:, :p]
        r = self._R[:, p:]
        beta = self._solve_via_backsubstitution(R, r)

        # Compute MURFI-style normalized activation
        activation = self._compute_murfi_activation(X, y, beta)

        return activation

class FeedbackEstimator:
    """Real-time neurofeedback estimator using incremental GLM.

    Processes fMRI volumes in real-time, fitting a GLM to remove nuisance
    signals and estimate BOLD activation. Supports multiple masks with
    optional voxel efficiency weighting.

    Parameters
    ----------
    mask_paths : list of Path or str
        Paths to NIfTI mask files. Each mask defines a region of interest.
    n_baseline : int, optional
        Number of initial volumes for baseline period. Default is 25 (~30s at
        TR=1.2s). Used for voxel weight computation.
    mask_names : list of str, optional
        Names for each mask. If not provided, uses mask filenames.
    use_efficiency_weights : bool, optional
        If True, use inverse-variance weighting for ROI aggregation (computed
        from baseline). If False, use simple mean. Default is False (matching
        MURFI's default behavior).
    task_regressor : ndarray or Path, optional
        Task regressor values for each volume (HRF-convolved block design).
        Can be:
        - ndarray of shape (n_volumes,): task regressor values
        - Path to a MURFI-style design matrix file (.mat ASCII format)
        If provided, this is treated as a regressor of interest and NOT
        subtracted from the error signal (MURFI behavior).

    Attributes
    ----------
    n_volumes : int
        Number of volumes processed so far.
    timeseries : dict of str to list of float
        Mean activation timeseries for each mask.

    Notes
    -----
    The GLM design matrix structure:
    - Column 0: Task regressor (of interest, NOT subtracted from error) - if provided
    - Column 1 (or 0 if no task): Intercept (constant) - nuisance
    - Column 2 (or 1): Linear drift - nuisance
    - Columns 3-8 (or 2-7): 6 motion parameters (differential) - nuisance

    Activation is computed following MURFI (Hinds et al., 2011):
    - Error = y - X_nuisance @ beta_nuisance (removing nuisance only)
    - Deviation = sqrt(RSS / t) where RSS is residual sum of squares
    - Activation = error / deviation

    Motion parameters are extracted directly from DICOM headers (Siemens private tags
    0021,115a for rotation and 0021,115b for translation in PerFrameFunctionalGroupsSequence).
    """

    N_NUISANCE_REGRESSORS = 8  # intercept + drift + 6 motion

    def __init__(
        self,
        mask_paths: Sequence[Path | str],
        n_baseline: int = 25,
        mask_names: Optional[list[str]] = None,
        use_efficiency_weights: bool = False,
        task_regressor: Optional[NDArray[np.float64] | Path | str] = None,
    ) -> None:
        if len(mask_paths) == 0:
            raise ValueError("At least one mask path required")

        self.n_baseline = n_baseline
        self.n_volumes = 0
        self.use_efficiency_weights = use_efficiency_weights

        # Load task regressor if provided
        self._task_regressor: Optional[NDArray[np.float64]] = None
        if task_regressor is not None:
            self._task_regressor = self._load_task_regressor(task_regressor)
            self._n_regressors = self.N_NUISANCE_REGRESSORS + 1  # task + nuisance
            self._n_nuisance = self.N_NUISANCE_REGRESSORS  # only nuisance columns
            self._task_col_idx = 0  # task regressor is first column
            self._nuisance_start_idx = 1  # nuisance starts at column 1
        else:
            self._n_regressors = self.N_NUISANCE_REGRESSORS
            self._n_nuisance = self.N_NUISANCE_REGRESSORS  # all are nuisance
            self._task_col_idx = None
            self._nuisance_start_idx = 0

        # Set up mask names
        mask_paths_resolved: list[Path] = [Path(p) for p in mask_paths]
        if mask_names is None:
            mask_names = [p.stem for p in mask_paths_resolved]
        if len(mask_names) != len(mask_paths_resolved):
            raise ValueError("mask_names must match length of mask_paths")
        self.mask_names = mask_names

        # Load masks and build voxel index mapping
        self._load_masks(mask_paths_resolved)

        # Initialize single GLM for all voxels (MURFI-style)
        self._glm = IncrementalGLM(
            n_regressors=self._n_regressors,
            n_nuisance=self._n_nuisance,
            n_outputs=self._n_voxels,
        )

        # Motion parameter tracking for differential motion
        self._prev_motion: Optional[NDArray[np.float64]] = None

        # Baseline activations storage for weight computation (if enabled)
        self._baseline_activations: list[NDArray[np.float64]] = []

        # Voxel weights per mask (computed after baseline if enabled)
        self._weights: dict[str, NDArray[np.float64]] = {}

        # Timeseries storage
        self.timeseries: dict[str, list[float]] = {name: [] for name in mask_names}

        logger.info(
            f"FeedbackEstimator initialized with {len(mask_paths)} masks, "
            f"{self._n_voxels} total voxels, "
            f"task_regressor={'enabled' if self._task_regressor is not None else 'disabled'}, "
            f"efficiency_weights={'enabled' if use_efficiency_weights else 'disabled'}"
        )

    def _load_task_regressor(
        self, source: NDArray[np.float64] | Path | str
    ) -> NDArray[np.float64]:
        """Load task regressor from array or MURFI design matrix file."""
        if isinstance(source, np.ndarray):
            return source.astype(np.float64).ravel()

        # Load from file (MURFI-style ASCII design matrix)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Task regressor file not found: {path}")

        # Parse MURFI design matrix format
        # First line is header with column names
        # Subsequent lines are space-separated values
        task_values = []
        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = line.split()
            if len(values) > 0:
                # First column is the task regressor
                task_values.append(float(values[0]))

        return np.array(task_values, dtype=np.float64)

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
        # Load first mask to get reference shape
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

            # Try to extract motion parameters from pattern: Motion: x,y,z,rx,ry,rz
            pattern = r'Motion:\s*([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)'
            match = re.search(pattern, image_comments)

            if match:
                values = [float(m) for m in match.groups()]
                return np.array(values, dtype=np.float64)

            logger.warning(f"No motion parameters found in {json_path}, using zeros")
            return np.zeros(6, dtype=np.float64)

        except Exception as e:
            logger.error(f"Error reading JSON {json_path}: {e}")
            return np.zeros(6, dtype=np.float64)

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
            Design row. If task regressor provided:
              [task, intercept, drift, motion_diff...]
            Otherwise:
              [intercept, drift, motion_diff...]
        """
        # Compute relative/differential motion (MURFI uses motion_diff)
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

        if self._task_regressor is not None:
            # With task regressor: [task, intercept, drift, motion_diff...]
            if vol_idx < len(self._task_regressor):
                X[0] = self._task_regressor[vol_idx]
            else:
                # Extrapolate with last value if we exceed design length
                X[0] = self._task_regressor[-1] if len(self._task_regressor) > 0 else 0.0
            X[1] = 1.0  # intercept
            X[2] = float(vol_idx + 1)  # linear drift (1-indexed volume number)
            X[3:9] = motion_diff  # differential motion
        else:
            # Without task regressor: [intercept, drift, motion_diff...]
            X[0] = 1.0  # intercept
            X[1] = float(vol_idx + 1)  # linear drift (1-indexed volume number)
            X[2:8] = motion_diff  # differential motion

        return X

    def _compute_weights(self) -> None:
        """Compute voxel efficiency weights from baseline activations."""
        if not self.use_efficiency_weights:
            return

        # Stack baseline activations: shape (n_baseline, n_voxels)
        baseline_array = np.array(self._baseline_activations)

        # Compute variance per voxel
        voxel_variance = np.var(baseline_array, axis=0, ddof=1)

        # Compute weights for each mask
        for name, idx in self._mask_voxel_idx.items():
            mask_variance = voxel_variance[idx]

            # Inverse variance weighting (handle zero variance)
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / mask_variance
                weights[~np.isfinite(weights)] = 0.0

            # Normalize to sum to 1
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                # Fallback to uniform weights
                weights = np.ones_like(weights) / len(weights)
                logger.warning(f"Mask '{name}': using uniform weights (zero variance)")

            self._weights[name] = weights

        logger.info("Voxel efficiency weights computed from baseline")

        # Clear baseline activations to free memory
        self._baseline_activations = []

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
        img: Nifti1Image = nib.loadsave.load(nifti_path)  # type: ignore[assignment]
        data = np.asarray(img.dataobj, dtype=np.float64)

        # Verify shape matches masks
        if data.shape[:3] != self._shape:
            raise ValueError(
                f"Volume shape {data.shape[:3]} != mask shape {self._shape}"
            )

        # Extract voxel values at all mask locations using flat indexing
        data_flat = data.reshape(-1)
        flat_indices = np.ravel_multi_index(self._voxel_indices, self._shape)
        voxel_values = data_flat[flat_indices]

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

        # ===== 3. Build design matrix and update GLM =====
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

        # During baseline, store activations for weight computation (if enabled)
        if self.use_efficiency_weights and self.n_volumes <= self.n_baseline:
            self._baseline_activations.append(result.copy())

            # Compute weights when baseline completes
            if self.n_volumes == self.n_baseline:
                self._compute_weights()

        # ===== 4. Compute weighted activations per mask =====
        activations: dict[str, float] = {}
        for name, idx in self._mask_voxel_idx.items():
            mask_values = result[idx]

            if self.use_efficiency_weights and name in self._weights:
                # Use efficiency weights if enabled and computed
                weighted_mean = float(np.sum(self._weights[name] * mask_values))
            else:
                # Use simple mean (MURFI default)
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


class NeurofeedbackCoordinator:
    """Coordinates real-time neurofeedback by watching for DICOMs.

    Monitors a directory for incoming DICOM files, converts them to NIfTI,
    processes through FeedbackEstimator, and logs results to CSV.

    Parameters
    ----------
    watch_dir : Path or str
        Directory to watch for incoming DICOM files.
    output_dir : Path or str
        Directory for NIfTI outputs and CSV log.
    mask_paths : list of Path or str
        Paths to NIfTI mask files for feedback estimation.
    n_baseline : int, optional
        Number of initial volumes for baseline. Default is 25.
    n_volumes : int, optional
        Total number of volumes to process. Default is 200.
    mask_names : list of str, optional
        Names for each mask. If not provided, uses mask filenames.
    task_regressor : ndarray or Path, optional
        Task regressor for the GLM. See FeedbackEstimator for details.

    Attributes
    ----------
    n_processed : int
        Number of volumes processed so far.

    Notes
    -----
    Uses watchdog for efficient filesystem event detection. Each new DICOM
    triggers conversion via dcm2niix and processing through FeedbackEstimator.
    Results are logged to CSV after each volume.

    Examples
    --------
    >>> coordinator = NeurofeedbackCoordinator(
    ...     watch_dir='/path/to/incoming',
    ...     output_dir='/path/to/output',
    ...     mask_paths=['/path/to/dmn_mask.nii.gz'],
    ...     n_volumes=100
    ... )
    >>> coordinator.run()
    """

    def __init__(
        self,
        watch_dir: Path | str,
        output_dir: Path | str,
        mask_paths: list[Path | str],
        n_baseline: int = 25,
        n_volumes: int = 200,
        mask_names: Optional[list[str]] = None,
        task_regressor: Optional[NDArray[np.float64] | Path | str] = None,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.n_baseline = n_baseline
        self.n_volumes = n_volumes

        # Validate watch directory
        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Watch directory not found: {self.watch_dir}")
        if not self.watch_dir.is_dir():
            raise ValueError(f"Watch path is not a directory: {self.watch_dir}")

        # Create output directories
        self.nifti_dir = self.output_dir / "01_nifti"
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
            mask_names=mask_names,
            task_regressor=task_regressor,
        )

        # Processing state
        self.n_processed = 0
        self._processed_files: set[str] = set()
        self._pending_files: list[Path] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None
        self.design_matrix: list[NDArray[np.float64]] = []

        # CSV log setup
        self._csv_path = self.output_dir / "feedback_log.csv"

        logger.info(
            f"NeurofeedbackCoordinator initialized: "
            f"watching {self.watch_dir}, {n_volumes} volumes expected"
        )

    def _init_csv(self) -> None:
        """Initialize CSV log file with header."""
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        header = ["volume", "time", "is_baseline"] + self.mask_names
        self._csv_writer.writerow(header)
        self._csv_file.flush()

    def _close_csv(self) -> None:
        """Close CSV log file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _on_new_file(self, file_path: Path) -> None:
        """Callback when a new file is detected."""
        with self._lock:
            if file_path.name not in self._processed_files:
                self._pending_files.append(file_path)

    def _is_valid_dicom(self, file_path: Path) -> bool:
        """Check if file is a valid DICOM."""
        try:
            pydicom.dcmread(file_path, stop_before_pixels=True)
            return True
        except Exception:
            return False

    def _process_volume(self, dicom_path: Path) -> None:
        """Convert DICOM and process through estimator."""
        from .utils import dicom_to_nifti

        # Convert DICOM to NIfTI
        volume_num = self.n_processed + 1
        output_name = f"vol_{volume_num:04d}"

        try:
            nifti_path = dicom_to_nifti(
                dicom_dir=dicom_path,
                output_dir=self.nifti_dir,
                output_name=output_name,
                overwrite=True,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Failed to convert DICOM {dicom_path}: {e}")
            return

        # Derive JSON sidecar path from nifti_path
        json_path = Path(str(nifti_path).replace('.nii.gz', '.json').replace('.nii', '.json'))

        # Process through estimator
        try:
            activations = self._estimator.process_volume(nifti_path, json_path)
            self.design_matrix.append(self._estimator._last_design_row)
        except Exception as e:
            logger.error(f"Failed to process volume {volume_num}: {e}")
            return

        self.n_processed += 1
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        is_baseline = self.n_processed <= self.n_baseline

        # Log to CSV
        row = [self.n_processed, f"{elapsed:.2f}", is_baseline]
        row.extend([f"{activations[name]:.4f}" for name in self.mask_names])
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
        if self._csv_file is not None:
            self._csv_file.flush()

        # Log to console
        status = "BASELINE" if is_baseline else "ACTIVE"
        activation_str = "  ".join(
            f"{name}={activations[name]:+.3f}" for name in self.mask_names
        )
        logger.info(
            f"[{elapsed:7.2f}s] Vol {self.n_processed:4d}/{self.n_volumes} "
            f"({status}): {activation_str}"
        )

    def run(self) -> None:
        """Run the coordinator until n_volumes processed or interrupted.

        Blocks until all volumes are processed or KeyboardInterrupt.
        """
        # Initialize CSV
        self._init_csv()

        # Set up watchdog observer
        event_handler = _DicomEventHandler(self._on_new_file)
        observer = Observer()
        observer.schedule(event_handler, str(self.watch_dir), recursive=False)
        observer.start()

        logger.info(f"Watching {self.watch_dir} for DICOMs...")
        self._start_time = time.time()

        try:
            while self.n_processed < self.n_volumes and not self._stop_event.is_set():
                # Check for pending files
                file_to_process = None
                with self._lock:
                    if self._pending_files:
                        # Sort by filename and take first
                        self._pending_files.sort()
                        candidate = self._pending_files[0]
                        if self._is_valid_dicom(candidate):
                            file_to_process = self._pending_files.pop(0)
                            self._processed_files.add(file_to_process.name)
                        else:
                            # Remove invalid file from pending
                            self._pending_files.pop(0)

                if file_to_process:
                    self._process_volume(file_to_process)
                else:
                    # Brief sleep to avoid busy-waiting
                    time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            observer.stop()
            observer.join()
            self._close_csv()

            elapsed = time.time() - self._start_time if self._start_time else 0.0
            logger.info(
                f"Session complete: {self.n_processed} volumes in {elapsed:.1f}s"
            )
            logger.info(f"Log saved to: {self._csv_path}")

    def stop(self) -> None:
        """Signal the coordinator to stop."""
        self._stop_event.set()

