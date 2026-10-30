"""
Functional network mask extraction pipeline for real-time fMRI neurofeedback.

This module extracts participant-specific functional network masks from
fMRI data. These masks define the brain regions used for computing
neurofeedback signals during subsequent task sessions.

Two extraction approaches are provided:

1. **ICA-based (resting-state)**: ``mask_extraction_rest()`` runs ICA on
   resting-state data and matches components to Yeo atlas networks.

2. **GLM-based (task localizer)**: ``mask_extraction_task()`` fits a GLM to
   task localizer data and thresholds contrast maps into masks. Supports two
   thresholding strategies:

   - **top_n** (default): Selects the top N voxels by stat magnitude.
   - **cluster**: Atlas-guided iterative cluster thresholding. Restricts the
     stat map to an anatomical ROI (e.g., Harvard-Oxford STG) and iteratively
     raises the stat threshold until the largest contiguous cluster fits the
     target size. See ``fetch_atlas_mask()`` and ``cluster_threshold_stat_map()``.

Key Components
--------------
mask_extraction_rest : ICA pipeline function
    End-to-end workflow for mask extraction from resting-state data.

mask_extraction_task : GLM pipeline function
    End-to-end workflow for mask extraction from task localizer data.

run_ica : ICA decomposition
    Runs CanICA (nilearn) on preprocessed fMRI data to extract
    spatially independent components.

spatial_correlation : Network matching
    Computes spatial correlations between ICA components and atlas
    networks to identify which component best represents each target network.

MaskSelection : Data class
    Tracks selection state for each mask (auto/manual, ICA component used,
    correlation values) for reproducibility.

Workflow
--------
The pipeline follows these steps:

1. **ICA Decomposition** (01_ica/)
   Run CanICA on preprocessed resting-state data to extract spatially
   independent components. Default: 20 components.

2. **Registration to MNI** (02_mnireg/)
   Compute transformation between participant space and MNI152 space.
   Pull the Yeo atlas into participant space for correlation.

3. **Network Selection** (03_masks/)
   For each target network (e.g., ContA, DefaultA):
   - Compute spatial correlation between each ICA component and atlas network
   - Select the component with highest correlation
   - Threshold to retain ~2000 highest-loading voxels
   - Save as binary mask

4. **QC Report** (qc/)
   Generate visualizations for quality control: component overlays,
   correlation heatmaps, mask coverage.

Why ICA-Based Masks?
--------------------
Group atlases (like Yeo) are derived from large samples and represent
population averages. Individual brains show systematic deviations from
these averages due to:
- Anatomical variability (sulcal patterns, cortical folding)
- Functional variability (individual network topography)

ICA-based masks capture each participant's actual network boundaries,
improving signal specificity for neurofeedback. The atlas is only used
to identify which component corresponds to which network.

Output Files
------------
After extraction, the root directory contains::

    root_dir/
    ├── 01_ica/
    │   ├── {participant}_components.nii.gz    # 4D ICA components
    │   ├── {participant}_ica_info.json        # ICA parameters
    │   └── {participant}_correlations.csv     # Component × network correlations
    ├── 02_mnireg/
    │   ├── {participant}_to_MNI.nii.gz        # Participant→MNI warped
    │   ├── Yeo_to_{participant}.nii.gz        # Atlas in participant space
    │   └── xfm/                               # ANTs transform files
    ├── 03_masks/
    │   ├── {participant}_{network}_mask.nii.gz  # Per-network masks
    │   └── mask_selections.json               # Selection metadata
    └── qc/
        └── mask_extraction_qc.html            # QC report

Dependencies
------------
External:
    - ANTs (via ANTsPy) : Registration to MNI space

Python:
    - nilearn : CanICA and atlas fetching
    - nibabel : NIfTI I/O
    - numpy, pandas : Data manipulation
    - scipy : Spatial correlation

Notes
-----
**Component Selection:**
The auto-selection algorithm chooses the ICA component with highest absolute
spatial correlation to each target network. Manual override is supported via
the GUI for cases where auto-selection fails.

**Thresholding:**
Masks are thresholded to retain a fixed number of voxels (default 2000) by
selecting the highest-loading voxels within each component. This provides
consistent mask sizes across networks and participants.

**Multi-Network Masks:**
For some networks (e.g., DMN), the signal may be split across multiple ICA
components. The GUI supports combining two components into a single mask.

See Also
--------
preprocessing : Prepare resting-state data for mask extraction
feedback : Use extracted masks for real-time neurofeedback
qc.generate_mask_extraction_report : Generate QC visualizations

References
----------
.. [1] Varoquaux, G., Sadaghiani, S., Pinel, P., Kleinschmidt, A., Poline, J. B.,
       & Thirion, B. (2010). A group model for stable multi-subject ICA on fMRI
       datasets. NeuroImage, 51(1), 288-299.

.. [2] Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
       Hollinshead, M., ... & Buckner, R. L. (2011). The organization of the
       human cerebral cortex estimated by intrinsic functional connectivity.
       Journal of Neurophysiology, 106(3), 1125-1165.

.. [3] Hacker, C. D., Laumann, T. O., Szrama, N. P., Baldassarre, A., Snyder, A. Z.,
       Leuthardt, E. C., & Corbetta, M. (2013). Resting state network estimation
       in individual subjects. NeuroImage, 82, 616-633.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# --- Standard Library ---
import contextlib
import json  # JSON I/O for selection metadata
import logging
import re  # Regex for filename parsing
import threading  # Cooperative cancellation signal for long-running steps
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Union, cast

# NOTE: nilearn is imported lazily inside the functions that need it to avoid
# ~8s import cost at module load time. The GUI imports this module at startup
# but doesn't need nilearn until mask extraction actually runs.
import nibabel as nib

# --- Third-Party: Neuroimaging ---
import numpy as np
import pandas as pd
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from scipy.ndimage import label as ndimage_label  # Connected component labeling

# --- Local Imports ---
from pineuro import qc
from pineuro.utils import (
    # Timing
    Timer,  # Performance timing
    apply_xfm,  # Apply transformation
    apply_xfm_sform_to_mni,  # Quick MNI alignment via sform
    # Registration helpers (ANTs-backed)
    compute_mean,  # Compute temporal mean of 4D volume
    compute_xfm,  # Compute transformation matrix
    compute_xfm_to_mni,  # Compute transform to MNI space
    # DICOM helpers
    dicom_to_nifti,  # DICOM conversion (for reference coregistration)
    # Atlas helpers
    get_yeo_labels,  # Get Yeo network names (7 or 17 networks)
)

# --- Module Logger ---
logger = logging.getLogger(__name__)


class MaskExtractionStopped(Exception):
    """Raised when a caller-supplied ``stop_event`` is set mid-pipeline.

    Signals a cooperative cancellation (e.g. the GUI "Stop" button) rather than
    an error, so callers can treat it as a clean stop instead of a failure. The
    ICA fit, which is not interruptible in-process, runs in a child process that
    is terminated when the stop is requested — see :func:`run_ica`.
    """


def _terminate_process_tree(pid: int, timeout: float = 5.0) -> None:
    """Terminate a process and all of its descendants.

    CanICA runs with ``n_jobs=-1``, so the worker process spawns joblib/loky
    grandchildren; terminating only the worker would orphan them. psutil walks
    the tree, sends terminate, then kills anything still alive after ``timeout``.
    """
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True)
    procs.append(parent)
    for p in procs:
        with contextlib.suppress(psutil.NoSuchProcess):
            p.terminate()
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            p.kill()


def _run_canica(
    input_file: str,
    mask_file: str,
    n_components: int,
    smooth_fwhm: float | None,
    highpass: float | None,
    lowpass: float | None,
    t_r: float,
    random_state: int,
    memory_dir: str,
    components_file: str,
    metadata_file: str,
) -> None:
    """Build CanICA, fit it, and write components + metadata to disk.

    Factored out so it can run either inline (no ``stop_event``) or inside a
    killable child process (see :func:`run_ica`). All arguments are plain,
    picklable values so the child can be spawned cross-platform.
    """
    from nilearn.decomposition import CanICA

    canica: Any = CanICA(
        mask=mask_file,
        n_components=n_components,
        smoothing_fwhm=smooth_fwhm,
        high_pass=highpass,
        low_pass=lowpass,
        threshold=None,  # type: ignore[arg-type]
        t_r=t_r,
        n_jobs=-1,  # Use all CPUs
        random_state=random_state,
        memory=memory_dir,
        memory_level=2,
        verbose=1,
    )
    canica.fit(input_file)
    components_img: SpatialImage = canica.components_img_
    save(components_img, components_file)
    logger.info(f"Saved components to: {components_file}")
    np.savez(
        metadata_file,
        n_components=n_components,
        components_shape=components_img.shape,
        input_file=str(input_file),
    )
    logger.info(f"Saved metadata to: {metadata_file}")


def _canica_worker(args: tuple, result_queue: Any) -> None:
    """Child-process entry point: run :func:`_run_canica`, report ok/error.

    The components are written to disk by ``_run_canica``; only a tiny status
    tuple travels back over the queue, so there is no large-payload join hazard.
    """
    try:
        _run_canica(*args)
        result_queue.put(("ok", None))
    except Exception as e:  # noqa: BLE001 - forward any failure to the parent
        result_queue.put(("error", f"{type(e).__name__}: {e}"))


def _remove_partial_outputs(*paths: Path) -> None:
    """Delete any partially-written outputs left by an interrupted run."""
    for p in paths:
        with contextlib.suppress(OSError):
            Path(p).unlink(missing_ok=True)


def _run_canica_cancellable(
    args: tuple,
    stop_event: threading.Event,
    components_file: Path,
    metadata_file: Path,
) -> None:
    """Run :func:`_run_canica` in a child process, killing it on ``stop_event``.

    nilearn's ``fit`` is a single opaque call with no cancellation hook, so the
    only way to interrupt it is to run it in a separate process and terminate
    that process (and its joblib workers) when the GUI requests a stop. ``spawn``
    is used so the child is a clean interpreter on every platform (a ``fork``
    after nibabel/ANTs threads are live is unsafe).
    """
    import multiprocessing as mp
    import queue as queue_mod

    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue()
    proc = ctx.Process(target=_canica_worker, args=(args, result_queue), daemon=False)
    proc.start()

    result: tuple | None = None
    try:
        while True:
            try:
                result = result_queue.get(timeout=0.5)
                break
            except queue_mod.Empty:
                pass
            if stop_event.is_set():
                logger.info("Stop requested — terminating ICA subprocess")
                _terminate_process_tree(proc.pid)  # type: ignore[arg-type]
                proc.join(timeout=10)
                _remove_partial_outputs(components_file, metadata_file)
                raise MaskExtractionStopped("Mask extraction (ICA) stopped by user")
            if not proc.is_alive():
                # Worker exited; collect its (possibly already-queued) result.
                try:
                    result = result_queue.get(timeout=2)
                except queue_mod.Empty:
                    result = ("error", "ICA worker exited without reporting a result")
                break
    finally:
        if proc.is_alive():
            _terminate_process_tree(proc.pid)  # type: ignore[arg-type]
        proc.join(timeout=10)

    status, detail = result if result is not None else ("error", "ICA worker produced no result")
    if status != "ok":
        raise RuntimeError(f"CanICA computation failed: {detail}")


def _json_default(obj):
    """JSON serializer for numpy types produced by the mask pipeline."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================
# These data structures track the selection state for each network mask.
# They enable reproducibility (re-running with the same selections) and
# support the GUI's ability to manually override automatic selections.


@dataclass
class MaskSelection:
    """Tracks the selection state for a single network mask.

    Attributes
    ----------
    network_name : str
        Name of the network (e.g., "ContA", "DefaultA")
    source_type : str
        One of: "ica" for single ICA component, "ica_combined" for two merged
        ICA components, "yeo_atlas" for atlas fallback (per-network, after ICA),
        or "mni_atlas" for full MNI atlas masks (no ICA/localizer required)
    component_index : int or None
        Primary ICA component index if source_type in ("ica", "ica_combined"),
        None for atlas masks
    secondary_component_index : int or None
        Second ICA component index if source_type == "ica_combined", None otherwise
    correlation : float
        Spatial correlation with Yeo atlas for primary component (0.0 for atlas masks)
    secondary_correlation : float or None
        Spatial correlation for secondary component if source_type == "ica_combined"
    selection_reason : str
        How this selection was made: "auto", "manual", "manual_combined", or "yeo_fallback"
    """

    network_name: str
    source_type: str  # "ica", "ica_combined", "yeo_atlas", or "mni_atlas"
    component_index: int | None
    correlation: float
    selection_reason: str  # "auto", "manual", "manual_combined", "yeo_fallback"
    secondary_component_index: int | None = None
    secondary_correlation: float | None = None


def save_mask_selections(masks_dir: Union[str, Path], selections: dict[str, MaskSelection]) -> Path:
    """Save mask selections to JSON file.

    Parameters
    ----------
    masks_dir : str or Path
        Directory containing the masks (typically 03_masks/)
    selections : dict
        Dictionary mapping network names to MaskSelection objects

    Returns
    -------
    Path
        Path to the saved JSON file
    """
    masks_dir = Path(masks_dir)
    output_file = masks_dir / "mask_selections.json"

    # Convert dataclasses to dicts for JSON serialization
    data = {name: asdict(sel) for name, sel in selections.items()}

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)

    logger.info(f"Saved mask selections to {output_file}")
    return output_file


def load_mask_selections(masks_dir: Union[str, Path]) -> dict[str, MaskSelection]:
    """Load mask selections from JSON file.

    Parameters
    ----------
    masks_dir : str or Path
        Directory containing the masks (typically 03_masks/)

    Returns
    -------
    dict
        Dictionary mapping network names to MaskSelection objects.
        Returns empty dict if file doesn't exist.
    """
    masks_dir = Path(masks_dir)
    selections_file = masks_dir / "mask_selections.json"

    if not selections_file.exists():
        return {}

    with open(selections_file) as f:
        data = json.load(f)

    # Task-based selections are wrapped: {"type": "task", "selections": {...}}
    # Return empty for task format — it uses TaskMaskSelection, not MaskSelection
    if data.get("type") == "task":
        return {}

    return {name: MaskSelection(**sel_data) for name, sel_data in data.items()}


def get_alternative_components(
    correlation_df: pd.DataFrame, network_name: str, top_n: int = 5
) -> list[tuple[int, float]]:
    """Get the top N ICA components for a network, ranked by correlation.

    Parameters
    ----------
    correlation_df : pd.DataFrame
        Spatial correlation matrix (components × networks)
    network_name : str
        Name of the target network
    top_n : int
        Number of top components to return

    Returns
    -------
    list of (int, float)
        List of (component_index, correlation) tuples, sorted by correlation descending
    """
    if network_name not in correlation_df.columns:
        raise ValueError(f"Network '{network_name}' not found in correlation matrix")

    correlations = correlation_df[network_name]
    # Get indices sorted by correlation (descending)
    # Use numpy argsort to avoid pandas FutureWarning with float-dtype index slicing
    corr_values = correlations.values
    sorted_indices = np.argsort(corr_values)[::-1][:top_n]

    return [(int(idx), float(corr_values[idx])) for idx in sorted_indices]


def compute_mask_network_correlation(
    mask_file: Union[str, Path],
    yeo_subject_space: Union[str, Path],
    network_name: str,
    n_networks: int = 17,
) -> float:
    """Compute spatial correlation between a mask and a specific Yeo network.

    This is used to compute the correlation for combined ICA component masks,
    where the individual component correlations don't reflect the combined result.

    Parameters
    ----------
    mask_file : str or Path
        Path to the mask file (binary or continuous values)
    yeo_subject_space : str or Path
        Path to Yeo atlas in subject space
    network_name : str
        Name of the target network (e.g., "DefaultA")
    n_networks : int
        Yeo parcellation (7 or 17 networks)

    Returns
    -------
    float
        Pearson correlation coefficient between the mask and the network
    """
    mask_file = Path(mask_file)
    yeo_subject_space = Path(yeo_subject_space)

    from nilearn.image import resample_to_img

    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    if not yeo_subject_space.exists():
        raise FileNotFoundError(f"Yeo atlas not found: {yeo_subject_space}")

    # Get network label index
    labels = get_yeo_labels(n_networks=n_networks)
    if network_name not in labels:
        raise ValueError(f"Network '{network_name}' not in Yeo-{n_networks} labels")
    network_index = labels.index(network_name) + 1  # 1-based indexing

    # Load mask
    mask_img = cast(Nifti1Image, load(mask_file))
    mask_data = mask_img.get_fdata().ravel()

    # Load and resample Yeo atlas to mask space
    atlas_img = cast(Nifti1Image, load(yeo_subject_space))
    atlas_resampled = resample_to_img(atlas_img, mask_img, interpolation="nearest")
    atlas_data = atlas_resampled.get_fdata().astype(int).ravel()

    # Create binary network mask
    network_mask = (atlas_data == network_index).astype(float)

    # Compute Pearson correlation
    # Demean both vectors
    mask_dm = mask_data - mask_data.mean()
    network_dm = network_mask - network_mask.mean()

    # Compute correlation
    mask_norm = np.sqrt((mask_dm**2).sum())
    network_norm = np.sqrt((network_dm**2).sum())

    if mask_norm == 0 or network_norm == 0:
        return 0.0

    correlation = (mask_dm @ network_dm) / (mask_norm * network_norm)

    return float(correlation)


def extract_yeo_mask_for_network(
    yeo_subject_space: Union[str, Path],
    network_name: str,
    output_file: Union[str, Path],
    n_networks: int = 17,
    threshold: int = 2000,
    overwrite: bool = False,
) -> Path:
    """Extract and threshold a mask directly from Yeo atlas in subject space.

    This is the fallback option when ICA-derived masks are rejected. The Yeo
    atlas (already registered to subject space) is used to create a binary
    mask for the specified network.

    Parameters
    ----------
    yeo_subject_space : str or Path
        Path to Yeo atlas in subject space (e.g., 02_mnireg/Yeo_to_{ref}.nii.gz)
    network_name : str
        Name of the target network (e.g., "DefaultA")
    output_file : str or Path
        Path to save the thresholded mask
    n_networks : int
        Yeo parcellation (7 or 17 networks)
    threshold : int
        Number of voxels to keep in the thresholded mask
    overwrite : bool
        If True, overwrite existing output

    Returns
    -------
    Path
        Path to the saved mask file
    """
    yeo_subject_space = Path(yeo_subject_space)
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        logger.info(f"Output exists and overwrite=False, skipping: {output_file}")
        return output_file

    if not yeo_subject_space.exists():
        raise FileNotFoundError(f"Yeo atlas not found: {yeo_subject_space}")

    # Get network label index from name
    labels = get_yeo_labels(n_networks=n_networks)
    if network_name not in labels:
        raise ValueError(f"Network '{network_name}' not in Yeo-{n_networks} labels: {labels}")

    network_index = labels.index(network_name) + 1  # Yeo uses 1-based indexing

    # Load atlas and extract network region
    atlas_img = cast(Nifti1Image, load(yeo_subject_space))
    atlas_data = atlas_img.get_fdata()

    # Create binary mask for this network
    network_mask = (atlas_data == network_index).astype(float)
    n_voxels = int(network_mask.sum())

    if n_voxels == 0:
        raise ValueError(f"No voxels found for network '{network_name}' (index {network_index})")

    logger.info(f"Yeo atlas '{network_name}': {n_voxels} voxels")

    # For Yeo atlas masks, keep all voxels - don't threshold/subsample
    # The atlas already defines anatomically meaningful network boundaries
    # Random subsampling would destroy spatial coherence
    binary_mask = network_mask.astype("uint8")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save(Nifti1Image(binary_mask, atlas_img.affine, atlas_img.header), output_file)
    logger.info(f"Saved Yeo atlas mask ({int(binary_mask.sum())} voxels) to {output_file}")

    return output_file


def split_mask_by_hemisphere(
    input_mask: Union[str, Path],
    output_left: Union[str, Path],
    output_right: Union[str, Path],
    threshold: int = 2000,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Split a network mask into left and right hemisphere masks with thresholding.

    Takes an unthresholded ICA component mask, splits it by hemisphere using
    world coordinates, then thresholds each hemisphere separately to retain
    the top N voxels. This ensures each hemisphere has the full threshold
    count (e.g., 2000 voxels each).

    Uses MNI/RAS convention: x < 0 = left hemisphere, x > 0 = right hemisphere.
    Voxels at exactly x=0 (midline) are excluded from both hemispheres.

    Parameters
    ----------
    input_mask : str or Path
        Path to input unthresholded network mask (ICA component, not _thr file)
    output_left : str or Path
        Path to save left hemisphere thresholded mask
    output_right : str or Path
        Path to save right hemisphere thresholded mask
    threshold : int
        Number of voxels to keep in each hemisphere (default: 2000)
    overwrite : bool
        If True, overwrite existing outputs

    Returns
    -------
    tuple[Path, Path]
        (left_mask_path, right_mask_path)

    Raises
    ------
    FileNotFoundError
        If input_mask does not exist
    """
    input_mask = Path(input_mask)
    output_left = Path(output_left)
    output_right = Path(output_right)

    if not input_mask.exists():
        raise FileNotFoundError(f"Input mask not found: {input_mask}")

    if output_left.exists() and output_right.exists() and not overwrite:
        logger.info("Hemisphere masks exist and overwrite=False, skipping")
        return output_left, output_right

    # Load unthresholded mask (ICA component with continuous values)
    mask_img = cast(Nifti1Image, load(input_mask))
    mask_data = mask_img.get_fdata()
    affine = mask_img.affine

    # Create hemisphere masks based on world coordinates
    # Get all voxel indices
    all_indices = np.indices(mask_data.shape).reshape(3, -1).T  # (N, 3)
    world_coords = nib.affines.apply_affine(affine, all_indices)
    x_coords = world_coords[:, 0].reshape(mask_data.shape)

    # Create left/right hemisphere masks (boolean)
    left_hemi_mask = x_coords < 0
    right_hemi_mask = x_coords > 0

    # Apply hemisphere masks to get component values in each hemisphere
    left_values = mask_data.copy()
    left_values[~left_hemi_mask] = 0

    right_values = mask_data.copy()
    right_values[~right_hemi_mask] = 0

    # Threshold each hemisphere separately
    def threshold_hemisphere(data: np.ndarray, n_voxels: int) -> np.ndarray:
        """Keep top N positive voxels by value."""
        flat = data.ravel()
        # Pre-filter to positive voxels only, then select top N
        positive_indices = np.where(flat > 0)[0]
        if len(positive_indices) <= n_voxels:
            # Keep all positive voxels if fewer than threshold
            result = (flat > 0).astype("uint8")
            return result.reshape(data.shape)
        # Select top N positive voxels by value
        positive_values = flat[positive_indices]
        top_n_within_positive = np.argpartition(positive_values, -n_voxels)[-n_voxels:]
        result = np.zeros(flat.shape, dtype="uint8")
        result[positive_indices[top_n_within_positive]] = 1
        return result.reshape(data.shape)

    left_thresholded = threshold_hemisphere(left_values, threshold)
    right_thresholded = threshold_hemisphere(right_values, threshold)

    n_left = int(left_thresholded.sum())
    n_right = int(right_thresholded.sum())

    logger.info(
        f"Split and thresholded {input_mask.name}: {n_left} left voxels, {n_right} right voxels (threshold={threshold})"
    )

    # Save
    output_left.parent.mkdir(parents=True, exist_ok=True)
    output_right.parent.mkdir(parents=True, exist_ok=True)
    save(Nifti1Image(left_thresholded, affine, mask_img.header), output_left)
    save(Nifti1Image(right_thresholded, affine, mask_img.header), output_right)

    logger.info(f"Saved left hemisphere mask to {output_left}")
    logger.info(f"Saved right hemisphere mask to {output_right}")

    return output_left, output_right


def combine_ica_components(
    components_file: Union[str, Path],
    component_idx_1: int,
    component_idx_2: int,
    output_file: Union[str, Path],
    threshold: int = 2000,
    overwrite: bool = False,
) -> Path:
    """Combine two ICA components into a single thresholded mask.

    Z-scores each component map before averaging to ensure equal contribution
    from both components (regardless of their original value ranges), then
    applies thresholding to retain the top N voxels.

    This is useful when ICA splits a network into two components (e.g.,
    left/right ContA, frontal/occipital DMN). Z-scoring before averaging
    ensures both components contribute equally to the final mask.

    Parameters
    ----------
    components_file : str or Path
        Path to 4D ICA components NIfTI file
    component_idx_1 : int
        Index of first ICA component
    component_idx_2 : int
        Index of second ICA component
    output_file : str or Path
        Path to save combined thresholded mask
    threshold : int
        Number of voxels to keep (default: 2000)
    overwrite : bool
        Overwrite existing output

    Returns
    -------
    Path
        Path to saved combined mask file

    Raises
    ------
    FileNotFoundError
        If components_file does not exist
    ValueError
        If component indices are out of range or identical
    """
    components_file = Path(components_file)
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        logger.info(f"Output exists and overwrite=False, skipping: {output_file}")
        return output_file

    if not components_file.exists():
        raise FileNotFoundError(f"Components file not found: {components_file}")

    if component_idx_1 == component_idx_2:
        raise ValueError(f"Cannot combine component with itself: {component_idx_1}")

    # Load ICA components
    components_img = cast(Nifti1Image, load(components_file))
    components_data = components_img.get_fdata()
    n_components = components_data.shape[3]

    # Validate indices
    for idx in [component_idx_1, component_idx_2]:
        if idx < 0 or idx >= n_components:
            raise ValueError(f"Component index {idx} out of range (0 to {n_components - 1})")

    # Extract both raw component maps
    comp1_data = components_data[..., component_idx_1]
    comp2_data = components_data[..., component_idx_2]

    # Z-score each component before averaging to ensure equal contribution
    # This prevents one component from dominating if it has a larger range
    comp1_std = comp1_data.std()
    comp2_std = comp2_data.std()
    if comp1_std > 0:
        comp1_norm = (comp1_data - comp1_data.mean()) / comp1_std
    else:
        comp1_norm = comp1_data - comp1_data.mean()
    if comp2_std > 0:
        comp2_norm = (comp2_data - comp2_data.mean()) / comp2_std
    else:
        comp2_norm = comp2_data - comp2_data.mean()

    # Average the z-scored component maps
    combined = (comp1_norm + comp2_norm) / 2

    logger.info(
        f"Combining ICA components {component_idx_1} and {component_idx_2} "
        f"(z-scoring, averaging, then thresholding to {threshold} voxels)"
    )

    # Threshold to keep top N positive voxels
    flat = combined.ravel()
    positive_indices = np.where(flat > 0)[0]
    if len(positive_indices) <= threshold:
        thresholded = (flat > 0).astype("uint8").reshape(combined.shape)
    else:
        positive_values = flat[positive_indices]
        top_n = np.argpartition(positive_values, -threshold)[-threshold:]
        result = np.zeros(flat.shape, dtype="uint8")
        result[positive_indices[top_n]] = 1
        thresholded = result.reshape(combined.shape)
    n_voxels = int(thresholded.sum())

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save(Nifti1Image(thresholded, components_img.affine, components_img.header), output_file)
    logger.info(f"Saved combined mask ({n_voxels} voxels) to {output_file}")

    return output_file


def combine_ica_components_unthresholded(
    components_file: Union[str, Path],
    component_idx_1: int,
    component_idx_2: int,
    output_file: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Combine two ICA components into a single unthresholded map.

    Z-scores each component map before averaging to ensure equal contribution
    from both components (regardless of their original value ranges). Used to
    save an unthresholded version alongside the thresholded mask.

    Parameters
    ----------
    components_file : str or Path
        Path to 4D ICA components NIfTI file
    component_idx_1 : int
        Index of first ICA component
    component_idx_2 : int
        Index of second ICA component
    output_file : str or Path
        Path to save combined (unthresholded) map
    overwrite : bool
        Overwrite existing output

    Returns
    -------
    Path
        Path to saved combined map file
    """
    components_file = Path(components_file)
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        logger.info(f"Output exists and overwrite=False, skipping: {output_file}")
        return output_file

    if not components_file.exists():
        raise FileNotFoundError(f"Components file not found: {components_file}")

    # Load ICA components
    components_img = cast(Nifti1Image, load(components_file))
    components_data = components_img.get_fdata()

    # Extract and z-score each component before averaging
    comp1_data = components_data[..., component_idx_1]
    comp2_data = components_data[..., component_idx_2]

    # Z-score to ensure equal contribution from each component
    comp1_std = comp1_data.std()
    comp2_std = comp2_data.std()
    if comp1_std > 0:
        comp1_norm = (comp1_data - comp1_data.mean()) / comp1_std
    else:
        comp1_norm = comp1_data - comp1_data.mean()
    if comp2_std > 0:
        comp2_norm = (comp2_data - comp2_data.mean()) / comp2_std
    else:
        comp2_norm = comp2_data - comp2_data.mean()

    combined = (comp1_norm + comp2_norm) / 2

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save(Nifti1Image(combined, components_img.affine, components_img.header), output_file)
    logger.info(f"Saved unthresholded combined map (z-scored) to {output_file}")

    return output_file


def regenerate_mask_from_selection(
    participant_id: str,
    session_id: str,
    masks_dir: Union[str, Path],
    selection: MaskSelection,
    components_file: Union[str, Path],
    yeo_subject_space: Union[str, Path],
    n_networks: int = 17,
    threshold: int = 2000,
) -> Path:
    """Regenerate a single network mask based on user's selection.

    This function is called after the user modifies mask selections in the
    mask review dialog. It regenerates the mask file based on the user's
    choice (different ICA component or Yeo atlas fallback).

    Parameters
    ----------
    participant_id : str
        Participant ID
    session_id : str
        BIDS session identifier (e.g., 'ses-01')
    masks_dir : str or Path
        Directory containing masks (typically 03_masks/)
    selection : MaskSelection
        The user's selection for this network
    components_file : str or Path
        Path to ICA components file (4D NIfTI)
    yeo_subject_space : str or Path
        Path to Yeo atlas in subject space
    n_networks : int
        Yeo parcellation (7 or 17 networks)
    threshold : int
        Number of voxels to keep in the thresholded mask

    Returns
    -------
    Path
        Path to the regenerated thresholded mask file
    """
    masks_dir = Path(masks_dir)
    network_name = selection.network_name
    bids_prefix = f"{participant_id}_{session_id}"

    # Output files
    network_file = masks_dir / f"{bids_prefix}_{network_name}.nii.gz"
    thr_file = masks_dir / f"{bids_prefix}_{network_name}_thr.nii.gz"

    if selection.source_type in ("yeo_atlas", "mni_atlas"):
        # Use Yeo/MNI atlas mask
        label = "MNI atlas" if selection.source_type == "mni_atlas" else "Yeo atlas (fallback)"
        logger.info(f"Regenerating {network_name} from {label}")
        extract_yeo_mask_for_network(
            yeo_subject_space=yeo_subject_space,
            network_name=network_name,
            output_file=thr_file,
            n_networks=n_networks,
            threshold=threshold,
            overwrite=True,
        )
        # Also save an unthresholded version (full atlas region)
        labels = get_yeo_labels(n_networks=n_networks)
        network_index = labels.index(network_name) + 1
        atlas_img = cast(Nifti1Image, load(yeo_subject_space))
        atlas_data = atlas_img.get_fdata()
        network_mask = (atlas_data == network_index).astype(float)
        save(Nifti1Image(network_mask, atlas_img.affine, atlas_img.header), network_file)

    elif selection.source_type == "ica":
        # Use specified ICA component
        component_idx = selection.component_index
        if component_idx is None:
            raise ValueError(f"ICA selection for {network_name} missing component_index")

        logger.info(f"Regenerating {network_name} from ICA component {component_idx}")

        # Load ICA components
        components_file = Path(components_file)
        if not components_file.exists():
            raise FileNotFoundError(f"Components file not found: {components_file}")

        components_img = cast(Nifti1Image, load(components_file))
        components_data = components_img.get_fdata()

        if component_idx >= components_data.shape[3]:
            raise ValueError(f"Component index {component_idx} out of range (max: {components_data.shape[3] - 1})")

        # Extract component and save unthresholded
        ic_data = components_data[..., component_idx]
        ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header)
        save(ic_img, network_file)

        # Threshold and save (pre-filter to positive voxels, then select top N)
        flat = ic_data.ravel()
        positive_indices = np.where(flat > 0)[0]
        if len(positive_indices) <= threshold:
            thresholded = (flat > 0).astype("uint8").reshape(ic_data.shape)
        else:
            positive_values = flat[positive_indices]
            top_n = np.argpartition(positive_values, -threshold)[-threshold:]
            result = np.zeros(flat.shape, dtype="uint8")
            result[positive_indices[top_n]] = 1
            thresholded = result.reshape(ic_data.shape)
        save(Nifti1Image(thresholded, components_img.affine, components_img.header), thr_file)

        logger.info(f"Saved regenerated mask ({int(thresholded.sum())} voxels) to {thr_file}")

    elif selection.source_type == "ica_combined":
        # Combine two ICA components
        component_idx_1 = selection.component_index
        component_idx_2 = selection.secondary_component_index

        if component_idx_1 is None or component_idx_2 is None:
            raise ValueError(
                f"Combined ICA selection for {network_name} requires both "
                f"component_index ({component_idx_1}) and "
                f"secondary_component_index ({component_idx_2})"
            )

        logger.info(f"Regenerating {network_name} from combined ICA components {component_idx_1} + {component_idx_2}")

        # Save thresholded combined mask
        combine_ica_components(
            components_file=components_file,
            component_idx_1=component_idx_1,
            component_idx_2=component_idx_2,
            output_file=thr_file,
            threshold=threshold,
            overwrite=True,
        )

        # Save unthresholded combined map
        combine_ica_components_unthresholded(
            components_file=components_file,
            component_idx_1=component_idx_1,
            component_idx_2=component_idx_2,
            output_file=network_file,
            overwrite=True,
        )

    else:
        raise ValueError(f"Unknown source_type: {selection.source_type}")

    return thr_file


# =============================================================================
# SECTION 2: ICA DECOMPOSITION
# =============================================================================
# ICA (Independent Component Analysis) identifies spatially independent
# patterns of brain activity from resting-state fMRI. Each component
# represents a coherent network of brain regions that activate together.
#
# Why CanICA?
# -----------
# We use nilearn's CanICA (Canonical ICA), which is designed specifically
# for fMRI data. It includes built-in regularization and handles the high
# dimensionality (many voxels) and temporal structure of fMRI data.
#
# Component Interpretation:
# -------------------------
# - High positive loadings: voxels that are part of the network
# - High negative loadings: anti-correlated voxels (can be included if desired)
# - Near-zero loadings: voxels not part of this network


def run_ica(
    input_file: Union[str, Path],
    mask_file: Union[str, Path],
    output_dir: Union[str, Path],
    output_name: str,
    n_components: int = 30,
    smooth_fwhm: float | None = None,
    highpass: float | None = None,
    lowpass: float | None = None,
    overwrite: bool = False,
    stop_event: threading.Event | None = None,
) -> tuple[Path, Path]:
    """
    Run Canonical ICA (CanICA) on fMRI data using nilearn.

    Parameters
    ----------
    input_file : str or Path
        Path to a NIfTI file
    mask_file : str or Path
        Path to a NIfTI file to use as a mask
    output_dir : str or Path
        Directory to save output files (components and reports)
    output_name : str
        Base name for output file (without extension)
    n_components : int, default = 30
        Number of independent components to extract
    smooth_fwhm : float, optional
        FWHM of Gaussian smoothing kernel in mm for ICA. If None, no smoothing is applied
    highpass : float, optional
        High-pass filter cutoff in Hz for ICA. If None, no high-pass filtering is applied
    lowpass : float, optional
        Low-pass filter cutoff in Hz for ICA. If None, no low-pass filtering is applied
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    stop_event : threading.Event, optional
        Cooperative cancellation signal. When provided, the ICA fit runs in a
        child process so it can be interrupted mid-computation: if the event is
        set, the child (and its joblib workers) is terminated, partial outputs
        are removed, and :class:`MaskExtractionStopped` is raised. When omitted,
        the fit runs inline (no subprocess) and always runs to completion.

    Returns
    -------
    components_file : Path
        Path to NIfTI file containing spatial components (4D)
    metadata_file : Path
        Path to numpy file containing component metadata and loadings

    Raises
    ------
    FileNotFoundError
        If input_files does not exist
    RuntimeError
        If CanICA command fails

    """

    logger.info("CanICA...")

    output_dir = Path(output_dir)
    components_file = output_dir / f"{output_name}_components.nii.gz"
    metadata_file = output_dir / f"{output_name}_metadata.npz"

    outputs = [components_file, metadata_file]
    if not overwrite and all(p.exists() for p in outputs):
        logger.info("All output files exist and overwrite=False. Skipping CanICA.")
        return components_file, metadata_file

    input_file = Path(input_file)
    logger.info("Inputs:")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    logger.info(f"  {str(input_file)}")
    mask_file = Path(mask_file)
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    logger.info(f"  {str(mask_file)}")

    logger.info("Output file:\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  {str(components_file)}")

    logger.info("Parameters:")
    logger.info(f"  n_components: {n_components}")

    img = cast(Nifti1Image, load(input_file))
    # Extract TR from 4th dimension of zooms (temporal spacing)
    tr = float(img.header.get_zooms()[3])

    # Convert nan to None (nilearn needs None to skip smoothing/filtering)
    import math

    if smooth_fwhm is not None and math.isnan(smooth_fwhm):
        smooth_fwhm = None
    if highpass is not None and math.isnan(highpass):
        highpass = None
    if lowpass is not None and math.isnan(lowpass):
        lowpass = None

    # All-picklable args so the fit can run inline or in a spawned child process.
    canica_args = (
        str(input_file),
        str(mask_file),
        n_components,
        smooth_fwhm,
        highpass,
        lowpass,
        tr,
        0,  # random_state — fixed for reproducibility
        str(output_dir / "cache"),
        str(components_file),
        str(metadata_file),
    )

    logger.info("Computing ICA components... (this may take several minutes)")
    with Timer("CanICA decomposition", "mask_extraction"):
        try:
            if stop_event is None:
                # No cancellation requested (CLI / tests): run inline, no subprocess overhead.
                _run_canica(*canica_args)
            else:
                # GUI: run in a killable child process so Stop interrupts the fit itself.
                _run_canica_cancellable(canica_args, stop_event, components_file, metadata_file)
        except MaskExtractionStopped:
            raise
        except Exception as e:
            error_msg = f"CanICA computation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    logger.info("CanICA computation completed successfully")

    # Log component dimensions (reload from disk; the fit may have run in a child process).
    components_data = np.asarray(cast(Nifti1Image, load(components_file)).dataobj)
    logger.info(f"Component dimensions: {components_data.shape}")
    logger.info(f"Number of components extracted: {components_data.shape[3]}")

    return components_file, metadata_file


def spatial_correlation(
    input_file: Union[str, Path],
    parcellation_file: Union[str, Path],
    mask_file: Union[str, Path],
    labels: list,
    output_file: Union[str, Path],
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Spatially correlate a NIfTI file with a parcellation (computed with numpy).

    Parameters
    ----------
    input_file : str or Path
        Path to 4D NIfTI file to compute spatial correlation with
    mask_file : str of Path
        Path to mask for input file
    parcellation_file : str or Path
        Path to 3D NIfTI parcellation file
    labels : List
        List of parcellation labels
    output_file : str or Path
        Path to save correlation results as CSV file
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

     Raises
    ------
    FileNotFoundError
        If components_file does not exist
    ValueError
        If n_components parameter is invalid

    """

    # Log the command
    logger.info("Computing spatial correlation...")

    # Convert to Path objects
    input_file = Path(input_file)
    mask_file = Path(mask_file)
    parcellation_file = Path(parcellation_file)
    from nilearn.image import get_data, resample_to_img

    output_file = Path(output_file)

    # Check if outputs exist and skip if overwrite=False
    if not overwrite and output_file.exists():
        logger.info("Output file exists and overwrite=False. Skipping spatial correlations.")
        corr_df = pd.read_csv(output_file)
        # Drop any auto-created index columns from prior saves
        corr_df = corr_df.loc[:, ~corr_df.columns.str.contains("^Unnamed")]
        return corr_df

    # Validate files
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    if not parcellation_file.exists():
        raise FileNotFoundError(f"File not found: {parcellation_file}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load input file
    logger.info(f"Loading: {input_file}...")
    img = cast(Nifti1Image, load(input_file))
    data = get_data(img)
    n = data.shape[3]

    # Load parcellation and resample (note: they should already be co-registered)
    logger.info(f"Loading: {parcellation_file}...")
    img_parcellation = cast(Nifti1Image, load(parcellation_file))
    img_parcellation_rs = resample_to_img(img_parcellation, img, interpolation="nearest")
    data_parcellation = get_data(img_parcellation_rs).astype(int)

    # Load mask
    logger.info(f"Loading: {mask_file}...")
    img_mask = cast(Nifti1Image, load(mask_file))
    data_mask = get_data(img_mask) > 0
    V = data_mask.sum()

    # Flatten components and labels inside the mask
    X = data[data_mask].reshape(V, n)
    data_parcellation = data_parcellation[data_mask]

    # Demean
    X_dm = X - X.mean(axis=0, keepdims=True)
    x_norm = np.sqrt((X_dm**2).sum(axis=0))

    # Compute spatial correlations
    logger.info("Correlating...")
    with Timer("Spatial correlation (vectorized)", "mask_extraction"):
        # Vectorized approach: compute all network correlations at once
        # Create matrix of all network masks: Y shape (m, V) where m is number of networks
        Y = np.stack([(data_parcellation == j + 1).astype(float) for j in range(len(labels))])

        # Demean all networks at once
        Y_dm = Y - Y.mean(axis=1, keepdims=True)

        # Compute all norms at once
        Y_norm = np.sqrt((Y_dm**2).sum(axis=1))

        # Single matrix multiply: (n, V) @ (V, m) = (n, m)
        num = X_dm.T @ Y_dm.T

        # Broadcast division: (n, 1) * (1, m) = (n, m)
        denom = x_norm[:, None] * Y_norm[None, :]
        # A zero norm means an ICA component has no variance within the mask, or a
        # network has no voxels in the mask -> the correlation is undefined. Report
        # those cases and treat the correlation as 0 so downstream component
        # selection (idxmax) cannot pick a NaN.
        if not np.all(x_norm > 0):
            logger.warning(f"{int(np.sum(x_norm == 0))} ICA component(s) have zero variance within the mask")
        if not np.all(Y_norm > 0):
            zero_nets = [labels[j] for j in np.where(Y_norm == 0)[0]]
            logger.warning(f"Network(s) with no voxels in the mask: {zero_nets}")
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation_matrix = num / denom
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Create DataFrame and save
    logger.info(f"Saving correlations to {output_file}")
    corr_df = pd.DataFrame(correlation_matrix, columns=labels)
    corr_df.to_csv(output_file, index=False)

    return corr_df


# =============================================================================
# SECTION 3: MAIN MASK EXTRACTION PIPELINE
# =============================================================================
# This section contains the high-level pipeline function that orchestrates
# the entire mask extraction process. It combines ICA, registration, and
# network selection into a single, reproducible workflow.


def _register_atlas_to_subject(
    ref_file: Path,
    mnireg_dir: Path,
    network_atlas: str,
    n_networks: int,
    overwrite: bool = False,
    progress_callback: Callable[[str, str], None] | None = None,
    *,
    xfm_prefix: Path | None = None,
) -> tuple[Path, list[str], list[str]]:
    """Register a network atlas from MNI space to participant subject space.

    Computes a nonlinear (SyN) transformation from the participant's BOLD
    reference to MNI152, then warps the atlas (Yeo or Schaefer) backward to
    participant space using the inverse transforms.

    This is a shared helper used by both ``mask_extraction_rest`` (ICA-based)
    and ``mask_extraction_mni`` (atlas-only fallback).

    Parameters
    ----------
    ref_file : Path
        BOLD reference image in participant space (used as registration input)
    mnireg_dir : Path
        Output directory for registration artifacts (typically ``02_mnireg/``)
    network_atlas : str
        Atlas identifier: ``"yeo_7"``, ``"yeo_17"``, ``"schaefer_100_7net"``, etc.
    n_networks : int
        Number of networks in the atlas (7 or 17)
    overwrite : bool
        If True, re-run registration even if outputs exist
    progress_callback : callable, optional
        Callback ``(step, detail)`` for progress updates
    xfm_prefix : Path, optional
        Prefix of a subject-to-MNI transform the caller has already computed
        (for example by :func:`mask_extraction_task`). It is reused as is
        (``overwrite`` then applies only to the warped atlas); the caller owns
        its freshness. By default the transform is computed under
        ``mnireg_dir``.

    Returns
    -------
    atlas_subject_space : Path
        Path to the atlas parcellation warped to participant space
    fwd_transform_files : list[str]
        Forward transforms (subject → MNI)
    inv_transform_files : list[str]
        Inverse transforms (MNI → subject)
    """
    from nilearn import datasets

    def emit(detail: str = ""):
        if detail:
            logger.info(detail)
        if progress_callback:
            progress_callback("MNI registration", detail)

    mnireg_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute transformation from participant space to MNI space
    if xfm_prefix is not None:
        output_prefix, xfm_overwrite = xfm_prefix, False
    else:
        output_prefix, xfm_overwrite = mnireg_dir / f"{ref_file.with_suffix('').stem}_to_MNI_boldref", overwrite
    emit("Computing transforms...")
    _mni_ref_file, sub2mni_fwd_xfm, sub2mni_inv_xfm = compute_xfm_to_mni(
        input_file=ref_file, output_prefix=output_prefix, overwrite=xfm_overwrite
    )

    atlas_display = ATLAS_SOURCES.get(network_atlas, {}).get("display_name", network_atlas)
    emit(f"Transforming {atlas_display} atlas to participant space...")

    # Step 2: Fetch atlas and warp to subject space
    if network_atlas.startswith("yeo"):
        yeo_atlas = datasets.fetch_atlas_yeo_2011(n_networks=n_networks, thickness="thick")
        yeo = yeo_atlas.maps
        yeo_mni = Path(yeo).with_suffix("").parent / Path(
            Path(yeo).with_suffix("").stem + "_interp_to_fMRIPrep_boldref.nii.gz"
        )

        atlas2sub = mnireg_dir / f"Yeo_to_{ref_file.with_suffix('').stem}.nii.gz"
        if not yeo_mni.exists() or overwrite:
            apply_xfm_sform_to_mni(input_file=yeo, output_file=yeo_mni, interp="nearest", overwrite=overwrite)

        if not atlas2sub.exists() or overwrite:
            apply_xfm(
                moving=yeo_mni,
                fixed=ref_file,
                transform_files=sub2mni_inv_xfm,
                output_file=atlas2sub,
                interpolator="nearestNeighbor",
                invert=False,
                overwrite=overwrite,
            )

    elif network_atlas.startswith("schaefer"):
        info = ATLAS_SOURCES[network_atlas]
        schaefer = datasets.fetch_atlas_schaefer_2018(
            n_rois=info["n_rois"],
            yeo_networks=n_networks,
            resolution_mm=2,
        )
        raw_labels = [lbl.decode() if isinstance(lbl, bytes) else str(lbl) for lbl in schaefer.labels]

        schaefer_maps = schaefer.maps
        schaefer_img = (
            schaefer_maps if isinstance(schaefer_maps, nib.Nifti1Image) else cast(Nifti1Image, nib.load(schaefer_maps))
        )
        schaefer_data = schaefer_img.get_fdata()

        network_labels = get_yeo_labels(n_networks)
        network_parcellation = np.zeros_like(schaefer_data)
        for parcel_idx, parcel_label in enumerate(raw_labels):
            parts = parcel_label.split("_")
            if len(parts) >= 3:
                net_name = parts[2]
            else:
                logger.warning(f"Unexpected Schaefer label format: {parcel_label}")
                continue
            try:
                net_idx = network_labels.index(net_name) + 1
            except ValueError:
                logger.warning(f"Network '{net_name}' from label '{parcel_label}' not in network labels")
                continue
            network_parcellation[schaefer_data == parcel_idx + 1] = net_idx

        schaefer_network_mni = mnireg_dir / f"Schaefer_{info['n_rois']}_{n_networks}net_networks_mni.nii.gz"
        if not schaefer_network_mni.exists() or overwrite:
            network_img = Nifti1Image(
                network_parcellation.astype(np.int16),
                schaefer_img.affine,
                schaefer_img.header,
            )
            save(network_img, schaefer_network_mni)

        atlas2sub = mnireg_dir / f"Schaefer_to_{ref_file.with_suffix('').stem}.nii.gz"
        if not atlas2sub.exists() or overwrite:
            apply_xfm(
                moving=schaefer_network_mni,
                fixed=ref_file,
                transform_files=sub2mni_inv_xfm,
                output_file=atlas2sub,
                interpolator="nearestNeighbor",
                invert=False,
                overwrite=overwrite,
            )
    else:
        raise ValueError(
            f"Unsupported network_atlas '{network_atlas}' for atlas registration. Use a Yeo or Schaefer atlas."
        )

    emit("Registration complete")
    return atlas2sub, sub2mni_fwd_xfm, sub2mni_inv_xfm


def mask_extraction_rest(
    participant_id: str,
    session_id: str,
    root_dir: Union[str, Path],
    input_file: Union[str, Path],
    mask_file: Union[str, Path],
    ref_file: Union[str, Path],
    networks: Union[str, list[str]] | None = None,
    network_atlas: str = "yeo_17",
    n_components: int = 30,
    threshold: int = 2000,
    smooth_fwhm: float | None = None,
    highpass: float | None = None,
    lowpass: float | None = None,
    overwrite: bool = False,
    qc_enabled: bool = True,
    qc_dir: Union[str, Path] | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    sham_hemisphere_network: str | None = None,
    stop_event: threading.Event | None = None,
) -> list[Path]:
    """
    Run the mask extraction pipeline for resting-state fMRI data.

    This function extracts participant-specific functional network masks using
    ICA decomposition and spatial correlation with network atlas parcellations.
    Supports Yeo (7/17) and Schaefer (100/400 x 7/17-net) atlases. The approach
    is preferred over using MNI masks directly as it accounts for individual
    anatomical and functional variability.

    Parameters
    ----------
    participant_id : str
        Participant ID
    root_dir : str or Path
        Path to the root directory for all outputs
    input_file : str or Path
        Path to NIfTI file to extract masks from
    mask_file : str or Path
        Path to NIfTI file to mask input_file
    ref_file : str or Path
        Path to NIfTI file to use as a BOLD reference in participant space
    networks : str or List of str
        Network targets to extract masks for (e.g., ["ContA", "DefaultA"])
    network_atlas : str, default="yeo_17"
        Network atlas for ICA component identification. Options:
        ``"yeo_7"``, ``"yeo_17"``, ``"schaefer_100_7net"``,
        ``"schaefer_100_17net"``, ``"schaefer_400_7net"``, ``"schaefer_400_17net"``.
    n_components : int, default=30
        Number of ICA components to extract
    threshold : int, default=2000
        Number of voxels to keep in the thresholded mask
    smooth_fwhm : float, optional
        FWHM of Gaussian smoothing kernel in mm for ICA. If None, no smoothing is applied
    highpass : float, optional
        High-pass filter cutoff in Hz for ICA. If None, no high-pass filtering is applied
    lowpass : float, optional
        Low-pass filter cutoff in Hz for ICA. If None, no low-pass filtering is applied
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    qc_enabled : bool, default=True
        If True, generate QC artifacts (PNG overlays, correlation heatmap, HTML report)
    qc_dir : str or Path, optional
        Custom QC output directory. Defaults to the parent of root_dir (e.g., base_dir/qc)
    progress_callback : callable, optional
        Callback function for progress updates. Called with (step, detail) arguments.
        Used by GUI to display real-time progress in log console.
    sham_hemisphere_network : str, optional
        If provided, generates hemisphere-split masks for this network (e.g., "SomMotA").
        Creates {participant}_{network}_left_thr.nii.gz and {participant}_{network}_right_thr.nii.gz
        files for use in hemisphere-split sham feedback.
    stop_event : threading.Event, optional
        Cooperative cancellation signal. The ICA fit is interruptible
        mid-computation (it runs in a killable subprocess; see :func:`run_ica`),
        and the registration / extraction steps are checked at their boundaries.
        When set, :class:`MaskExtractionStopped` is raised. If omitted, the
        pipeline always runs to completion.

    Returns
    -------
    thr_files
        List of extracted thresholded masks

    Raises
    ------
    FileNotFoundError
        If input_file does not exist
    ValueError
        If network_atlas or networks parameter is invalid

    """

    if networks is None:
        networks = ["ContA", "DefaultA"]

    from nilearn.image import get_data

    # Helper to emit progress to both logger and callback
    def emit_progress(step: str, detail: str = ""):
        if detail:
            logger.info(f"{step}: {detail}")
        else:
            logger.info(step)
        if progress_callback:
            progress_callback(step, detail)

    def check_stop():
        """Abort cooperatively at a step boundary if the caller requested a stop.

        The ICA fit interrupts mid-computation via its own subprocess (see
        ``run_ica``); these boundary checks cover the cheaper ANTs registration
        and mask-extraction steps that run between the ICA and the result.
        """
        if stop_event is not None and stop_event.is_set():
            raise MaskExtractionStopped("Mask extraction stopped by user")

    # Log the command
    emit_progress(f"Mask extraction for {participant_id} / {session_id}")
    bids_prefix = f"{participant_id}_{session_id}"

    # Convert to Path object and validate
    root_dir = Path(root_dir)
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    mask_file = Path(mask_file)
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    ref_file = Path(ref_file)
    if not ref_file.exists():
        raise FileNotFoundError(f"BOLD reference file not found: {ref_file}")

    # network_atlas is authoritative; the network count is derived from it.
    if network_atlas not in ATLAS_SOURCES:
        raise ValueError(f"Unknown network_atlas '{network_atlas}'. Available: {sorted(ATLAS_SOURCES)}")
    n_networks = ATLAS_SOURCES[network_atlas]["n_networks"]

    # Convert to list and validate targets against network labels
    if isinstance(networks, str):
        networks = [networks]
    labels = get_yeo_labels(n_networks=n_networks)
    for target in networks:
        if target not in labels:
            raise ValueError(
                f"Label '{target}' not in labels for {network_atlas} "
                f"({n_networks}-network parcellation). Available: {labels}"
            )

    # Step 1: ICA
    check_stop()
    emit_progress("Step 1/3: Running ICA decomposition", f"{n_components} components")
    ica_dir = root_dir / "01_ica"
    components_file, _ = run_ica(
        input_file=input_file,
        mask_file=mask_file,
        output_dir=ica_dir,
        output_name=input_file.with_suffix("").stem,
        n_components=n_components,
        smooth_fwhm=smooth_fwhm,
        highpass=highpass,
        lowpass=lowpass,
        overwrite=overwrite,
        stop_event=stop_event,
    )
    components_img = cast(Nifti1Image, load(components_file))
    components_data = get_data(components_img)
    emit_progress("Step 1/3", "ICA complete")

    # Step 2: MNI registration - bring Yeo atlas into participant space
    # Overview: We need the Yeo atlas in the same space as our ICA components
    # to perform spatial correlation. This requires: participant → MNI → Yeo → participant
    check_stop()
    emit_progress("Step 2/3: MNI registration", "Computing transforms...")
    mnireg_dir = root_dir / "02_mnireg"

    def _reg_progress(step: str, detail: str = ""):
        emit_progress("Step 2/3", detail)

    atlas2sub, sub2mni_fwd_xfm, sub2mni_inv_xfm = _register_atlas_to_subject(
        ref_file=ref_file,
        mnireg_dir=mnireg_dir,
        network_atlas=network_atlas,
        n_networks=n_networks,
        overwrite=overwrite,
        progress_callback=_reg_progress,
    )

    emit_progress("Step 2/3", "Registration complete")

    # Step 3: Extract network masks using ICA-atlas correlation
    # Now that ICA components and Yeo atlas are in same space, identify which
    # ICA components best represent each target network
    check_stop()
    emit_progress("Step 3/3: Extracting network masks", f"Targets: {', '.join(networks)}")
    masks_dir = root_dir / "03_masks"

    # Step 3a: Compute spatial correlation between ICA components and Yeo networks
    # Each ICA component is correlated with each Yeo network region
    # Result is a component × network matrix showing which components match which networks
    emit_progress("Step 3/3", "Computing spatial correlations...")
    sc_file = masks_dir / Path("network_IC_spatial_correlation.csv")
    sc_df = spatial_correlation(
        input_file=components_file,
        mask_file=mask_file,
        parcellation_file=atlas2sub,
        labels=labels,
        output_file=sc_file,
        overwrite=overwrite,
    )
    chosen_components = {target: int(sc_df[target].idxmax()) for target in networks}

    # Save initial mask selections (auto-selected based on correlation)
    initial_selections: dict[str, MaskSelection] = {}
    for target in networks:
        component_idx = chosen_components[target]
        correlation = float(sc_df[target].iloc[component_idx])
        initial_selections[target] = MaskSelection(
            network_name=target,
            source_type="ica",
            component_index=component_idx,
            correlation=correlation,
            selection_reason="auto",
        )
    save_mask_selections(masks_dir, initial_selections)

    # Step 3b: Select ICA component with highest correlation for each target network
    # For each target network (e.g., DMN), find the ICA component that best matches it
    emit_progress("Step 3/3", "Selecting best ICA components for each network...")
    network_files = []
    for target in networks:
        network_file = masks_dir / Path(f"{bids_prefix}_{target}.nii.gz")
        if not network_file.exists() or overwrite:
            # Find component index with maximum correlation for this network
            ic_ind = chosen_components[target]
            # Extract that component's spatial map as the network mask
            ic_data = components_data[..., ic_ind]
            ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header)
            save(ic_img, network_file)
        else:
            logger.info("Output file exists and overwrite=False. Skipping network extraction.")
        network_files.append(network_file)

    # Step 3c: Threshold masks to retain only top N voxels by activation magnitude
    # Reduces each network mask to its strongest voxels for focused neurofeedback
    emit_progress("Step 3/3", f"Thresholding masks ({threshold} voxels)...")
    network_thr_files = []
    for network_file in network_files:
        thr_file = network_file.with_name(network_file.with_suffix("").stem + "_thr.nii.gz")
        if not thr_file.exists() or overwrite:
            img = cast(Nifti1Image, load(network_file))
            data = img.get_fdata()
            flat = data.ravel()
            # Pre-filter to positive voxels, then select top N
            positive_indices = np.where(flat > 0)[0]
            if len(positive_indices) <= threshold:
                thresholded = (flat > 0).astype("uint8").reshape(data.shape)
            else:
                positive_values = flat[positive_indices]
                top_n = np.argpartition(positive_values, -threshold)[-threshold:]
                result = np.zeros(flat.shape, dtype="uint8")
                result[positive_indices[top_n]] = 1
                thresholded = result.reshape(data.shape)
            save(Nifti1Image(thresholded, img.affine, img.header), thr_file)
        else:
            logger.info("Output file exists and overwrite=False. Skipping network thresholding.")
        network_thr_files.append(thr_file)

    emit_progress("Step 3/3", f"Complete ({len(network_thr_files)} masks extracted)")

    # Step 3d: Generate hemisphere-split masks for sham (if configured)
    # Each hemisphere gets its own threshold count (e.g., 2000 voxels each)
    if sham_hemisphere_network:
        emit_progress("Generating hemisphere-split masks", f"Network: {sham_hemisphere_network}")

        # Check if this network was already extracted (unthresholded)
        network_file = masks_dir / f"{bids_prefix}_{sham_hemisphere_network}.nii.gz"

        if not network_file.exists():
            # Network not in networks, need to extract it first
            # Check if it's a valid Yeo network
            if sham_hemisphere_network not in labels:
                logger.warning(
                    f"Sham hemisphere network '{sham_hemisphere_network}' not in "
                    f"{network_atlas} labels. Skipping hemisphere split."
                )
            else:
                # Extract this network for sham (unthresholded)
                emit_progress("Extracting sham network", sham_hemisphere_network)
                ic_ind = int(sc_df[sham_hemisphere_network].idxmax())
                if not network_file.exists() or overwrite:
                    ic_data = components_data[..., ic_ind]
                    ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header)
                    save(ic_img, network_file)

        # Now split into hemispheres (using unthresholded mask, thresholding each hemisphere)
        if network_file.exists():
            left_file = masks_dir / f"{bids_prefix}_{sham_hemisphere_network}_left_thr.nii.gz"
            right_file = masks_dir / f"{bids_prefix}_{sham_hemisphere_network}_right_thr.nii.gz"
            split_mask_by_hemisphere(network_file, left_file, right_file, threshold=threshold, overwrite=overwrite)
            emit_progress(
                "Hemisphere split", f"Created {left_file.name} and {right_file.name} ({threshold} voxels each)"
            )
        else:
            logger.warning(f"Could not find or create mask for {sham_hemisphere_network}")

    # Generate QC report
    if qc_enabled:
        emit_progress("Generating QC report...")
        qc_output_dir = Path(qc_dir) if qc_dir is not None else root_dir.parent / "qc"

        # Use high-level API to generate comprehensive QC report
        network_names = [str(target) for target in networks]
        report_path = qc.generate_mask_extraction_report(
            mask_extraction_dir=root_dir,
            participant_id=participant_id,
            session_id=session_id,
            network_names=network_names,
            output_dir=qc_output_dir,
            n_networks=n_networks,
            overwrite=overwrite,
            open_browser=False,
            ref_file=ref_file,
        )
        emit_progress("QC report", f"Saved to {report_path.name}")

    return network_thr_files


def mask_extraction_mni(
    participant_id: str,
    session_id: str,
    root_dir: Union[str, Path],
    ref_file: Union[str, Path],
    networks: Union[str, list[str]] | None = None,
    network_atlas: str = "yeo_17",
    threshold: int = 2000,
    overwrite: bool = False,
    qc_enabled: bool = True,
    qc_dir: Union[str, Path] | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    sham_hemisphere_network: str | None = None,
) -> list[Path]:
    """Extract network masks directly from an MNI atlas (no localizer/ICA required).

    This is a fallback pathway that skips the localizer collection, preprocessing,
    and ICA/GLM steps. Instead, it registers a network atlas (Yeo or Schaefer)
    from MNI space to the participant's reference scan and extracts binary masks
    for each target network directly from the warped atlas.

    The resulting masks are less specific than ICA-derived masks (they reflect
    population-average network boundaries rather than individual functional
    topography), but allow a neurofeedback session to proceed when the standard
    localizer pipeline fails or is impractical.

    Output structure matches ``mask_extraction_rest`` exactly, so downstream
    steps (coregistration, feedback) work without modification.

    Parameters
    ----------
    participant_id : str
        Participant ID
    root_dir : str or Path
        Root directory for outputs (typically ``mask_extraction/``)
    ref_file : str or Path
        BOLD reference in participant space. Can be a boldref from preprocessing
        or the mean of a reference scan collection.
    networks : str or list of str, optional
        Target networks to extract (e.g., ``["ContA", "DefaultA"]``).
        Defaults to ``["ContA", "DefaultA"]``.
    network_atlas : str, default="yeo_17"
        Atlas identifier. See ``_register_atlas_to_subject`` for options.
    threshold : int, default=2000
        Number of voxels per mask (passed to ``extract_yeo_mask_for_network``).
    overwrite : bool, default=False
        Re-run even if outputs exist.
    qc_enabled : bool, default=True
        Generate QC overlays and report.
    qc_dir : str or Path, optional
        Custom QC output directory.
    progress_callback : callable, optional
        Callback ``(step, detail)`` for GUI progress updates.
    sham_hemisphere_network : str, optional
        If set, generate hemisphere-split masks for this network.

    Returns
    -------
    list of Path
        Thresholded mask files (``*_thr.nii.gz``)
    """
    if networks is None:
        networks = ["ContA", "DefaultA"]

    # Helper to emit progress to both logger and callback
    def emit_progress(step: str, detail: str = ""):
        if detail:
            logger.info(f"{step}: {detail}")
        else:
            logger.info(step)
        if progress_callback:
            progress_callback(step, detail)

    emit_progress(f"MNI atlas mask extraction for {participant_id} / {session_id}")
    bids_prefix = f"{participant_id}_{session_id}"

    # Convert to Path and validate
    root_dir = Path(root_dir)
    ref_file = Path(ref_file)
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_file}")

    # network_atlas is authoritative; the network count is derived from it.
    if network_atlas not in ATLAS_SOURCES:
        raise ValueError(f"Unknown network_atlas '{network_atlas}'. Available: {sorted(ATLAS_SOURCES)}")
    n_networks = ATLAS_SOURCES[network_atlas]["n_networks"]

    # Validate targets
    if isinstance(networks, str):
        networks = [networks]
    labels = get_yeo_labels(n_networks=n_networks)
    for target in networks:
        if target not in labels:
            raise ValueError(
                f"Label '{target}' not in labels for {network_atlas} "
                f"({n_networks}-network parcellation). Available: {labels}"
            )

    # Step 1/2: Register atlas to subject space
    emit_progress("Step 1/2: MNI registration", "Computing transforms...")
    mnireg_dir = root_dir / "02_mnireg"

    def _reg_progress(step: str, detail: str = ""):
        emit_progress("Step 1/2", detail)

    atlas2sub, _fwd_xfm, inv_xfm = _register_atlas_to_subject(
        ref_file=ref_file,
        mnireg_dir=mnireg_dir,
        network_atlas=network_atlas,
        n_networks=n_networks,
        overwrite=overwrite,
        progress_callback=_reg_progress,
    )

    emit_progress("Step 1/2", "Registration complete")

    # Step 2/2: Extract network masks from atlas
    emit_progress("Step 2/2: Extracting network masks", f"Targets: {', '.join(networks)}")
    masks_dir = root_dir / "03_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    network_thr_files: list[Path] = []
    for target in networks:
        thr_file = masks_dir / f"{bids_prefix}_{target}_thr.nii.gz"
        if not thr_file.exists() or overwrite:
            extract_yeo_mask_for_network(
                yeo_subject_space=atlas2sub,
                network_name=target,
                output_file=thr_file,
                n_networks=n_networks,
                threshold=threshold,
                overwrite=True,
            )
        network_thr_files.append(thr_file)

    emit_progress("Step 2/2", f"Complete ({len(network_thr_files)} masks extracted)")

    # Save mask selections metadata
    selections: dict[str, MaskSelection] = {}
    for target in networks:
        selections[target] = MaskSelection(
            network_name=target,
            source_type="mni_atlas",
            component_index=None,
            correlation=0.0,
            selection_reason="mni_fallback",
        )
    save_mask_selections(masks_dir, selections)

    # Hemisphere-split masks for sham
    if sham_hemisphere_network:
        emit_progress("Generating hemisphere-split masks", f"Network: {sham_hemisphere_network}")

        if sham_hemisphere_network not in labels:
            logger.warning(
                f"Sham hemisphere network '{sham_hemisphere_network}' not in "
                f"{network_atlas} labels. Skipping hemisphere split."
            )
        else:
            # Extract sham network from atlas (unthresholded for splitting)
            sham_network_file = masks_dir / f"{bids_prefix}_{sham_hemisphere_network}.nii.gz"
            network_index = labels.index(sham_hemisphere_network) + 1
            atlas_img = cast(Nifti1Image, load(atlas2sub))
            atlas_data = atlas_img.get_fdata()
            sham_mask = (atlas_data == network_index).astype(float)
            save(Nifti1Image(sham_mask, atlas_img.affine, atlas_img.header), sham_network_file)

            # Also save the thresholded version if not already in targets
            if sham_hemisphere_network not in networks:
                sham_thr_file = masks_dir / f"{bids_prefix}_{sham_hemisphere_network}_thr.nii.gz"
                extract_yeo_mask_for_network(
                    yeo_subject_space=atlas2sub,
                    network_name=sham_hemisphere_network,
                    output_file=sham_thr_file,
                    n_networks=n_networks,
                    threshold=threshold,
                    overwrite=overwrite,
                )

            left_file = masks_dir / f"{bids_prefix}_{sham_hemisphere_network}_left_thr.nii.gz"
            right_file = masks_dir / f"{bids_prefix}_{sham_hemisphere_network}_right_thr.nii.gz"
            split_mask_by_hemisphere(sham_network_file, left_file, right_file, threshold=threshold, overwrite=overwrite)
            emit_progress(
                "Hemisphere split", f"Created {left_file.name} and {right_file.name} ({threshold} voxels each)"
            )

    # Generate QC report
    if qc_enabled:
        emit_progress("Generating QC report...")
        qc_output_dir = Path(qc_dir) if qc_dir is not None else root_dir.parent / "qc"

        network_names = [str(target) for target in networks]
        report_path = qc.generate_mask_extraction_report(
            mask_extraction_dir=root_dir,
            participant_id=participant_id,
            session_id=session_id,
            network_names=network_names,
            output_dir=qc_output_dir,
            n_networks=n_networks,
            overwrite=overwrite,
            open_browser=False,
            ref_file=ref_file,
        )
        emit_progress("QC report", f"Saved to {report_path.name}")

    return network_thr_files


# =============================================================================
# SECTION 3A: TISSUE MASK EXTRACTION (WM/CSF for nuisance regression)
# =============================================================================
# Creates eroded white matter and CSF masks from MNI tissue probability maps.
# These are used as nuisance regressors in the real-time GLM (mean WM/CSF
# signal per volume). No T1 structural scan required — masks are derived from
# MNI template priors warped to participant functional space.


def create_tissue_masks(
    ref_file: Union[str, Path],
    output_dir: Union[str, Path],
    participant_id: str,
    session_id: str,
    inv_transform_files: list[str] | None = None,
    overwrite: bool = False,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Path]:
    """Create eroded WM and CSF masks in participant functional space.

    Uses MNI152NLin2009cAsym tissue probability maps from templateflow,
    warped to participant space via ANTs SyN registration. Masks are
    thresholded at 0.9 probability and eroded by 2 voxels to avoid
    partial volume contamination from gray matter.

    Parameters
    ----------
    ref_file : str or Path
        BOLD reference file in participant space (boldref from preprocessing).
    output_dir : str or Path
        Directory for tissue mask outputs (typically mask_extraction/05_tissue/).
    participant_id : str
        Participant ID for file naming.
    session_id : str
        BIDS session identifier (e.g., 'ses-01') for file naming.
    inv_transform_files : list of str, optional
        Inverse ANTs transform files (MNI → subject direction) from a prior
        ``compute_xfm_to_mni`` call. If provided, the MNI registration step
        is skipped (reuses existing transforms from mask_extraction_rest).
        If None, a fresh SyN registration is computed.
    overwrite : bool
        Whether to overwrite existing outputs.
    progress_callback : callable, optional
        Progress callback ``(step, detail)`` for GUI integration.

    Returns
    -------
    dict with keys ``'wm'`` and ``'csf'``, values are Paths to eroded
    binary masks in participant functional space.

    Raises
    ------
    ValueError
        If resulting masks are empty after erosion.
    """
    from templateflow import api as tflow

    ref_file = Path(ref_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def emit_progress(step: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback(step, detail)

    bids_prefix = f"{participant_id}_{session_id}"
    # Output file paths
    wm_prob_file = output_dir / f"{bids_prefix}_tissue_wm_prob.nii.gz"
    csf_prob_file = output_dir / f"{bids_prefix}_tissue_csf_prob.nii.gz"
    wm_mask_file = output_dir / f"{bids_prefix}_tissue_wm_mask.nii.gz"
    csf_mask_file = output_dir / f"{bids_prefix}_tissue_csf_mask.nii.gz"

    # Check if final outputs exist (skip if not overwriting)
    if not overwrite and wm_mask_file.exists() and csf_mask_file.exists():
        logger.info("Tissue masks already exist, skipping creation")
        return {"wm": wm_mask_file, "csf": csf_mask_file}

    # Step 1: Fetch MNI tissue probability maps from templateflow
    emit_progress("Fetching tissue priors", "MNI152NLin2009cAsym WM/CSF")
    wm_mni = Path(str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="WM", suffix="probseg")))
    csf_mni = Path(str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="CSF", suffix="probseg")))
    logger.info(f"WM prior: {wm_mni.name}")
    logger.info(f"CSF prior: {csf_mni.name}")

    # Step 2: Compute or reuse MNI registration
    if inv_transform_files is None:
        emit_progress("Computing MNI registration", "ANTs SyN (may take 1-2 min)")
        output_prefix = output_dir / f"{ref_file.with_suffix('').stem}_to_MNI_tissue"
        _warped, _fwd_xfm, inv_transform_files = compute_xfm_to_mni(
            input_file=ref_file,
            output_prefix=output_prefix,
            overwrite=overwrite,
        )
        logger.info("Computed fresh MNI registration for tissue masks")
    else:
        logger.info("Reusing existing MNI inverse transforms for tissue masks")

    # Step 3: Warp tissue probability maps MNI -> subject space
    emit_progress("Warping tissue priors", "MNI -> subject space")
    for mni_path, out_path, tissue_name in [
        (wm_mni, wm_prob_file, "WM"),
        (csf_mni, csf_prob_file, "CSF"),
    ]:
        if not overwrite and out_path.exists():
            logger.info(f"{tissue_name} probability map already exists: {out_path.name}")
            continue
        apply_xfm(
            moving=mni_path,
            fixed=ref_file,
            transform_files=inv_transform_files,
            output_file=out_path,
            interpolator="linear",
            overwrite=overwrite,
        )
        logger.info(f"Warped {tissue_name} probability map to subject space")

    # Step 4: Threshold and erode to create binary masks
    emit_progress("Creating binary masks", "Threshold 0.9, adaptive erosion")
    result = {}
    for prob_file, mask_file, tissue_name in [
        (wm_prob_file, wm_mask_file, "wm"),
        (csf_prob_file, csf_mask_file, "csf"),
    ]:
        if not overwrite and mask_file.exists():
            logger.info(f"{tissue_name.upper()} mask already exists: {mask_file.name}")
            result[tissue_name] = mask_file
            continue

        # Load probability map and threshold
        prob_img = cast(Nifti1Image, nib.loadsave.load(prob_file))
        prob_data = np.asarray(prob_img.dataobj)
        binary_data = (prob_data >= 0.9).astype(np.uint8)
        n_pre_erosion = int(binary_data.sum())

        # Save pre-erosion binary for debugging
        binary_img = Nifti1Image(binary_data, prob_img.affine, prob_img.header)
        pre_erosion_file = prob_file.parent / prob_file.name.replace("_prob.", "_binary.")
        nib.loadsave.save(binary_img, str(pre_erosion_file))

        # Adaptive erosion: try 2 voxels, fall back to 1, fall back to none
        # CSF is thin at BOLD resolution and may not survive aggressive erosion
        from pineuro.utils import erode_mask

        n_post_erosion = 0
        erosion_used = 0
        for n_erode in [2, 1]:
            erode_mask(pre_erosion_file, mask_file, n_erode=n_erode, overwrite=True)
            eroded_img = cast(Nifti1Image, nib.loadsave.load(mask_file))
            n_post_erosion = int((np.asarray(eroded_img.dataobj) > 0).sum())
            if n_post_erosion > 0:
                erosion_used = n_erode
                break

        if n_post_erosion == 0:
            # No erosion: use the thresholded binary directly
            import shutil

            shutil.copy2(pre_erosion_file, mask_file)
            n_post_erosion = n_pre_erosion
            erosion_used = 0
            logger.warning(f"{tissue_name.upper()} mask: skipped erosion (too few voxels at BOLD resolution)")

        if n_post_erosion == 0:
            raise ValueError(
                f"{tissue_name.upper()} mask is empty after thresholding "
                f"({n_pre_erosion} voxels). "
                f"The BOLD reference may have poor MNI registration."
            )
        logger.info(f"{tissue_name.upper()} mask: {n_pre_erosion} -> {n_post_erosion} voxels (erosion={erosion_used})")
        result[tissue_name] = mask_file

    emit_progress("Tissue masks complete", f"WM and CSF masks in {output_dir.name}/")
    return result


# =============================================================================
# SECTION 3B: TASK-BASED MASK EXTRACTION (GLM CONTRASTS)
# =============================================================================
# Alternative to ICA-based extraction: uses a task localizer and GLM contrasts
# to identify network-specific voxels. Each contrast map (e.g., "nback - rest")
# is directly thresholded into a binary mask, without Yeo atlas matching.
#
# This approach is more targeted when a suitable task localizer is available,
# since it directly measures task-evoked activation rather than inferring
# networks from resting-state correlations.


# --- Atlas Source Definitions ---
# Atlas metadata lives in atlas_registry.py (lightweight, no heavy deps)
# so the GUI can import it without triggering nibabel/scipy/nilearn.
from pineuro.atlas_registry import ATLAS_SOURCES

# Cache for on-demand atlas label fetching
_ATLAS_LABELS_CACHE: dict[str, list[str]] = {}


def get_atlas_labels(atlas_source: str) -> list[str]:
    """Fetch region/network labels for an atlas from nilearn. Cached after first call.

    Parameters
    ----------
    atlas_source : str
        Key into ``ATLAS_SOURCES`` (e.g., ``"yeo_7"``, ``"schaefer_400_7net"``,
        ``"harvard_oxford"``, ``"aal"``).

    Returns
    -------
    list[str]
        Region or network label names.

    Raises
    ------
    ValueError
        If ``atlas_source`` is not in ``ATLAS_SOURCES``.
    """
    from nilearn import datasets

    if atlas_source in _ATLAS_LABELS_CACHE:
        return _ATLAS_LABELS_CACHE[atlas_source]

    if atlas_source not in ATLAS_SOURCES:
        raise ValueError(f"Unknown atlas_source '{atlas_source}'. Available: {', '.join(sorted(ATLAS_SOURCES.keys()))}")

    if atlas_source.startswith("yeo"):
        n = ATLAS_SOURCES[atlas_source]["n_networks"]
        labels = get_yeo_labels(n)
    elif atlas_source.startswith("schaefer"):
        info = ATLAS_SOURCES[atlas_source]
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=info["n_rois"],
            yeo_networks=info["n_networks"],
            resolution_mm=2,
        )
        labels = [lbl.decode() if isinstance(lbl, bytes) else str(lbl) for lbl in atlas.labels]
    elif atlas_source == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        labels = [lbl for lbl in atlas.labels if lbl != "Background"]
    elif atlas_source == "aal":
        atlas = datasets.fetch_atlas_aal(version="SPM12")
        labels = list(atlas.labels)
    else:
        raise ValueError(f"Unknown atlas_source: {atlas_source}")

    _ATLAS_LABELS_CACHE[atlas_source] = labels
    return labels


def fetch_atlas_mask(
    output_file: Union[str, Path],
    atlas_source: str | None = None,
    regions: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Fetch an atlas from nilearn and extract specified regions as a binary mask.

    Fetches the atlas identified by ``atlas_source`` and extracts the named
    ``regions`` directly.

    Parameters
    ----------
    output_file : str or Path
        Path to save the binary MNI-space mask NIfTI.
    atlas_source : str, optional
        Key into ``ATLAS_SOURCES`` (e.g., ``"harvard_oxford"``, ``"aal"``,
        ``"schaefer_400_7net"``).
    regions : list[str], optional
        Region labels to include in the mask. Required with ``atlas_source``.
    overwrite : bool
        If True, regenerate even if ``output_file`` exists.

    Returns
    -------
    Path
        Path to the saved binary mask in MNI space.

    Raises
    ------
    ValueError
        If ``atlas_source`` + ``regions`` is not provided, or if region labels
        are not found in the atlas.
    """
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        logger.info(f"Atlas mask already exists: {output_file.name}")
        return output_file

    # --- Resolve atlas data and target regions ---
    if atlas_source == "gray_matter":
        # Gray matter probability mask from templateflow (thresholded at 0.5)
        import templateflow.api as tflow

        gm_mni = Path(str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="GM", suffix="probseg")))
        gm_img = cast(Nifti1Image, load(gm_mni))
        gm_data = gm_img.get_fdata()
        binary_mask = (gm_data > 0.5).astype(np.uint8)
        n_voxels = int(binary_mask.sum())
        logger.info(f"Gray matter mask: {n_voxels} voxels in MNI space (threshold > 0.5)")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save(Nifti1Image(binary_mask, gm_img.affine, gm_img.header), output_file)
        return output_file
    elif atlas_source is not None and regions is not None:
        atlas_img, atlas_data, labels, label_indices = _fetch_atlas_and_resolve_regions(atlas_source, regions)
        atlas_label = f"{atlas_source} ({len(regions)} regions)"
    else:
        raise ValueError("atlas_source + regions must be provided")

    # Create binary mask: 1 where atlas label is in target regions, 0 elsewhere
    binary_mask = np.isin(atlas_data, label_indices).astype(np.uint8)
    n_voxels = int(binary_mask.sum())
    logger.info(f"Atlas '{atlas_label}': {len(label_indices)} regions, {n_voxels} voxels in MNI space")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save(Nifti1Image(binary_mask, atlas_img.affine, atlas_img.header), output_file)
    return output_file


def _fetch_atlas_and_resolve_regions(
    atlas_source: str,
    target_regions: list[str],
) -> tuple[nib.Nifti1Image, np.ndarray, list, list[int]]:
    """Fetch an atlas image and resolve region names to integer label indices.

    Parameters
    ----------
    atlas_source : str
        Atlas source key (e.g., ``"harvard_oxford"``, ``"aal"``, ``"schaefer_400_7net"``).
    target_regions : list[str]
        Region label names to extract.

    Returns
    -------
    tuple of (atlas_img, atlas_data, labels, label_indices)
        - atlas_img: the NIfTI image
        - atlas_data: numpy array of integer labels
        - labels: all label names from the atlas
        - label_indices: integer values in atlas_data corresponding to target_regions
    """
    from nilearn import datasets

    if atlas_source == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_maps = atlas.maps
        atlas_img = atlas_maps if isinstance(atlas_maps, nib.Nifti1Image) else cast(Nifti1Image, nib.load(atlas_maps))
        atlas_data = atlas_img.get_fdata()
        labels = list(atlas.labels)
        # Harvard-Oxford: label index = voxel value
        label_indices = []
        for region in target_regions:
            try:
                idx = labels.index(region)
                label_indices.append(idx)
            except ValueError:
                raise ValueError(
                    f"Region '{region}' not found in Harvard-Oxford labels. "
                    f"Available: {[lbl for lbl in labels if lbl != 'Background']}"
                ) from None

    elif atlas_source == "aal":
        atlas = datasets.fetch_atlas_aal(version="SPM12")
        atlas_maps = atlas.maps
        atlas_img = atlas_maps if isinstance(atlas_maps, nib.Nifti1Image) else cast(Nifti1Image, nib.load(atlas_maps))
        atlas_data = atlas_img.get_fdata()
        labels = list(atlas.labels)
        indices = [int(i) for i in atlas.indices]
        # AAL: uses non-sequential atlas.indices for label→voxel mapping
        label_indices = []
        for region in target_regions:
            try:
                label_pos = labels.index(region)
                label_indices.append(indices[label_pos])
            except ValueError:
                raise ValueError(f"Region '{region}' not found in AAL labels. Available: {labels}") from None

    elif atlas_source.startswith("schaefer"):
        info = ATLAS_SOURCES[atlas_source]
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=info["n_rois"],
            yeo_networks=info["n_networks"],
            resolution_mm=2,
        )
        atlas_maps = atlas.maps
        atlas_img = atlas_maps if isinstance(atlas_maps, nib.Nifti1Image) else cast(Nifti1Image, nib.load(atlas_maps))
        atlas_data = atlas_img.get_fdata()
        raw_labels = [lbl.decode() if isinstance(lbl, bytes) else str(lbl) for lbl in atlas.labels]
        labels = raw_labels
        # Schaefer: 1-indexed (labels[0] → voxel value 1)
        label_indices = []
        for region in target_regions:
            try:
                label_pos = raw_labels.index(region)
                label_indices.append(label_pos + 1)
            except ValueError:
                raise ValueError(
                    f"Region '{region}' not found in Schaefer labels. Available (first 10): {raw_labels[:10]}..."
                ) from None

    else:
        raise ValueError(f"Unsupported atlas_source for mask extraction: {atlas_source}")

    return atlas_img, atlas_data, labels, label_indices


def cluster_threshold_stat_map(
    stat_data: np.ndarray,
    atlas_data: np.ndarray,
    target_n_voxels: int,
    start_threshold: float = 2.0,
    step: float = 0.1,
    max_iterations: int = 200,
    return_all_clusters: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find cluster(s) within an atlas-masked stat map.

    Iteratively raises the statistical threshold until the selected cluster(s)
    within the atlas region fit within ``target_n_voxels``. This mirrors the
    approach used in FSL-based STG extraction pipelines where ``cluster`` is
    called repeatedly with increasing thresholds.

    When ``return_all_clusters=False`` (default), only the single largest
    contiguous cluster is returned. When ``True``, all clusters above the
    final threshold are returned combined — suitable for whole-brain atlases
    (e.g. gray matter) where multiple disconnected activation foci are expected.
    The stopping criterion also changes: with ``return_all_clusters=False`` the
    loop stops when the *largest* cluster fits; with ``True`` it stops when the
    *total* suprathreshold voxel count fits.

    Parameters
    ----------
    stat_data : np.ndarray
        3D array of statistic values (t or z scores).
    atlas_data : np.ndarray
        3D binary array (0 or 1) defining the anatomical ROI.
        Must have the same shape as ``stat_data``.
    target_n_voxels : int
        Desired maximum cluster size.
    start_threshold : float
        Starting stat threshold for the iterative search.
    step : float
        Amount to increase threshold per iteration.
    max_iterations : int
        Safety limit to prevent infinite loops.
    return_all_clusters : bool
        If False (default), return only the single largest contiguous cluster.
        If True, return all clusters above the final threshold combined.
        Use True for whole-brain atlases (e.g. gray matter).

    Returns
    -------
    binary_mask : np.ndarray
        3D uint8 array with 1 for the selected cluster, 0 elsewhere.
    info : dict
        Metadata about the thresholding process:

        - ``final_threshold`` : float — threshold that produced the cluster
        - ``n_iterations`` : int — number of iterations performed
        - ``n_clusters_at_final`` : int — clusters at the final threshold
        - ``cluster_size`` : int — voxels in the selected cluster
        - ``atlas_voxel_count`` : int — total voxels in the atlas mask
        - ``outcome`` : str — one of ``"success"``,
          ``"already_under_target"``, ``"no_clusters_found"``,
          ``"empty_atlas"``, ``"max_iterations_reached"``

    Raises
    ------
    ValueError
        If ``stat_data`` and ``atlas_data`` have different shapes.
    """
    if stat_data.shape != atlas_data.shape:
        raise ValueError(f"Shape mismatch: stat_data {stat_data.shape} vs atlas_data {atlas_data.shape}")

    # Restrict stat map to atlas region
    masked_stat = stat_data * atlas_data
    atlas_voxel_count = int(atlas_data.sum())

    if atlas_voxel_count == 0:
        return np.zeros(stat_data.shape, dtype=np.uint8), {
            "outcome": "empty_atlas",
            "final_threshold": start_threshold,
            "n_iterations": 0,
            "n_clusters_at_final": 0,
            "cluster_size": 0,
            "atlas_voxel_count": 0,
        }

    threshold = start_threshold
    _prev_labeled = None
    _prev_n_clusters = 0
    labeled = None
    largest_label = 0
    largest_size = 0
    n_clusters = 0

    for iteration in range(max_iterations):
        above_threshold = masked_stat >= threshold

        if not above_threshold.any():
            if iteration == 0:
                return np.zeros(stat_data.shape, dtype=np.uint8), {
                    "outcome": "no_clusters_found",
                    "final_threshold": threshold,
                    "n_iterations": iteration + 1,
                    "n_clusters_at_final": 0,
                    "cluster_size": 0,
                    "atlas_voxel_count": atlas_voxel_count,
                }
            # Step back to previous threshold where clusters existed
            threshold -= step
            above_threshold = masked_stat >= threshold
            labeled, n_clusters = ndimage_label(above_threshold.astype(np.int32))
            cluster_sizes = np.bincount(labeled.ravel())[1:]
            if cluster_sizes.size == 0:
                # Defensive: the stepped-back threshold yielded no clusters either.
                return np.zeros(stat_data.shape, dtype=np.uint8), {
                    "outcome": "no_clusters_found",
                    "final_threshold": float(threshold),
                    "n_iterations": iteration + 1,
                    "n_clusters_at_final": 0,
                    "cluster_size": 0,
                    "atlas_voxel_count": atlas_voxel_count,
                }
            largest_label = int(cluster_sizes.argmax()) + 1
            largest_size = int(cluster_sizes.max())
            if return_all_clusters:
                binary_mask = (labeled > 0).astype(np.uint8)
                selected_size = int(binary_mask.sum())
            else:
                binary_mask = (labeled == largest_label).astype(np.uint8)
                selected_size = largest_size
            return binary_mask, {
                "outcome": "success",
                "final_threshold": float(threshold),
                "n_iterations": iteration + 1,
                "n_clusters_at_final": n_clusters,
                "cluster_size": selected_size,
                "atlas_voxel_count": atlas_voxel_count,
            }

        labeled, n_clusters = ndimage_label(above_threshold.astype(np.int32))

        if n_clusters == 0:
            return np.zeros(stat_data.shape, dtype=np.uint8), {
                "outcome": "no_clusters_found",
                "final_threshold": float(threshold),
                "n_iterations": iteration + 1,
                "n_clusters_at_final": 0,
                "cluster_size": 0,
                "atlas_voxel_count": atlas_voxel_count,
            }

        cluster_sizes = np.bincount(labeled.ravel())[1:]  # skip background
        largest_label = int(cluster_sizes.argmax()) + 1  # 1-based
        largest_size = int(cluster_sizes.max())
        total_size = int(above_threshold.sum())

        size_fits = total_size <= target_n_voxels if return_all_clusters else largest_size <= target_n_voxels

        if size_fits:
            if return_all_clusters:
                binary_mask = (labeled > 0).astype(np.uint8)
                selected_size = total_size
            else:
                binary_mask = (labeled == largest_label).astype(np.uint8)
                selected_size = largest_size
            return binary_mask, {
                "outcome": "success" if iteration > 0 else "already_under_target",
                "final_threshold": float(threshold),
                "n_iterations": iteration + 1,
                "n_clusters_at_final": n_clusters,
                "cluster_size": selected_size,
                "atlas_voxel_count": atlas_voxel_count,
            }

        # Too large — increase threshold
        _prev_labeled = labeled
        _prev_n_clusters = n_clusters
        threshold += step

    # Max iterations reached — return current cluster(s)
    if labeled is None:
        return np.zeros(stat_data.shape, dtype=np.uint8), {
            "outcome": "no_iterations",
            "final_threshold": float(threshold),
            "n_iterations": 0,
            "n_clusters_at_final": 0,
            "cluster_size": 0,
            "atlas_voxel_count": atlas_voxel_count,
        }
    if return_all_clusters:
        binary_mask = (labeled > 0).astype(np.uint8)
        selected_size = int(binary_mask.sum())
    else:
        binary_mask = (labeled == largest_label).astype(np.uint8)
        selected_size = largest_size
    return binary_mask, {
        "outcome": "max_iterations_reached",
        "final_threshold": float(threshold),
        "n_iterations": max_iterations,
        "n_clusters_at_final": n_clusters,
        "cluster_size": selected_size,
        "atlas_voxel_count": atlas_voxel_count,
    }


def cluster_extent_threshold(
    stat_data: np.ndarray,
    stat_threshold: float = 2.0,
    min_cluster_size: int = 50,
    max_total_voxels: int | None = None,
    atlas_data: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Threshold a stat map and keep clusters above a minimum size.

    Classic fMRI cluster-extent thresholding: apply a fixed stat threshold,
    identify contiguous clusters, and discard clusters smaller than
    ``min_cluster_size``. Optionally restrict to a spatial mask (e.g., grey
    matter) before clustering.

    Parameters
    ----------
    stat_data : np.ndarray
        3D array of statistic values (t or z scores).
    stat_threshold : float
        Stat value cutoff. Voxels with values >= this are candidates.
    min_cluster_size : int
        Minimum number of contiguous voxels for a cluster to survive.
    max_total_voxels : int or None
        Optional safety cap. If the total surviving voxels exceed this, the
        highest-stat voxels within the surviving clusters are kept up to the
        cap (top-N by stat value). Must be a positive integer. ``None``
        disables the cap.
    atlas_data : np.ndarray or None
        Optional 3D binary mask (same shape as ``stat_data``). When provided,
        the stat map is intersected with this mask before clustering. Useful
        for restricting to grey matter.

    Returns
    -------
    binary_mask : np.ndarray
        3D uint8 array with 1 for surviving voxels, 0 elsewhere.
    info : dict
        Metadata about the thresholding:

        - ``outcome`` : str — ``"success"``, ``"no_suprathreshold_voxels"``,
          ``"no_clusters_survive"``, or ``"capped"``
        - ``stat_threshold`` : float
        - ``min_cluster_size`` : int
        - ``n_clusters_input`` : int — clusters before size filtering
        - ``largest_cluster_input`` : int — size of the largest cluster before
          size filtering (0 if none)
        - ``n_clusters_surviving`` : int — clusters after size filtering
        - ``cluster_sizes`` : list[int] — sizes of surviving clusters (descending)
        - ``total_voxels`` : int — total voxels in the output mask
        - ``atlas_voxel_count`` : int or None — voxels in atlas mask (if used)
        - ``capped`` : bool — whether ``max_total_voxels`` was applied

    Raises
    ------
    ValueError
        If ``stat_data`` and ``atlas_data`` have different shapes.
    """
    if atlas_data is not None and stat_data.shape != atlas_data.shape:
        raise ValueError(f"Shape mismatch: stat_data {stat_data.shape} vs atlas_data {atlas_data.shape}")
    if max_total_voxels is not None and max_total_voxels <= 0:
        raise ValueError(f"max_total_voxels must be a positive integer or None, got {max_total_voxels}")

    atlas_voxel_count = int(atlas_data.sum()) if atlas_data is not None else None

    # Apply stat threshold
    above = stat_data >= stat_threshold

    # Optionally restrict to atlas region
    if atlas_data is not None:
        above = above & (atlas_data > 0.5)

    if not above.any():
        return np.zeros(stat_data.shape, dtype=np.uint8), {
            "outcome": "no_suprathreshold_voxels",
            "stat_threshold": stat_threshold,
            "min_cluster_size": min_cluster_size,
            "n_clusters_input": 0,
            "largest_cluster_input": 0,
            "n_clusters_surviving": 0,
            "cluster_sizes": [],
            "total_voxels": 0,
            "atlas_voxel_count": atlas_voxel_count,
            "capped": False,
        }

    # Label contiguous clusters
    labeled, n_clusters = ndimage_label(above.astype(np.int32))
    sizes = np.bincount(labeled.ravel())[1:]  # skip background (label 0)
    largest_input = int(sizes.max()) if sizes.size else 0

    # Filter by minimum cluster size
    surviving_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_cluster_size]

    if not surviving_labels:
        return np.zeros(stat_data.shape, dtype=np.uint8), {
            "outcome": "no_clusters_survive",
            "stat_threshold": stat_threshold,
            "min_cluster_size": min_cluster_size,
            "n_clusters_input": n_clusters,
            "largest_cluster_input": largest_input,
            "n_clusters_surviving": 0,
            "cluster_sizes": [],
            "total_voxels": 0,
            "atlas_voxel_count": atlas_voxel_count,
            "capped": False,
        }

    # Candidate region: all voxels in clusters that passed the size filter
    surviving_region = np.isin(labeled, surviving_labels)

    # Apply max_total_voxels cap by keeping the highest-stat voxels (top-N).
    # Keeping whole clusters largest-first would discard a single cluster that
    # alone exceeds the cap, emptying the mask; top-N respects the cap exactly.
    capped = False
    n_surviving = int(surviving_region.sum())
    if max_total_voxels is not None and n_surviving > max_total_voxels:
        capped = True
        flat_stat = stat_data.ravel()
        region_idx = np.flatnonzero(surviving_region.ravel())
        region_vals = flat_stat[region_idx]
        top = np.argpartition(region_vals, -max_total_voxels)[-max_total_voxels:]
        keep = np.zeros(flat_stat.shape, dtype=bool)
        keep[region_idx[top]] = True
        binary_mask = keep.reshape(stat_data.shape).astype(np.uint8)
    else:
        binary_mask = surviving_region.astype(np.uint8)

    total_voxels = int(binary_mask.sum())

    # Recompute cluster stats on the FINAL mask so reported numbers match the
    # saved output (top-N selection can split a cluster).
    final_labeled, _ = ndimage_label(binary_mask)
    final_sizes = sorted(np.bincount(final_labeled.ravel())[1:].tolist(), reverse=True)

    outcome = "capped" if capped else "success"
    return binary_mask, {
        "outcome": outcome,
        "stat_threshold": stat_threshold,
        "min_cluster_size": min_cluster_size,
        "n_clusters_input": n_clusters,
        "largest_cluster_input": largest_input,
        "n_clusters_surviving": len(final_sizes),
        "cluster_sizes": final_sizes,
        "total_voxels": total_voxels,
        "atlas_voxel_count": atlas_voxel_count,
        "capped": capped,
    }


def compute_cluster_table(
    binary_mask: np.ndarray,
    stat_data: np.ndarray,
    affine: np.ndarray,
    atlas_data: np.ndarray | None = None,
    atlas_labels: list[str] | None = None,
    max_clusters: int = 50,
) -> list[dict[str, Any]]:
    """Identify connected clusters in a binary mask and label them.

    Parameters
    ----------
    binary_mask : np.ndarray
        3D binary mask (0/1).
    stat_data : np.ndarray
        3D stat map (same shape) for peak values.
    affine : np.ndarray
        4×4 NIfTI affine for coordinate conversion.
    atlas_data : np.ndarray, optional
        Integer-labeled parcellation in same space (0 = background).
    atlas_labels : list[str], optional
        Region names indexed by atlas integer values. Index 0 is background.
    max_clusters : int
        Maximum clusters to return (sorted by size descending).

    Returns
    -------
    list[dict]
        Per-cluster info: cluster_id, n_voxels, peak_stat, peak_ijk,
        peak_xyz, centroid_xyz, anatomical_label, label_overlap.
    """
    labeled, n_clusters = ndimage_label(binary_mask.astype(np.int32))
    if n_clusters == 0:
        return []

    clusters = []
    for cluster_id in range(1, n_clusters + 1):
        mask_c = labeled == cluster_id
        n_voxels = int(mask_c.sum())

        # Peak stat within this cluster
        stat_in_cluster = stat_data * mask_c
        peak_flat = int(np.argmax(stat_in_cluster))
        peak_ijk = np.unravel_index(peak_flat, stat_data.shape)
        peak_stat = float(stat_data[peak_ijk])

        # Centroid
        coords = np.argwhere(mask_c)
        centroid_ijk = tuple(float(round(v, 1)) for v in coords.mean(axis=0))

        # Convert to mm via affine
        peak_xyz = tuple(float(round(v, 1)) for v in (affine @ [*peak_ijk, 1])[:3])
        centroid_xyz = tuple(float(round(v, 1)) for v in (affine @ [*centroid_ijk, 1])[:3])

        entry = {
            "cluster_id": cluster_id,
            "n_voxels": n_voxels,
            "peak_stat": round(peak_stat, 3),
            "peak_ijk": list(peak_ijk),
            "peak_xyz": list(peak_xyz),
            "centroid_xyz": list(centroid_xyz),
        }

        # Anatomical labeling via atlas overlap
        if atlas_data is not None and atlas_labels is not None:
            atlas_vals = atlas_data[mask_c].astype(int)
            overlap: dict[str, int] = {}
            for val in np.unique(atlas_vals):
                if val == 0:
                    continue  # skip background
                if val < len(atlas_labels):
                    label = atlas_labels[val]
                else:
                    label = f"region_{val}"
                overlap[label] = int((atlas_vals == val).sum())
            if overlap:
                entry["anatomical_label"] = max(overlap, key=lambda k: overlap[k])
                entry["label_overlap"] = overlap
            else:
                entry["anatomical_label"] = "Unlabeled"
                entry["label_overlap"] = {}
        else:
            entry["anatomical_label"] = ""
            entry["label_overlap"] = {}

        clusters.append(entry)

    # Sort by size descending, limit
    clusters.sort(key=lambda c: c["n_voxels"], reverse=True)  # type: ignore[arg-type, return-value]
    # Re-number after sorting
    for i, c in enumerate(clusters[:max_clusters]):
        c["cluster_id"] = i + 1
    return clusters[:max_clusters]


@dataclass
class TaskMaskSelection:
    """Tracks the selection state for a single task-based GLM mask.

    Attributes
    ----------
    mask_name : str
        User-assigned name for this mask (e.g., "CEN", "DMN")
    source_type : str
        ``"glm_contrast"`` for a mask thresholded from the contrast map. A mask
        replaced at review is ``"atlas_network"`` (a network warped from MNI),
        ``"mni_file"`` (a user mask warped from MNI), ``"native_file"`` (a user
        mask already in the participant's BOLD reference space, e.g. an edited
        copy of the mask), or ``"atlas_roi"`` (the localizer's search atlas itself)
    contrast_definition : str
        The contrast expression (e.g., "nback - rest")
    stat_type : str
        "t" or "z" - the type of statistical map used
    peak_stat : float
        Peak statistic value in the unthresholded map
    n_voxels_above_zero : int
        Number of positive voxels before thresholding
    n_voxels_thresholded : int
        Number of voxels in the final mask
    threshold_method : str
        How the mask was thresholded (``"top_n"``, ``"cluster"``, or
        ``"cluster_extent"``)
    selection_reason : str
        ``"auto"`` for the study's own thresholding. Operator changes at review
        are ``"operator_rethreshold"`` (new threshold parameters),
        ``"operator_exclude_clusters"`` (clusters left out of the mask),
        ``"fallback_configured"`` (the study's fallback mask),
        ``"operator_replace"`` (a network or file chosen at review), or
        ``"atlas_roi"`` (the search atlas)
    atlas_used : str or None
        Atlas name or path used for cluster thresholding, if applicable
    final_cluster_threshold : float or None
        Stat threshold that produced the final cluster (cluster method only)
    n_clusters_found : int or None
        Number of clusters at the final threshold (cluster method only)
    cluster_outcome : str or None
        Outcome of cluster thresholding (cluster method only)
    cluster_table : list of dict or None
        Per-cluster breakdown: size, peak stat, coordinates, anatomical label
    min_cluster_size : int or None
        Minimum cluster size used (cluster_extent method only)
    max_total_voxels : int or None
        Maximum total voxels cap used (cluster_extent method only)
    n_clusters_input : int or None
        Clusters at the cluster-forming threshold before the size filter
        (cluster_extent method only)
    largest_cluster_input : int or None
        Size of the largest of those clusters (cluster_extent method only)
    threshold_params : dict or None
        The thresholding parameters actually used for this mask (None for a
        replaced mask)
    source : str or None
        What a replaced mask came from, e.g. ``"yeo_17:DefaultA"`` or the path
        of an MNI-space mask file
    excluded_clusters : list of dict or None
        Clusters an operator left out of the mask: ``n_voxels``, ``peak_stat``
        and ``peak_ijk`` of each (None when no exclusion was requested)
    """

    mask_name: str
    source_type: str = "glm_contrast"
    contrast_definition: str = ""
    stat_type: str = "t"
    peak_stat: float = 0.0
    n_voxels_above_zero: int = 0
    n_voxels_thresholded: int = 0
    threshold_method: str = "top_n"
    selection_reason: str = "auto"
    atlas_used: str | None = None
    final_cluster_threshold: float | None = None
    n_clusters_found: int | None = None
    cluster_outcome: str | None = None
    cluster_table: list[dict[str, Any]] | None = None
    min_cluster_size: int | None = None
    max_total_voxels: int | None = None
    n_clusters_input: int | None = None
    largest_cluster_input: int | None = None
    threshold_params: dict[str, Any] | None = None
    source: str | None = None
    excluded_clusters: list[dict[str, Any]] | None = None


def save_task_mask_selections(masks_dir: Union[str, Path], selections: dict[str, TaskMaskSelection]) -> Path:
    """Save task-based mask selections to JSON file.

    Parameters
    ----------
    masks_dir : str or Path
        Directory containing the masks (typically 03_masks/)
    selections : dict
        Dictionary mapping mask names to TaskMaskSelection objects

    Returns
    -------
    Path
        Path to the saved JSON file
    """
    masks_dir = Path(masks_dir)
    output_file = masks_dir / "mask_selections.json"

    data = {"type": "task", "selections": {name: asdict(sel) for name, sel in selections.items()}}

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)

    logger.info(f"Saved task mask selections to {output_file}")
    return output_file


def load_task_mask_selections(masks_dir: Union[str, Path]) -> dict[str, TaskMaskSelection]:
    """Load task-based mask selections written by :func:`save_task_mask_selections`.

    Returns an empty dict when the file is missing or is not a task-format file.
    Unknown keys are ignored, so a file written by a newer pineuro still loads.
    """
    selections_file = Path(masks_dir) / "mask_selections.json"
    if not selections_file.exists():
        return {}
    with open(selections_file) as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("type") != "task":
        return {}
    known = {f.name for f in fields(TaskMaskSelection)}
    return {
        name: TaskMaskSelection(**{k: v for k, v in entry.items() if k in known})
        for name, entry in data.get("selections", {}).items()
    }


MASK_OVERRIDES_FILENAME = "mask_overrides.json"


def load_mask_overrides(root_dir: Union[str, Path]) -> dict[str, dict[str, Any]]:
    """Load the per-participant task-mask overrides saved at review.

    Parameters
    ----------
    root_dir : str or Path
        The localizer's mask-extraction root (the ``root_dir`` passed to
        :func:`mask_extraction_task`).

    Returns
    -------
    dict
        ``{mask_name: override}`` in the ``mask_overrides`` format of
        :func:`mask_extraction_task`; empty when none are saved.
    """
    overrides_file = Path(root_dir) / MASK_OVERRIDES_FILENAME
    if not overrides_file.exists():
        return {}
    with open(overrides_file) as f:
        data = json.load(f)
    return dict(data) if isinstance(data, dict) else {}


def fallback_override(spec: dict[str, str]) -> dict[str, Any]:
    """The ``mask_overrides`` entry that applies a study's configured fallback mask.

    ``spec`` is an ``extraction.fallback_masks`` value: ``{"atlas", "network"}``
    or ``{"file"}``.
    """
    if "file" in spec:
        return {"kind": "mni_file", "file": spec["file"], "selection_reason": "fallback_configured"}
    return {
        "kind": "atlas_network",
        "atlas": spec["atlas"],
        "network": spec["network"],
        "selection_reason": "fallback_configured",
    }


def save_mask_overrides(root_dir: Union[str, Path], overrides: dict[str, dict[str, Any]]) -> Path:
    """Save the per-participant task-mask overrides; an empty dict removes the file."""
    overrides_file = Path(root_dir) / MASK_OVERRIDES_FILENAME
    if not overrides:
        overrides_file.unlink(missing_ok=True)
        return overrides_file
    overrides_file.parent.mkdir(parents=True, exist_ok=True)
    with open(overrides_file, "w") as f:
        json.dump(overrides, f, indent=2, default=_json_default)
    logger.info(f"Saved mask overrides to {overrides_file}")
    return overrides_file


# Thresholding parameters each method uses, with the mask_extraction_task defaults.
_METHOD_PARAM_DEFAULTS: dict[str, Any] = {
    "threshold": 2000,
    "cluster_start_threshold": 2.0,
    "cluster_step": 0.1,
    "cluster_return_all": False,
    "min_cluster_size": 50,
    "max_total_voxels": None,
}

# Fields a "rethreshold" override may change, per thresholding method. The method
# itself cannot be changed per mask.
_RETHRESHOLD_FIELDS: dict[str, tuple[str, ...]] = {
    "top_n": ("threshold",),
    "cluster": ("threshold", "cluster_start_threshold", "cluster_step", "cluster_return_all"),
    "cluster_extent": ("cluster_start_threshold", "min_cluster_size", "max_total_voxels"),
}

# Override kind -> selection_reason recorded when the override does not name one.
_OVERRIDE_DEFAULT_REASONS: dict[str, str] = {
    "rethreshold": "operator_rethreshold",
    "exclude_clusters": "operator_exclude_clusters",
    "atlas_network": "operator_replace",
    "mni_file": "operator_replace",
    "native_file": "operator_replace",
    "atlas_roi": "atlas_roi",
}


def _atlas_qc_label(atlas_mask: Union[str, Path] | None, atlas_source: str | None) -> str:
    """Label used in the ``03_masks/{prefix}_atlas_{label}.nii.gz`` copy of the search atlas."""
    return Path(atlas_mask).name if atlas_mask is not None else str(atlas_source)


def threshold_task_stat_map(
    stat_data: np.ndarray,
    *,
    threshold_method: str,
    threshold: int = 2000,
    cluster_start_threshold: float = 2.0,
    cluster_step: float = 0.1,
    cluster_return_all: bool = False,
    min_cluster_size: int = 50,
    max_total_voxels: int | None = None,
    atlas_data: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Threshold one GLM contrast map into a binary mask.

    This is the thresholding step of :func:`mask_extraction_task`, shared with
    :func:`update_task_mask` so that a mask re-thresholded at review is computed
    exactly as the pipeline computes it. See :func:`mask_extraction_task` for
    the methods and parameters.

    Returns
    -------
    binary_mask : np.ndarray
        3D uint8 mask.
    info : dict
        ``outcome`` (None for top_n), ``n_voxels``, ``n_positive``,
        ``peak_stat``, ``final_cluster_threshold``, ``n_clusters_found``,
        ``n_clusters_input``, ``largest_cluster_input`` and
        ``threshold_params`` (the method and parameter values used).
    """
    positive_mask = stat_data > 0
    n_positive = int(positive_mask.sum())
    peak_stat = float(np.nanmax(stat_data)) if n_positive > 0 else 0.0
    info: dict[str, Any] = {
        "outcome": None,
        "n_positive": n_positive,
        "peak_stat": peak_stat,
        "final_cluster_threshold": None,
        "n_clusters_found": None,
        "n_clusters_input": None,
        "largest_cluster_input": None,
    }

    if threshold_method == "top_n":
        if n_positive == 0:
            binary_mask = np.zeros(stat_data.shape, dtype=np.uint8)
        elif n_positive <= threshold:
            binary_mask = positive_mask.astype(np.uint8)
        else:
            # Standard case: threshold to top N positive voxels
            flat = stat_data.ravel()
            positive_indices = np.where(flat > 0)[0]
            positive_values = flat[positive_indices]
            top_n = np.argpartition(positive_values, -threshold)[-threshold:]
            result = np.zeros(flat.shape, dtype=bool)
            result[positive_indices[top_n]] = True
            binary_mask = result.astype(np.uint8).reshape(stat_data.shape)
        info["threshold_params"] = {"threshold_method": "top_n", "threshold": threshold}

    elif threshold_method == "cluster_extent":
        binary_mask, extent_info = cluster_extent_threshold(
            stat_data=stat_data,
            stat_threshold=cluster_start_threshold,
            min_cluster_size=min_cluster_size,
            max_total_voxels=max_total_voxels,
            atlas_data=atlas_data,
        )
        info.update(
            outcome=extent_info["outcome"],
            final_cluster_threshold=cluster_start_threshold,
            n_clusters_found=extent_info.get("n_clusters_surviving"),
            n_clusters_input=extent_info["n_clusters_input"],
            largest_cluster_input=extent_info["largest_cluster_input"],
            threshold_params={
                "threshold_method": "cluster_extent",
                "cluster_start_threshold": cluster_start_threshold,
                "min_cluster_size": min_cluster_size,
                "max_total_voxels": max_total_voxels,
            },
        )

    elif threshold_method == "cluster":
        if atlas_data is None:
            raise ValueError("threshold_method='cluster' needs atlas_data to define the search region")
        binary_mask, cluster_info = cluster_threshold_stat_map(
            stat_data=stat_data,
            atlas_data=atlas_data,
            target_n_voxels=threshold,
            start_threshold=cluster_start_threshold,
            step=cluster_step,
            return_all_clusters=cluster_return_all,
        )
        outcome = cluster_info["outcome"]
        if outcome in ("no_clusters_found", "empty_atlas"):
            # Fallback: top-N on atlas-masked stat data
            atlas_masked_stat = stat_data * atlas_data
            atlas_positive = atlas_masked_stat > 0
            n_atlas_positive = int(atlas_positive.sum())
            if n_atlas_positive == 0:
                binary_mask = np.zeros(stat_data.shape, dtype=np.uint8)
            elif n_atlas_positive <= threshold:
                binary_mask = atlas_positive.astype(np.uint8)
            else:
                flat = atlas_masked_stat.ravel()
                positive_indices = np.where(flat > 0)[0]
                positive_values = flat[positive_indices]
                top_n = np.argpartition(positive_values, -threshold)[-threshold:]
                result = np.zeros(flat.shape, dtype=bool)
                result[positive_indices[top_n]] = True
                binary_mask = result.astype(np.uint8).reshape(stat_data.shape)
            outcome = "fallback_top_n"
        info.update(
            outcome=outcome,
            final_cluster_threshold=cluster_info.get("final_threshold"),
            n_clusters_found=cluster_info.get("n_clusters_at_final"),
            threshold_params={
                "threshold_method": "cluster",
                "threshold": threshold,
                "cluster_start_threshold": cluster_start_threshold,
                "cluster_step": cluster_step,
                "cluster_return_all": cluster_return_all,
            },
        )

    else:
        raise ValueError(f"threshold_method must be 'top_n', 'cluster', or 'cluster_extent', got '{threshold_method}'")

    info["n_voxels"] = int(binary_mask.sum())
    return binary_mask, info


def _validate_mask_override(mask_name: str, override: dict[str, Any], threshold_method: str) -> None:
    """Raise ValueError if ``override`` is not a valid ``mask_overrides`` entry."""
    kind = override.get("kind") if isinstance(override, dict) else None
    if kind not in _OVERRIDE_DEFAULT_REASONS:
        raise ValueError(
            f"Override for '{mask_name}' needs 'kind' set to one of {sorted(_OVERRIDE_DEFAULT_REASONS)}, got {override!r}"
        )
    extra = set(override) - {"kind", "selection_reason"}
    if kind == "rethreshold":
        allowed = set(_RETHRESHOLD_FIELDS[threshold_method])
        if not extra or not extra <= allowed:
            raise ValueError(
                f"Re-threshold override for '{mask_name}' may set {sorted(allowed)} for "
                f"threshold_method='{threshold_method}', got {sorted(extra)}"
            )
    elif kind == "exclude_clusters":
        allowed = set(_RETHRESHOLD_FIELDS[threshold_method])
        peaks = override.get("peaks")
        valid_peaks = (
            isinstance(peaks, (list, tuple))
            and len(peaks) > 0
            and all(
                isinstance(peak, (list, tuple))
                and len(peak) == 3
                and all(isinstance(v, (int, np.integer)) for v in peak)
                for peak in peaks
            )
        )
        if not valid_peaks or not (extra - {"peaks"}) <= allowed:
            raise ValueError(
                f"Cluster exclusion for '{mask_name}' needs 'peaks' (a list of [i, j, k] voxels) and may set "
                f"{sorted(allowed)} for threshold_method='{threshold_method}', got {override!r}"
            )
    elif kind == "atlas_network":
        atlas, network = override.get("atlas"), override.get("network")
        if extra != {"atlas", "network"} or ATLAS_SOURCES.get(str(atlas), {}).get("type") != "network":
            raise ValueError(
                f"Atlas-network override for '{mask_name}' needs a network atlas and a network, got {override!r}"
            )
        labels = get_yeo_labels(n_networks=ATLAS_SOURCES[str(atlas)]["n_networks"])
        if network not in labels:
            raise ValueError(f"Network '{network}' is not in {atlas}: {labels}")
    elif kind in ("mni_file", "native_file"):
        if extra != {"file"} or not override.get("file"):
            raise ValueError(f"File override for '{mask_name}' needs a 'file', got {override!r}")
    elif extra:
        raise ValueError(f"Atlas-ROI override for '{mask_name}' takes no parameters, got {sorted(extra)}")


def _exclude_clusters(
    mask_name: str, binary_mask: np.ndarray, stat_data: np.ndarray, peaks: Iterable[Iterable[int]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Remove the clusters containing the given peak voxels from a thresholded mask.

    Clusters are face-connected, as in :func:`compute_cluster_table`, so a removal
    matches its row in the cluster table. Nothing is added back, so the result is
    the input mask minus those clusters. A peak outside the mask (the stat map
    changed since the review) is ignored with a warning.

    Returns
    -------
    kept_mask : np.ndarray
        The mask without the excluded clusters.
    removed : list of dict
        ``n_voxels``, ``peak_stat`` and ``peak_ijk`` of each removed cluster.
    """
    labeled, _ = ndimage_label(binary_mask.astype(np.int32))
    kept = binary_mask.copy()
    removed: list[dict[str, Any]] = []
    removed_labels: set[int] = set()
    for peak in peaks:
        ijk = tuple(int(v) for v in peak)
        in_bounds = all(0 <= v < size for v, size in zip(ijk, labeled.shape, strict=True))
        cluster_label = int(labeled[ijk]) if in_bounds else 0
        if cluster_label == 0:
            logger.warning(f"Excluded cluster peak {list(ijk)} is not in '{mask_name}' any more; ignoring it")
            continue
        if cluster_label in removed_labels:
            continue
        removed_labels.add(cluster_label)
        cluster = labeled == cluster_label
        kept[cluster] = 0
        removed.append(
            {
                "n_voxels": int(cluster.sum()),
                "peak_stat": round(float(stat_data[cluster].max()), 3),
                "peak_ijk": list(ijk),
            }
        )
    return kept, removed


def single_cluster_image(mask_file: Union[str, Path], peak_ijk: Iterable[int]) -> tuple[Nifti1Image, int]:
    """One cluster of a saved mask, for showing which voxels a cluster-table row covers.

    Parameters
    ----------
    mask_file : str or Path
        A thresholded binary mask.
    peak_ijk : iterable of int
        A voxel in the cluster (its ``peak_ijk`` in the cluster table).

    Returns
    -------
    cluster_img : Nifti1Image
        A binary mask of that cluster only.
    n_voxels : int
        Its size.

    Raises
    ------
    ValueError
        If the voxel is not in the mask.
    """
    img = cast(Nifti1Image, load(mask_file))
    labeled, _ = ndimage_label((np.asarray(img.dataobj) > 0).astype(np.int32))
    ijk = tuple(int(v) for v in peak_ijk)
    cluster_label = int(labeled[ijk])
    if cluster_label == 0:
        raise ValueError(f"Voxel {list(ijk)} is not in {Path(mask_file).name}")
    cluster = (labeled == cluster_label).astype(np.uint8)
    return Nifti1Image(cluster, img.affine, img.header), int(cluster.sum())


def _replacement_mask(
    mask_name: str,
    override: dict[str, Any],
    stat_img: Nifti1Image,
    *,
    atlas_data: np.ndarray | None,
    atlas_used: str | None,
    ref_file: Union[str, Path] | None,
    mnireg_dir: Path,
    bids_prefix: str,
    overwrite: bool,
) -> tuple[np.ndarray, str, str]:
    """Build a replacement mask on the stat-map grid; return ``(mask, source_type, source)``."""
    from nilearn.image import resample_to_img

    kind = override["kind"]
    if kind == "atlas_roi":
        if atlas_data is None:
            raise ValueError(f"An atlas-ROI mask for '{mask_name}' needs the localizer's search atlas")
        return (atlas_data > 0.5).astype(np.uint8), "atlas_roi", f"atlas_roi:{atlas_used}"

    if kind == "native_file":
        # Already in the participant's space: used as is, on exactly the stat map's grid
        native_mask = Path(override["file"])
        if not native_mask.exists():
            raise FileNotFoundError(f"Participant-space mask not found: {native_mask}")
        native_img = cast(Nifti1Image, load(native_mask))
        if native_img.shape[:3] != stat_img.shape[:3] or not np.allclose(native_img.affine, stat_img.affine, atol=1e-3):
            raise ValueError(
                f"{native_mask.name} is not on this participant's BOLD reference grid "
                f"(shape {native_img.shape[:3]} vs {stat_img.shape[:3]}). A participant-space mask must be "
                "in the localizer's BOLD reference space, e.g. an edited copy of the mask."
            )
        return (np.asarray(native_img.dataobj) > 0).astype(np.uint8), "native_file", str(native_mask)

    if ref_file is None:
        raise ValueError(f"Replacing '{mask_name}' with an MNI-space mask needs ref_file to warp it")
    ref_file = Path(ref_file)
    # The subject-to-MNI transform mask_extraction_task computes for cluster labelling
    xfm_prefix = mnireg_dir / f"{bids_prefix}_to_MNI_boldref"

    if kind == "atlas_network":
        atlas, network = override["atlas"], override["network"]
        n_networks = ATLAS_SOURCES[atlas]["n_networks"]
        # One folder per atlas: the warped parcellation's name does not encode 7 vs 17 networks
        fallback_dir = mnireg_dir / f"fallback_{atlas}"
        atlas_subject, _, _ = _register_atlas_to_subject(
            ref_file, fallback_dir, atlas, n_networks, overwrite=overwrite, xfm_prefix=xfm_prefix
        )
        mask_file = extract_yeo_mask_for_network(
            atlas_subject,
            network,
            fallback_dir / f"{bids_prefix}_{network}.nii.gz",
            n_networks=n_networks,
            overwrite=overwrite,
        )
        source_type, source = "atlas_network", f"{atlas}:{network}"
    else:  # mni_file
        mni_mask = Path(override["file"])
        if not mni_mask.exists():
            raise FileNotFoundError(f"Fallback mask file not found: {mni_mask}")
        _mni_ref, _fwd_xfm, inv_xfm = compute_xfm_to_mni(input_file=ref_file, output_prefix=xfm_prefix, overwrite=False)
        fallback_dir = mnireg_dir / "fallback_file"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        mask_file = fallback_dir / f"{bids_prefix}_{mask_name}.nii.gz"
        _warp_mask_to_reference(
            mni_mask,
            ref_file,
            inv_xfm,
            mask_file,
            fallback_dir / f"{bids_prefix}_{mask_name}_transformed.nii.gz",
            overwrite=overwrite,
        )
        source_type, source = "mni_file", str(mni_mask)

    resampled = resample_to_img(cast(Nifti1Image, load(mask_file)), stat_img, interpolation="nearest")
    return (resampled.get_fdata() > 0.5).astype(np.uint8), source_type, source


def _derive_task_mask(
    mask_name: str,
    contrast_expr: str,
    stat_img: Nifti1Image,
    *,
    stat_type: str,
    threshold_method: str,
    method_params: dict[str, Any],
    override: dict[str, Any] | None,
    atlas_data: np.ndarray | None,
    atlas_used: str | None,
    ref_file: Union[str, Path] | None,
    mnireg_dir: Path,
    bids_prefix: str,
    overwrite: bool = False,
) -> tuple[np.ndarray, TaskMaskSelection, dict[str, Any]]:
    """Derive one task mask, applying its override if any.

    Used by both :func:`mask_extraction_task` and :func:`update_task_mask`, so the
    pipeline and the review path cannot compute a mask differently. Returns the
    mask, its selection record (without the cluster table) and the
    thresholding info (empty for a replaced mask).
    """
    stat_data = stat_img.get_fdata()
    kind = override["kind"] if override else None

    if kind in (None, "rethreshold", "exclude_clusters"):
        params = dict(method_params)
        if override:
            params.update({k: override[k] for k in _RETHRESHOLD_FIELDS[threshold_method] if k in override})
        binary_mask, info = threshold_task_stat_map(
            stat_data, threshold_method=threshold_method, atlas_data=atlas_data, **params
        )
        excluded: list[dict[str, Any]] | None = None
        if override and kind == "exclude_clusters":
            binary_mask, excluded = _exclude_clusters(mask_name, binary_mask, stat_data, override["peaks"])
            info["n_voxels"] = int(binary_mask.sum())
            if info["n_clusters_found"] is not None:
                info["n_clusters_found"] = int(ndimage_label(binary_mask.astype(np.int32))[1])
        selection = TaskMaskSelection(
            mask_name=mask_name,
            contrast_definition=contrast_expr,
            stat_type=stat_type,
            peak_stat=info["peak_stat"],
            n_voxels_above_zero=info["n_positive"],
            n_voxels_thresholded=info["n_voxels"],
            threshold_method=threshold_method,
            selection_reason=override.get("selection_reason", _OVERRIDE_DEFAULT_REASONS[override["kind"]])
            if override
            else "auto",
            atlas_used=atlas_used if threshold_method != "top_n" else None,
            final_cluster_threshold=info["final_cluster_threshold"],
            n_clusters_found=info["n_clusters_found"],
            cluster_outcome=info["outcome"],
            n_clusters_input=info["n_clusters_input"],
            largest_cluster_input=info["largest_cluster_input"],
            threshold_params=info["threshold_params"],
            excluded_clusters=excluded,
        )
        if threshold_method == "cluster_extent":
            selection.min_cluster_size = params["min_cluster_size"]
            selection.max_total_voxels = params["max_total_voxels"]
        return binary_mask, selection, info

    assert override is not None
    binary_mask, source_type, source = _replacement_mask(
        mask_name,
        override,
        stat_img,
        atlas_data=atlas_data,
        atlas_used=atlas_used,
        ref_file=ref_file,
        mnireg_dir=mnireg_dir,
        bids_prefix=bids_prefix,
        overwrite=overwrite,
    )
    n_positive = int((stat_data > 0).sum())
    selection = TaskMaskSelection(
        mask_name=mask_name,
        source_type=source_type,
        contrast_definition=contrast_expr,
        stat_type=stat_type,
        peak_stat=float(np.nanmax(stat_data)) if n_positive > 0 else 0.0,
        n_voxels_above_zero=n_positive,
        n_voxels_thresholded=int(binary_mask.sum()),
        threshold_method=threshold_method,
        selection_reason=override.get("selection_reason", _OVERRIDE_DEFAULT_REASONS[override["kind"]]),
        atlas_used=atlas_used if kind == "atlas_roi" else None,
        source=source,
    )
    return binary_mask, selection, {}


def _log_threshold_outcome(
    mask_name: str,
    contrast_expr: str,
    info: dict[str, Any],
    emit_progress: Callable[..., None],
    step_label: str,
) -> None:
    """Log the outcome of :func:`threshold_task_stat_map` for one mask."""
    params = info["threshold_params"]
    method = params["threshold_method"]
    n_kept = info["n_voxels"]
    outcome = info["outcome"]

    if method == "top_n":
        if info["n_positive"] == 0:
            logger.warning(f"No positive voxels in contrast '{mask_name}' ({contrast_expr}). Saving empty mask.")
        elif info["n_positive"] <= params["threshold"]:
            logger.warning(
                f"Contrast '{mask_name}' has only {info['n_positive']} positive voxels "
                f"(requested {params['threshold']}). Keeping all positive voxels."
            )
    elif method == "cluster_extent":
        start = params["cluster_start_threshold"]
        if outcome == "success":
            emit_progress(
                step_label, f"{mask_name}: {n_kept} voxels ({info['n_clusters_found']} clusters, stat>={start})"
            )
        elif outcome == "no_suprathreshold_voxels":
            logger.warning(f"No voxels above threshold {start} for '{mask_name}'")
        elif outcome == "no_clusters_survive":
            logger.warning(
                f"All {info['n_clusters_input']} clusters below "
                f"min_cluster_size={params['min_cluster_size']} for '{mask_name}' "
                f"(largest {info['largest_cluster_input']} voxels)"
            )
        elif outcome == "capped":
            emit_progress(step_label, f"{mask_name}: {n_kept} voxels ({info['n_clusters_found']} clusters, capped)")
    else:
        if outcome in ("success", "already_under_target"):
            emit_progress(
                step_label,
                f"{mask_name}: {n_kept} voxels "
                f"(cluster at z>={info['final_cluster_threshold']:.1f}, "
                f"{info['n_clusters_found']} clusters)",
            )
        elif outcome == "fallback_top_n":
            logger.warning(f"Cluster thresholding failed for '{mask_name}'. Fell back to top-N within atlas region.")
            emit_progress(step_label, f"{mask_name}: {outcome} — {n_kept} voxels (fallback)")
        elif outcome == "max_iterations_reached":
            logger.warning(
                f"Cluster thresholding for '{mask_name}' hit max iterations. "
                f"Using largest cluster ({n_kept} voxels, target was {params['threshold']})."
            )


def mask_extraction_task(
    participant_id: str,
    session_id: str,
    root_dir: Union[str, Path],
    input_file: Union[str, Path],
    mask_file: Union[str, Path],
    events: Union[str, Path, pd.DataFrame],
    contrasts: dict[str, str],
    tr: float,
    threshold: int = 2000,
    threshold_method: str = "top_n",
    atlas_mask: Union[str, Path] | None = None,
    atlas_source: str | None = None,
    atlas_regions: list[str] | None = None,
    ref_file: Union[str, Path] | None = None,
    cluster_start_threshold: float = 2.0,
    cluster_step: float = 0.1,
    cluster_return_all: bool = False,
    min_cluster_size: int = 50,
    max_total_voxels: int | None = None,
    stat_type: str = "t",
    smoothing_fwhm: float | None = 5.0,
    high_pass: float = 0.01,
    drift_model: str = "cosine",
    hrf_model: str = "spm",
    include_reverse: bool = False,
    overwrite: bool = False,
    qc_enabled: bool = True,
    qc_dir: Union[str, Path] | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    mask_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    """
    Run the mask extraction pipeline for task-based fMRI data using GLM contrasts.

    This function extracts participant-specific functional masks from task localizer
    data. A whole-brain GLM is fit using nilearn's FirstLevelModel, and user-specified
    contrasts are computed to produce statistical maps. Each contrast map is then
    thresholded into a binary mask using one of two strategies:

    - **top_n** (default): Selects the top N voxels by stat value. Simple and fast,
      but the resulting mask may contain scattered, non-contiguous voxels.
    - **cluster**: Atlas-guided iterative cluster thresholding. Restricts the stat
      map to an anatomical ROI, then iteratively raises the stat threshold until
      the selected cluster(s) fit the target size. By default returns the single
      largest contiguous cluster (suitable for anatomical ROIs like STG). Set
      ``cluster_return_all=True`` to return all clusters combined (suitable for
      whole-brain atlases like gray matter).
    - **cluster_extent**: Classic cluster-extent thresholding. Applies a fixed stat
      threshold, identifies contiguous clusters, and discards clusters smaller than
      ``min_cluster_size``. An optional atlas mask (e.g., grey matter) can restrict
      the search space. Unlike ``cluster``, this method does not target a specific
      voxel count — the output size is determined by the stat threshold and the
      data itself.

    Parameters
    ----------
    participant_id : str
        Participant ID (e.g., "sub-01")
    root_dir : str or Path
        Path to the root directory for all outputs
    input_file : str or Path
        Path to preprocessed 4D NIfTI file (output of preprocessing())
    mask_file : str or Path
        Path to brain mask NIfTI file (output of preprocessing())
    events : str, Path, or pd.DataFrame
        BIDS-style events with columns: onset (seconds), duration (seconds),
        trial_type (condition name). If a path, reads as tab-separated file.
    contrasts : dict[str, str]
        Mapping of mask name to contrast expression string.
        Example: {"CEN": "nback - rest", "DMN": "rest - nback"}.
        Expressions are parsed by nilearn and support arithmetic
        (e.g., "2*nback - rest - fixation").
    tr : float
        Repetition time in seconds
    threshold : int
        Number of voxels to retain per mask (default: 2000). For ``top_n``,
        this is the exact count selected. For ``cluster``, this is the maximum
        cluster size target.
    threshold_method : str
        ``"top_n"`` (default) selects the top N voxels by stat value.
        ``"cluster"`` uses atlas-guided iterative cluster thresholding.
        ``"cluster_extent"`` applies a fixed stat threshold and keeps
        clusters above ``min_cluster_size``.
    atlas_mask : str, Path, or None
        Path to a binary atlas NIfTI in subject space. Used with
        ``threshold_method="cluster"`` to constrain the search region.
        Mutually exclusive with ``atlas_source``.
    atlas_source : str or None
        Key into ``ATLAS_SOURCES`` (e.g., ``"harvard_oxford"``, ``"aal"``,
        ``"schaefer_400_7net"``). Used with ``atlas_regions`` to specify
        which regions to extract. Mutually exclusive with ``atlas_mask``.
        Requires ``ref_file`` for registration to subject space.
    atlas_regions : list[str] or None
        Region labels from the atlas to include in the mask. Required
        when ``atlas_source`` is provided.
    ref_file : str, Path, or None
        BOLD reference image for atlas registration when ``atlas_source`` is
        provided. Typically the temporal mean from preprocessing.
    cluster_start_threshold : float
        Starting stat threshold for iterative cluster search (default: 2.0).
        Only used when ``threshold_method="cluster"``.
    cluster_step : float
        Amount to increase the threshold per iteration (default: 0.1).
        Only used when ``threshold_method="cluster"``.
    cluster_return_all : bool
        If True, return all clusters above the final threshold combined rather
        than only the largest single cluster. Use with whole-brain atlases
        (e.g. ``atlas_source="gray_matter"``). Default: False.
        Only used when ``threshold_method="cluster"``.
    min_cluster_size : int
        Minimum number of contiguous voxels for a cluster to survive
        (default: 50). Only used when ``threshold_method="cluster_extent"``.
    max_total_voxels : int or None
        Optional safety cap on total mask size. If the total surviving voxels
        exceed this, the highest-stat voxels within the surviving clusters are
        kept up to the cap. ``None`` (default) disables the cap.
        Only used when ``threshold_method="cluster_extent"``.
    stat_type : str
        Type of statistical map: "t" or "z" (default: "t")
    smoothing_fwhm : float or None
        Spatial smoothing FWHM in mm (default: 5.0). None disables smoothing.
    high_pass : float
        High-pass filter cutoff in Hz (default: 0.01)
    drift_model : str
        Drift model for nuisance regression: "cosine", "polynomial", or None
        (default: "cosine")
    hrf_model : str
        HRF model: "spm", "spm + derivative", "glover", etc. (default: "spm")
    include_reverse : bool
        When True, generate a reverse (negated) mask for each contrast by
        negating the stat map before thresholding. For contrast "A - B", the
        reverse captures "B - A". Reverse masks are named "{mask_name}_reverse".
        Enables dual-network feedback from a single-contrast localizer.
    overwrite : bool
        If True, rerun even if outputs exist (default: False)
    qc_enabled : bool
        If True, generate QC report (default: True)
    qc_dir : str, Path, or None
        Custom QC output directory. If None, uses {root_dir}/../qc/
    progress_callback : callable or None
        Function(step: str, detail: str) for progress updates
    mask_overrides : dict or None
        Per-mask changes recorded when an operator reviews the masks (see
        :func:`update_task_mask`), keyed by mask name. Each entry has a
        ``kind`` and an optional ``selection_reason``:

        - ``{"kind": "rethreshold", ...}`` thresholds this mask with other
          values of the method's own parameters (``threshold`` for top_n;
          ``threshold``, ``cluster_start_threshold``, ``cluster_step`` and
          ``cluster_return_all`` for cluster; ``cluster_start_threshold``,
          ``min_cluster_size`` and ``max_total_voxels`` for cluster_extent).
        - ``{"kind": "exclude_clusters", "peaks": [[i, j, k], ...], ...}``
          thresholds this mask (optionally with the same fields as
          ``rethreshold``), then removes the whole clusters containing those
          voxels. A peak outside the mask is ignored with a warning.
        - ``{"kind": "atlas_network", "atlas": "yeo_17", "network": "DefaultA"}``
          replaces the mask with a whole atlas network warped from MNI space.
        - ``{"kind": "mni_file", "file": "/path/mask_mni.nii.gz"}`` replaces it
          with an MNI-space mask file warped to subject space.
        - ``{"kind": "native_file", "file": "/path/mask.nii.gz"}`` replaces it
          with a mask already on the participant's BOLD reference grid (e.g. an
          edited copy of the mask), used as is.
        - ``{"kind": "atlas_roi"}`` replaces it with the search atlas.

        The MNI-space replacements need ``ref_file``. Overrides naming an
        unknown mask are ignored with a warning.

    Returns
    -------
    list[Path]
        Paths to thresholded binary mask files ({participant}_{mask_name}_thr.nii.gz)

    Raises
    ------
    ValueError
        If events is empty, missing required columns, or contrast references
        unknown conditions. Also if cluster parameters are invalid.
    FileNotFoundError
        If input_file, mask_file, or events path does not exist

    Examples
    --------
    Top-N thresholding (default):

    >>> mask_files = mask_extraction_task(
    ...     participant_id='sub-01',
    ...     root_dir='/data/sub-01/mask_extraction',
    ...     input_file='/data/sub-01/preprocessing/sub-01_task-nback_preproc.nii.gz',
    ...     mask_file='/data/sub-01/preprocessing/sub-01_task-nback_preproc_mask.nii.gz',
    ...     events=pd.DataFrame({
    ...         'onset': [0, 30, 60, 90],
    ...         'duration': [30, 30, 30, 30],
    ...         'trial_type': ['rest', 'nback', 'rest', 'nback']
    ...     }),
    ...     contrasts={"CEN": "nback - rest", "DMN": "rest - nback"},
    ...     tr=1.2,
    ... )

    Atlas-guided cluster thresholding (STG from self/other localizer):

    >>> mask_files = mask_extraction_task(
    ...     participant_id='sub-01',
    ...     root_dir='/data/sub-01/mask_extraction',
    ...     input_file='/data/sub-01/preprocessing/sub-01_task-selfother_preproc.nii.gz',
    ...     mask_file='/data/sub-01/preprocessing/sub-01_task-selfother_preproc_mask.nii.gz',
    ...     events='/data/sub-01/events/sub-01_task-selfother_events.tsv',
    ...     contrasts={"STG": "self - other"},
    ...     tr=1.2,
    ...     threshold=200,
    ...     threshold_method="cluster",
    ...     atlas_source="harvard_oxford",
    ...     atlas_regions=[
    ...         "Superior Temporal Gyrus, anterior division",
    ...         "Superior Temporal Gyrus, posterior division",
    ...     ],
    ...     ref_file='/data/sub-01/preprocessing/sub-01_task-selfother_mean.nii.gz',
    ... )

    See Also
    --------
    mask_extraction_rest : ICA-based extraction from resting-state data
    coregister_reference : Align masks to feedback scanner position
    fetch_atlas_mask : Fetch anatomical atlas regions from nilearn
    cluster_threshold_stat_map : Iterative cluster thresholding algorithm
    """
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.image import resample_to_img

    # --- Setup ---
    if not contrasts:
        raise ValueError("contrasts dict is empty — at least one contrast is required")

    root_dir = Path(root_dir)
    input_file = Path(input_file)
    mask_file = Path(mask_file)

    def emit_progress(step: str, detail: str = ""):
        if progress_callback:
            progress_callback(step, detail)
        if detail:
            logger.info(f"{step}: {detail}")
        else:
            logger.info(step)

    emit_progress("Task-based mask extraction", f"Participant: {participant_id}")

    # --- Validate inputs ---
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    if not contrasts:
        raise ValueError("contrasts must be a non-empty dict (e.g., {'CEN': 'nback - rest'})")

    # Load events
    if isinstance(events, (str, Path)):
        events_path = Path(events)
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        events_df = pd.read_csv(events_path, sep="\t")
    elif isinstance(events, pd.DataFrame):
        events_df = events.copy()
    else:
        raise TypeError(f"events must be a path or DataFrame, got {type(events)}")

    # Validate events columns
    required_cols = {"onset", "duration", "trial_type"}
    missing_cols = required_cols - set(events_df.columns)
    if missing_cols:
        raise ValueError(
            f"Events missing required columns: {missing_cols}. "
            f"Expected BIDS format with columns: onset, duration, trial_type"
        )
    if len(events_df) == 0:
        raise ValueError("Events DataFrame is empty")

    available_conditions = sorted(events_df["trial_type"].unique())
    emit_progress("Events loaded", f"{len(events_df)} events, conditions: {available_conditions}")

    # --- Validate threshold_method and cluster-specific params ---
    if threshold_method not in ("top_n", "cluster", "cluster_extent"):
        raise ValueError(f"threshold_method must be 'top_n', 'cluster', or 'cluster_extent', got '{threshold_method}'")
    if threshold_method == "cluster":
        if atlas_mask is None and atlas_source is None:
            raise ValueError(
                "When threshold_method='cluster', one of atlas_mask or atlas_source "
                "must be provided to define the anatomical ROI."
            )
        if atlas_mask is not None and atlas_source is not None:
            raise ValueError("Provide only one of atlas_mask or atlas_source, not both.")
        if atlas_source is not None and ref_file is None:
            raise ValueError(
                "ref_file is required when atlas_source is provided, to register the atlas from MNI to subject space."
            )

    # --- Validate per-mask overrides ---
    all_mask_names = list(contrasts) + ([f"{name}_reverse" for name in contrasts] if include_reverse else [])
    overrides: dict[str, dict[str, Any]] = {}
    for name, override in (mask_overrides or {}).items():
        if name not in all_mask_names:
            logger.warning(f"Ignoring override for unknown mask '{name}' (masks: {all_mask_names})")
            continue
        _validate_mask_override(name, override, threshold_method)
        overrides[name] = override
    override_kinds = {o["kind"] for o in overrides.values()}
    if override_kinds & {"atlas_network", "mni_file"} and ref_file is None:
        raise ValueError("ref_file is required to replace a mask with an MNI-space mask")
    uses_atlas = threshold_method == "cluster" or (
        threshold_method == "cluster_extent" and (atlas_mask is not None or atlas_source is not None)
    )
    if "atlas_roi" in override_kinds and not uses_atlas:
        raise ValueError("An atlas_roi override needs a search atlas (atlas_mask or atlas_source)")

    n_steps = 3  # 01_glm, 02_mnireg, 03_masks
    bids_prefix = f"{participant_id}_{session_id}"

    # =========================================================================
    # Step 1/{n_steps}: GLM Fit
    # =========================================================================
    glm_dir = root_dir / "01_glm"
    glm_dir.mkdir(parents=True, exist_ok=True)

    emit_progress(f"Step 1/{n_steps}: Fitting GLM", f"{len(contrasts)} contrasts to compute")

    # Check if all outputs already exist
    stat_map_paths = {}
    all_exist = True
    for mask_name in contrasts:
        stat_path = glm_dir / f"{bids_prefix}_{mask_name}_{stat_type}map.nii.gz"
        stat_map_paths[mask_name] = stat_path
        if not stat_path.exists():
            all_exist = False

    if all_exist and not overwrite:
        logger.info("All GLM stat maps exist and overwrite=False. Skipping GLM fit.")
        emit_progress(f"Step 1/{n_steps}", "Skipped (outputs exist)")
    else:
        with Timer("glm_fit", "mask_extraction_task"):
            model = FirstLevelModel(
                t_r=tr,
                mask_img=str(mask_file),
                smoothing_fwhm=smoothing_fwhm,
                high_pass=high_pass,
                drift_model=drift_model,
                hrf_model=hrf_model,
                noise_model="ar1",
                minimize_memory=False,
                n_jobs=1,
            )

            emit_progress(f"Step 1/{n_steps}", "Fitting model...")
            model.fit(str(input_file), events=events_df)

            # Save design matrix for QC and reproducibility
            design_matrix = model.design_matrices_[0]
            design_file = glm_dir / f"{bids_prefix}_design_matrix.tsv"
            design_matrix.to_csv(design_file, sep="\t", index=True)

            # Compute and save each contrast
            for mask_name, contrast_expr in contrasts.items():
                emit_progress(f"Step 1/{n_steps}", f"Computing contrast: {mask_name} = {contrast_expr}")

                # nilearn's contrast stat computation evaluates 1.0 / variance
                # over every voxel (including zero-variance voxels it masks out
                # afterwards), emitting a benign divide-by-zero RuntimeWarning.
                # Suppress it here -- the masked result is unaffected.
                with np.errstate(divide="ignore", invalid="ignore"):
                    stat_map = model.compute_contrast(
                        contrast_expr,
                        stat_type=stat_type,
                        output_type="stat",
                    )
                nib.save(stat_map, str(stat_map_paths[mask_name]))

            # Save GLM metadata
            glm_info = {
                "participant_id": participant_id,
                "session_id": session_id,
                "input_file": str(input_file),
                "tr": tr,
                "n_volumes": design_matrix.shape[0],
                "conditions": available_conditions,
                "contrasts": contrasts,
                "stat_type": stat_type,
                "hrf_model": hrf_model,
                "drift_model": drift_model,
                "high_pass": high_pass,
                "smoothing_fwhm": smoothing_fwhm,
                "design_matrix_columns": list(design_matrix.columns),
            }
            info_file = glm_dir / f"{bids_prefix}_glm_info.json"
            with open(info_file, "w") as f:
                json.dump(glm_info, f, indent=2, default=_json_default)

        emit_progress(f"Step 1/{n_steps}", "GLM fit complete")

    # Generate reverse contrasts by negating each forward stat map
    if include_reverse:
        for mask_name in list(contrasts.keys()):
            reverse_name = f"{mask_name}_reverse"
            forward_path = stat_map_paths[mask_name]
            reverse_path = glm_dir / f"{bids_prefix}_{reverse_name}_{stat_type}map.nii.gz"
            stat_map_paths[reverse_name] = reverse_path

            if reverse_path.exists() and not overwrite:
                logger.info(f"Reverse stat map exists: {reverse_path.name}")
                continue

            forward_img = cast(Nifti1Image, load(forward_path))
            reverse_data = -forward_img.get_fdata()
            reverse_img = Nifti1Image(reverse_data, forward_img.affine, forward_img.header)
            nib.save(reverse_img, str(reverse_path))
            logger.info(f"Generated reverse contrast: {reverse_name} (negated {mask_name})")

    # Build combined contrasts dict for thresholding (forward + any reverse)
    all_contrasts: dict[str, str] = dict(contrasts)
    if include_reverse:
        for mask_name, contrast_expr in contrasts.items():
            all_contrasts[f"{mask_name}_reverse"] = f"-({contrast_expr})"

    # =========================================================================
    # Step 2/3: MNI Registration (atlas + Harvard-Oxford parcellation)
    # =========================================================================
    mnireg_dir = root_dir / "02_mnireg"
    mnireg_dir.mkdir(parents=True, exist_ok=True)

    atlas_subject_space_data = None  # Set below if cluster/cluster_extent mode
    atlas_inv_xfm: list[str] | None = None  # Subject<-MNI transform, if computed for the atlas

    _has_atlas = any(x is not None for x in (atlas_mask, atlas_source))
    if threshold_method == "cluster" or (threshold_method == "cluster_extent" and _has_atlas):
        emit_progress(f"Step 2/{n_steps}: Preparing atlas mask")

        if atlas_mask is not None:
            # User-provided atlas — use directly (assumed in subject space)
            atlas_mask_path = Path(atlas_mask)
            if not atlas_mask_path.exists():
                raise FileNotFoundError(f"Atlas mask not found: {atlas_mask_path}")
            atlas_img = cast(Nifti1Image, load(atlas_mask_path))
            atlas_label = atlas_mask_path.name
            emit_progress(f"Step 2/{n_steps}", f"Using user-provided atlas: {atlas_label}")
        else:
            # Fetch atlas and register to subject space
            assert ref_file is not None  # validated above
            ref_file = Path(ref_file)

            assert atlas_source is not None  # validated above
            fetch_label = atlas_source
            atlas_mni_path = mnireg_dir / f"{atlas_source}_mni.nii.gz"
            atlas_mni_path = fetch_atlas_mask(
                output_file=atlas_mni_path,
                atlas_source=atlas_source,
                regions=atlas_regions or [],
                overwrite=overwrite,
            )

            emit_progress(f"Step 2/{n_steps}", f"Fetched {fetch_label} atlas in MNI space")

            # Sub-step: Register to subject space via ANTs
            xfm_prefix = mnireg_dir / f"{bids_prefix}_to_MNI_boldref"
            mni_ref_file, sub2mni_fwd_xfm, sub2mni_inv_xfm = compute_xfm_to_mni(
                input_file=ref_file,
                output_prefix=xfm_prefix,
                overwrite=overwrite,
            )
            atlas_inv_xfm = sub2mni_inv_xfm

            # Sub-step: Apply inverse transform (MNI -> subject space)
            atlas_subject_path = mnireg_dir / f"{fetch_label}_to_{bids_prefix}.nii.gz"
            apply_xfm(
                moving=atlas_mni_path,
                fixed=ref_file,
                transform_files=sub2mni_inv_xfm,
                output_file=atlas_subject_path,
                interpolator="nearestNeighbor",
                invert=False,
                overwrite=overwrite,
            )
            atlas_img = cast(Nifti1Image, load(atlas_subject_path))
            atlas_label = fetch_label
            emit_progress(f"Step 2/{n_steps}", "Atlas registered to subject space")

        # Resample atlas to stat map resolution if needed
        first_stat_path = stat_map_paths[list(contrasts.keys())[0]]
        stat_ref_img = cast(Nifti1Image, load(first_stat_path))
        atlas_resampled_img = resample_to_img(atlas_img, stat_ref_img, interpolation="nearest")
        atlas_subject_space_data = (atlas_resampled_img.get_fdata() > 0.5).astype(np.float64)

        # Save atlas-in-subject-space for QC
        masks_dir = root_dir / "03_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        atlas_qc_path = masks_dir / f"{bids_prefix}_atlas_{atlas_label}.nii.gz"
        save(
            Nifti1Image(
                atlas_subject_space_data.astype(np.uint8),
                stat_ref_img.affine,
                stat_ref_img.header,
            ),
            atlas_qc_path,
        )
        emit_progress(f"Step 2/{n_steps}", f"Atlas mask: {int(atlas_subject_space_data.sum())} voxels in subject space")

    # =========================================================================
    # Harvard-Oxford parcellation for anatomical cluster labeling
    # =========================================================================
    ho_subject_data = None
    ho_labels = None

    if ref_file is not None:
        from nilearn import datasets

        label_step = f"Step 2/{n_steps}"
        emit_progress(label_step, "Registering Harvard-Oxford atlas for cluster labeling...")

        # Fetch Harvard-Oxford cortical parcellation (integer-labeled, MNI space)
        ho_atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        ho_maps = ho_atlas.maps
        ho_img = ho_maps if isinstance(ho_maps, nib.Nifti1Image) else cast(Nifti1Image, nib.load(ho_maps))
        ho_labels = list(ho_atlas.labels)  # Index 0 = "Background"

        ho_mni_path = mnireg_dir / "harvard_oxford_cort_mni.nii.gz"
        if not ho_mni_path.exists() or overwrite:
            save(ho_img, ho_mni_path)

        # Compute MNI <-> subject transforms, reusing the atlas step's if it ran
        ref_file_path = Path(ref_file)
        if atlas_inv_xfm is not None:
            inv_xfm = atlas_inv_xfm
        else:
            xfm_prefix = mnireg_dir / f"{bids_prefix}_to_MNI_boldref"
            _mni_ref, _fwd_xfm, inv_xfm = compute_xfm_to_mni(
                input_file=ref_file_path,
                output_prefix=xfm_prefix,
                overwrite=overwrite,
            )

        # Apply inverse transform: MNI -> subject space (nearestNeighbor preserves labels)
        ho_subject_path = mnireg_dir / f"harvard_oxford_cort_to_{bids_prefix}.nii.gz"
        apply_xfm(
            moving=ho_mni_path,
            fixed=ref_file_path,
            transform_files=inv_xfm,
            output_file=ho_subject_path,
            interpolator="nearestNeighbor",
            invert=False,
            overwrite=overwrite,
        )

        ho_subject_img = cast(Nifti1Image, load(ho_subject_path))

        # Resample to stat map grid
        first_stat_path = stat_map_paths[list(contrasts.keys())[0]]
        stat_ref_img = cast(Nifti1Image, load(first_stat_path))
        ho_resampled = resample_to_img(ho_subject_img, stat_ref_img, interpolation="nearest")
        ho_subject_data = np.round(ho_resampled.get_fdata()).astype(int)

        emit_progress(label_step, f"Harvard-Oxford: {len(ho_labels) - 1} cortical regions in subject space")

    # =========================================================================
    # Step {n_steps}/{n_steps}: Threshold to Binary Masks
    # =========================================================================
    masks_dir = root_dir / "03_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    step_label = f"Step {n_steps}/{n_steps}"
    if threshold_method == "top_n":
        emit_progress(f"{step_label}: Thresholding masks", f"{threshold} voxels per mask")
    elif threshold_method == "cluster_extent":
        emit_progress(
            f"{step_label}: Cluster-extent thresholding",
            f"stat>={cluster_start_threshold}, min_cluster={min_cluster_size}",
        )
    else:
        emit_progress(
            f"{step_label}: Cluster thresholding", f"target={threshold} voxels, start_z={cluster_start_threshold}"
        )

    network_thr_files = []
    selections: dict[str, TaskMaskSelection] = {}
    previous_selections = load_task_mask_selections(masks_dir)
    method_params = {
        "threshold": threshold,
        "cluster_start_threshold": cluster_start_threshold,
        "cluster_step": cluster_step,
        "cluster_return_all": cluster_return_all,
        "min_cluster_size": min_cluster_size,
        "max_total_voxels": max_total_voxels,
    }
    _atlas_label = atlas_source or atlas_mask
    atlas_used = str(_atlas_label) if _atlas_label else None

    for mask_name, contrast_expr in all_contrasts.items():
        stat_map_path = stat_map_paths[mask_name]

        # Save unthresholded map to masks dir
        unthr_file = masks_dir / f"{bids_prefix}_{mask_name}.nii.gz"
        thr_file = masks_dir / f"{bids_prefix}_{mask_name}_thr.nii.gz"

        if thr_file.exists() and not overwrite:
            logger.info(f"Thresholded mask exists and overwrite=False: {thr_file.name}")
            network_thr_files.append(thr_file)

            img = cast(Nifti1Image, load(thr_file))
            n_voxels = int(img.get_fdata().sum())

            # Keep the recorded selection, which may hold an operator's override
            previous = previous_selections.get(mask_name)
            if previous is not None:
                previous.n_voxels_thresholded = n_voxels
                selections[mask_name] = previous
                continue

            # Without one, still record selection metadata from the existing file
            stat_img = cast(Nifti1Image, load(stat_map_path))
            stat_data = stat_img.get_fdata()
            selections[mask_name] = TaskMaskSelection(
                mask_name=mask_name,
                contrast_definition=contrast_expr,
                stat_type=stat_type,
                peak_stat=float(np.nanmax(stat_data)),
                n_voxels_above_zero=int(np.sum(stat_data > 0)),
                n_voxels_thresholded=n_voxels,
                threshold_method=threshold_method,
            )
            continue

        emit_progress(step_label, f"Thresholding {mask_name}...")

        # Load stat map
        stat_img = cast(Nifti1Image, load(stat_map_path))
        stat_data = stat_img.get_fdata()

        # Copy unthresholded map
        save(Nifti1Image(stat_data, stat_img.affine, stat_img.header), unthr_file)

        if threshold_method == "cluster":
            # Save atlas-masked stat map for QC
            assert atlas_subject_space_data is not None
            atlas_masked_file = masks_dir / f"{bids_prefix}_{mask_name}_atlas_masked.nii.gz"
            save(
                Nifti1Image(stat_data * atlas_subject_space_data, stat_img.affine, stat_img.header),
                atlas_masked_file,
            )

        binary_mask, selection, threshold_info = _derive_task_mask(
            mask_name,
            contrast_expr,
            stat_img,
            stat_type=stat_type,
            threshold_method=threshold_method,
            method_params=method_params,
            override=overrides.get(mask_name),
            atlas_data=atlas_subject_space_data,
            atlas_used=atlas_used,
            ref_file=ref_file,
            mnireg_dir=mnireg_dir,
            bids_prefix=bids_prefix,
            overwrite=overwrite,
        )
        selections[mask_name] = selection
        n_kept = selection.n_voxels_thresholded
        peak_stat = selection.peak_stat
        if threshold_info:
            _log_threshold_outcome(mask_name, contrast_expr, threshold_info, emit_progress, step_label)
        else:
            emit_progress(step_label, f"{mask_name}: replaced with {selection.source}")

        # Compute cluster table with anatomical labels
        if n_kept > 0:
            cluster_table = compute_cluster_table(
                binary_mask=binary_mask,
                stat_data=stat_data,
                affine=stat_img.affine,
                atlas_data=ho_subject_data,
                atlas_labels=ho_labels,
            )
            selection.cluster_table = cluster_table
            n_clusters_found = len(cluster_table)
            logger.info(f"{mask_name}: {n_clusters_found} clusters in {n_kept} voxels")

        save(Nifti1Image(binary_mask, stat_img.affine, stat_img.header), thr_file)
        network_thr_files.append(thr_file)

        emit_progress(step_label, f"{mask_name}: {n_kept} voxels (peak {stat_type}={peak_stat:.2f})")
        if n_kept == 0:
            emit_progress(step_label, f"WARNING: {mask_name} is empty (0 voxels)")

    # Save selection metadata
    save_task_mask_selections(masks_dir, selections)

    emit_progress(step_label, f"Complete ({len(network_thr_files)} masks extracted)")

    # Generate QC report
    if qc_enabled:
        emit_progress("Generating QC report...")
        qc_output_dir = Path(qc_dir) if qc_dir is not None else root_dir.parent / "qc"
        mask_names = list(all_contrasts.keys())
        report_path = qc.generate_task_mask_extraction_report(
            mask_extraction_dir=root_dir,
            participant_id=participant_id,
            session_id=session_id,
            mask_names=mask_names,
            contrasts=all_contrasts,
            stat_type=stat_type,
            output_dir=qc_output_dir,
            overwrite=overwrite,
            open_browser=False,
            ref_file=ref_file,
        )
        emit_progress("QC report", f"Saved to {report_path.name}")

    return network_thr_files


def _subject_harvard_oxford(
    mnireg_dir: Path, bids_prefix: str, stat_img: Nifti1Image
) -> tuple[np.ndarray | None, list[str] | None]:
    """Load the Harvard-Oxford labels :func:`mask_extraction_task` warped to subject space, if present."""
    ho_subject_path = mnireg_dir / f"harvard_oxford_cort_to_{bids_prefix}.nii.gz"
    if not ho_subject_path.exists():
        return None, None
    from nilearn import datasets
    from nilearn.image import resample_to_img

    ho_labels = list(datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm").labels)
    ho_resampled = resample_to_img(cast(Nifti1Image, load(ho_subject_path)), stat_img, interpolation="nearest")
    return np.round(ho_resampled.get_fdata()).astype(int), ho_labels


def _task_mask_from_cache(
    params: dict[str, Any], mask_name: str, override: dict[str, Any] | None
) -> tuple[np.ndarray, TaskMaskSelection, Nifti1Image]:
    """Derive one task mask from the cached GLM outputs; return ``(mask, selection, stat_img)``."""
    root_dir = Path(params["root_dir"])
    bids_prefix = f"{params['participant_id']}_{params['session_id']}"
    stat_type = params.get("stat_type", "t")
    threshold_method = params.get("threshold_method", "top_n")
    atlas_mask, atlas_source = params.get("atlas_mask"), params.get("atlas_source")

    contrasts: dict[str, str] = dict(params["contrasts"])
    if params.get("include_reverse"):
        contrasts.update({f"{name}_reverse": f"-({expr})" for name, expr in params["contrasts"].items()})
    if mask_name not in contrasts:
        raise ValueError(f"Unknown mask '{mask_name}' (masks: {list(contrasts)})")
    if override is not None:
        _validate_mask_override(mask_name, override, threshold_method)

    stat_path = root_dir / "01_glm" / f"{bids_prefix}_{mask_name}_{stat_type}map.nii.gz"
    if not stat_path.exists():
        raise FileNotFoundError(f"Stat map not found (run mask extraction first): {stat_path}")
    stat_img = cast(Nifti1Image, load(stat_path))

    atlas_data = None
    if threshold_method == "cluster" or (threshold_method == "cluster_extent" and (atlas_mask or atlas_source)):
        atlas_file = root_dir / "03_masks" / f"{bids_prefix}_atlas_{_atlas_qc_label(atlas_mask, atlas_source)}.nii.gz"
        if not atlas_file.exists():
            raise FileNotFoundError(f"Search atlas not found (run mask extraction first): {atlas_file}")
        atlas_data = (cast(Nifti1Image, load(atlas_file)).get_fdata() > 0.5).astype(np.float64)

    _atlas_label = atlas_source or atlas_mask
    binary_mask, selection, _info = _derive_task_mask(
        mask_name,
        contrasts[mask_name],
        stat_img,
        stat_type=stat_type,
        threshold_method=threshold_method,
        method_params={k: params.get(k, default) for k, default in _METHOD_PARAM_DEFAULTS.items()},
        override=override,
        atlas_data=atlas_data,
        atlas_used=str(_atlas_label) if _atlas_label else None,
        ref_file=params.get("ref_file"),
        mnireg_dir=root_dir / "02_mnireg",
        bids_prefix=bids_prefix,
    )
    return binary_mask, selection, stat_img


def preview_task_mask(
    params: dict[str, Any], mask_name: str, override: dict[str, Any] | None
) -> tuple[Nifti1Image, TaskMaskSelection]:
    """The mask :func:`update_task_mask` would write, without writing anything.

    Parameters are as for :func:`update_task_mask`.

    Returns
    -------
    mask_img : Nifti1Image
        The previewed binary mask, on the stat map's grid.
    selection : TaskMaskSelection
        Its selection record, with an unlabelled cluster table.
    """
    binary_mask, selection, stat_img = _task_mask_from_cache(params, mask_name, override)
    if selection.n_voxels_thresholded > 0:
        selection.cluster_table = compute_cluster_table(binary_mask, stat_img.get_fdata(), stat_img.affine)
    return Nifti1Image(binary_mask, stat_img.affine, stat_img.header), selection


def update_task_mask(params: dict[str, Any], mask_name: str, override: dict[str, Any] | None) -> TaskMaskSelection:
    """Re-derive one task mask from the cached GLM outputs, without refitting the GLM.

    The review path for :func:`mask_extraction_task`: it reads the saved stat
    map (``01_glm``) and search atlas (``03_masks``), applies ``override`` exactly
    as ``mask_extraction_task(mask_overrides={mask_name: override})`` would, and
    writes the mask and its ``mask_selections.json`` entry. The caller records
    the override (see :func:`save_mask_overrides`) so a replay reproduces it.
    :func:`preview_task_mask` gives the same mask without writing it.

    Parameters
    ----------
    params : dict
        The keyword arguments the localizer's :func:`mask_extraction_task` call
        used (the ``PARAMS`` of its ``reproduce_masks.py``).
    mask_name : str
        The mask to re-derive (a contrast name, or ``"{name}_reverse"``).
    override : dict or None
        A ``mask_overrides`` entry, or None to restore the study's parameters.

    Returns
    -------
    TaskMaskSelection
        The selection record of the new mask.
    """
    binary_mask, selection, stat_img = _task_mask_from_cache(params, mask_name, override)
    root_dir = Path(params["root_dir"])
    bids_prefix = f"{params['participant_id']}_{params['session_id']}"
    masks_dir = root_dir / "03_masks"

    n_kept = selection.n_voxels_thresholded
    if n_kept > 0:
        ho_data, ho_labels = _subject_harvard_oxford(root_dir / "02_mnireg", bids_prefix, stat_img)
        selection.cluster_table = compute_cluster_table(
            binary_mask, stat_img.get_fdata(), stat_img.affine, atlas_data=ho_data, atlas_labels=ho_labels
        )
    masks_dir.mkdir(parents=True, exist_ok=True)
    save(
        Nifti1Image(binary_mask, stat_img.affine, stat_img.header), masks_dir / f"{bids_prefix}_{mask_name}_thr.nii.gz"
    )
    selections = load_task_mask_selections(masks_dir)
    selections[mask_name] = selection
    save_task_mask_selections(masks_dir, selections)
    logger.info(f"Updated task mask {mask_name}: {n_kept} voxels ({selection.selection_reason})")
    return selection


# =============================================================================
# SECTION 3C: CONFIG-DRIVEN DISPATCH (rest / task / mni)
# =============================================================================


def run_mask_extraction_from_config(
    mode: str,
    params: dict[str, Any],
    *,
    progress_callback: Callable[[str, str], None] | None = None,
    stop_event: threading.Event | None = None,
) -> list[Path]:
    """Dispatch to the right ``mask_extraction_*`` function by mode.

    Single source of truth for "how to call mask extraction". Both the live GUI
    pipeline and the generated ``reproduce_masks.py`` script call this, so neither
    can drift from the underlying function signatures.

    Parameters
    ----------
    mode : str
        One of ``"rest"`` (ICA), ``"task"`` (GLM), or ``"mni"`` (atlas fallback).
    params : dict
        The science-defining keyword arguments for the chosen function
        (``participant_id``, ``session_id``, ``root_dir``, inputs, atlas/threshold
        settings, ``overwrite``, ``qc_enabled``, ...). Forwarded as ``**params``,
        so this helper needs no edits when a function gains a parameter.
    progress_callback : callable, optional
        Runtime progress hook. Not part of ``params`` (not serializable).
    stop_event : threading.Event, optional
        Cooperative cancellation signal. Only ``mode="rest"`` supports it.

    Returns
    -------
    list[Path]
        Paths to the thresholded network mask files.
    """
    if mode == "rest":
        return mask_extraction_rest(**params, progress_callback=progress_callback, stop_event=stop_event)
    if mode == "task":
        return mask_extraction_task(**params, progress_callback=progress_callback)
    if mode == "mni":
        return mask_extraction_mni(**params, progress_callback=progress_callback)
    raise ValueError(f"Unknown mask extraction mode: {mode!r}. Expected 'rest', 'task', or 'mni'.")


# =============================================================================
# SECTION 3D: MANUAL ROI IMPORT
# =============================================================================
# For researchers with pre-existing NIfTI masks (FreeSurfer, fMRIPrep,
# hand-drawn ROIs, standard atlases). Bypasses the ICA/GLM extraction pipeline
# and imports masks directly into 03_masks/ for use in the feedback pipeline.
#
# Supports three input spaces:
# - "native": masks already in functional/EPI space (copied + binarized)
# - "anat": masks drawn on T1/structural (ANTs Rigid T1→EPI registration)
# - "mni": masks in MNI152 (ANTs SyN EPI→MNI, inverse applied to masks)


def _warp_mask_to_reference(
    mask_path: Path,
    ref_file: Union[str, Path],
    transform_files: list[str],
    out_file: Path,
    transformed_file: Path,
    *,
    overwrite: bool = False,
) -> Nifti1Image:
    """Warp a mask onto ``ref_file``'s grid (nearest neighbour), binarize it, and save it to ``out_file``.

    ``transformed_file`` holds the intermediate warped image and is removed afterwards.
    """
    apply_xfm(
        moving=mask_path,
        fixed=ref_file,
        transform_files=transform_files,
        output_file=transformed_file,
        interpolator="nearestNeighbor",
        overwrite=overwrite,
    )

    # Load transformed mask and binarize
    img = cast(Nifti1Image, load(transformed_file))
    data = np.asarray(img.get_fdata())

    binarized = Nifti1Image((data > 0).astype(np.uint8), img.affine, img.header)
    save(binarized, out_file)

    # Clean up intermediate transformed file
    if transformed_file.exists() and transformed_file != out_file:
        transformed_file.unlink()
    return binarized


def import_manual_masks(
    participant_id: str,
    session_id: str,
    root_dir: Union[str, Path],
    mask_paths: list[Union[str, Path]],
    mask_names: list[str],
    ref_file: Union[str, Path],
    mask_space: str = "native",
    anat_file: Union[str, Path] | None = None,
    overwrite: bool = False,
    qc_enabled: bool = False,
    qc_dir: Union[str, Path] | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> list[Path]:
    """Import pre-existing NIfTI masks into the feedback pipeline.

    Copies (and optionally registers) user-provided NIfTI masks into
    ``{root_dir}/03_masks/`` with standardised naming so that downstream
    coregistration and feedback can proceed normally.

    Parameters
    ----------
    participant_id : str
        Participant identifier (e.g. ``"sub-01"``).
    root_dir : str or Path
        Mask extraction root directory. Output goes to ``root_dir/03_masks/``.
    mask_paths : list of str or Path
        NIfTI files to import (one per mask). Must match ``mask_names`` in
        length.
    mask_names : list of str
        Human-readable names for each mask (e.g. ``["ContA", "DefaultA"]``).
        Used in output file naming and feedback display.
    ref_file : str or Path
        Native-space BOLD/EPI reference volume. Used as the registration
        target for ``"anat"`` and ``"mni"`` modes, and later by
        ``coregister_reference()``.
    mask_space : str
        Input space of the masks:

        - ``"native"`` — masks are already in functional/EPI space (same
          space as *ref_file*). Validated, binarised, and copied.
        - ``"anat"`` — masks are drawn on a T1/structural image. Rigid-body
          registration (T1→EPI) is computed via ANTs and applied to each mask.
          Requires *anat_file*.
        - ``"mni"`` — masks are in MNI152 standard space. SyN registration
          (EPI→MNI) is computed and the inverse is applied to each mask.

    anat_file : str or Path, optional
        Required when ``mask_space="anat"``. The T1/structural image on which
        the masks were drawn.
    overwrite : bool
        If ``True``, re-import even when output files already exist.
    qc_enabled : bool
        If ``True``, generate a mask-extraction QC report (overlays + HTML)
        via :func:`pineuro.qc.generate_mask_extraction_report` after import.
    qc_dir : str or Path, optional
        QC output directory. Defaults to ``root_dir.parent / "qc"`` when
        ``qc_enabled`` is set.
    progress_callback : callable, optional
        ``(step: str, detail: str) -> None`` for progress reporting.

    Returns
    -------
    list of Path
        Paths to the imported mask files in ``03_masks/``.

    Raises
    ------
    ValueError
        If *mask_paths* and *mask_names* lengths differ, *mask_space* is
        invalid, or *anat_file* is missing when required.
    FileNotFoundError
        If any input file does not exist.
    """
    root_dir = Path(root_dir)
    ref_file = Path(ref_file)
    bids_prefix = f"{participant_id}_{session_id}"

    from nilearn.image import resample_to_img

    def emit_progress(step: str, detail: str) -> None:
        logger.info(f"[Manual Import] {step}: {detail}")
        if progress_callback is not None:
            progress_callback(step, detail)

    # ---- Validation ----
    if not mask_paths:
        raise ValueError("mask_paths must contain at least one mask")
    if len(mask_paths) != len(mask_names):
        raise ValueError(f"mask_paths ({len(mask_paths)}) and mask_names ({len(mask_names)}) must have the same length")
    if len(set(mask_names)) != len(mask_names):
        raise ValueError(f"mask_names must be unique (they become output filenames); got {mask_names}")
    if any(not str(n).strip() for n in mask_names):
        raise ValueError(f"mask_names must all be non-empty; got {mask_names}")
    if mask_space not in ("native", "anat", "mni"):
        raise ValueError(f"mask_space must be 'native', 'anat', or 'mni', got '{mask_space}'")
    if mask_space == "anat" and anat_file is None:
        raise ValueError("anat_file is required when mask_space='anat'")
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_file}")
    for mp in mask_paths:
        if not Path(mp).exists():
            raise FileNotFoundError(f"Mask file not found: {mp}")
    if mask_space == "anat" and anat_file is not None and not Path(anat_file).exists():
        raise FileNotFoundError(f"Anatomical file not found: {anat_file}")

    masks_dir = root_dir / "03_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # ---- Space-specific registration ----
    xfm_files: list[str] | None = None

    if mask_space == "anat":
        anat_file = Path(anat_file)  # type: ignore[arg-type]
        reg_dir = root_dir / "02_anat_reg"
        reg_dir.mkdir(parents=True, exist_ok=True)

        emit_progress("Registration", "Computing T1→EPI rigid transform...")
        xfm_prefix = reg_dir / f"{bids_prefix}_anat_to_epi"
        _, xfm_files = compute_xfm(
            moving=anat_file,
            fixed=ref_file,
            output_prefix=xfm_prefix,
            type_of_transform="Rigid",
            overwrite=overwrite,
        )
        emit_progress("Registration", "T1→EPI transform computed")

    elif mask_space == "mni":
        reg_dir = root_dir / "02_mnireg"
        reg_dir.mkdir(parents=True, exist_ok=True)

        emit_progress("Registration", "Computing EPI→MNI SyN transform...")
        xfm_prefix = reg_dir / f"{bids_prefix}_epi_to_mni"
        _, _fwd_xfm, inv_xfm_files = compute_xfm_to_mni(
            input_file=ref_file,
            output_prefix=xfm_prefix,
            type_of_transform="SyN",
            overwrite=overwrite,
        )
        xfm_files = inv_xfm_files
        emit_progress("Registration", "EPI→MNI transform computed")

    # ---- Import each mask ----
    output_files: list[Path] = []

    for i, (mask_path, mask_name) in enumerate(zip(mask_paths, mask_names, strict=False), 1):
        mask_path = Path(mask_path)
        out_file = masks_dir / f"{bids_prefix}_{mask_name}_thr.nii.gz"

        if out_file.exists() and not overwrite:
            logger.info(f"Output exists, skipping: {out_file.name}")
            output_files.append(out_file)
            continue

        emit_progress(f"Mask {i}/{len(mask_paths)}", f"Importing {mask_name}...")

        if mask_space == "native":
            # Load and validate
            img = cast(Nifti1Image, load(mask_path))
            data = np.asarray(img.get_fdata())

            if data.ndim != 3:
                raise ValueError(f"Mask must be 3D, got {data.ndim}D: {mask_path}")

            # Warn on shape mismatch with ref_file
            ref_img = cast(Nifti1Image, load(ref_file))
            if data.shape != ref_img.shape[:3]:
                logger.warning(
                    f"Shape mismatch for {mask_name}: mask {data.shape} vs "
                    f"ref {ref_img.shape[:3]}. Resampling to reference grid."
                )
                emit_progress(
                    f"Mask {i}/{len(mask_paths)}",
                    f"{mask_name}: shape {data.shape} != reference {ref_img.shape[:3]}, resampling to reference grid",
                )
                img = resample_to_img(img, ref_img, interpolation="nearest")
                data = np.asarray(img.get_fdata())

            # Binarize
            binarized = (data > 0).astype(np.uint8)
            save(Nifti1Image(binarized, img.affine, img.header), out_file)

        else:
            # anat or mni: apply registration transform
            assert xfm_files is not None
            _warp_mask_to_reference(
                mask_path,
                ref_file,
                xfm_files,
                out_file,
                masks_dir / f"{bids_prefix}_{mask_name}_transformed.nii.gz",
                overwrite=overwrite,
            )

        n_voxels = int(np.sum(cast(Nifti1Image, nib.load(out_file)).get_fdata() > 0))
        emit_progress(f"Mask {i}/{len(mask_paths)}", f"{mask_name}: {n_voxels} voxels")
        output_files.append(out_file)

    # Save selections JSON for review dialog and persistence.
    # Per-mask dicts are written at the top level (matching the MNI-atlas
    # convention) so qc.generate_mask_extraction_report and the GUI mask
    # review dialog detect them via ``source_type``; scalar metadata keys
    # sit alongside and are skipped by ``isinstance(v, dict)`` guards.
    selections_data: dict[str, Any] = {
        "type": "manual",
        "ref_file": str(ref_file),
        "mask_space": mask_space,
    }
    for mask_name, out_file in zip(mask_names, output_files, strict=False):
        n_vox = int(np.sum(cast(Nifti1Image, nib.load(out_file)).get_fdata() > 0))
        selections_data[mask_name] = {
            "n_voxels_thresholded": n_vox,
            "source": "manual_import",
            "source_type": "manual_import",
            "mask_space": mask_space,
        }
    selections_file = masks_dir / "mask_selections.json"
    with open(selections_file, "w") as f:
        json.dump(selections_data, f, indent=2, default=_json_default)
    logger.info(f"Saved manual mask selections to {selections_file}")

    emit_progress("Complete", f"Imported {len(output_files)} masks to {masks_dir}")

    if qc_enabled:
        try:
            emit_progress("QC", "Generating mask extraction QC report...")
            report_path = qc.generate_mask_extraction_report(
                mask_extraction_dir=root_dir,
                participant_id=participant_id,
                session_id=session_id,
                network_names=list(mask_names),
                output_dir=Path(qc_dir) if qc_dir is not None else None,
                overwrite=overwrite,
                open_browser=False,
                ref_file=ref_file,
            )
            emit_progress("QC", f"QC report saved to {report_path}")
        except Exception as exc:  # QC failure must not abort a successful import
            logger.warning(f"Manual mask QC report generation failed: {exc}")
            emit_progress("QC", f"QC report generation failed: {exc}")

    return output_files


# =============================================================================
# SECTION 4: REFERENCE SCAN COREGISTRATION
# =============================================================================
# Between sessions, the participant's head position in the scanner may change.
# The reference scan coregistration addresses this by:
#
# 1. Acquiring a brief reference EPI scan at the start of each new session
#    (typically 2-4 volumes)
# 2. Registering this to the original resting-state reference
# 3. Applying the inverse transform to bring masks into the new position
#
# This ensures the network masks align with the participant's current brain
# position, even if they've moved since the original resting-state scan.


def coregister_reference(
    participant_id: str,
    session_id: str,
    root_dir: Union[str, Path],
    input_dicom: Union[str, Path],
    network_files: Union[str, Path, Sequence[Union[str, Path]]],
    ref_file: Union[str, Path],
    overwrite: bool = False,
    run_id: int | None = None,
    output_subdir: str = "01_reference",
) -> list[Path]:
    """
    Coregister extracted networks to participant's current position in the scanner
    using a reference scan acquisition.

    Parameters
    ----------
    participant_id : str
        Participant ID
    root_dir : str or Path
        Path to the root directory for all outputs
    input_dicom : str or Path
        Path to a single DICOM folder
    network_files : str or Path or list of str of Path
        Path to a single network file, or list of multiple network files
    ref_file : str or Path
        Path to NIfTI file to use as a BOLD reference for network space
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.
    run_id : int, optional
        Explicit run number to use. If None, derived from the DICOM folder
        name (run-XX). Falls back to the latest run (max of existing outputs
        and inferred run), or the next available run if none are found.
    output_subdir : str, default="01_reference"
        Subdirectory name under root_dir for outputs. Use "04_reference" for
        initial coregistration (mask_extraction), "01_reference" for realtime
        coregistration (feedback).

    Returns
    -------
    network_coreg_files
        List of co-registered networks

    Raises
    ------
    FileNotFoundError
        If input_dicoms, network_files, or ref_file does not exist or is empty

    """

    # Log the command
    logger.info(f"Running preprocessing for {participant_id} / {session_id}...")
    bids_prefix = f"{participant_id}_{session_id}"

    # Convert to Path object
    root_dir = Path(root_dir)

    # Convert to Path objects and validate
    input_path = Path(input_dicom)
    logger.info(f"Input DICOM folder: {str(input_path)}")
    if not input_path.exists():
        raise FileNotFoundError(f"Folder not found: {input_path}")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {input_path}")
    if not any(input_path.iterdir()):
        raise FileNotFoundError(f"Directory is empty: {input_path}")

    # Convert outputs to list if single file provided
    if isinstance(network_files, (str, Path)):
        network_files = [network_files]

    # Convert to Path objects and validate
    network_paths: list[Path] = []
    logger.info("Input network files:\n")
    for f in network_files:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        network_paths.append(path)
        logger.info(f"  {str(path)}")

    # Convert to Path object and validate
    ref_file = Path(ref_file)
    if not ref_file.exists():
        raise FileNotFoundError(f"File not found: {ref_file}")

    # Step 1: Determine run number (use latest reference if multiple exist)
    reference_dir = root_dir / output_subdir
    reference_dir.mkdir(parents=True, exist_ok=True)

    def _extract_run_number_from_path(path: Path) -> int | None:
        for part in reversed(path.parts):
            match = re.search(r"run-(\d+)", part)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
        return None

    pattern = re.compile(r"run-(\d+)")
    existing_runs: list[int] = []
    for file in reference_dir.glob(f"{bids_prefix}_task-reference_run-*.nii.gz"):
        match = pattern.search(file.stem)
        if match:
            try:
                existing_runs.append(int(match.group(1)))
            except ValueError:
                continue

    latest_existing_run = max(existing_runs) if existing_runs else None
    inferred_run = _extract_run_number_from_path(input_path) if run_id is None else None

    if run_id is not None:
        run_number = run_id
    elif inferred_run is not None:
        # Prefer the latest of inferred and existing runs
        run_number = inferred_run if latest_existing_run is None else max(inferred_run, latest_existing_run)
    else:
        # No hint from path; pick the next run after the latest existing
        run_number = (latest_existing_run or 0) + 1

    run_str = str(run_number).zfill(2)
    logger.info(f"Using reference run {run_str} (existing runs: {sorted(existing_runs) if existing_runs else 'none'})")

    # Step 2: Convert DICOMs to NIfTI
    nifti_file = dicom_to_nifti(
        dicom_dir=input_path,
        output_dir=reference_dir,
        output_name=f"{bids_prefix}_task-reference_run-{run_str}",
        overwrite=overwrite,
    )

    # Step 3: Compute mean
    mean_file = reference_dir / Path(f"{bids_prefix}_task-reference_run-{run_str}_mean.nii.gz")
    compute_mean(input_file=nifti_file, output_file=mean_file, overwrite=overwrite)

    # Step 4: Compute transformation (ANTs, ~28x faster than FLIRT)
    xfm_dir = reference_dir / "xfm"
    ref_stem = ref_file.stem.replace(".nii", "")
    output_prefix = xfm_dir / Path(f"{bids_prefix}_{ref_stem}_to_task-reference_run-{run_str}_mean")
    _, xfm_files = compute_xfm(
        moving=ref_file, fixed=mean_file, output_prefix=output_prefix, type_of_transform="Rigid", overwrite=overwrite
    )

    # Step 5: Apply transformation
    mask_dir = reference_dir / "masks"
    network_coreg_files = []
    for network in network_paths:
        output_file = mask_dir / (network.with_suffix("").stem + f"_to_task-reference_run-{run_str}_mean.nii.gz")
        apply_xfm(
            moving=network,
            fixed=mean_file,
            transform_files=xfm_files,
            output_file=output_file,
            interpolator="nearestNeighbor",
            overwrite=overwrite,
        )
        network_coreg_files.append(output_file)

    return network_coreg_files


def generate_reference_qc_overlays(
    participant_id: str,
    session_id: str,
    session_dir: Path,
    ref_file: Path,
    mask_extraction_dir: Path,
    network_coreg_files: Sequence[Path],
    run_str: str,
    n_networks: int = 17,
    overwrite: bool = False,
    reference_subdir: str = "01_reference",
) -> None:
    """
    Generate QC overlays in reference space using the Yeo atlas warped to
    the reference scan.

    Parameters
    ----------
    participant_id : str
        Participant ID
    session_dir : Path
        Session directory (mask_extraction for run-01, feedback for run-02+)
    ref_file : Path
        Path to BOLD reference file
    mask_extraction_dir : Path
        Path to mask extraction directory (for Yeo atlas)
    network_coreg_files : Sequence[Path]
        List of coregistered network mask files
    run_str : str
        Run number string (e.g., "01" or "02")
    n_networks : int, default=17
        Yeo parcellation (7 or 17 networks)
    overwrite : bool, default=False
        If True, overwrite existing files
    reference_subdir : str, default="01_reference"
        Subdirectory name under session_dir where reference outputs are stored.
        Use "04_reference" for initial coregistration (mask_extraction),
        "01_reference" for realtime coregistration (feedback).
    """
    if not network_coreg_files:
        return

    reference_dir = session_dir / reference_subdir
    bids_prefix = f"{participant_id}_{session_id}"
    mean_file = reference_dir / f"{bids_prefix}_task-reference_run-{run_str}_mean.nii.gz"
    if not mean_file.exists():
        logger.info(f"Skipping reference QC overlays; mean file not found at {mean_file}")
        return

    xfm_dir = reference_dir / "xfm"
    ref_stem = ref_file.stem.replace(".nii", "")
    xfm_files = sorted(xfm_dir.glob(f"{bids_prefix}_{ref_stem}_to_task-reference_run-{run_str}_mean_*"))
    if not xfm_files:
        logger.info(f"Skipping reference QC overlays; transforms not found in {xfm_dir}")
        return

    # Warp Yeo atlas to reference space if available (ICA workflow).
    # For task-based workflows the Yeo atlas may not exist; in that case
    # we skip atlas warping and use the mask itself for slice selection.
    yeo_sub = mask_extraction_dir / "02_mnireg" / f"Yeo_to_{ref_file.with_suffix('').stem}.nii.gz"
    atlas_reference = None
    labels: list[str] = []
    if yeo_sub.exists():
        atlas_reference = reference_dir / "masks" / f"Yeo_to_task-reference_run-{run_str}_mean.nii.gz"
        apply_xfm(
            moving=yeo_sub,
            fixed=mean_file,
            transform_files=[str(f) for f in xfm_files],
            output_file=atlas_reference,
            interpolator="nearestNeighbor",
            overwrite=overwrite,
        )
        atlas_data = cast(Nifti1Image, load(atlas_reference)).get_fdata()
        n_labels = int(np.max(atlas_data))
        labels = get_yeo_labels(n_networks=17 if n_labels > 7 else 7)
    else:
        logger.info(f"Yeo atlas not found at {yeo_sub}; using mask-based slice selection")

    qc_dir = session_dir.parent / "qc"
    overlays_dir = qc_dir / "overlays" / "reference"
    overlays: list[tuple[str, Path]] = []
    for network in network_coreg_files:
        stem = network.stem
        try:
            target = stem.split(f"{bids_prefix}_", 1)[1].split("_thr")[0]
        except IndexError:
            target = stem

        atlas_slices = qc.select_slices_from_atlas(
            atlas_path=atlas_reference if atlas_reference else network,
            labels=labels,
            target_label=str(target),
            top_n_per_plane=1,
            mask_path=network,  # Use mask for slice selection
        )
        overlay_png = overlays_dir / f"{stem}_overlay.png"
        qc.plot_mask_overlay_tripanel(
            mask_path=network,
            ref_path=mean_file,
            out_path=overlay_png,
            slices=atlas_slices,
            title=f"{target} (reference run {run_str})",
        )
        overlays.append((target, overlay_png))

    # Regenerate QC report to include reference overlays
    logger.info(f"Generated {len(overlays)} reference overlays in {overlays_dir}")

    # Extract network names from the overlay list
    network_names = [name for name, _ in overlays]

    # Regenerate report with reference overlays. Use the ICA report generator
    # (which knows about reference overlays); for task-based workflows this may
    # fail gracefully since there are no ICA outputs, but the overlay PNGs
    # are already written and available for the QC viewer.
    try:
        logger.info("Regenerating QC report with reference overlays...")
        qc.generate_mask_extraction_report(
            mask_extraction_dir=mask_extraction_dir,
            participant_id=participant_id,
            session_id=session_id,
            network_names=network_names,
            output_dir=qc_dir,
            n_networks=n_networks,
            overwrite=False,  # Skip existing images, just update HTML
            open_browser=False,
            ref_file=ref_file,
        )
    except Exception as e:
        logger.info(f"Skipping ICA QC report regeneration (task-based workflow): {e}")
