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
   For each target network (e.g., DefaultA, ContA):
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
import json  # JSON I/O for selection metadata
import logging
import re  # Regex for filename parsing
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

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
    dicom_to_nifti,  # DICOM conversion (for 2vol coregistration)
    # Atlas helpers
    get_yeo_labels,  # Get Yeo network names (7 or 17 networks)
)

# --- Module Logger ---
logger = logging.getLogger(__name__)


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
        Name of the network (e.g., "DefaultA", "ContA")
    source_type : str
        One of: "ica" for single ICA component, "ica_combined" for two merged
        ICA components, or "yeo_atlas" for atlas fallback
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
    source_type: str  # "ica", "ica_combined", or "yeo_atlas"
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
        json.dump(data, f, indent=2)

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

    # Handle backward compatibility for older selections without combined fields
    selections = {}
    for name, sel_data in data.items():
        # Add default values for new fields if missing
        if "secondary_component_index" not in sel_data:
            sel_data["secondary_component_index"] = None
        if "secondary_correlation" not in sel_data:
            sel_data["secondary_correlation"] = None
        selections[name] = MaskSelection(**sel_data)

    return selections


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
    n_yeo: int = 17,
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
    n_yeo : int
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
    labels = get_yeo_labels(n_networks=n_yeo)
    if network_name not in labels:
        raise ValueError(f"Network '{network_name}' not in Yeo-{n_yeo} labels")
    network_index = labels.index(network_name) + 1  # 1-based indexing

    # Load mask
    mask_img = load(mask_file)
    mask_data = mask_img.get_fdata().ravel()

    # Load and resample Yeo atlas to mask space
    atlas_img = load(yeo_subject_space)
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
    n_yeo: int = 17,
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
    n_yeo : int
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
    labels = get_yeo_labels(n_networks=n_yeo)
    if network_name not in labels:
        raise ValueError(f"Network '{network_name}' not in Yeo-{n_yeo} labels: {labels}")

    network_index = labels.index(network_name) + 1  # Yeo uses 1-based indexing

    # Load atlas and extract network region
    atlas_img = load(yeo_subject_space)
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
    mask_img = load(input_mask)
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
        """Keep top N voxels by absolute value."""
        flat = data.ravel()
        abs_flat = np.abs(flat)
        if n_voxels >= np.count_nonzero(abs_flat):
            # Keep all nonzero voxels if fewer than threshold
            return (abs_flat > 0).astype("uint8").reshape(data.shape)
        # Find threshold value for top N
        threshold_indices = np.argpartition(abs_flat, -n_voxels)[-n_voxels:]
        result = np.zeros(flat.shape, dtype="uint8")
        result[threshold_indices] = 1
        # Only keep where original was nonzero (positive values after thresholding)
        result = result & (flat > 0)
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
    components_img = load(components_file)
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

    # Threshold to keep top N voxels
    flat = combined.ravel()
    mask = np.zeros(flat.shape, dtype=bool)
    mask[np.argpartition(flat, -threshold)[-threshold:]] = True
    mask = mask.reshape(combined.shape)

    # Create binary mask (only keep positive values after thresholding)
    thresholded = ((combined * mask) > 0).astype("uint8")
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
    components_img = load(components_file)
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
    masks_dir: Union[str, Path],
    selection: MaskSelection,
    components_file: Union[str, Path],
    yeo_subject_space: Union[str, Path],
    n_yeo: int = 17,
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
    masks_dir : str or Path
        Directory containing masks (typically 03_masks/)
    selection : MaskSelection
        The user's selection for this network
    components_file : str or Path
        Path to ICA components file (4D NIfTI)
    yeo_subject_space : str or Path
        Path to Yeo atlas in subject space
    n_yeo : int
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

    # Output files
    network_file = masks_dir / f"{participant_id}_{network_name}.nii.gz"
    thr_file = masks_dir / f"{participant_id}_{network_name}_thr.nii.gz"

    if selection.source_type == "yeo_atlas":
        # Use Yeo atlas fallback
        logger.info(f"Regenerating {network_name} from Yeo atlas (fallback)")
        extract_yeo_mask_for_network(
            yeo_subject_space=yeo_subject_space,
            network_name=network_name,
            output_file=thr_file,
            n_yeo=n_yeo,
            threshold=threshold,
            overwrite=True,
        )
        # Also save an unthresholded version (full atlas region)
        labels = get_yeo_labels(n_networks=n_yeo)
        network_index = labels.index(network_name) + 1
        atlas_img = load(yeo_subject_space)
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

        components_img = load(components_file)
        components_data = components_img.get_fdata()

        if component_idx >= components_data.shape[3]:
            raise ValueError(f"Component index {component_idx} out of range (max: {components_data.shape[3] - 1})")

        # Extract component and save unthresholded
        ic_data = components_data[..., component_idx]
        ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header)
        save(ic_img, network_file)

        # Threshold and save
        flat = ic_data.ravel()
        mask = np.zeros(flat.shape, dtype=bool)
        mask[np.argpartition(flat, -threshold)[-threshold:]] = True
        mask = mask.reshape(ic_data.shape)
        thresholded = ((ic_data * mask) > 0).astype("uint8")
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

    from nilearn.decomposition import CanICA

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

    img = load(input_file)
    # Extract TR from 4th dimension of zooms (temporal spacing)
    # type: ignore - nibabel typing doesn't recognize header.get_zooms()[3]
    tr = img.header.get_zooms()[3]  # type: ignore

    # Convert nan to None (nilearn needs None to skip smoothing/filtering)
    import math

    if smooth_fwhm is not None and math.isnan(smooth_fwhm):
        smooth_fwhm = None
    if highpass is not None and math.isnan(highpass):
        highpass = None
    if lowpass is not None and math.isnan(lowpass):
        lowpass = None

    # Initialize CanICA
    logger.info("Initializing CanICA...")
    canica: Any = CanICA(
        mask=mask_file,
        n_components=n_components,
        smoothing_fwhm=smooth_fwhm,  # type: ignore - nilearn accepts None or float
        high_pass=highpass,
        low_pass=lowpass,
        threshold=None,  # type: ignore - nilearn accepts None or float
        t_r=tr,
        n_jobs=-1,  # Use all CPUs
        random_state=0,  # For reproducibility
        memory=str(output_dir / "cache"),
        memory_level=2,
        verbose=1,
    )

    # Fit CanICA
    logger.info("Computing ICA components... (this may take several minutes)")
    with Timer("CanICA decomposition", "mask_extraction"):
        try:
            canica.fit(input_file)
            logger.info("CanICA computation completed successfully")
        except Exception as e:
            error_msg = f"CanICA computation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # Get components as image
    components_img: SpatialImage = canica.components_img_  # type: ignore

    # Save components
    save(components_img, components_file)
    logger.info(f"Saved components to: {components_file}")

    # Save metadata
    np.savez(
        metadata_file, n_components=n_components, components_shape=components_img.shape, input_file=str(input_file)
    )
    logger.info(f"Saved metadata to: {metadata_file}")

    # Log component information
    components_data = components_img.get_fdata()
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
    Spatially coreralte a NIfTI file with a parcellation using FSL's fslcc.

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
    img = load(input_file)
    data = get_data(img)
    n = data.shape[3]

    # Load parcellation and resample (note: they should already be co-registered)
    logger.info(f"Loading: {parcellation_file}...")
    img_parcellation = load(parcellation_file)
    img_parcellation_rs = resample_to_img(img_parcellation, img, interpolation="nearest")
    data_parcellation = get_data(img_parcellation_rs).astype(int)

    # Load mask
    logger.info(f"Loading: {mask_file}...")
    img_mask = load(mask_file)
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
        correlation_matrix = num / (x_norm[:, None] * Y_norm[None, :])

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


def mask_extraction_rest(
    participant_id: str,
    root_dir: Union[str, Path],
    input_file: Union[str, Path],
    mask_file: Union[str, Path],
    ref_file: Union[str, Path],
    n_yeo: int = 17,
    yeo_targets: Union[str, list[str]] | None = None,
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
    n_yeo : int, default=17
        Yeo network parcellation to use. Options: 7 or 17.
        Ignored when ``network_atlas`` explicitly specifies the atlas.
    yeo_targets : str or List of str
        Network targets to extract masks for (e.g., ["DefaultA", "ContA"])
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

    Returns
    -------
    thr_files
        List of extracted thresholded masks

    Raises
    ------
    FileNotFoundError
        If input_file does not exist
    ValueError
        If network_atlas, n_yeo, or yeo_targets parameter is invalid

    """

    if yeo_targets is None:
        yeo_targets = ["DefaultA", "ContA"]

    from nilearn import datasets
    from nilearn.image import get_data

    # Helper to emit progress to both logger and callback
    def emit_progress(step: str, detail: str = ""):
        if detail:
            logger.info(f"{step}: {detail}")
        else:
            logger.info(step)
        if progress_callback:
            progress_callback(step, detail)

    # Log the command
    emit_progress(f"Mask extraction for {participant_id}")

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

    # Derive n_networks from network_atlas (overrides n_yeo for backward compat)
    if network_atlas in ATLAS_SOURCES:
        n_networks = ATLAS_SOURCES[network_atlas]["n_networks"]
    else:
        # Fallback: treat n_yeo as authoritative (backward compat)
        if n_yeo not in (7, 17):
            raise ValueError(f"n_yeo must be 7 or 17, got {n_yeo}")
        n_networks = n_yeo
        network_atlas = f"yeo_{n_yeo}"

    # Convert to list and validate targets against network labels
    if isinstance(yeo_targets, str):
        yeo_targets = [yeo_targets]
    labels = get_yeo_labels(n_networks=n_networks)
    for target in yeo_targets:
        if target not in labels:
            raise ValueError(
                f"Label '{target}' not in labels for {network_atlas} "
                f"({n_networks}-network parcellation). Available: {labels}"
            )

    # Step 1: ICA
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
    )
    components_img = load(components_file)
    components_data = get_data(components_img)
    emit_progress("Step 1/3", "ICA complete")

    # Step 2: MNI registration - bring Yeo atlas into participant space
    # Overview: We need the Yeo atlas in the same space as our ICA components
    # to perform spatial correlation. This requires: participant → MNI → Yeo → participant
    emit_progress("Step 2/3: MNI registration", "Computing transforms...")
    mnireg_dir = root_dir / "02_mnireg"

    # Step 2a: Compute transformation from participant space to MNI space
    # Uses ANTs SyN (nonlinear) registration between BOLD reference volumes
    output_prefix = mnireg_dir / Path(f"{ref_file.with_suffix('').stem}_to_MNI_boldref")
    mni_ref_file, sub2mni_fwd_xfm, sub2mni_inv_xfm = compute_xfm_to_mni(
        input_file=ref_file, output_prefix=output_prefix, overwrite=overwrite
    )

    atlas_display = ATLAS_SOURCES.get(network_atlas, {}).get("display_name", network_atlas)
    emit_progress("Step 2/3", f"Transforming {atlas_display} atlas to participant space...")

    if network_atlas.startswith("yeo"):
        # Yeo atlas path (existing behavior)
        yeo_atlas = datasets.fetch_atlas_yeo_2011(n_networks=n_networks, thickness="thick")
        yeo = yeo_atlas.maps
        yeo_mni = Path(yeo).with_suffix("").parent / Path(
            Path(yeo).with_suffix("").stem + "_interp_to_fMRIPrep_boldref.nii.gz"
        )

        atlas2sub = mnireg_dir / Path(f"Yeo_to_{ref_file.with_suffix('').stem}.nii.gz")
        if not yeo_mni.exists() or overwrite:
            apply_xfm_sform_to_mni(input_file=yeo, output_file=yeo_mni, interp="nearestneighbour", overwrite=overwrite)

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
        # Schaefer atlas: fetch, group parcels by network, register to subject space
        info = ATLAS_SOURCES[network_atlas]
        schaefer = datasets.fetch_atlas_schaefer_2018(
            n_rois=info["n_rois"],
            yeo_networks=n_networks,
            resolution_mm=2,
        )
        raw_labels = [lbl.decode() if isinstance(lbl, bytes) else str(lbl) for lbl in schaefer.labels]

        # Load Schaefer parcellation
        schaefer_maps = schaefer.maps
        schaefer_img = schaefer_maps if isinstance(schaefer_maps, nib.Nifti1Image) else nib.load(schaefer_maps)
        schaefer_data = schaefer_img.get_fdata()

        # Group parcels by network name: "7Networks_LH_Default_Temp_1" → "Default"
        network_labels = get_yeo_labels(n_networks)
        network_parcellation = np.zeros_like(schaefer_data)
        for parcel_idx, parcel_label in enumerate(raw_labels):
            # Extract network name from label (3rd underscore-delimited segment)
            parts = parcel_label.split("_")
            if len(parts) >= 3:
                net_name = parts[2]
            else:
                logger.warning(f"Unexpected Schaefer label format: {parcel_label}")
                continue
            try:
                net_idx = network_labels.index(net_name) + 1  # 1-indexed
            except ValueError:
                logger.warning(f"Network '{net_name}' from label '{parcel_label}' not in network labels")
                continue
            network_parcellation[schaefer_data == parcel_idx + 1] = net_idx

        # Save network-level parcellation as NIfTI in MNI space
        schaefer_network_mni = mnireg_dir / f"Schaefer_{info['n_rois']}_{n_networks}net_networks_mni.nii.gz"
        if not schaefer_network_mni.exists() or overwrite:
            network_img = Nifti1Image(
                network_parcellation.astype(np.int16),
                schaefer_img.affine,
                schaefer_img.header,
            )
            save(network_img, schaefer_network_mni)

        # Register to subject space
        atlas2sub = mnireg_dir / Path(f"Schaefer_to_{ref_file.with_suffix('').stem}.nii.gz")
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
            f"Unsupported network_atlas '{network_atlas}' for resting-state extraction. Use a Yeo or Schaefer atlas."
        )

    emit_progress("Step 2/3", "Registration complete")

    # Step 3: Extract network masks using ICA-atlas correlation
    # Now that ICA components and Yeo atlas are in same space, identify which
    # ICA components best represent each target network
    emit_progress("Step 3/3: Extracting network masks", f"Targets: {', '.join(yeo_targets)}")
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
    chosen_components = {target: int(sc_df[target].idxmax()) for target in yeo_targets}

    # Save initial mask selections (auto-selected based on correlation)
    initial_selections: dict[str, MaskSelection] = {}
    for target in yeo_targets:
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
    for target in yeo_targets:
        network_file = masks_dir / Path(f"{participant_id}_{target}.nii.gz")
        if not network_file.exists() or overwrite:
            # Find component index with maximum correlation for this network
            ic_ind = chosen_components[target]
            # Extract that component's spatial map as the network mask
            ic_data = components_data[..., ic_ind]  # type: ignore
            ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header)  # type: ignore
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
            img: SpatialImage = load(network_file)  # type: ignore
            data = img.get_fdata()
            flat = data.ravel()
            # Use argpartition to efficiently find top N voxels without full sort
            mask = np.zeros(flat.shape, dtype=bool)
            mask[np.argpartition(flat, -threshold)[-threshold:]] = True
            mask = mask.reshape(data.shape)
            thresholded = ((data * mask) > 0).astype("uint8")
            save(Nifti1Image(thresholded, img.affine, img.header), thr_file)  # type: ignore
        else:
            logger.info("Output file exists and overwrite=False. Skipping network thresholding.")
        network_thr_files.append(thr_file)

    emit_progress("Step 3/3", f"Complete ({len(network_thr_files)} masks extracted)")

    # Step 3d: Generate hemisphere-split masks for sham (if configured)
    # Each hemisphere gets its own threshold count (e.g., 2000 voxels each)
    if sham_hemisphere_network:
        emit_progress("Generating hemisphere-split masks", f"Network: {sham_hemisphere_network}")

        # Check if this network was already extracted (unthresholded)
        network_file = masks_dir / f"{participant_id}_{sham_hemisphere_network}.nii.gz"

        if not network_file.exists():
            # Network not in yeo_targets, need to extract it first
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
            left_file = masks_dir / f"{participant_id}_{sham_hemisphere_network}_left_thr.nii.gz"
            right_file = masks_dir / f"{participant_id}_{sham_hemisphere_network}_right_thr.nii.gz"
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
        network_names = [str(target) for target in yeo_targets]
        report_path = qc.generate_mask_extraction_report(
            mask_extraction_dir=root_dir,
            participant_id=participant_id,
            network_names=network_names,
            output_dir=qc_output_dir,
            n_yeo=n_networks,
            overwrite=overwrite,
            open_browser=False,
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

    # Output file paths
    wm_prob_file = output_dir / f"{participant_id}_tissue_wm_prob.nii.gz"
    csf_prob_file = output_dir / f"{participant_id}_tissue_csf_prob.nii.gz"
    wm_mask_file = output_dir / f"{participant_id}_tissue_wm_mask.nii.gz"
    csf_mask_file = output_dir / f"{participant_id}_tissue_csf_mask.nii.gz"

    # Check if final outputs exist (skip if not overwriting)
    if not overwrite and wm_mask_file.exists() and csf_mask_file.exists():
        logger.info("Tissue masks already exist, skipping creation")
        return {"wm": wm_mask_file, "csf": csf_mask_file}

    # Step 1: Fetch MNI tissue probability maps from templateflow
    emit_progress("Fetching tissue priors", "MNI152NLin2009cAsym WM/CSF")
    wm_mni = Path(str(tflow.get(
        "MNI152NLin2009cAsym", resolution=2, label="WM", suffix="probseg"
    )))
    csf_mni = Path(str(tflow.get(
        "MNI152NLin2009cAsym", resolution=2, label="CSF", suffix="probseg"
    )))
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

    # Step 3: Warp tissue probability maps MNI → subject space
    emit_progress("Warping tissue priors", "MNI → subject space")
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
            interpolator='linear',
            overwrite=overwrite,
        )
        logger.info(f"Warped {tissue_name} probability map to subject space")

    # Step 4: Threshold and erode to create binary masks
    emit_progress("Creating binary masks", "Threshold 0.9, erode 2 voxels")
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
        prob_img = nib.loadsave.load(prob_file)
        prob_data = np.asarray(prob_img.dataobj)
        binary_data = (prob_data >= 0.9).astype(np.uint8)
        n_pre_erosion = int(binary_data.sum())

        # Save pre-erosion binary for debugging
        binary_img = Nifti1Image(binary_data, prob_img.affine, prob_img.header)
        pre_erosion_file = prob_file.parent / prob_file.name.replace("_prob.", "_binary.")
        nib.loadsave.save(binary_img, str(pre_erosion_file))

        # Erode using existing utility (2 voxels, 6-connected)
        from pineuro.utils import erode_mask
        erode_mask(pre_erosion_file, mask_file, overwrite=True)

        # Validate non-empty
        eroded_img = nib.loadsave.load(mask_file)
        n_post_erosion = int((np.asarray(eroded_img.dataobj) > 0).sum())
        if n_post_erosion == 0:
            raise ValueError(
                f"{tissue_name.upper()} mask is empty after erosion "
                f"({n_pre_erosion} voxels before erosion). "
                f"The BOLD reference may have poor MNI registration."
            )
        logger.info(
            f"{tissue_name.upper()} mask: {n_pre_erosion} → {n_post_erosion} voxels "
            f"(after erosion)"
        )
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


# --- Legacy Atlas Region Definitions ---
# Curated combined entries kept for backward compatibility with atlas_name API.
# New code should use atlas_source + regions via fetch_atlas_mask().

_ATLAS_COMBINED_ENTRIES: dict[str, dict[str, Any]] = {
    "harvard_oxford_stg": {
        "description": "Superior Temporal Gyrus (anterior + posterior)",
        "atlas_type": "harvard_oxford",
        "atlas_fetch": "cort-maxprob-thr25-2mm",
        "regions": [
            "Superior Temporal Gyrus, anterior division",
            "Superior Temporal Gyrus, posterior division",
        ],
    },
    "harvard_oxford_ifg": {
        "description": "Inferior Frontal Gyrus (pars triangularis + opercularis)",
        "atlas_type": "harvard_oxford",
        "atlas_fetch": "cort-maxprob-thr25-2mm",
        "regions": [
            "Inferior Frontal Gyrus, pars triangularis",
            "Inferior Frontal Gyrus, pars opercularis",
        ],
    },
    "harvard_oxford_mtg": {
        "description": "Middle Temporal Gyrus (anterior + posterior + temporo-occipital)",
        "atlas_type": "harvard_oxford",
        "atlas_fetch": "cort-maxprob-thr25-2mm",
        "regions": [
            "Middle Temporal Gyrus, anterior division",
            "Middle Temporal Gyrus, posterior division",
            "Middle Temporal Gyrus, temporo-occipital part",
        ],
    },
    "harvard_oxford_smg": {
        "description": "Supramarginal Gyrus (anterior + posterior)",
        "atlas_type": "harvard_oxford",
        "atlas_fetch": "cort-maxprob-thr25-2mm",
        "regions": [
            "Supramarginal Gyrus, anterior division",
            "Supramarginal Gyrus, posterior division",
        ],
    },
    "harvard_oxford_cingulate": {
        "description": "Cingulate Gyrus (anterior + posterior)",
        "atlas_type": "harvard_oxford",
        "atlas_fetch": "cort-maxprob-thr25-2mm",
        "regions": [
            "Cingulate Gyrus, anterior division",
            "Cingulate Gyrus, posterior division",
        ],
    },
}


def get_atlas_region_registry() -> dict[str, dict[str, Any]]:
    """Get the legacy atlas region registry (curated combined entries only).

    Returns the static curated combined entries without fetching any atlas
    data from nilearn. This avoids the startup lag that previously occurred
    when Harvard-Oxford labels were fetched eagerly.

    For individual regions, use :func:`get_atlas_labels` with an atlas_source
    key instead.

    Returns
    -------
    dict
        Mapping of atlas_name to dict with "description", "atlas_fetch",
        and "regions" keys.
    """
    return dict(_ATLAS_COMBINED_ENTRIES)


# Keep ATLAS_REGION_REGISTRY as a reference to the combined entries for
# backward compatibility.
ATLAS_REGION_REGISTRY = _ATLAS_COMBINED_ENTRIES


def fetch_atlas_mask(
    output_file: Union[str, Path],
    atlas_name: str | None = None,
    atlas_source: str | None = None,
    regions: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Fetch an atlas from nilearn and extract specified regions as a binary mask.

    Supports two modes:

    1. **Legacy** (``atlas_name``): Looks up a curated combined entry from
       :func:`get_atlas_region_registry` (e.g., ``"harvard_oxford_stg"``).
    2. **New** (``atlas_source`` + ``regions``): Fetches any supported atlas
       and extracts the named regions directly.

    Parameters
    ----------
    output_file : str or Path
        Path to save the binary MNI-space mask NIfTI.
    atlas_name : str, optional
        Legacy key into the curated atlas registry.
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
        If neither ``atlas_name`` nor ``atlas_source`` is provided, or if
        region labels are not found in the atlas.
    """
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        logger.info(f"Atlas mask already exists: {output_file.name}")
        return output_file

    # --- Resolve atlas data and target regions ---
    if atlas_source is not None and regions is not None:
        # New pathway: fetch atlas by source key
        atlas_img, atlas_data, labels, label_indices = _fetch_atlas_and_resolve_regions(atlas_source, regions)
        atlas_label = f"{atlas_source} ({len(regions)} regions)"
    elif atlas_name is not None:
        # Legacy pathway: look up curated combined entry
        registry = get_atlas_region_registry()
        if atlas_name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(f"Unknown atlas_name '{atlas_name}'. Available: {available}")
        entry = registry[atlas_name]
        target_regions = entry["regions"]
        atlas_type = entry.get("atlas_type", "harvard_oxford")
        atlas_img, atlas_data, labels, label_indices = _fetch_atlas_and_resolve_regions(atlas_type, target_regions)
        atlas_label = atlas_name
    else:
        raise ValueError("Either atlas_name or (atlas_source + regions) must be provided")

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
        atlas_img = atlas_maps if isinstance(atlas_maps, nib.Nifti1Image) else nib.load(atlas_maps)
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
        atlas_img = atlas_maps if isinstance(atlas_maps, nib.Nifti1Image) else nib.load(atlas_maps)
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
        atlas_img = atlas_maps if isinstance(atlas_maps, nib.Nifti1Image) else nib.load(atlas_maps)
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the largest contiguous cluster within an atlas-masked stat map.

    Iteratively raises the statistical threshold until the largest spatially
    contiguous cluster within the atlas region is at most ``target_n_voxels``
    in size. This mirrors the approach used in FSL-based STG extraction
    pipelines where ``cluster`` is called repeatedly with increasing thresholds.

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
            largest_label = int(cluster_sizes.argmax()) + 1
            largest_size = int(cluster_sizes.max())
            binary_mask = (labeled == largest_label).astype(np.uint8)
            return binary_mask, {
                "outcome": "success",
                "final_threshold": float(threshold),
                "n_iterations": iteration + 1,
                "n_clusters_at_final": n_clusters,
                "cluster_size": largest_size,
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

        if largest_size <= target_n_voxels:
            binary_mask = (labeled == largest_label).astype(np.uint8)
            return binary_mask, {
                "outcome": "success" if iteration > 0 else "already_under_target",
                "final_threshold": float(threshold),
                "n_iterations": iteration + 1,
                "n_clusters_at_final": n_clusters,
                "cluster_size": largest_size,
                "atlas_voxel_count": atlas_voxel_count,
            }

        # Cluster too large — increase threshold
        _prev_labeled = labeled
        _prev_n_clusters = n_clusters
        threshold += step

    # Max iterations reached — return the current largest cluster
    binary_mask = (labeled == largest_label).astype(np.uint8)
    return binary_mask, {
        "outcome": "max_iterations_reached",
        "final_threshold": float(threshold),
        "n_iterations": max_iterations,
        "n_clusters_at_final": n_clusters,
        "cluster_size": largest_size,
        "atlas_voxel_count": atlas_voxel_count,
    }


@dataclass
class TaskMaskSelection:
    """Tracks the selection state for a single task-based GLM mask.

    Attributes
    ----------
    mask_name : str
        User-assigned name for this mask (e.g., "CEN", "DMN")
    source_type : str
        Always "glm_contrast" for task-based masks
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
        How the mask was thresholded (``"top_n"`` or ``"cluster"``)
    selection_reason : str
        Always "auto" for task-based (no manual selection needed)
    atlas_used : str or None
        Atlas name or path used for cluster thresholding, if applicable
    final_cluster_threshold : float or None
        Stat threshold that produced the final cluster (cluster method only)
    n_clusters_found : int or None
        Number of clusters at the final threshold (cluster method only)
    cluster_outcome : str or None
        Outcome of cluster thresholding (cluster method only)
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


def save_task_mask_selections(masks_dir: Union[str, Path], selections: dict[str, TaskMaskSelection]) -> Path:
    """Save task-based mask selections to JSON file.

    Parameters
    ----------
    masks_dir : str or Path
        Directory containing the masks (typically 02_masks/)
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
        json.dump(data, f, indent=2)

    logger.info(f"Saved task mask selections to {output_file}")
    return output_file


def mask_extraction_task(
    participant_id: str,
    root_dir: Union[str, Path],
    input_file: Union[str, Path],
    mask_file: Union[str, Path],
    events: Union[str, Path, pd.DataFrame],
    contrasts: dict[str, str],
    tr: float,
    threshold: int = 2000,
    threshold_method: str = "top_n",
    atlas_mask: Union[str, Path] | None = None,
    atlas_name: str | None = None,
    atlas_source: str | None = None,
    atlas_regions: list[str] | None = None,
    ref_file: Union[str, Path] | None = None,
    cluster_start_threshold: float = 2.0,
    cluster_step: float = 0.1,
    stat_type: str = "t",
    smoothing_fwhm: float | None = 5.0,
    high_pass: float = 0.01,
    drift_model: str = "cosine",
    hrf_model: str = "spm",
    overwrite: bool = False,
    qc_enabled: bool = True,
    qc_dir: Union[str, Path] | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
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
      map to an anatomical ROI (e.g., Harvard-Oxford STG), then iteratively raises
      the stat threshold until the largest spatially contiguous cluster fits the
      target size. Guarantees a single contiguous cluster within the atlas region.

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
    atlas_mask : str, Path, or None
        Path to a binary atlas NIfTI in subject space. Used with
        ``threshold_method="cluster"`` to constrain the search region.
        Mutually exclusive with ``atlas_name``.
    atlas_name : str or None
        Legacy: name of a curated combined atlas entry (e.g.,
        ``"harvard_oxford_stg"``). See :func:`get_atlas_region_registry`.
        Requires ``ref_file`` for registration to subject space.
        Mutually exclusive with ``atlas_mask`` and ``atlas_source``.
    atlas_source : str or None
        Key into ``ATLAS_SOURCES`` (e.g., ``"harvard_oxford"``, ``"aal"``,
        ``"schaefer_400_7net"``). Used with ``atlas_regions`` to specify
        which regions to extract. Mutually exclusive with ``atlas_name``.
    atlas_regions : list[str] or None
        Region labels from the atlas to include in the mask. Required
        when ``atlas_source`` is provided.
    ref_file : str, Path, or None
        BOLD reference image for atlas registration when ``atlas_name`` is
        provided. Typically the temporal mean from preprocessing.
    cluster_start_threshold : float
        Starting stat threshold for iterative cluster search (default: 2.0).
        Only used when ``threshold_method="cluster"``.
    cluster_step : float
        Amount to increase the threshold per iteration (default: 0.1).
        Only used when ``threshold_method="cluster"``.
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
    overwrite : bool
        If True, rerun even if outputs exist (default: False)
    qc_enabled : bool
        If True, generate QC report (default: True)
    qc_dir : str, Path, or None
        Custom QC output directory. If None, uses {root_dir}/../qc/
    progress_callback : callable or None
        Function(step: str, detail: str) for progress updates

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
    ...     atlas_name="harvard_oxford_stg",
    ...     ref_file='/data/sub-01/preprocessing/sub-01_task-selfother_mean.nii.gz',
    ... )

    See Also
    --------
    mask_extraction_rest : ICA-based extraction from resting-state data
    coregister_2vol : Align masks to feedback scanner position
    fetch_atlas_mask : Fetch anatomical atlas regions from nilearn
    cluster_threshold_stat_map : Iterative cluster thresholding algorithm
    """
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.image import resample_to_img

    # --- Setup ---
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
    if threshold_method not in ("top_n", "cluster"):
        raise ValueError(f"threshold_method must be 'top_n' or 'cluster', got '{threshold_method}'")
    if threshold_method == "cluster":
        if atlas_mask is None and atlas_name is None and atlas_source is None:
            raise ValueError(
                "When threshold_method='cluster', one of atlas_mask, atlas_name, "
                "or atlas_source must be provided to define the anatomical ROI."
            )
        n_sources = sum(x is not None for x in [atlas_mask, atlas_name, atlas_source])
        if n_sources > 1:
            raise ValueError("Provide only one of atlas_mask, atlas_name, or atlas_source, not multiple.")
        if atlas_name is not None and ref_file is None:
            raise ValueError(
                "ref_file is required when atlas_name is provided, to register the atlas from MNI to subject space."
            )
        registry = get_atlas_region_registry()
        if atlas_name is not None and atlas_name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(f"Unknown atlas_name '{atlas_name}'. Available: {available}")

    n_steps = 3 if threshold_method == "cluster" else 2

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
        stat_path = glm_dir / f"{participant_id}_{mask_name}_{stat_type}map.nii.gz"
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
            design_file = glm_dir / f"{participant_id}_design_matrix.tsv"
            design_matrix.to_csv(design_file, sep="\t", index=True)

            # Compute and save each contrast
            for mask_name, contrast_expr in contrasts.items():
                emit_progress(f"Step 1/{n_steps}", f"Computing contrast: {mask_name} = {contrast_expr}")

                stat_map = model.compute_contrast(
                    contrast_expr,
                    stat_type=stat_type,
                    output_type="stat",
                )
                nib.save(stat_map, str(stat_map_paths[mask_name]))

            # Save GLM metadata
            glm_info = {
                "participant_id": participant_id,
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
            info_file = glm_dir / f"{participant_id}_glm_info.json"
            with open(info_file, "w") as f:
                json.dump(glm_info, f, indent=2)

        emit_progress(f"Step 1/{n_steps}", "GLM fit complete")

    # =========================================================================
    # Step 2/3: Atlas Registration (cluster mode only)
    # =========================================================================
    atlas_subject_space_data = None  # Set below if cluster mode

    if threshold_method == "cluster":
        mnireg_dir = root_dir / "02_mnireg"
        mnireg_dir.mkdir(parents=True, exist_ok=True)

        emit_progress(f"Step 2/{n_steps}: Preparing atlas mask")

        if atlas_mask is not None:
            # User-provided atlas — use directly (assumed in subject space)
            atlas_mask_path = Path(atlas_mask)
            if not atlas_mask_path.exists():
                raise FileNotFoundError(f"Atlas mask not found: {atlas_mask_path}")
            atlas_img = load(atlas_mask_path)
            atlas_label = atlas_mask_path.name
            emit_progress(f"Step 2/{n_steps}", f"Using user-provided atlas: {atlas_label}")
        else:
            # Fetch atlas and register to subject space
            assert ref_file is not None  # validated above
            ref_file = Path(ref_file)

            # Determine fetch params from atlas_source or legacy atlas_name
            if atlas_source is not None:
                fetch_label = atlas_source
                atlas_mni_path = mnireg_dir / f"{atlas_source}_mni.nii.gz"
                atlas_mni_path = fetch_atlas_mask(
                    output_file=atlas_mni_path,
                    atlas_source=atlas_source,
                    regions=atlas_regions or [],
                    overwrite=overwrite,
                )
            else:
                assert atlas_name is not None  # validated above
                fetch_label = atlas_name
                atlas_mni_path = mnireg_dir / f"{atlas_name}_mni.nii.gz"
                atlas_mni_path = fetch_atlas_mask(
                    output_file=atlas_mni_path,
                    atlas_name=atlas_name,
                    overwrite=overwrite,
                )

            emit_progress(f"Step 2/{n_steps}", f"Fetched {fetch_label} atlas in MNI space")

            # Sub-step: Register to subject space via ANTs
            xfm_prefix = mnireg_dir / f"{participant_id}_to_MNI_boldref"
            mni_ref_file, sub2mni_fwd_xfm, sub2mni_inv_xfm = compute_xfm_to_mni(
                input_file=ref_file,
                output_prefix=xfm_prefix,
                overwrite=overwrite,
            )

            # Sub-step: Apply inverse transform (MNI -> subject space)
            atlas_subject_path = mnireg_dir / f"{fetch_label}_to_{participant_id}.nii.gz"
            apply_xfm(
                moving=atlas_mni_path,
                fixed=ref_file,
                transform_files=sub2mni_inv_xfm,
                output_file=atlas_subject_path,
                interpolator="nearestNeighbor",
                invert=False,
                overwrite=overwrite,
            )
            atlas_img = load(atlas_subject_path)
            atlas_label = fetch_label
            emit_progress(f"Step 2/{n_steps}", "Atlas registered to subject space")

        # Resample atlas to stat map resolution if needed
        first_stat_path = stat_map_paths[list(contrasts.keys())[0]]
        stat_ref_img: SpatialImage = load(first_stat_path)  # type: ignore
        atlas_resampled_img = resample_to_img(atlas_img, stat_ref_img, interpolation="nearest")
        atlas_subject_space_data = (atlas_resampled_img.get_fdata() > 0.5).astype(np.float64)

        # Save atlas-in-subject-space for QC
        masks_dir = root_dir / "03_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        atlas_qc_path = masks_dir / f"{participant_id}_atlas_{atlas_label}.nii.gz"
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
    # Step {n_steps}/{n_steps}: Threshold to Binary Masks
    # =========================================================================
    if threshold_method == "cluster":
        masks_dir = root_dir / "03_masks"
    else:
        masks_dir = root_dir / "02_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    step_label = f"Step {n_steps}/{n_steps}"
    if threshold_method == "top_n":
        emit_progress(f"{step_label}: Thresholding masks", f"{threshold} voxels per mask")
    else:
        emit_progress(
            f"{step_label}: Cluster thresholding", f"target={threshold} voxels, start_z={cluster_start_threshold}"
        )

    network_thr_files = []
    selections: dict[str, TaskMaskSelection] = {}

    for mask_name, contrast_expr in contrasts.items():
        stat_map_path = stat_map_paths[mask_name]

        # Save unthresholded map to masks dir
        unthr_file = masks_dir / f"{participant_id}_{mask_name}.nii.gz"
        thr_file = masks_dir / f"{participant_id}_{mask_name}_thr.nii.gz"

        if thr_file.exists() and not overwrite:
            logger.info(f"Thresholded mask exists and overwrite=False: {thr_file.name}")
            network_thr_files.append(thr_file)

            # Still record selection metadata from existing file
            img: SpatialImage = load(thr_file)  # type: ignore
            n_voxels = int(img.get_fdata().sum())
            stat_img: SpatialImage = load(stat_map_path)  # type: ignore
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
        stat_img: SpatialImage = load(stat_map_path)  # type: ignore
        stat_data = stat_img.get_fdata()

        # Copy unthresholded map
        save(Nifti1Image(stat_data, stat_img.affine, stat_img.header), unthr_file)  # type: ignore

        positive_mask = stat_data > 0
        n_positive = int(positive_mask.sum())
        peak_stat = float(np.nanmax(stat_data)) if n_positive > 0 else 0.0

        if threshold_method == "top_n":
            # === TOP-N THRESHOLDING (existing logic) ===
            if n_positive == 0:
                logger.warning(f"No positive voxels in contrast '{mask_name}' ({contrast_expr}). Saving empty mask.")
                binary_mask = np.zeros(stat_data.shape, dtype=np.uint8)
                n_kept = 0
            elif n_positive <= threshold:
                logger.warning(
                    f"Contrast '{mask_name}' has only {n_positive} positive voxels "
                    f"(requested {threshold}). Keeping all positive voxels."
                )
                binary_mask = positive_mask.astype(np.uint8)
                n_kept = n_positive
            else:
                # Standard case: threshold to top N voxels
                flat = stat_data.ravel()
                mask = np.zeros(flat.shape, dtype=bool)
                mask[np.argpartition(flat, -threshold)[-threshold:]] = True
                mask = mask.reshape(stat_data.shape)
                # Intersect with positive voxels only
                binary_mask = (mask & positive_mask).astype(np.uint8)
                n_kept = int(binary_mask.sum())

            selections[mask_name] = TaskMaskSelection(
                mask_name=mask_name,
                contrast_definition=contrast_expr,
                stat_type=stat_type,
                peak_stat=peak_stat,
                n_voxels_above_zero=n_positive,
                n_voxels_thresholded=n_kept,
                threshold_method="top_n",
            )

        else:
            # === CLUSTER-BASED THRESHOLDING ===
            assert atlas_subject_space_data is not None

            # Save atlas-masked stat map for QC
            atlas_masked_stat = stat_data * atlas_subject_space_data
            atlas_masked_file = masks_dir / f"{participant_id}_{mask_name}_atlas_masked.nii.gz"
            save(
                Nifti1Image(atlas_masked_stat, stat_img.affine, stat_img.header),  # type: ignore
                atlas_masked_file,
            )

            binary_mask, cluster_info = cluster_threshold_stat_map(
                stat_data=stat_data,
                atlas_data=atlas_subject_space_data,
                target_n_voxels=threshold,
                start_threshold=cluster_start_threshold,
                step=cluster_step,
            )
            n_kept = int(binary_mask.sum())
            outcome = cluster_info["outcome"]

            if outcome in ("success", "already_under_target"):
                emit_progress(
                    step_label,
                    f"{mask_name}: {n_kept} voxels "
                    f"(cluster at z>={cluster_info['final_threshold']:.1f}, "
                    f"{cluster_info['n_clusters_at_final']} clusters)",
                )
            elif outcome in ("no_clusters_found", "empty_atlas"):
                logger.warning(
                    f"Cluster thresholding failed for '{mask_name}': {outcome}. "
                    f"Falling back to top-N within atlas region."
                )
                # Fallback: top-N on atlas-masked stat data
                atlas_positive = atlas_masked_stat > 0
                n_atlas_positive = int(atlas_positive.sum())
                if n_atlas_positive == 0:
                    binary_mask = np.zeros(stat_data.shape, dtype=np.uint8)
                    n_kept = 0
                elif n_atlas_positive <= threshold:
                    binary_mask = atlas_positive.astype(np.uint8)
                    n_kept = n_atlas_positive
                else:
                    flat = atlas_masked_stat.ravel()
                    mask = np.zeros(flat.shape, dtype=bool)
                    mask[np.argpartition(flat, -threshold)[-threshold:]] = True
                    mask = mask.reshape(stat_data.shape)
                    binary_mask = (mask & atlas_positive).astype(np.uint8)
                    n_kept = int(binary_mask.sum())
                outcome = "fallback_top_n"
                emit_progress(step_label, f"{mask_name}: {outcome} — {n_kept} voxels (fallback)")
            elif outcome == "max_iterations_reached":
                logger.warning(
                    f"Cluster thresholding for '{mask_name}' hit max iterations. "
                    f"Using largest cluster ({n_kept} voxels, target was {threshold})."
                )

            selections[mask_name] = TaskMaskSelection(
                mask_name=mask_name,
                contrast_definition=contrast_expr,
                stat_type=stat_type,
                peak_stat=peak_stat,
                n_voxels_above_zero=n_positive,
                n_voxels_thresholded=n_kept,
                threshold_method="cluster",
                atlas_used=str(atlas_name or atlas_mask),
                final_cluster_threshold=cluster_info.get("final_threshold"),
                n_clusters_found=cluster_info.get("n_clusters_at_final"),
                cluster_outcome=outcome,
            )

        save(Nifti1Image(binary_mask, stat_img.affine, stat_img.header), thr_file)  # type: ignore
        network_thr_files.append(thr_file)

        emit_progress(step_label, f"{mask_name}: {n_kept} voxels (peak {stat_type}={peak_stat:.2f})")

    # Save selection metadata
    save_task_mask_selections(masks_dir, selections)

    emit_progress(step_label, f"Complete ({len(network_thr_files)} masks extracted)")

    # Generate QC report
    if qc_enabled:
        emit_progress("Generating QC report...")
        qc_output_dir = Path(qc_dir) if qc_dir is not None else root_dir.parent / "qc"
        mask_names = list(contrasts.keys())
        report_path = qc.generate_task_mask_extraction_report(
            mask_extraction_dir=root_dir,
            participant_id=participant_id,
            mask_names=mask_names,
            contrasts=contrasts,
            stat_type=stat_type,
            output_dir=qc_output_dir,
            overwrite=overwrite,
            open_browser=False,
        )
        emit_progress("QC report", f"Saved to {report_path.name}")

    return network_thr_files


# =============================================================================
# SECTION 3B: MANUAL ROI IMPORT
# =============================================================================
# For researchers with pre-existing NIfTI masks (FreeSurfer, fMRIPrep,
# hand-drawn ROIs, standard atlases). Bypasses the ICA/GLM extraction pipeline
# and imports masks directly into 03_masks/ for use in the feedback pipeline.
#
# Supports three input spaces:
# - "native": masks already in functional/EPI space (copied + binarized)
# - "anat": masks drawn on T1/structural (ANTs Rigid T1→EPI registration)
# - "mni": masks in MNI152 (ANTs SyN EPI→MNI, inverse applied to masks)


def import_manual_masks(
    participant_id: str,
    root_dir: Union[str, Path],
    mask_paths: list[Union[str, Path]],
    mask_names: list[str],
    ref_file: Union[str, Path],
    mask_space: str = "native",
    anat_file: Union[str, Path] | None = None,
    overwrite: bool = False,
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
        ``coregister_2vol()``.
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

    from nilearn.image import resample_to_img

    def emit_progress(step: str, detail: str) -> None:
        logger.info(f"[Manual Import] {step}: {detail}")
        if progress_callback is not None:
            progress_callback(step, detail)

    # ---- Validation ----
    if len(mask_paths) != len(mask_names):
        raise ValueError(f"mask_paths ({len(mask_paths)}) and mask_names ({len(mask_names)}) must have the same length")
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
        xfm_prefix = reg_dir / f"{participant_id}_anat_to_epi"
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
        xfm_prefix = reg_dir / f"{participant_id}_epi_to_mni"
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
        out_file = masks_dir / f"{participant_id}_{mask_name}_thr.nii.gz"

        if out_file.exists() and not overwrite:
            logger.info(f"Output exists, skipping: {out_file.name}")
            output_files.append(out_file)
            continue

        emit_progress(f"Mask {i}/{len(mask_paths)}", f"Importing {mask_name}...")

        if mask_space == "native":
            # Load and validate
            img: SpatialImage = load(mask_path)  # type: ignore
            data = np.asarray(img.get_fdata())

            if data.ndim != 3:
                raise ValueError(f"Mask must be 3D, got {data.ndim}D: {mask_path}")

            # Warn on shape mismatch with ref_file
            ref_img: SpatialImage = load(ref_file)  # type: ignore
            if data.shape != ref_img.shape[:3]:
                logger.warning(
                    f"Shape mismatch for {mask_name}: mask {data.shape} vs "
                    f"ref {ref_img.shape[:3]}. Resampling to reference grid."
                )
                img = resample_to_img(img, ref_img, interpolation="nearest")
                data = np.asarray(img.get_fdata())

            # Binarize
            binarized = (data > 0).astype(np.uint8)
            save(Nifti1Image(binarized, img.affine, img.header), out_file)  # type: ignore

        else:
            # anat or mni: apply registration transform
            assert xfm_files is not None

            transformed_file = masks_dir / f"{participant_id}_{mask_name}_transformed.nii.gz"
            apply_xfm(
                moving=mask_path,
                fixed=ref_file,
                transform_files=xfm_files,
                output_file=transformed_file,
                interpolator="nearestNeighbor",
                overwrite=overwrite,
            )

            # Load transformed mask and binarize
            img = load(transformed_file)  # type: ignore
            data = np.asarray(img.get_fdata())

            binarized = (data > 0).astype(np.uint8)
            save(Nifti1Image(binarized, img.affine, img.header), out_file)  # type: ignore

            # Clean up intermediate transformed file
            if transformed_file.exists() and transformed_file != out_file:
                transformed_file.unlink()

        n_voxels = int(np.sum(nib.load(out_file).get_fdata() > 0))
        emit_progress(f"Mask {i}/{len(mask_paths)}", f"{mask_name}: {n_voxels} voxels")
        output_files.append(out_file)

    # Save selections JSON for review dialog and persistence
    selections_data = {
        "type": "manual",
        "ref_file": str(ref_file),
        "mask_space": mask_space,
        "selections": {},
    }
    for mask_name, out_file in zip(mask_names, output_files, strict=False):
        n_vox = int(np.sum(nib.load(out_file).get_fdata() > 0))
        selections_data["selections"][mask_name] = {
            "n_voxels_thresholded": n_vox,
            "source": "manual_import",
            "mask_space": mask_space,
        }
    selections_file = masks_dir / "mask_selections.json"
    with open(selections_file, "w") as f:
        json.dump(selections_data, f, indent=2)
    logger.info(f"Saved manual mask selections to {selections_file}")

    emit_progress("Complete", f"Imported {len(output_files)} masks to {masks_dir}")
    return output_files


# =============================================================================
# SECTION 4: 2-VOLUME COREGISTRATION
# =============================================================================
# Between sessions, the participant's head position in the scanner may change.
# The "2-volume" coregistration addresses this by:
#
# 1. Acquiring a brief (2-volume) EPI scan at the start of each new session
# 2. Registering this to the original resting-state reference
# 3. Applying the inverse transform to bring masks into the new position
#
# This ensures the network masks align with the participant's current brain
# position, even if they've moved since the original resting-state scan.
#
# The name "2vol" refers to the minimal acquisition needed - just 2 volumes
# are enough to create a reliable registration target (averaged to reduce noise).


def coregister_2vol(
    participant_id: str,
    root_dir: Union[str, Path],
    input_dicom: Union[str, Path],
    network_files: Union[str, Path, Sequence[Union[str, Path]]],
    ref_file: Union[str, Path],
    overwrite: bool = False,
    run_id: int | None = None,
    output_subdir: str = "01_2vol",
) -> list[Path]:
    """
    Coregister extracted networks to participant's current position in the scanner
    using a two volume (2vol) acquisition.

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
    output_subdir : str, default="01_2vol"
        Subdirectory name under root_dir for outputs. Use "04_2vol" for
        initial coregistration (mask_extraction), "01_2vol" for realtime
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
    logger.info(f"Running preprocessing for {participant_id}...")

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

    # Step 1: Determine run number (use latest 2vol if multiple exist)
    twovol_dir = root_dir / output_subdir
    twovol_dir.mkdir(parents=True, exist_ok=True)

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
    for file in twovol_dir.glob(f"{participant_id}_task-2vol_run-*.nii.gz"):
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
    logger.info(f"Using 2vol run {run_str} (existing runs: {sorted(existing_runs) if existing_runs else 'none'})")

    # Step 2: Convert DICOMs to NIfTI
    nifti_file = dicom_to_nifti(
        dicom_dir=input_path,
        output_dir=twovol_dir,
        output_name=f"{participant_id}_task-2vol_run-{run_str}",
        overwrite=overwrite,
    )

    # Step 3: Compute mean
    mean_file = twovol_dir / Path(f"{participant_id}_task-2vol_run-{run_str}_mean.nii.gz")
    compute_mean(input_file=nifti_file, output_file=mean_file, overwrite=overwrite)

    # Step 4: Compute transformation (ANTs, ~28x faster than FLIRT)
    xfm_dir = twovol_dir / "xfm"
    output_prefix = xfm_dir / Path(f"{participant_id}_boldref_to_task-2vol_run-{run_str}_mean")
    _, xfm_files = compute_xfm(
        moving=ref_file, fixed=mean_file, output_prefix=output_prefix, type_of_transform="Rigid", overwrite=overwrite
    )

    # Step 5: Apply transformation
    mask_dir = twovol_dir / "masks"
    network_coreg_files = []
    for network in network_paths:
        output_file = mask_dir / (network.with_suffix("").stem + f"_to_task-2vol_run-{run_str}_mean.nii.gz")
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


def generate_2vol_qc_overlays(
    participant_id: str,
    session_dir: Path,
    ref_file: Path,
    mask_extraction_dir: Path,
    network_coreg_files: Sequence[Path],
    run_str: str,
    n_yeo: int = 17,
    overwrite: bool = False,
    twovol_subdir: str = "01_2vol",
) -> None:
    """
    Generate QC overlays in 2vol space using the Yeo atlas warped to 2vol.

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
    n_yeo : int, default=17
        Yeo parcellation (7 or 17 networks)
    overwrite : bool, default=False
        If True, overwrite existing files
    twovol_subdir : str, default="01_2vol"
        Subdirectory name under session_dir where 2vol outputs are stored.
        Use "04_2vol" for initial coregistration (mask_extraction),
        "01_2vol" for realtime coregistration (feedback).
    """
    if not network_coreg_files:
        return

    twovol_dir = session_dir / twovol_subdir
    mean_file = twovol_dir / f"{participant_id}_task-2vol_run-{run_str}_mean.nii.gz"
    if not mean_file.exists():
        logger.info(f"Skipping 2vol QC overlays; mean file not found at {mean_file}")
        return

    xfm_dir = twovol_dir / "xfm"
    xfm_files = sorted(xfm_dir.glob(f"{participant_id}_boldref_to_task-2vol_run-{run_str}_mean_*"))
    if not xfm_files:
        logger.info(f"Skipping 2vol QC overlays; transforms not found in {xfm_dir}")
        return

    # Warp Yeo atlas to 2vol space if available (ICA workflow).
    # For task-based workflows the Yeo atlas may not exist; in that case
    # we skip atlas warping and use the mask itself for slice selection.
    yeo_sub = mask_extraction_dir / "02_mnireg" / f"Yeo_to_{ref_file.with_suffix('').stem}.nii.gz"
    atlas_2vol = None
    labels: list[str] = []
    if yeo_sub.exists():
        atlas_2vol = twovol_dir / "masks" / f"Yeo_to_task-2vol_run-{run_str}_mean.nii.gz"
        apply_xfm(
            moving=yeo_sub,
            fixed=mean_file,
            transform_files=xfm_files,
            output_file=atlas_2vol,
            interpolator="nearestNeighbor",
            overwrite=overwrite,
        )
        atlas_data = load(atlas_2vol).get_fdata()
        n_labels = int(np.max(atlas_data))
        labels = get_yeo_labels(n_networks=17 if n_labels > 7 else 7)
    else:
        logger.info(f"Yeo atlas not found at {yeo_sub}; using mask-based slice selection")

    qc_dir = session_dir.parent / "qc"
    overlays_dir = qc_dir / "overlays" / "2vol"
    overlays: list[tuple[str, Path]] = []
    for network in network_coreg_files:
        stem = network.stem
        try:
            target = stem.split(f"{participant_id}_", 1)[1].split("_thr")[0]
        except IndexError:
            target = stem

        atlas_slices = qc.select_slices_from_atlas(
            atlas_path=atlas_2vol if atlas_2vol else network,
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
            title=f"{target} (2vol run {run_str})",
        )
        overlays.append((target, overlay_png))

    # Regenerate QC report to include 2vol overlays
    logger.info(f"Generated {len(overlays)} 2vol overlays in {overlays_dir}")

    # Extract network names from the overlay list
    network_names = [name for name, _ in overlays]

    # Regenerate report with 2vol overlays. Use the ICA report generator
    # (which knows about 2vol overlays); for task-based workflows this may
    # fail gracefully since there are no ICA outputs, but the overlay PNGs
    # are already written and available for the QC viewer.
    try:
        logger.info("Regenerating QC report with 2vol overlays...")
        qc.generate_mask_extraction_report(
            mask_extraction_dir=mask_extraction_dir,
            participant_id=participant_id,
            network_names=network_names,
            output_dir=qc_dir,
            n_yeo=n_yeo,
            overwrite=False,  # Skip existing images, just update HTML
            open_browser=False,
        )
    except Exception as e:
        logger.info(f"Skipping ICA QC report regeneration (task-based workflow): {e}")
