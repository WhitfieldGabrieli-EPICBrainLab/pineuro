"""
Functional network mask extraction pipeline for real-time fMRI neurofeedback.

Summary
-------
Extract participant-specific network masks from resting-state fMRI using ICA
and spatial correlation with reference atlases.

Functions
---------
- run_ica: Canonical ICA decomposition on fMRI data
- spatial_correlation: Correlate ICA components with parcellations
- mask_extraction_rest: End-to-end mask extraction workflow

Workflow
--------
- Run ICA on resting-state data within a mask
- Register reference atlas to participant space
- Correlate components with atlas networks and select targets
- Threshold and save network masks for feedback

Dependencies
------------
nilearn, ANTsPy, nibabel, numpy, pandas, scipy

Notes
-----
Outputs follow a standardized layout:
- 01_ica : ICA components and metadata
- 02_mnireg : MNI registrations and transforms
- 03_masks : Extracted and thresholded network masks

This expects FSL and nilearn installed and available; external commands are
invoked via the utilities in this package.
Further reading: see docs/references.md for reference list.

References
----------
[1] Varoquaux, G., Sadaghiani, S., Pinel, P., Kleinschmidt, A., Poline, J. B., 
    & Thirion, B. (2010). A group model for stable multi-subject ICA on fMRI 
    datasets. NeuroImage, 51(1), 288-299.
[2] Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, 
    S. M. (2012). FSL. NeuroImage, 62(2), 782-790.
[3] Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D., 
    Hollinshead, M., ... & Buckner, R. L. (2011). The organization of the 
    human cerebral cortex estimated by intrinsic functional connectivity. 
    Journal of Neurophysiology, 106(3), 1125-1165.
"""

# Standard library imports
import json
import re
from dataclasses import dataclass, asdict
from typing import Union, Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from nilearn.decomposition import CanICA
from nilearn import datasets
from nilearn.image import get_data, resample_to_img
import nibabel as nib
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from scipy.stats import pearsonr

# Local imports
from pineuro import qc
from pineuro.utils import (
    compute_mean, compute_xfm, compute_xfm_to_mni, apply_xfm,
    get_yeo_labels, dicom_to_nifti, apply_xfm_sform_to_mni, Timer
)

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)


@dataclass
class MaskSelection:
    """Tracks the selection state for a single network mask.

    Attributes
    ----------
    network_name : str
        Name of the network (e.g., "DefaultA", "ContA")
    source_type : str
        Either "ica" for ICA-derived mask or "yeo_atlas" for atlas fallback
    component_index : int or None
        ICA component index if source_type == "ica", None for atlas masks
    correlation : float
        Spatial correlation with Yeo atlas (0.0 for atlas masks)
    selection_reason : str
        How this selection was made: "auto", "manual", or "yeo_fallback"
    """
    network_name: str
    source_type: str  # "ica" or "yeo_atlas"
    component_index: Optional[int]
    correlation: float
    selection_reason: str  # "auto", "manual", "yeo_fallback"


def save_mask_selections(
    masks_dir: Union[str, Path],
    selections: Dict[str, MaskSelection]
) -> Path:
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

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved mask selections to {output_file}")
    return output_file


def load_mask_selections(masks_dir: Union[str, Path]) -> Dict[str, MaskSelection]:
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

    with open(selections_file, 'r') as f:
        data = json.load(f)

    return {
        name: MaskSelection(**sel_data)
        for name, sel_data in data.items()
    }


def get_alternative_components(
    correlation_df: pd.DataFrame,
    network_name: str,
    top_n: int = 5
) -> List[Tuple[int, float]]:
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


def extract_yeo_mask_for_network(
    yeo_subject_space: Union[str, Path],
    network_name: str,
    output_file: Union[str, Path],
    n_yeo: int = 17,
    threshold: int = 2000,
    overwrite: bool = False
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
    overwrite: bool = False
) -> Tuple[Path, Path]:
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
        logger.info(f"Hemisphere masks exist and overwrite=False, skipping")
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
        f"Split and thresholded {input_mask.name}: "
        f"{n_left} left voxels, {n_right} right voxels (threshold={threshold})"
    )

    # Save
    output_left.parent.mkdir(parents=True, exist_ok=True)
    output_right.parent.mkdir(parents=True, exist_ok=True)
    save(Nifti1Image(left_thresholded, affine, mask_img.header), output_left)
    save(Nifti1Image(right_thresholded, affine, mask_img.header), output_right)

    logger.info(f"Saved left hemisphere mask to {output_left}")
    logger.info(f"Saved right hemisphere mask to {output_right}")

    return output_left, output_right


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
            overwrite=True
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
            raise ValueError(
                f"Component index {component_idx} out of range "
                f"(max: {components_data.shape[3] - 1})"
            )

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

    else:
        raise ValueError(f"Unknown source_type: {selection.source_type}")

    return thr_file


def run_ica(
        input_file: Union[str, Path],
        mask_file: Union[str, Path],
        output_dir: Union[str, Path],
        output_name: str,
        n_components: int = 20,
        smooth_fwhm: Optional[float] = None,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        overwrite: bool = False
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
    n_components : int, default = 20
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

    logger.info(f"CanICA...")

    output_dir = Path(output_dir)
    components_file = output_dir / f"{output_name}_components.nii.gz"
    metadata_file = output_dir / f"{output_name}_metadata.npz"

    outputs = [components_file, metadata_file]
    if not overwrite:
        if all(p.exists() for p in outputs):
            logger.info(f"All output files exist and overwrite=False. Skipping CanICA.")
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

    logger.info(f"Output file:\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  {str(components_file)}")

    logger.info("Parameters:")
    logger.info(f"  n_components: {n_components}")

    img = load(input_file)
    # Extract TR from 4th dimension of zooms (temporal spacing)
    # type: ignore - nibabel typing doesn't recognize header.get_zooms()[3]
    tr = img.header.get_zooms()[3] # type: ignore

    # Initialize CanICA
    logger.info("Initializing CanICA...")
    canica: Any = CanICA(
        mask=mask_file,
        n_components=n_components,
        smoothing_fwhm=smooth_fwhm, # type: ignore - nilearn accepts None or float
        high_pass=highpass,
        low_pass=lowpass,
        threshold=None,  # type: ignore - nilearn accepts None or float
        t_r=tr,
        n_jobs=-1, # Use all CPUs
        random_state=0, # For reproducibility
        memory=str(output_dir / "cache"),
        memory_level=2,
        verbose=1
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
        metadata_file,
        n_components=n_components,
        components_shape=components_img.shape,
        input_file=str(input_file)
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
        labels: List,
        output_file: Union[str, Path],
        overwrite: bool = False
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
    logger.info(f"Computing spatial correlation...")
    
    # Convert to Path objects
    input_file = Path(input_file)
    mask_file = Path(mask_file)
    parcellation_file = Path(parcellation_file)
    output_file = Path(output_file)

    # Check if outputs exist and skip if overwrite=False
    if not overwrite:
        if output_file.exists():
            logger.info(f"Output file exists and overwrite=False. Skipping spatial correlations.")
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
    img_parcellation_rs = resample_to_img(img_parcellation, img, interpolation='nearest')
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
        Y = np.stack([(data_parcellation == j+1).astype(float) for j in range(len(labels))])

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

def mask_extraction_rest(
        participant_id: str,
        root_dir: Union[str, Path],
        input_file: Union[str, Path],
        mask_file: Union[str, Path],
        ref_file: Union[str, Path],
        n_yeo: int = 17,
        yeo_targets : Union[str, List[str]] = ["DefaultA", "ContA"],
        n_components: int = 20,
        threshold: int = 2000,
        smooth_fwhm: Optional[float] = None,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        overwrite: bool = False,
        qc_enabled: bool = True,
        qc_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None,
        sham_hemisphere_network: Optional[str] = None
) -> List[Path]:
    """
    Run the mask extraction pipeline for resting-state fMRI data.

    This function extracts participant-specific functional network masks using
    ICA decomposition and spatial correlation with Yeo atlas networks. The
    approach is preferred over using MNI masks directly as it accounts for
    individual anatomical and functional variability. Fallback to MNI masks
    may be implemented in future versions for cases where participant data
    quality is insufficient for reliable ICA decomposition.

    Parameters
    ----------
    participant_id : str
        Participant ID
    task : str
        Task name (e.g., rest)
    root_dir : str or Path
        Path to the root directory for all outputs
    input_file : str or Path
        Path to NIfTI file to extract masks from
    mask_file : str or Path
        Path to NIfTI file to mask input_file
    ref_file : str or Path
        Path to NIfTI file to use as a BOLD reference in participant space
    n_yeo : int, default=17
        Yeo network parcellation to use. Options: 7 or 17
    yeo_targets : str or List of str
        Yeo network targets to extract masks for
    n_components : int, default=20
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
        If n_yeo or yeo_targets parameter is invalid

    """

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
    
    # Validate n_networks
    if n_yeo not in [7, 17]:
        raise ValueError(f"n_networks must be 7 or 17, got {n_yeo}")
    
    # Convert to list and validate
    if isinstance(yeo_targets, str):
        yeo_targets = [yeo_targets]
    labels = get_yeo_labels(n_networks=n_yeo)
    for target in yeo_targets:
        if target not in labels:
            raise ValueError(f"Label {target} not in labels for Yeo {n_yeo} network parcellation")
    
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
        overwrite=overwrite
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
    # Uses ANTs rigid+affine registration between BOLD reference volumes
    output_prefix = mnireg_dir / Path(f"{ref_file.with_suffix('').stem}_to_MNI_boldref")
    mni_ref_file, sub2mni_xfm = compute_xfm_to_mni(
        input_file=ref_file,
        output_prefix=output_prefix,
        overwrite=overwrite
    )

    emit_progress("Step 2/3", f"Transforming Yeo {n_yeo}-network atlas to participant space...")
    yeo = datasets.fetch_atlas_yeo_2011()[f"thick_{str(n_yeo)}"]
    yeo_mni = Path(yeo).with_suffix('').parent / Path(Path(yeo).with_suffix('').stem + '_interp_to_fMRIPrep_boldref.nii.gz')

    yeo2sub = mnireg_dir / Path(f"Yeo_to_{ref_file.with_suffix('').stem}.nii.gz")
    if not yeo_mni.exists() or overwrite:
        apply_xfm_sform_to_mni(input_file=yeo, output_file=yeo_mni, interp='nearestneighbour', overwrite=overwrite)

    if not yeo2sub.exists() or overwrite:
        # Step 2c: Transform Yeo atlas from MNI space to participant space
        # Inverts the participant→MNI transformation computed in Step 2a
        # Uses nearest neighbor interpolation to preserve integer network labels
        apply_xfm(
            moving=yeo_mni,
            fixed=ref_file,
            transform_files=sub2mni_xfm,
            output_file=yeo2sub,
            interpolator='nearestNeighbor',
            invert=True,  # Apply inverse transform: MNI → participant
            overwrite=overwrite
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
    sc_file = masks_dir / Path(f"Yeo_IC_spatial_correlation.csv")
    sc_df = spatial_correlation(input_file=components_file, mask_file=mask_file, parcellation_file=yeo2sub, labels=labels, output_file=sc_file, overwrite=overwrite)
    chosen_components = {target: int(sc_df[target].idxmax()) for target in yeo_targets}

    # Save initial mask selections (auto-selected based on correlation)
    initial_selections: Dict[str, MaskSelection] = {}
    for target in yeo_targets:
        component_idx = chosen_components[target]
        correlation = float(sc_df[target].iloc[component_idx])
        initial_selections[target] = MaskSelection(
            network_name=target,
            source_type="ica",
            component_index=component_idx,
            correlation=correlation,
            selection_reason="auto"
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
            ic_data = components_data[..., ic_ind] # type: ignore
            ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header) # type: ignore
            save(ic_img, network_file)
        else:
            logger.info(f"Output file exists and overwrite=False. Skipping network extraction.")
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
            logger.info(f"Output file exists and overwrite=False. Skipping network thresholding.")
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
                    f"Yeo {n_yeo} labels. Skipping hemisphere split."
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
            split_mask_by_hemisphere(
                network_file, left_file, right_file,
                threshold=threshold, overwrite=overwrite
            )
            emit_progress(
                "Hemisphere split",
                f"Created {left_file.name} and {right_file.name} ({threshold} voxels each)"
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
            n_yeo=n_yeo,
            overwrite=overwrite,
            open_browser=False
        )
        emit_progress("QC report", f"Saved to {report_path.name}")

    return network_thr_files

def coregister_2vol(
        participant_id: str,
        root_dir: Union[str, Path],
        input_dicom: Union[str, Path],
        network_files: Union[str, Path, Sequence[Union[str, Path]]],
        ref_file: Union[str, Path],
        overwrite: bool = False,
        run_id: Optional[int] = None,
        output_subdir: str = "01_2vol"
) -> List[Path]:
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
    logger.info(f"Input network files:\n")
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

    def _extract_run_number_from_path(path: Path) -> Optional[int]:
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
        overwrite=overwrite
    )

    # Step 3: Compute mean
    mean_file = twovol_dir / Path(f"{participant_id}_task-2vol_run-{run_str}_mean.nii.gz")
    compute_mean(input_file=nifti_file, output_file=mean_file, overwrite=overwrite)

    # Step 4: Compute transformation (ANTs, ~28x faster than FLIRT)
    xfm_dir = twovol_dir / "xfm"
    output_prefix = xfm_dir / Path(f"{participant_id}_boldref_to_task-2vol_run-{run_str}_mean")
    _, xfm_files = compute_xfm(moving=ref_file, fixed=mean_file, output_prefix=output_prefix, type_of_transform='Rigid', overwrite=overwrite)

    # Step 5: Apply transformation
    mask_dir = twovol_dir / "masks"
    network_coreg_files = []
    for network in network_paths:
        output_file = mask_dir / (network.with_suffix("").stem + f"_to_task-2vol_run-{run_str}_mean.nii.gz")
        apply_xfm(moving=network, fixed=mean_file, transform_files=xfm_files, output_file=output_file, interpolator='nearestNeighbor', overwrite=overwrite)
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

    yeo_sub = mask_extraction_dir / "02_mnireg" / f"Yeo_to_{ref_file.with_suffix('').stem}.nii.gz"
    if not yeo_sub.exists():
        logger.info(f"Skipping 2vol QC overlays; atlas not found at {yeo_sub}")
        return

    atlas_2vol = twovol_dir / "masks" / f"Yeo_to_task-2vol_run-{run_str}_mean.nii.gz"
    apply_xfm(
        moving=yeo_sub,
        fixed=mean_file,
        transform_files=xfm_files,
        output_file=atlas_2vol,
        interpolator='nearestNeighbor',
        overwrite=overwrite
    )

    atlas_data = load(atlas_2vol).get_fdata()
    n_labels = int(np.max(atlas_data))
    labels = get_yeo_labels(n_networks=17 if n_labels > 7 else 7)

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
            atlas_path=atlas_2vol,
            labels=labels,
            target_label=str(target),
            top_n_per_plane=1,
            mask_path=network  # Use mask for slice selection
        )
        overlay_png = overlays_dir / f"{stem}_overlay.png"
        qc.plot_mask_overlay_tripanel(
            mask_path=network,
            ref_path=mean_file,
            out_path=overlay_png,
            slices=atlas_slices,
            title=f"{target} (2vol run {run_str})"
        )
        overlays.append((target, overlay_png))

    # Regenerate QC report to include 2vol overlays
    logger.info(f"Generated {len(overlays)} 2vol overlays in {overlays_dir}")
    logger.info("Regenerating QC report with 2vol overlays...")

    # Extract network names from the overlay list
    network_names = [name for name, _ in overlays]

    # Regenerate report with 2vol overlays and open in browser
    qc.generate_mask_extraction_report(
        mask_extraction_dir=mask_extraction_dir,
        participant_id=participant_id,
        network_names=network_names,
        output_dir=qc_dir,
        n_yeo=n_yeo,
        overwrite=True,  # Force regeneration to include new 2vol overlays
        open_browser=False
    )
