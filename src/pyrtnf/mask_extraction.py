"""


"""

# Standard library imports
from typing import Union, Any, List
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from nilearn.decomposition import CanICA
from nilearn import datasets
from nilearn.image import get_data, resample_to_img
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from scipy.stats import pearsonr

# Local imports
from pyrtnf.utils import compute_mean, compute_xfm_to_mni, invert_xfm, apply_xfm, get_yeo_labels

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

def run_ica(
        input_file: Union[str, Path],
        mask_file: Union[str, Path],
        output_dir: Union[str, Path],
        output_name: str,
        n_components: int = 30,
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
    n_components : int, default = 30
        Number of independent components to extract
    overwrite : bool, default=True
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

    # Log the command
    logger.info(f"CanICA...")

    # Convert to Path object
    output_dir = Path(output_dir)
    components_file = output_dir / f"{output_name}_components.nii.gz"
    metadata_file = output_dir / f"{output_name}_metadata.npz"
    
    outputs = [components_file, metadata_file]
    if not overwrite:
        if all(p.exists() for p in outputs):
            logger.info(f"All output files exist and overwrite=False. Skipping CanICA.")
            return components_file, metadata_file

    # Convert to Path object and validate
    input_file = Path(input_file)
    logger.info("Inputs:")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    logger.info(f"  {str(input_file)}")
    mask_file = Path(mask_file)
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    logger.info(f"  {str(mask_file)}")

    # Create output directory
    logger.info(f"Output file:\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  {str(components_file)}")

    # Log parameters
    logger.info("Parameters:")
    logger.info(f"  n_components: {n_components}")

    # Initialize CanICA
    logger.info("Initializing CanICA...")
    canica: Any = CanICA(
        mask=mask_file,
        n_components=n_components,
        smoothing_fwhm=None,
        n_jobs=-1, # Use all CPUs
        random_state=0, # For reproducibility
        memory=str(output_dir / "cache"),
        memory_level=2,
        verbose=1
    )

    # Fit CanICA
    logger.info("Computing ICA components... (this may take several minutes)")
    try:
        canica.fit(input_file)
        logger.info("CanICA computation completed successfully")
    except Exception as e:
        error_msg = f"CanICA computation failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # Get components as image
    components_img = canica.components_img_

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
        labels: List,
        output_file: Union[str, Path]

) -> pd.DataFrame:
    """
    Spatially coreralte a NIfTI file with a parcellation using FSL's fslcc.

    Parameters
    ----------
    input_file : str or Path
        Path to first 4D NIfTI file
    parcellation_file : str or Path
        Path to 3D NIfTI parcellation file
    labels : List
        List of parcellation labels
    output_file : str or Path
        Path to save correlation results as CSV file
    
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
    parcellation_file = Path(parcellation_file)
    output_file = Path(output_file)

    # Validate files
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not parcellation_file.exists():
        raise FileNotFoundError(f"File not found: {parcellation_file}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load files
    logger.info(f"Loading: {input_file}...")
    img = load(input_file)  
    data = get_data(img) 
    n = data.shape[3]
    logger.info(f"Loading: {parcellation_file}...")
    img_parcellation = load(parcellation_file)  
    
    # Resample (note: they should already be co-registered)
    logger.info("Resampling...")
    img_parcellation_rs = resample_to_img(img_parcellation, img, interpolation='nearest')
    data_parcellation = get_data(img_parcellation_rs)
    mask = data_parcellation > 0

    # Compute spatial correlations
    logger.info("Correlating...")
    correlation_matrix: np.ndarray = np.zeros((n, len(labels)))
    for idx in range(n):
        x = data[..., idx]
        for pdx, id in enumerate(labels):
            network_map = (data_parcellation == id).astype(float)
            corr, _ = pearsonr(x[mask], network_map[mask]) # type: ignore
            correlation_matrix[idx, pdx] = corr

    # Create DataFrame and save
    logger.info(f"Saving correlations to{output_file}")
    corr_df = pd.DataFrame(correlation_matrix, columns=labels)
    corr_df.to_csv(output_file)

    return corr_df

def mask_extraction_rest(
        participant_id: str,
        root_dir: Union[str, Path],
        input_file: Union[str, Path],
        mask_file: Union[str, Path],
        ref_file: Union[str, Path],
        n_yeo: int = 17,
        yeo_targets : Union[str, List[str]] = ["DefaultA", "ContA"]
) -> List[Path]:
    """
    Run the mask extraction pipeline for resting-state fMRI data.

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

    Raises
    ------
    FileNotFoundError
        If input_file does not exist
    ValueError
        If n_yeo or yeo_targets parameter is invalid

    """

    # Log the command
    logger.info(f"Extracting resting-state masks for {participant_id}...")
    
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
    ica_dir = root_dir / "01_ica"
    components_file, _ = run_ica(input_file=input_file, mask_file=mask_file, output_dir=ica_dir, output_name=input_file.with_suffix("").stem)
    components_img = load(components_file)
    components_data = get_data(components_img)

    # Step 2: MNI registration
    mnireg_dir = root_dir / "02_mnireg"

    # Step 2a: Register fMRI BOLD reference to MNI BOLD reference
    output_name = mnireg_dir / Path(f"{ref_file.with_suffix("").stem}_to_MNI_boldref")
    _, sub2mni_xfm = compute_xfm_to_mni(input_file=ref_file, output_name=output_name, cost='corratio', dof=6)

    # Step 2c: Invert transformation
    mni2sub_xfm = mnireg_dir / Path(f"MNI_boldref_to_{ref_file.with_suffix("").stem}.mat")
    invert_xfm(input_file=sub2mni_xfm, output_file=mni2sub_xfm)

    # Step 2d: Register the Yeo parcellation to participant space
    yeo = datasets.fetch_atlas_yeo_2011()[f"thick_{str(n_yeo)}"]
    yeo2sub = mnireg_dir / Path(f"{Path(yeo).with_suffix("").stem}_to_{ref_file.with_suffix("").stem}.nii.gz")
    apply_xfm(input_file=yeo, reference_file=ref_file, xfm_file=mni2sub_xfm, output_file=yeo2sub, interp='nearestneighbour')

    # Step 3: Extract masks
    masks_dir = root_dir / "03_masks"
    
    # Step 3a: Spatial correlation
    sc_file = masks_dir / Path(f"{Path(yeo).with_suffix("").stem}_IC_spatial_correlation.csv")
    sc_df = spatial_correlation(input_file=components_file, parcellation_file=yeo2sub, labels=labels, output_file=sc_file)

    # Step 3b: Extract component with maximum correlation
    mask_files = []
    for target in yeo_targets:
        mask_file = masks_dir / Path(f"{participant_id}_{target}.nii.gz")
        ic_ind = sc_df[target].idmax()
        ic_data = components_data[..., ic_ind]
        ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header) # type: ignore
        save(ic_img, mask_file)
        mask_files.append(mask_file)

    return mask_files

    

    







