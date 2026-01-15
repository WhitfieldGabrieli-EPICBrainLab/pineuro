"""
Functional network mask extraction pipeline for real-time fMRI neurofeedback.

This module provides functions for extracting participant-specific functional network masks from resting-state fMRI data using Independent Component Analysis (ICA) and spatial correlation with reference brain networks. The pipeline identifies network-specific brain regions for use as neurofeedback targets.

Key Functions
-------------
run_ica : Perform Canonical ICA decomposition on fMRI data
spatial_correlation : Compute spatial correlation between components and parcellations
mask_extraction_rest : Complete mask extraction pipeline for resting-state data

Dependencies
------------
- nilearn[1]
- FSL[2]
- nibabel
- numpy
- pandas
- scipy

Notes
-----
The mask extraction pipeline creates a standardized directory structure:
- 01_ica : ICA components and metadata
- 02_mnireg : MNI space registrations and transformations
- 03_masks : Extracted and thresholded network masks

The pipeline registers reference atlases (e.g., Yeo networks) from MNI152 space
to participant space, performs ICA decomposition, and identifies components
corresponding to target functional networks through spatial correlation.

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
from typing import Union, Any, List, Optional, Sequence
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from nilearn.decomposition import CanICA
from nilearn import datasets
from nilearn.image import get_data, resample_to_img
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from scipy.stats import pearsonr

# Local imports
from pyrtnf.utils import compute_mean, compute_xfm, compute_xfm_to_mni, invert_xfm, apply_xfm, get_yeo_labels, dicom_to_nifti, apply_xfm_sform_to_mni

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

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

    # Extract the TR
    img = load(input_file)
    tr = img.header.get_zooms()[3] # type: ignore

    # Initialize CanICA
    logger.info("Initializing CanICA...")
    canica: Any = CanICA(
        mask=mask_file,
        n_components=n_components,
        smoothing_fwhm=smooth_fwhm, # type: ignore
        high_pass=highpass,
        low_pass=lowpass,
        threshold=None,  # type: ignore
        t_r=tr,
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
    correlation_matrix: np.ndarray = np.zeros((n, len(labels)))
    for j, net_id in enumerate(labels):
        
        # Extract and demean
        y = (data_parcellation == j+1).astype(float)
        y_dm = y - y.mean()
        y_norm = np.sqrt((y_dm**2).sum())

        # Numerator
        num = X_dm.T @ y_dm

        # Denominator
        den = x_norm * y_norm

        # Compute the correlation
        corr = num / den
        correlation_matrix[:, j] = corr

    # Create DataFrame and save
    logger.info(f"Saving correlations to {output_file}")
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
        yeo_targets : Union[str, List[str]] = ["DefaultA", "ContA"],
        n_components: int = 20,
        threshold: int = 2000,
        smooth_fwhm: Optional[float] = None,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        overwrite: bool = False
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

    # Step 2: MNI registration
    mnireg_dir = root_dir / "02_mnireg"

    # Step 2a: Register fMRI BOLD reference to MNI BOLD reference
    output_name = mnireg_dir / Path(f"{ref_file.with_suffix('').stem}_to_MNI_boldref")
    _, sub2mni_xfm = compute_xfm_to_mni(input_file=ref_file, output_name=output_name, cost='corratio', dof=6, overwrite=overwrite)

    # Step 2c: Invert transformation
    mni2sub_xfm = mnireg_dir / Path(f"MNI_boldref_to_{ref_file.with_suffix('').stem}.mat")
    invert_xfm(input_file=sub2mni_xfm, output_file=mni2sub_xfm, overwrite=overwrite)

    
    yeo2sub = mnireg_dir / Path(f"Yeo_to_{ref_file.with_suffix('').stem}.nii.gz")
    if not yeo2sub.exists() or overwrite:
        
        # Step 2d: Interpolate the Yeo parcellation into MNI space
        yeo = datasets.fetch_atlas_yeo_2011()[f"thick_{str(n_yeo)}"]
        yeo_mni = Path(yeo).with_suffix('').parent / Path(Path(yeo).with_suffix('').stem + '_interp_to_fMRIPrep_boldref.nii.gz')
        apply_xfm_sform_to_mni(input_file=yeo, output_file=yeo_mni, interp='nearestneighbour', overwrite=overwrite)
    
        # Step 2e: Register the Yeo parcellation to participant space
        apply_xfm(input_file=yeo_mni, reference_file=ref_file, xfm_file=mni2sub_xfm, output_file=yeo2sub, interp='nearestneighbour', overwrite=overwrite)

    # Step 3: Extract masks
    masks_dir = root_dir / "03_masks"
    
    # Step 3a: Spatial correlation
    sc_file = masks_dir / Path(f"Yeo_IC_spatial_correlation.csv")
    sc_df = spatial_correlation(input_file=components_file, mask_file=mask_file, parcellation_file=yeo2sub, labels=labels, output_file=sc_file, overwrite=overwrite)

    # Step 3b: Extract component with maximum correlation
    network_files = []
    for target in yeo_targets:
        network_file = masks_dir / Path(f"{participant_id}_{target}.nii.gz")
        if not network_file.exists() or overwrite:
            ic_ind = sc_df[target].idxmax()
            ic_data = components_data[..., ic_ind] # type: ignore
            ic_img = Nifti1Image(ic_data, components_img.affine, components_img.header) # type: ignore
            save(ic_img, network_file)
        else:
            logger.info(f"Output file exists and overwrite=False. Skipping network extraction.")
        network_files.append(network_file)

    # Step 3c: Threshold by number of voxels
    network_thr_files = []
    for network_file in network_files:
        thr_file = network_file.with_name(network_file.with_suffix("").stem + "_thr.nii.gz")
        if not thr_file.exists() or overwrite:
            img: SpatialImage = load(network_file)  # type: ignore
            data = img.get_fdata()
            flat = data.ravel()
            mask = np.zeros(flat.shape, dtype=bool)
            mask[np.argpartition(flat, -threshold)[-threshold:]] = True
            mask = mask.reshape(data.shape)
            thresholded = ((data * mask) > 0).astype("uint8")
            save(Nifti1Image(thresholded, img.affine, img.header), thr_file)  # type: ignore
        else:
            logger.info(f"Output file exists and overwrite=False. Skipping network thresholding.")
        network_thr_files.append(thr_file)
        
    return network_thr_files

def coregister_2vol(
        participant_id: str,
        root_dir: Union[str, Path],
        input_dicom: Union[str, Path],
        network_files: Union[str, Path, Sequence[Union[str, Path]]],
        ref_file: Union[str, Path],
        overwrite: bool = False
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

    # Step 1: Check for existing 2vol and get largest run number
    # Current behavior: Increments run number if overwrite=True
    # Future enhancement: Could track 2vol acquisitions by timestamp or session metadata
    # for more robust handling of multiple coregistration runs
    twovol_dir = root_dir / "01_2vol"
    last_run = 0
    if twovol_dir.exists() and overwrite:
        pattern = f"{participant_id}_task-2vol_run-*.nii.gz"
        matching_files = list(twovol_dir.glob(pattern))
        run_numbers = []
        for file in matching_files:
            for part in file.stem.split('_'):
                if part.startswith('run-'):
                    try:
                        run_numbers.append(int(part.split('-')[1]))
                    except ValueError:
                        pass
        if run_numbers:
            last_run = max(run_numbers)
    else:
        twovol_dir.mkdir(parents=True, exist_ok=True)
    next_run = last_run + 1

    # Step 2: Convert DICOMs to NIfTI
    nifti_file = dicom_to_nifti(
        dicom_dir=input_path,
        output_dir=twovol_dir,
        output_name=f"{participant_id}_task-2vol_run-{str(next_run).zfill(2)}",
        overwrite=overwrite
    )

    # Step 3: Compute mean
    mean_file = twovol_dir / Path(f"{participant_id}_task-2vol_run-{str(next_run).zfill(2)}_mean.nii.gz")
    compute_mean(input_file=nifti_file, output_file=mean_file, overwrite=overwrite)
    
    # Step 4: Compute transformation
    xfm_dir = twovol_dir / "xfm"
    output_name = xfm_dir / Path(f"{participant_id}_boldref_to_task-2vol_run-{str(next_run).zfill(2)}_mean")
    _, xfm_file = compute_xfm(input_file=ref_file, reference_file=mean_file, output_name=output_name, cost='corratio', dof=6, overwrite=overwrite)

    # Step 5: Apply transformation
    mask_dir = twovol_dir / "masks"
    network_coreg_files = []
    for network in network_paths:
        output_file = mask_dir / (network.with_suffix("").stem + f"_to_task-2vol_run-{str(next_run).zfill(2)}_mean.nii.gz")
        apply_xfm(input_file=network, reference_file=mean_file, xfm_file=xfm_file, output_file=output_file, interp='nearestneighbour', overwrite=overwrite)
        network_coreg_files.append(output_file)

    return network_coreg_files



