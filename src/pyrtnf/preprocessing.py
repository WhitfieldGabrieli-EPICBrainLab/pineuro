"""
Real-time fMRI preprocessing pipeline for neurofeedback research.

This module provides core preprocessing functions for real-time fMRI neurofeedback studies, including DICOM conversion, motion correction, brain extraction, and multi-run coregistration. The preprocessing pipeline creates standardized output structures organized by processing step.

Key Functions
-------------
dicom_to_nifti : Convert DICOM files to NIfTI format
motion_correction : Apply motion correction using FSL's mcflirt
brain_extraction : Perform brain extraction using FSL's bet
preprocessing : Complete preprocessing pipeline 

Dependencies
------------
- FSL[1]
- nibabel
- numpy

Notes
-----
The preprocessing pipeline creates a standardized directory structure:
- 01_nifti : Converted NIfTI files
- 02_motion_correction : Motion-corrected data
- 03_brain_extraction : Means and brain extracted data
- 04_coregistered : Coregistered and concatenated runs

References
----------
[1] Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. NeuroImage, 62(2), 782-790.

"""

# Standard library imports
from typing import Union, List, Optional
from pathlib import Path
import shutil
import copy

# Third-party imports
import numpy as np
import nibabel as nib
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from nilearn.image import smooth_img
from scipy.signal import butter, filtfilt

# Local imports
from pyrtnf.utils import run_command, dicom_to_nifti, compute_xfm, apply_xfm, compute_mean, apply_mask

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

def motion_correction(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Perform motion correction on fMRI data using FSL's mcflirt.

    Parameters
    ----------
    input_file : str or Path
        Path to input 4D NIfTI file containing fMRI timeseries
    output_file : str or Path
        Path for output motion-corrected NIfTI file. Parent directory will be created if it doesn't exist
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """

    # Build mcflirt command
    cmd = [
        'mcflirt',
        '-in', str(input_file),
        '-out', str(output_file)
    ]

    # Run mcflirt
    run_command(
        cmd_name='mcflirt',
        cmd=cmd,
        description='Motion correction',
        inputs=input_file,
        outputs=output_file,
        overwrite=overwrite
    )

def concat(
        input_files: List[Union[str, Path]],
        output_file: Union[str, Path],
        overwrite: bool = False
) -> None:
    """
    Concatenate a set of fMRI data along the time axis.

    Parameters
    ----------
    input_files : List of str or Path
        List of path to input NIfTI files
    output_file: str or Path
        Path to save concatenated data
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping concatenation.")
        return
    
    logger.info("Concatenating...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all images and concatenate
    images = [nib.load(str(f)) for f in input_files]
    
    # Get data arrays and concatenate along time axis
    data_arrays = [img.get_fdata() for img in images]
    concatenated = np.concatenate(data_arrays, axis=-1)
    
    # Use affine and header from first image
    first_img = images[0]
    out_img = Nifti1Image(concatenated, first_img.affine, first_img.header)
    
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")

def spatial_smoothing(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        fwhm: float = 5.0,
        overwrite: bool = False
) -> None:
    """
    Apply Gaussian spatial smoothing to fMRI data using nilearn.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_file : str or Path
        Path for output smoothed NIfTI file. Parent directory will be created if it doesn't exist
    fwhm : float, default=5.0
        Full width at half maximum (FWHM) of the Gaussian smoothing kernel in mm
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping spatial smoothing.")
        return
    
    logger.info("Spatial smoothing...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use nilearn for smoothing
    smoothed = smooth_img(str(input_file), fwhm=fwhm)
    smoothed.to_filename(str(output_path))
    logger.info(f"Output: {output_path}")

def temporal_filtering(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        overwrite: bool = False
) -> None:
    """
    Apply temporal bandpass filtering to fMRI data using scipy Butterworth filter.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_file : str or Path
        Path for output filtered NIfTI file. Parent directory will be created if it doesn't exist
    highpass : float, optional
        High-pass filter cutoff in Hz. If None, no high-pass filtering is applied
    lowpass : float, optional
        Low-pass filter cutoff in Hz. If None, no low-pass filtering is applied
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    output_path = Path(output_file)
    
    # Check if output exists
    if not overwrite and output_path.exists():
        logger.info(f"Output file exists and overwrite=False. Skipping temporal filtering.")
        return
    
    # Load image
    img = nib.load(str(input_file))
    data = img.get_fdata()
    
    # Get TR from header (sampling rate)
    tr = img.header.get_zooms()[3]  # type: ignore
    fs = 1.0 / tr  # Sampling frequency in Hz
    nyquist = fs / 2.0
    
    logger.info(f"Temporal filtering (TR={tr}s, fs={fs:.3f}Hz)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine filter type and cutoffs
    if highpass is not None and lowpass is not None:
        # Bandpass filter
        btype = 'bandpass'
        Wn = [highpass / nyquist, lowpass / nyquist]
    elif highpass is not None:
        # High-pass filter
        btype = 'highpass'
        Wn = highpass / nyquist
    elif lowpass is not None:
        # Low-pass filter  
        btype = 'lowpass'
        Wn = lowpass / nyquist
    else:
        # No filtering requested
        logger.info("No filter cutoffs specified. Saving unchanged.")
        save(img, output_path)
        return
    
    # Design Butterworth filter (order 2 to match FSL behavior)
    b, a = butter(2, Wn, btype=btype)
    
    # Apply filter along time axis (axis=-1)
    # Reshape to 2D for filtering, then reshape back
    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])
    
    # Apply zero-phase filtering
    filtered_2d = filtfilt(b, a, data_2d, axis=-1)
    filtered = filtered_2d.reshape(original_shape)
    
    # Save output
    out_img = Nifti1Image(filtered, img.affine, img.header)
    save(out_img, output_path)
    logger.info(f"Output: {output_path}")

def brain_extraction(
        input_file: Union[str, Path],
        output_dir: Union[str, Path],
        overwrite: bool = False
) -> tuple[Path, Path]:
    """
    Perform brain extraction using FSL's bet.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_dir : str or Path
        Directory where the NIfTI file will be saved. Will be created if it doesn't exist
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    
    # Construct the expected outputs
    output_name = Path(input_file).with_suffix("").stem
    brain_file = Path(output_dir) / f"{output_name}_brain.nii.gz"
    mask_file = Path(output_dir) / f"{output_name}_brain_mask.nii.gz"
    
    # Build bet command
    cmd = [
        'bet',
        str(input_file),
        str(brain_file),
        "-m"
    ]

    # Run bet
    run_command(
        cmd_name='bet',
        cmd=cmd,
        description='Brain extraction',
        inputs=input_file,
        outputs=[brain_file, mask_file],
        overwrite=overwrite
    )

    return brain_file, mask_file

def preprocessing(
        participant_id: str,
        task: str,
        root_dir: Union[str, Path],
        input_dicoms: Union[str, Path, List[Union[str, Path]]],
        overwrite: bool = False
) -> tuple[Path, Path, Path]:
    """
    Run the preprocessing pipeline for fMRI data (task or rest).

    Parameters
    ----------
    participant_id : str
        Participant ID
    task : str
        Task name (e.g., rest)
    root_dir : str or Path
        Path to the root directory for all outputs
    input_dicoms : str or Path or list of str or Path
        Path to a single DICOM folder, or list of paths to multiple files
    overwrite : bool, default=False
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    Returns
    -------
    preproc_file : Path
        Preprocessed file
    mask_file : Path
        Mask file
    ref_file : Path
        BOLD reference file

    Raises
    ------
    FileNotFoundError
        If any folder in input_dicoms does not exist or is empty

    """

    # Log the command
    logger.info(f"Running preprocessing for {participant_id}...")
    
    # Convert to Path object
    root_dir = Path(root_dir)
    
    # Convert inputs to list if single file provided
    if isinstance(input_dicoms, (str, Path)):
        input_dicoms = [input_dicoms]
    
    # Convert to strings and validate
    input_paths: list[str] = []
    logger.info(f"Input DICOM folders:\n")
    for f in input_dicoms:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"Folder not found: {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")
        if not any(path.iterdir()):
            raise FileNotFoundError(f"Directory is empty: {path}")
        input_paths.append(str(path))
        logger.info(f"  {str(path)}")
    
    # Step 1: Convert DICOMs to NIfTI
    logger.info("Step 1/4: Converting DICOMs to NIfTI...")
    nifti_dir = root_dir / "01_nifti"
    nifti_files = []
    for run, path in enumerate(input_paths):
        logger.info(f"  Processing run {run+1}/{len(input_paths)}...")
        nifti_file = dicom_to_nifti(
            dicom_dir=path,
            output_dir=nifti_dir,
            output_name=f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}",
            overwrite=overwrite
        )
        nifti_files.append(nifti_file)
    logger.info(f"✓ DICOM to NIfTI conversion complete ({len(nifti_files)} runs)")

    # Step 2: Motion correction
    logger.info("Step 2/4: Running motion correction...")
    mcf_dir = root_dir / "02_motion_correction"
    mcf_files = []
    for run, nifti in enumerate(nifti_files):
        logger.info(f"  Processing run {run+1}/{len(nifti_files)}...")
        mcf_file = mcf_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mcf.nii.gz")
        motion_correction(input_file=nifti, output_file=mcf_file, overwrite=overwrite)
        mcf_files.append(mcf_file)
    logger.info(f"✓ Motion correction complete ({len(mcf_files)} runs)")

    # Step 3: Brain extract
    logger.info("Step 3/4: Computing temporal means and brain extraction...")
    bet_dir = root_dir / "03_brain_extraction"

    # Step 3a: Compute mean of motion corrected data
    mean_files = []
    for run, nifti in enumerate(mcf_files):
        mean_file = bet_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean.nii.gz")
        compute_mean(input_file=nifti, output_file=mean_file, overwrite=overwrite)
        mean_files.append(mean_file)
    
    # Step 3b: Skull-strip first run
    logger.info(f"  Extracting brain mask from run 1...")
    sub_ref = mean_files[0]
    _, mean_mask = brain_extraction(input_file=sub_ref, output_dir=bet_dir, overwrite=overwrite)
    logger.info(f"✓ Brain extraction complete")

    # Step 4: Co-registration across runs (if more than one)
    logger.info("Step 4/4: Coregistering runs and applying masks...")
    coreg_dir = root_dir / "04_coregistered"
    coreg_file = coreg_dir / Path(f"{participant_id}_task-{task}.nii.gz")
    xfm_dir = coreg_dir / "xfm"

    if len(mcf_files) == 1:
        logger.info(f"  Single run detected, skipping coregistration")
        # Just use the first run
        coreg_file = mcf_files[0]
    
    else:
        logger.info(f"  Coregistering {len(mcf_files)} runs to run 1...")
        # Step 4a: Co-register to first run's mean
        sub_ref = mean_files[0]
        concat_files = []
        for run, mean_file in enumerate(mean_files):
            
            if run == 0:

                # Store first run for concatenation
                concat_files.append(mcf_files[run])
            
            if run > 0:
                logger.info(f"    Coregistering run {run+1}/{len(mean_files)}...")
                # Compute transformation using the means
                output_name = xfm_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean_to_run-01_mean")
                _, xfm_file = compute_xfm(input_file=mean_file, reference_file=sub_ref, output_name=output_name, cost='corratio', dof=6, overwrite=overwrite)

                # Apply transformation
                output_file = coreg_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_to_run-01.nii.gz")
                apply_xfm(input_file=mcf_files[run], reference_file=sub_ref, xfm_file=xfm_file, output_file=output_file, interp='nearestneighbour', overwrite=overwrite)
                concat_files.append(output_file)

        # Step 5b: Concatenate
        # Current behavior: Concatenates all available runs sequentially
        # Incomplete sequences: If a run fails preprocessing, it should be excluded
        # from concat_files prior to this step (currently assumes all runs succeed)
        logger.info(f"  Concatenating {len(concat_files)} runs...")
        concat(input_files=concat_files, output_file=coreg_file, overwrite=overwrite)

    # Copy mask 
    logger.info(f"  Applying brain mask to outputs...")
    mask_file = root_dir / Path(f"{participant_id}_task-{task}_preproc_mask.nii.gz")
    if not mask_file.exists() or overwrite:
        shutil.copy(mean_mask, mask_file)

    # Apply mask to preprocessed file
    preproc_file = root_dir / Path(f"{participant_id}_task-{task}_preproc.nii.gz")
    if not preproc_file.exists() or overwrite:
        apply_mask(input_file=coreg_file, mask_file=mask_file, output_file=preproc_file, overwrite=overwrite)

    # Apply mask to BOLD reference
    ref_file = root_dir / Path(f"{participant_id}_task-{task}_boldref.nii.gz")
    if not ref_file.exists() or overwrite:
        apply_mask(input_file=sub_ref, mask_file=mask_file, output_file=ref_file, overwrite=overwrite)
    
    logger.info(f"✓ Preprocessing complete!")
    logger.info(f"  Preprocessed file: {preproc_file.name}")
    logger.info(f"  Brain mask: {mask_file.name}")
    logger.info(f"  BOLD reference: {ref_file.name}")
    
    # Return preprocessed file
    return preproc_file, mask_file, ref_file
