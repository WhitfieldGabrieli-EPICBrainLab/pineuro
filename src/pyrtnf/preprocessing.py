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
from nibabel.loadsave import load

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
    overwrite : bool, default=True
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
    Concatenate a set of fMRI data.

    Parameters
    ----------
    input_files : List of str or Path
        List of path to input NIfTI files
    output_file: str or Path
        Path to save concatenated data
    overwrite : bool, default=True
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    
    # Build fslmerge command
    cmd: List[str] = [
        'fslmerge', '-t',
        str(output_file),
    ] + [str(f) for f in input_files]

    # Run fslmerge
    run_command(
        cmd_name='fslmerge',
        cmd=cmd,
        description='Concatenating',
        inputs=input_files, 
        outputs=output_file,
        overwrite=overwrite
    )

def spatial_smoothing(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        fwhm: float = 5.0,
        overwrite: bool = False
) -> None:
    """
    Apply Gaussian spatial smoothing to fMRI data using FSL's fslmaths.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_file : str or Path
        Path for output smoothed NIfTI file. Parent directory will be created if it doesn't exist
    fwhm : float, default=5.0
        Full width at half maximum (FWHM) of the Gaussian smoothing kernel in mm
    overwrite : bool, default=True
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """

    # Build fslmaths command
    cmd = [
        'fslmaths',
        str(input_file),
        '-s', str(fwhm / 2.355),  # Convert FWHM to sigma (FWHM = 2.355 * sigma)
        str(output_file)
    ]

    # Run fslmaths
    run_command(
        cmd_name='fslmaths',
        cmd=cmd,
        description='Spatial smoothing',
        inputs=input_file,
        outputs=output_file,
        overwrite=overwrite
    )

def temporal_filtering(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
        overwrite: bool = False
) -> None:
    """
    Apply temporal bandpass filtering to fMRI data using FSL's fslmaths.

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
    overwrite : bool, default=True
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """

    # Load TR from header
    img = load(input_file)
    tr = img.header.get_zooms()[3] # type: ignore
    logger.info(f"Using TR from header: {tr}s")

    # Build fslmaths command
    cmd = ['fslmaths', str(input_file)]
    
    # Add high-pass filter
    if highpass is not None:
        sigma_hp = 1 / (2 * highpass * tr)  # Convert Hz to sigma in volumes
        cmd.extend(['-bptf', str(sigma_hp), '-1'])
    
    # Add low-pass filter
    if lowpass is not None:
        sigma_lp = 1 / (2 * lowpass * tr)
        if highpass is None:
            cmd.extend(['-bptf', '-1', str(sigma_lp)])
        else:
            # Replace -1 with actual lowpass value
            cmd[-1] = str(sigma_lp)
    
    cmd.append(str(output_file))

    # Run fslmaths
    run_command(
        cmd_name='fslmaths',
        cmd=cmd,
        description='Temporal filtering',
        inputs=input_file,
        outputs=output_file,
        overwrite=overwrite
    )

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
    overwrite : bool, default=True
        If True, always run command and overwrite existing outputs.
        If False, skip command if all output files already exist.

    """
    
    # Construct the expected outputs
    output_name = Path(input_file).with_suffix("").stem
    brain_file = Path(output_dir) /f"{output_name}_brain.nii.gz"
    mask_file = Path(output_dir) /f"{output_name}_brain_mask.nii.gz"
    
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
    overwrite : bool, default=True
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
    nifti_dir = root_dir / "01_nifti"
    nifti_files = []
    for run, path in enumerate(input_paths):
        nifti_file = dicom_to_nifti(
            dicom_dir=path,
            output_dir=nifti_dir,
            output_name=f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}",
            overwrite=overwrite
        )
        nifti_files.append(nifti_file)

    # Step 2: Motion correction
    mcf_dir = root_dir / "02_motion_correction"
    mcf_files = []
    for run, nifti in enumerate(nifti_files):
        mcf_file = mcf_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mcf.nii.gz")
        motion_correction(input_file=nifti, output_file=mcf_file, overwrite=overwrite)
        mcf_files.append(mcf_file)

    # Step 3: Brain extract
    bet_dir = root_dir / "03_brain_extraction"

    # Step 3a: Compute mean of motion corrected data
    mean_files = []
    for run, nifti in enumerate(mcf_files):
        mean_file = bet_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean.nii.gz")
        compute_mean(input_file=nifti, output_file=mean_file, overwrite=overwrite)
        mean_files.append(mean_file)
    
    # Step 3b: Skull-strip first run
    sub_ref = mean_files[0]
    _, mean_mask = brain_extraction(input_file=sub_ref, output_dir=bet_dir, overwrite=overwrite)

    # Step 4: Co-registration across runs (if more than one)
    coreg_dir = root_dir / "04_coregistered"
    coreg_file = coreg_dir / Path(f"{participant_id}_task-{task}.nii.gz")
    xfm_dir = coreg_dir / "xfm"

    if len(mcf_files) == 1:

        # Just use the first run
        coreg_file = mcf_files[0]
    
    else:
        
        # Step 4a: Co-register to first run's mean
        sub_ref = mean_files[0]
        concat_files = []
        for run, mean_file in enumerate(mean_files):
            
            if run == 0:

                # Store first run for concatenation
                concat_files.append(mcf_files[run])
            
            if run > 0:
                
                # Compute transformation using the means
                output_name = xfm_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean_to_run-01_mean")
                _, xfm_file = compute_xfm(input_file=mean_file, reference_file=sub_ref, output_name=output_name, cost='corratio', dof=6, overwrite=overwrite)

                # Apply transformation
                output_file = coreg_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_to_run-01.nii.gz")
                apply_xfm(input_file=mcf_files[run], reference_file=sub_ref, xfm_file=xfm_file, output_file=output_file, interp='nearestneighbour', overwrite=overwrite)
                concat_files.append(output_file)

        # Step 5b: Concatenate
        # TODO: Handle incomplete sequences 
        concat(input_files=concat_files, output_file=coreg_file, overwrite=overwrite)

    # Copy mask 
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
    
    # Return preprocessed file
    return preproc_file, mask_file, ref_file
