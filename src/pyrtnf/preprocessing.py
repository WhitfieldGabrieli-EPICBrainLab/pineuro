"""
Real-time fMRI preprocessing pipeline for neurofeedback research.

This module provides core preprocessing functions for real-time fMRI neurofeedback studies, including DICOM conversion, motion correction, spatial smoothing, temporal filtering, and multi-run coregistration. The preprocessing pipeline creates standardized output structures organized by processing step.

Key Functions
-------------
dicom_to_nifti : Convert DICOM files to NIfTI format
motion_correction : Apply motion correction using FSL's mcflirt
spatial_smoothing : Apply Gaussian spatial smoothing
temporal_filtering : Apply temporal bandpass filtering
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
- 03_smoothed : Spatially smoothed data (optional)
- 04_filtered : Temporally filtered data (optional)
- 05_coregistered : Coregistered and concatenated runs

References
----------
[1] Jenkinson, M., Beckmann, C. F., Behrens, T. E., Woolrich, M. W., & Smith, S. M. (2012). FSL. NeuroImage, 62(2), 782-790.

"""

# Standard library imports
from typing import Union, List, Optional
from pathlib import Path
import shutil

# Third-party imports
from nibabel.loadsave import load

# Local imports
from pyrtnf.utils import run_command, compute_xfm, apply_xfm, compute_mean

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

def dicom_to_nifti(
        dicom_dir: Union[str, Path],
        output_dir: Union[str, Path],
        output_name: str
) -> Path:
    """
    Convert DICOM files to NIfTI format using dcm2niix.

    Parameters
    ----------
    dicom_dir : str or Path
        Directory containing DICOM files to convert.
    output_dir : str or Path
        Directory where the NIfTI file will be saved. Will be created if it doesn't exist
    output_name : str
        Base name for output file (without extension)

    Returns
    -------
    nifti_path : Path
        Path to the created NIfTI file

    """
    
    # Build command
    cmd = ['dcm2niix', '-z', 'y', '-o', str(output_dir), '-f', str(output_name), str(dicom_dir)]

    # Set output filename
    nifti_path = Path(output_dir) /f"{output_name}.nii.gz"

    # Run dcm2niix
    run_command(
        cmd_name='dcm2niix',
        cmd=cmd,
        description='Converting DICOM to NIfTI',
        inputs=dicom_dir,
        outputs=nifti_path)

    return nifti_path

def motion_correction(
        input_file: Union[str, Path],
        output_file: Union[str, Path]
) -> None:
    """
    Perform motion correction on fMRI data using FSL's mcflirt.

    Parameters
    ----------
    input_file : str or Path
        Path to input 4D NIfTI file containing fMRI timeseries
    output_file : str or Path
        Path for output motion-corrected NIfTI file. Parent directory will be created if it doesn't exist

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
        outputs=output_file
    )

def concat(
        input_files: List[Union[str, Path]],
        output_file: Union[str, Path]
) -> None:
    """
    Concatenate a set of fMRI data.

    Parameters
    ----------
    input_files : List of str or Path
        List of path to input NIfTI files
    output_file: str or Path
        Path to save concatenated data

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
        outputs=output_file
    )

def spatial_smoothing(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        fwhm: float = 5.0
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
        outputs=output_file
    )

def temporal_filtering(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
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
        outputs=output_file
    )

def brain_extraction(
        input_file: Union[str, Path],
        output_dir: Union[str, Path]
) -> tuple[Path, Path]:
    """
    Perform brain extraction using FSL's bet.

    Parameters
    ----------
    input_file : str or Path
        Path to input NIfTI file
    output_dir : str or Path
        Directory where the NIfTI file will be saved. Will be created if it doesn't exist

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
        outputs=[brain_file, mask_file]
    )

    return brain_file, mask_file

def preprocessing(
        participant_id: str,
        task: str,
        root_dir: Union[str, Path],
        input_dicoms: Union[str, Path, List[Union[str, Path]]],
        smooth_fwhm: Optional[float] = None,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None
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
    smooth_fwhm : float, optional
        FWHM of Gaussian smoothing kernel in mm. If None, no smoothing is applied
    highpass : float, optional
        High-pass filter cutoff in Hz. If None, no high-pass filtering is applied
    lowpass : float, optional
        Low-pass filter cutoff in Hz. If None, no low-pass filtering is applied

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
    
    # Convert to Path objects and validate
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
            output_name=f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}"
        )
        nifti_files.append(nifti_file)

    # Step 2: Motion correction
    mcf_dir = root_dir / "02_motion_correction"
    
    # Step 2a: Perform motion correction
    mcf_files = []
    for run, nifti in enumerate(nifti_files):
        mcf_file = mcf_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mcf.nii.gz")
        motion_correction(input_file=nifti, output_file=mcf_file)
        mcf_files.append(mcf_file)

    # Step 2b: Compute mean of motion corrected data
    mean_files = []
    for run, nifti in enumerate(mcf_files):
        mean_file = mcf_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean.nii.gz")
        compute_mean(input_file=nifti, output_file=mean_file)
        mean_files.append(mean_file)

    # Step 2c: Skull-strip first run, to use as mask
    _, mean_mask = brain_extraction(input_file=mean_files[0], output_dir=mcf_dir)

    # Step 3: Spatial smoothing
    if smooth_fwhm is not None:
        smooth_dir = root_dir / "03_smoothed"
        smooth_files = []
        for run, mcf_file in enumerate(mcf_files):
            smooth_file = smooth_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_smooth.nii.gz")
            spatial_smoothing(input_file=mcf_file, output_file=smooth_file, fwhm=smooth_fwhm)
            smooth_files.append(smooth_file)
        current_files = smooth_files
    else:
        current_files = mcf_files

    # Step 4: Temporal filtering
    if highpass is not None or lowpass is not None:
        filter_dir = root_dir / "04_filtered"
        filter_files = []
        for run, current_file in enumerate(current_files):
            filter_file = filter_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_filtered.nii.gz")
            temporal_filtering(input_file=current_file, output_file=filter_file, highpass=highpass, lowpass=lowpass)
            filter_files.append(filter_file)
        current_files = filter_files

    # Step 5: Co-registration across runs (if more than one)
    coreg_dir = root_dir / "05_coregistered"
    coreg_file = coreg_dir / Path(f"{participant_id}_task-{task}.nii.gz")
    xfm_dir = coreg_dir / "xfm"

    if len(current_files) == 1:
        
        # Just copy
        sub_ref = mean_files[0]
        coreg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(current_files[0], coreg_file)
    
    else:
        
        # Step 5a: Co-register to first run's mean
        sub_ref = mean_files[0]
        concat_files = []
        for run, mean_file in enumerate(mean_files):
            
            if run == 0:

                # Store first run for concatenation
                concat_files.append(current_files[run])
            
            if run > 0:
                
                # Compute transformation using the means
                output_name = xfm_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_mean_to_run-01_mean")
                _, xfm_file = compute_xfm(input_file=mean_file, reference_file=sub_ref, output_name=output_name, cost='corratio', dof=6)

                # Apply transformation
                output_file = coreg_dir / Path(f"{participant_id}_task-{task}_run-{str(run+1).zfill(2)}_to_run-01.nii.gz")
                apply_xfm(input_file=current_files[run], reference_file=sub_ref, xfm_file=xfm_file, output_file=output_file, interp='nearestneighbour')
                concat_files.append(output_file)

        # Step 5b: Concatenate
        # TODO: Handle incomplete sequences 
        concat(input_files=concat_files, output_file=coreg_file)

    # Copy final file to preprocessed file
    preproc_file = root_dir / Path(f"{participant_id}_task-{task}_preproc.nii.gz")
    shutil.copy(coreg_file, preproc_file)

    # Copy mask 
    mask_file = root_dir / Path(f"{participant_id}_task-{task}_preproc_mask.nii.gz")
    shutil.copy(mean_mask, mask_file)

    # Copy BOLD reference
    ref_file = root_dir / Path(f"{participant_id}_task-{task}_boldref.nii.gz")
    shutil.copy(sub_ref, ref_file)
    
    # Return preprocessed file
    return preproc_file, mask_file, ref_file
